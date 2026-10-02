from __future__ import annotations

import gc
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import Trainer, TrainingArguments

from common import (
    BLOCK_SIZE,
    DATA_DIR,
    FIGURES_DIR,
    MODEL_ID,
    MODELS_DIR,
    RESULTS_DIR,
    ROOT,
    SEED,
    PackedTextDataset,
    chinese_character_rate,
    compute_ppl,
    corpus_stats,
    ensure_dirs,
    load_model,
    load_tokenizer,
    read_jsonl,
    seed_everything,
    summarize_generation_rows,
    tokenization_stats,
    write_json,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_BLOCKS = 128


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_splits() -> dict[str, list[str]]:
    required = ["zh_train", "zh_dev", "zh_test", "en_test", "en_rehearsal", "pt_train", "pt_dev", "pt_test"]
    missing = [name for name in required if not (DATA_DIR / f"{name}.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed splits {missing}; run src/prepare_data.py first")
    return {name: read_jsonl(DATA_DIR / f"{name}.jsonl") for name in required}


def generation_experiment(model, tokenizer) -> tuple[list[dict], list[dict]]:
    prompts = {
        "classic": "Once upon a time, a quiet fox found a silver key beside the river.",
        "dialogue": 'Mia looked at the tiny dragon and asked, "Why are you hiding?"',
        "constraint": "Write a simple story about a patient robot that must not use magic.",
        "counterfactual": "In a village where shadows shine at noon, a child lost her blue kite.",
    }
    settings = {
        "greedy": {"do_sample": False},
        "beam4": {"do_sample": False, "num_beams": 4, "early_stopping": True},
        "low_temp": {"do_sample": True, "temperature": 0.45, "top_k": 50},
        "top_k50": {"do_sample": True, "temperature": 0.8, "top_k": 50},
        "nucleus_p90": {"do_sample": True, "temperature": 0.8, "top_p": 0.90, "top_k": 0},
        "high_temp_p95": {"do_sample": True, "temperature": 1.15, "top_p": 0.95, "top_k": 0},
    }
    rows: list[dict] = []
    model.eval()
    for prompt_name, prompt in prompts.items():
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        prompt_length = encoded["input_ids"].shape[1]
        for setting_name, kwargs in settings.items():
            for replicate in range(3):
                seed = SEED + replicate + 100 * list(prompts).index(prompt_name)
                torch.manual_seed(seed)
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=128,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **kwargs,
                    )[0]
                new_ids = output[prompt_length:]
                completion = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                rows.append(
                    {
                        "prompt_type": prompt_name,
                        "prompt": prompt,
                        "setting": setting_name,
                        "replicate": replicate,
                        "seed": seed,
                        "completion": completion,
                        "new_tokens": int(len(new_ids)),
                        "ended_with_eos": bool(len(new_ids) and new_ids[-1].item() == tokenizer.eos_token_id),
                    }
                )
    summary = summarize_generation_rows(rows)
    save_jsonl(RESULTS_DIR / "generation_samples.jsonl", rows)
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "generation_summary.csv", index=False)
    return rows, summary


def make_eval_sets(splits, tokenizer) -> dict[str, PackedTextDataset]:
    return {
        "zh_dev": PackedTextDataset(splits["zh_dev"], tokenizer, max_blocks=EVAL_BLOCKS),
        "zh_test": PackedTextDataset(splits["zh_test"], tokenizer, max_blocks=EVAL_BLOCKS),
        "en_test": PackedTextDataset(splits["en_test"], tokenizer, max_blocks=EVAL_BLOCKS),
        "pt_test": PackedTextDataset(splits["pt_test"], tokenizer, max_blocks=EVAL_BLOCKS),
    }


def evaluate_model(model, eval_sets, label: str) -> list[dict]:
    rows = []
    for split_name, dataset in eval_sets.items():
        metrics = compute_ppl(model, dataset, batch_size=16, device=DEVICE)
        rows.append({"model": label, "split": split_name, **metrics})
        print(label, split_name, metrics)
    return rows


def training_args(output_dir: Path, lr: float, steps: int, eval_steps: int) -> TrainingArguments:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        max_steps=steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.06 * steps)),
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        use_cpu=not torch.cuda.is_available(),
    )


def train_once(
    name: str,
    train_texts: list[str],
    dev_dataset: PackedTextDataset,
    tokenizer,
    lr: float,
    steps: int,
    save_model: bool,
) -> tuple[object, list[dict], float]:
    seed_everything()
    model = load_model(DEVICE)
    model.config.use_cache = False
    train_dataset = PackedTextDataset(
        train_texts,
        tokenizer,
        block_size=BLOCK_SIZE,
        max_blocks=max(steps * 16, 4096),
        seed=SEED,
        shuffle_blocks=True,
    )
    args = training_args(MODELS_DIR / f"_trainer_{name}", lr, steps, max(20, steps // 4))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
    )
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    logs = [{"variant": name, **row} for row in trainer.state.log_history]
    logs.append({"variant": name, "elapsed_seconds": elapsed, "train_loss_final": result.training_loss})
    if save_model:
        output = MODELS_DIR / name
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
    return model, logs, elapsed


def release_model(model) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_results(generation_summary: list[dict], training_logs: list[dict], ppl_rows: list[dict]) -> None:
    sns.set_theme(style="whitegrid", context="paper")

    gen = pd.DataFrame(generation_summary).sort_values("distinct_2")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.scatter(gen["repeat_4"], gen["distinct_2"], s=75, color="#31688e")
    for _, row in gen.iterrows():
        ax.annotate(row["setting"], (row["repeat_4"], row["distinct_2"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Repeated 4-gram fraction (lower is better)", ylabel="Distinct-2 (higher is better)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "decoding_frontier.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "decoding_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    logs = pd.DataFrame(training_logs)
    loss = logs.dropna(subset=["loss"]).copy() if "loss" in logs else pd.DataFrame()
    eval_loss = logs.dropna(subset=["eval_loss"]).copy() if "eval_loss" in logs else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    if not loss.empty:
        sns.lineplot(data=loss, x="step", y="loss", hue="variant", ax=ax, linewidth=1.6)
    if not eval_loss.empty:
        for variant, group in eval_loss.groupby("variant"):
            ax.scatter(group["step"], group["eval_loss"], marker="D", s=24, label=f"{variant} dev")
    ax.set(xlabel="Optimizer step", ylabel="Cross-entropy loss", title="Continual pre-training dynamics")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    ppl = pd.DataFrame(ppl_rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    sns.barplot(data=ppl, x="model", y="ppl", hue="split", ax=ax, palette="colorblind")
    ax.set_yscale("log")
    ax.set(xlabel="Checkpoint", ylabel="Perplexity (log scale)")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perplexity_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "perplexity_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def chinese_generation_probe(model, tokenizer, label: str) -> list[dict]:
    prompts = ["从前，有一只胆小的小熊", "小雨问机器人：\"你为什么难过？\"", "在没有太阳的村庄里"]
    rows = []
    model.eval()
    for index, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        torch.manual_seed(SEED + index)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]
        completion = tokenizer.decode(output[encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        rows.append(
            {
                "model": label,
                "prompt": prompt,
                "completion": completion,
                "chinese_character_rate": chinese_character_rate(completion),
            }
        )
    return rows


def main() -> None:
    ensure_dirs()
    seed_everything()
    splits = load_splits()
    tokenizer = load_tokenizer()
    eval_sets = make_eval_sets(splits, tokenizer)

    environment = {
        "model_id": MODEL_ID,
        "parameters": 41_689_600,
        "data_provenance": "provided story corpus",
        "seed": SEED,
        "block_size": BLOCK_SIZE,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    write_json(RESULTS_DIR / "environment.json", environment)

    stats = {name: corpus_stats(texts, tokenizer) for name, texts in splits.items()}
    stats["tokenization_efficiency"] = {
        "chinese": tokenization_stats(splits["zh_test"], tokenizer),
        "english": tokenization_stats(splits["en_test"], tokenizer),
    }
    write_json(RESULTS_DIR / "corpus_stats.json", stats)

    base = load_model(DEVICE)
    generation_rows, generation_summary = generation_experiment(base, tokenizer)
    ppl_rows = evaluate_model(base, eval_sets, "base")
    zh_probe_rows = chinese_generation_probe(base, tokenizer, "base")
    release_model(base)

    sweep_logs: list[dict] = []
    sweep_rows: list[dict] = []
    candidates = [5e-5, 2e-4, 8e-4]
    for lr in candidates:
        label = f"sweep_lr_{lr:.0e}"
        model, logs, elapsed = train_once(label, splits["zh_train"], eval_sets["zh_dev"], tokenizer, lr, 80, False)
        metrics = compute_ppl(model, eval_sets["zh_dev"], device=DEVICE)
        sweep_rows.append({"variant": label, "learning_rate": lr, "elapsed_seconds": elapsed, **metrics})
        sweep_logs.extend(logs)
        release_model(model)

    chosen = min(sweep_rows, key=lambda row: row["ppl"])
    chosen_lr = float(chosen["learning_rate"])

    final_steps = 1_200
    vanilla, vanilla_logs, vanilla_elapsed = train_once(
        "chinese_only", splits["zh_train"], eval_sets["zh_dev"], tokenizer, chosen_lr, final_steps, True
    )
    ppl_rows.extend(evaluate_model(vanilla, eval_sets, "Chinese-only"))
    zh_probe_rows.extend(chinese_generation_probe(vanilla, tokenizer, "Chinese-only"))
    release_model(vanilla)

    rehearsal_texts = list(splits["zh_train"]) + list(splits["en_rehearsal"][:1_000])
    random.Random(SEED).shuffle(rehearsal_texts)
    rehearsal, rehearsal_logs, rehearsal_elapsed = train_once(
        "rehearsal_zh_en", rehearsal_texts, eval_sets["zh_dev"], tokenizer, chosen_lr, final_steps, True
    )
    ppl_rows.extend(evaluate_model(rehearsal, eval_sets, "ZH+EN rehearsal"))
    zh_probe_rows.extend(chinese_generation_probe(rehearsal, tokenizer, "ZH+EN rehearsal"))
    release_model(rehearsal)

    training_logs = sweep_logs + vanilla_logs + rehearsal_logs
    save_jsonl(RESULTS_DIR / "training_log.jsonl", training_logs)
    pd.DataFrame(sweep_rows).to_csv(RESULTS_DIR / "lr_sweep.csv", index=False)
    pd.DataFrame(ppl_rows).to_csv(RESULTS_DIR / "perplexity.csv", index=False)
    pd.DataFrame(zh_probe_rows).to_csv(RESULTS_DIR / "chinese_generation_probe.csv", index=False)
    write_json(
        RESULTS_DIR / "selection.json",
        {
            "chosen_learning_rate": chosen_lr,
            "criterion": "lowest Chinese development perplexity after an equal 80-step budget",
            "final_steps": final_steps,
            "chinese_only_seconds": vanilla_elapsed,
            "rehearsal_seconds": rehearsal_elapsed,
        },
    )
    plot_results(generation_summary, training_logs, ppl_rows)
    print("Experiment complete. Chosen learning rate:", chosen_lr)


if __name__ == "__main__":
    main()

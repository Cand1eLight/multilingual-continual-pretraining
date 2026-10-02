from __future__ import annotations

import json
import random

import pandas as pd

from common import RESULTS_DIR, SEED, load_tokenizer, write_json
from run_experiments import (
    DEVICE,
    chinese_generation_probe,
    evaluate_model,
    load_splits,
    make_eval_sets,
    plot_results,
    release_model,
    save_jsonl,
    train_once,
)


FINAL_STEPS = 1_200


def main() -> None:
    splits = load_splits()
    tokenizer = load_tokenizer()
    eval_sets = make_eval_sets(splits, tokenizer)
    selection = json.loads((RESULTS_DIR / "selection.json").read_text(encoding="utf-8"))
    chosen_lr = float(selection["chosen_learning_rate"])

    old_ppl = pd.read_csv(RESULTS_DIR / "perplexity.csv")
    ppl_rows = old_ppl[old_ppl["model"] == "base"].to_dict("records")
    old_logs = []
    with (RESULTS_DIR / "training_log.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("variant", "")).startswith("sweep_"):
                old_logs.append(row)

    vanilla, vanilla_logs, vanilla_elapsed = train_once(
        "chinese_only", splits["zh_train"], eval_sets["zh_dev"], tokenizer, chosen_lr, FINAL_STEPS, True
    )
    ppl_rows.extend(evaluate_model(vanilla, eval_sets, "Chinese-only"))
    zh_probe_rows = chinese_generation_probe(vanilla, tokenizer, "Chinese-only")
    release_model(vanilla)

    rehearsal_texts = list(splits["zh_train"]) + list(splits["en_rehearsal"][:1_000])
    random.Random(SEED).shuffle(rehearsal_texts)
    rehearsal, rehearsal_logs, rehearsal_elapsed = train_once(
        "rehearsal_zh_en", rehearsal_texts, eval_sets["zh_dev"], tokenizer, chosen_lr, FINAL_STEPS, True
    )
    ppl_rows.extend(evaluate_model(rehearsal, eval_sets, "ZH+EN rehearsal"))
    zh_probe_rows.extend(chinese_generation_probe(rehearsal, tokenizer, "ZH+EN rehearsal"))
    release_model(rehearsal)

    logs = old_logs + vanilla_logs + rehearsal_logs
    save_jsonl(RESULTS_DIR / "training_log.jsonl", logs)
    pd.DataFrame(ppl_rows).to_csv(RESULTS_DIR / "perplexity.csv", index=False)
    pd.DataFrame(zh_probe_rows).to_csv(RESULTS_DIR / "chinese_generation_probe.csv", index=False)
    write_json(
        RESULTS_DIR / "selection.json",
        {
            "chosen_learning_rate": chosen_lr,
            "criterion": "lowest Chinese development perplexity after an equal 80-step budget",
            "final_steps": FINAL_STEPS,
            "chinese_only_seconds": vanilla_elapsed,
            "rehearsal_seconds": rehearsal_elapsed,
        },
    )
    generation_summary = pd.read_csv(RESULTS_DIR / "generation_summary.csv").to_dict("records")
    plot_results(generation_summary, logs, ppl_rows)
    print("Refined final models and refreshed all comparisons.")


if __name__ == "__main__":
    main()

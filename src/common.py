from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = Path(os.environ.get("CPT_RESULTS_DIR", ROOT / "runs" / "results"))
FIGURES_DIR = ROOT / "runs" / "figures"
MODELS_DIR = ROOT / "runs" / "models"
MODEL_ID = os.environ.get("CPT_MODEL_PATH", str(ROOT / "models" / "base"))
SEED = 7021
BLOCK_SIZE = 256


def ensure_dirs() -> None:
    for path in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = row["text"] if isinstance(row, dict) else str(row)
            if text and text.strip():
                texts.append(text.strip())
    return texts


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(device: str | None = None):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if device is not None:
        model.to(device)
    return model


class PackedTextDataset(Dataset):
    """Deterministically concatenate stories with EOS and return fixed token blocks."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        block_size: int = BLOCK_SIZE,
        max_blocks: int | None = None,
        seed: int = SEED,
        shuffle_blocks: bool = False,
    ) -> None:
        token_ids: list[int] = []
        eos = tokenizer.eos_token_id
        for start in range(0, len(texts), 128):
            batch = tokenizer(
                list(texts[start : start + 128]),
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            for ids in batch:
                token_ids.extend(ids)
                token_ids.append(eos)

        usable = len(token_ids) // block_size
        blocks = [token_ids[i * block_size : (i + 1) * block_size] for i in range(usable)]
        if shuffle_blocks:
            random.Random(seed).shuffle(blocks)
        if max_blocks is not None:
            blocks = blocks[:max_blocks]
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.blocks[index], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def compute_ppl(
    model,
    dataset: Dataset,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[str, float]:
    """Token-weighted perplexity over fixed blocks, excluding each block's first label."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output = model(**batch)
            valid_tokens = int(batch["attention_mask"][:, 1:].sum().item())
            total_nll += float(output.loss.item()) * valid_tokens
            total_tokens += valid_tokens
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "nll": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll < 80 else float("inf"),
        "tokens": total_tokens,
        "blocks": len(dataset),
    }


def tokenization_stats(texts: Sequence[str], tokenizer, limit: int = 500) -> dict[str, float]:
    subset = list(texts[:limit])
    chars = sum(len(x) for x in subset)
    token_count = sum(len(x) for x in tokenizer(subset, add_special_tokens=False)["input_ids"])
    return {
        "documents": len(subset),
        "characters": chars,
        "tokens": token_count,
        "tokens_per_character": token_count / max(chars, 1),
    }


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _word_tokens(text: str) -> list[str]:
    return [x.lower() for x in _WORD_RE.findall(text)]


def _ngram_repetition(tokens: Sequence[str], n: int = 4) -> float:
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def summarize_generation_rows(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(row)

    summaries: list[dict] = []
    for setting, items in grouped.items():
        token_lists = [_word_tokens(x["completion"]) for x in items]
        all_words = [w for seq in token_lists for w in seq]
        all_bigrams = [tuple(seq[i : i + 2]) for seq in token_lists for i in range(len(seq) - 1)]
        first_sentences = [re.split(r"[.!?]", x["completion"])[0].strip().lower() for x in items]
        summaries.append(
            {
                "setting": setting,
                "samples": len(items),
                "mean_words": float(np.mean([len(x) for x in token_lists])),
                "distinct_1": len(set(all_words)) / max(len(all_words), 1),
                "distinct_2": len(set(all_bigrams)) / max(len(all_bigrams), 1),
                "repeat_4": float(np.mean([_ngram_repetition(x, 4) for x in token_lists])),
                "eos_rate": float(np.mean([bool(x["ended_with_eos"]) for x in items])),
                "unique_opening_rate": len(set(first_sentences)) / max(len(first_sentences), 1),
            }
        )
    return summaries


def chinese_character_rate(text: str) -> float:
    visible = [c for c in text if not c.isspace()]
    chinese = [c for c in visible if "\u4e00" <= c <= "\u9fff"]
    return len(chinese) / max(len(visible), 1)


def corpus_stats(texts: Sequence[str], tokenizer) -> dict[str, float]:
    lengths = np.array([len(x) for x in texts], dtype=np.int64)
    token_lengths_list: list[int] = []
    for start in range(0, len(texts), 128):
        encoded = tokenizer(list(texts[start : start + 128]), add_special_tokens=False)["input_ids"]
        token_lengths_list.extend(len(ids) for ids in encoded)
    token_lengths = np.array(token_lengths_list, dtype=np.int64)
    return {
        "documents": len(texts),
        "characters": int(lengths.sum()),
        "median_characters": float(np.median(lengths)),
        "mean_tokens": float(token_lengths.mean()),
        "median_tokens": float(np.median(token_lengths)),
        "p95_tokens": float(np.percentile(token_lengths, 95)),
    }

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss

from common import MODEL_ID, RESULTS_DIR, ROOT, load_model, load_tokenizer, read_jsonl


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
BATCH_SIZE = 16


def document_perplexities(model, tokenizer, texts: list[str]) -> np.ndarray:
    """Assignment-compatible per-document PPL with stable float32 loss."""
    values: list[float] = []
    loss_fct = CrossEntropyLoss(reduction="none")
    model.eval()
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenizer(
            texts[start : start + BATCH_SIZE],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH - 1,
            return_tensors="pt",
        ).to(DEVICE)
        bos = torch.full(
            (batch.input_ids.shape[0], 1), tokenizer.bos_token_id,
            dtype=batch.input_ids.dtype, device=DEVICE,
        )
        input_ids = torch.cat([bos, batch.input_ids], dim=1)
        attention = torch.cat(
            [torch.ones_like(bos), batch.attention_mask], dim=1
        )
        with torch.inference_mode():
            logits = model(input_ids, attention_mask=attention).logits[..., :-1, :].float()
        labels = input_ids[..., 1:]
        mask = attention[..., 1:]
        losses = loss_fct(logits.transpose(1, 2), labels)
        nll = (losses * mask).sum(1) / mask.sum(1)
        values.extend(torch.exp(torch.clamp(nll, max=80)).cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    tokenizer = load_tokenizer()
    splits = {
        name: read_jsonl(ROOT / "data" / "processed" / f"{name}.jsonl")
        for name in ("en_test", "zh_test", "pt_test")
    }
    checkpoints = {
        "base": MODEL_ID,
        "Chinese-only": str(ROOT / "runs" / "models" / "chinese_only"),
        "ZH+EN rehearsal": str(ROOT / "runs" / "models" / "rehearsal_zh_en"),
    }
    rows = []
    for model_name, path in checkpoints.items():
        model = load_model(DEVICE) if model_name == "base" else __import__(
            "transformers"
        ).AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEVICE)
        for split_name, texts in splits.items():
            if split_name == "pt_test" and model_name != "base":
                continue
            ppls = document_perplexities(model, tokenizer, texts)
            rows.append({
                "model": model_name,
                "split": split_name,
                "documents": len(ppls),
                "max_length": MAX_LENGTH,
                "mean_document_ppl": float(ppls.mean()),
                "median_document_ppl": float(np.median(ppls)),
                "geometric_mean_ppl": float(math.exp(np.log(ppls).mean())),
                "p95_document_ppl": float(np.percentile(ppls, 95)),
            })
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "course_compute_ppl.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

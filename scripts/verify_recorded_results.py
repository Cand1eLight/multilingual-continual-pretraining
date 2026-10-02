"""Check saved PPL arithmetic and generation statistics with the standard library."""
import csv
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "recorded"


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError((a, b))


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    ppls = read_csv("perplexity.csv")
    assert len(ppls) == 12
    for row in ppls:
        close(math.exp(float(row["nll"])), row["ppl"])
        assert int(row["tokens"]) == 32640 and int(row["blocks"]) == 128
    samples = [json.loads(line) for line in (RESULTS / "generation_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(samples) == 72
    for saved in read_csv("generation_summary.csv"):
        rows = [row for row in samples if row["setting"] == saved["setting"]]
        assert len(rows) == int(saved["samples"]) == 12
        words = [[word.lower() for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", row["completion"])] for row in rows]
        flat = [word for sentence in words for word in sentence]
        bigrams = [tuple(sentence[i:i + 2]) for sentence in words for i in range(len(sentence) - 1)]
        repetitions = []
        for sentence in words:
            grams = [tuple(sentence[i:i + 4]) for i in range(max(0, len(sentence) - 3))]
            repetitions.append(1 - len(set(grams)) / len(grams) if grams else 0)
        openings = [re.split(r"[.!?]", row["completion"])[0].strip().lower() for row in rows]
        close(sum(map(len, words)) / len(words), saved["mean_words"])
        close(len(set(flat)) / len(flat), saved["distinct_1"])
        close(len(set(bigrams)) / len(bigrams), saved["distinct_2"])
        close(sum(repetitions) / len(repetitions), saved["repeat_4"])
        close(sum(row["ended_with_eos"] for row in rows) / len(rows), saved["eos_rate"])
        close(len(set(openings)) / len(rows), saved["unique_opening_rate"])
    documents = read_csv("course_compute_ppl.csv")
    assert len(documents) == 7
    assert all(int(row["documents"]) == 1000 and float(row["mean_document_ppl"]) > 0 for row in documents)
    print("Verified 12 fixed-token PPL rows and 72 generated continuations; 7 document-PPL rows have valid counts.")


if __name__ == "__main__":
    main()

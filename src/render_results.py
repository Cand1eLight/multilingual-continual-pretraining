from __future__ import annotations

import json

import pandas as pd

from common import RESULTS_DIR
from run_experiments import plot_results


def main() -> None:
    generation_summary = pd.read_csv(RESULTS_DIR / "generation_summary.csv").to_dict("records")
    with (RESULTS_DIR / "training_log.jsonl").open("r", encoding="utf-8") as handle:
        training_logs = [json.loads(line) for line in handle if line.strip()]
    ppl_rows = pd.read_csv(RESULTS_DIR / "perplexity.csv").to_dict("records")
    plot_results(generation_summary, training_logs, ppl_rows)
    print("Rendered all result figures.")


if __name__ == "__main__":
    main()

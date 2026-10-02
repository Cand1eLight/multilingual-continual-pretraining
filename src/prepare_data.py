"""Prepare the supplied story splits and an explicitly selected replay corpus."""
from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from common import DATA_DIR, ROOT, SEED, ensure_dirs, sha256, write_json

REPLAY_DATASET = "Gabrui/multilingual_TinyStories"
REPLAY_REVISION = "7fb5f327b907318dd95877a366e288f1971b64c1"
REPLAY_SEED = SEED + 3


def clean(text):
    return "\n".join(line.strip() for line in text.replace("\r", "").split("\n") if line.strip())


def stories(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}. See data/README.md.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for row in rows:
        text = row if isinstance(row, str) else row.get("text") or row.get("story") or row.get("content")
        if text and text.strip():
            output.append(clean(text))
    return output


def write_split(path, texts, source):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, text in enumerate(texts):
            stream.write(json.dumps({"id": index, "text": clean(text), "source": source}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--course-dir", type=Path, default=ROOT / "data" / "course")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rehearsal-file", type=Path)
    group.add_argument("--fetch-rehearsal", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    counts = {"zh_train": 10000, "zh_dev": 500, "zh_test": 1000, "en_test": 1000,
              "pt_train": 10000, "pt_dev": 500, "pt_test": 1000}
    data = {}
    for name, count in counts.items():
        raw = stories(args.course_dir / f"{name}.jsonl")
        if len(raw) < count:
            raise ValueError(f"{name}: expected at least {count} stories; got {len(raw)}")
        data[name] = raw[:count]
    if args.rehearsal_file:
        replay = stories(args.rehearsal_file)
    else:
        from datasets import load_dataset
        stream = load_dataset(REPLAY_DATASET, "english", split="train", streaming=True, revision=REPLAY_REVISION)
        replay = [clean(row["story"]) for row in islice(stream.shuffle(seed=REPLAY_SEED, buffer_size=50000), 1200)]
    if len(replay) < 1200:
        raise ValueError("The replay file must contain at least 1,200 stories.")
    data["en_rehearsal"] = replay[:1200]
    manifest = {"seed": SEED, "provenance": "supplied story corpus", "splits": {},
                "replay_dataset": REPLAY_DATASET, "replay_revision": REPLAY_REVISION}
    for name, texts in data.items():
        # Source labels are retained to preserve the recorded prepared-file hashes.
        source = f"{REPLAY_DATASET}/english" if name == "en_rehearsal" else "course-provided files under data/course"
        path = DATA_DIR / f"{name}.jsonl"
        write_split(path, texts, source)
        manifest["splits"][name] = {"documents": len(texts), "sha256": sha256(path), "path": path.relative_to(ROOT).as_posix()}
    write_json(DATA_DIR / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

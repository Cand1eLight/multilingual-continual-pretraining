# Input assets

The recorded experiment used a supplied 41,689,600-parameter checkpoint with
its unchanged tokenizer. Place its model files in `models/base/`, or set
`CPT_MODEL_PATH` to the checkpoint directory.

Place the supplied JSONL story files in `data/course/`:

| File | Stories used |
| --- | ---: |
| `zh_train.jsonl` | 10,000 |
| `zh_dev.jsonl` | 500 |
| `zh_test.jsonl` | 1,000 |
| `en_test.jsonl` | 1,000 |
| `pt_train.jsonl` | 10,000, retained as a source split |
| `pt_dev.jsonl` | 500, retained as a source split |
| `pt_test.jsonl` | 1,000 |

Each line can contain a story string or an object with `text`, `story`, or
`content`. Preprocessing strips empty lines and normalizes whitespace within
the same story. Source documents are not shuffled before evaluation.

The replay corpus is a separate English TinyStories training sample. Pass the
recorded JSONL file with `--rehearsal-file`; its prepared-file SHA256 appears in
`results/recorded/data_manifest.json`. The file contains 1,200 sampled stories;
the first 1,000 are used in the replay condition.

`--fetch-rehearsal` samples 1,200 stories from the fixed multilingual TinyStories
revision with seed 7024 and a 50,000-item shuffle buffer. Check the resulting
hash against the recorded manifest before treating it as an exact reproduction:
streaming order is an upstream dependency. A different replay file is valid for
a new experiment and should be reported with its new hash.

The public repository includes input metadata and hashes. It does not include
the teaching checkpoint or the raw story datasets.

### Table C - wall-clock cost model

Wall clock is arithmetic over terms that are either config-derived (exact, read from the merged config) or measured (with where the measurement came from). Both measured terms print as `NOT-RUN` until supplied -- this module will not guess a corpus length or a GPU throughput, because a wall-clock figure resting on a guessed denominator is precisely the kind of plausible number that reads as a measurement.

| config                                           | params trainable | epochs | seqs/step | tokens/step | train_seqs | steps   | sec/epoch | wall clock | basis   |
|--------------------------------------------------|------------------|--------|-----------|-------------|------------|---------|-----------|------------|---------|
| config/kaggle/data2vec_kaggle.yaml               | 126.4M           | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/kaggle/mlm_kaggle.yaml                    | 162.7M           | 30     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/kaggle/textspanjepa_kaggle.yaml           | 137.9M           | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/scaling/base_140m.yaml                    | 137.9M           | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/scaling/devices/dev1_bs512.yaml           | 139.1M           | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/scaling/devices/dev2_bs256.yaml           | 139.1M           | 50     | 256       | 131072      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/scaling/devices/dev4_bs128.yaml           | 139.1M           | 50     | 128       | 65536       | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/scaling/devices/dev8_bs64.yaml            | 139.1M           | 50     | 64        | 32768       | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/scaling/large_300m.yaml                   | 284.4M           | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/scaling/small_100m.yaml                   | 88.9M            | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/scaling/xsmall_30m.yaml                   | 32.3M            | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/tinystories/textspanjepa_tinystories.yaml | 88.9M            | 20     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/data2vec_wikitext_large.yaml     | 257.8M           | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/data2vec_wikitext_small.yaml     | 83.4M            | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/data2vec_wikitext_train.yaml     | 126.4M           | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/data2vec_wikitext_xsmall.yaml    | 30.8M            | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/mlm_wikitext_base.yaml           | 162.7M           | 30     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/mlm_wikitext_large.yaml          | 305.1M           | 30     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/mlm_wikitext_small.yaml          | 114.0M           | 30     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/mlm_wikitext_xsmall.yaml         | 49.5M            | 30     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/textspanjepa_wikitext_base.yaml  | 137.9M           | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |
| config/wikitext/textspanjepa_wikitext_small.yaml | 88.9M            | 50     | 512       | 262144      | NOT-RUN    | NOT-RUN | NOT-RUN   | NOT-RUN    | NOT-RUN |

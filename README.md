# SEAnet

SEAnet is a novel architecture especially designed for data series representation learning (DEA).

Codes were developed and tested under Linux environment.

## Train SEAnet

1. _**Compile Coconut Sampling**_

```bash
cd lib/
make
```

2. **Add a configuration file**

An example configuration for SEAnet is given in *conf/example.json*.
Two fields with *TO_BE_CHANGED* are required to get changed.

> **database_path**: indicates the dataset to be indexed \
> **query_path**: indicates the query set

Other fields could be left by default.
Please refer to *util/conf.py* for all possible configurations.

3. **Train SEAnet**

```bash
python run.py -C conf/example.json
```

## Approximate Similarity Search

The indexing and query answering of DEA is in https://github.com/qtwang/isax-modularized

## Cite this work

```latex
@inproceedings{kdd21-Wang-SEAnet,
  author    = {Wang, Qitong and 
               Palpanas, Themis},
  title     = {Deep Learning Embeddings for Data Series Similarity Search},
  booktitle = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Virtual Event, Singapore, August 14-18, 2021},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3447548.3467317},
  doi       = {10.1145/3447548.3467317},
  timestamp = {Thu, 05 Aug 2021 09:46:47 +0800}
}
```

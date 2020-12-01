# SEAnet

SEAnet is a novel architecture especially designed for data series representation learning (DEA).

Codes were developed and tested under Linux environment.

## Train SEAnet

1. _**Compile Coconut Sampling**_

```bash
cd lib/
make
```

2. _**Add a configuration file**_

An example configuration for SEAnet is given in *conf/example.json*.
Two fields with **TO_BE_CHANGED** are required to get changed.

> **database_path**: indicates the dataset to be indexed \
> **query_path**: indicates the query set

Other fields could be left by default.
Please refer to *util/conf.py* for all possible configurations.

3. _**Train SEAnet**_

```bash
python run.py -C conf/example.json
```

## Approximate Similarity Search on DEA

The indexing and query answering of DEA is in https://github.com/qtwang/isax-modularized.

## Cite this work

```latex
@online{web20-SEAnet,
  author = {Wang, Qitong and 
            Palpanas, Themis},
  title  = {SEAnet Codebase},
  url    = {https://github.com/qtwang/SEAnet},
  year   = {2020},
  timestamp = {Tue, 02 Dec 2020 01:59:59 +0100},
}
```

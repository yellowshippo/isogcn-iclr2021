# IsoGCN

The official implementation of IsoGCN, presented in the ICLR2021 paper [Isometric Transformation Invariant and Equivariant Graph Convolutional Networks](https://openreview.net/forum?id=FX0vR39SJ5q) [[arXiv](https://arxiv.org/abs/2005.06316)].

Please cite us as:

```
@inproceedings{
horie2021isometric,
title={Isometric Transformation Invariant and Equivariant Graph Convolutional Networks},
author={Masanobu Horie and Naoki Morita and Toshiaki Hishinuma and Yu Ihara and Naoto Mitsume},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=FX0vR39SJ5q}
}
```



## General notice

If some of the following steps not working, please modify `User settings` section in the `Makefile` to fit with your environment.



## Installation

You can either locally install locally or use Docker image. However, to generate an anisotropic nonlinear heat equation dataset, we recommend using Docker.

### Local install

We are using poetry, thus first install it following the instruction on https://python-poetry.org/docs/ . Then, update the `Makefile` to `PYTHON ?= 'poetry run python3'` or explicitly specify the `PYTHON` environment variable on the execution of the `make` command.

For GPU environment,

```
PYTHON='poetry run python3' make poetry
poetry install
PYTHON='poetry run python3' make install_pyg_gpu
```

For CPU environment,

```
PYTHON='poetry run python3' make poetry
poetry install
PYTHON='poetry run python3' make install_pyg_cpu
```

, and set `GPU_ID = -1` in the `Makefile`.

Also, optionally, please install [FrontISTR](https://github.com/FrontISTR/FrontISTR) and [gmsh](https://gitlab.onelab.info/gmsh/gmsh) to generate an anisotropic nonlinear heat equation dataset (which are installed in the Docker image). 



## Docker image

Please download the docker image via https://drive.google.com/file/d/1WDbdGdzlgo_vuaqo6Cj4kNju38HQyWCT/view?usp=sharing , then place the image in the `images` directory. After that, plsease `make in` to login the docker befor perfroming all the following processes.



## Differential operator dataset

### Data generation

```
make differential_data
```

### Training IsoGCN

```
make scalar2grad  # Scalar to gradient task
make scalar2grad  ADJ=5  # Scalar to gradient task with # hops = 5
make scalar2hessian  # Scalar to Hessian task
make grad2laplacian  # Gradient to Laplacian task
make grad2hessian  # Gradient to Hessian task
```

### Training baseline models

```
make scalar2grad_baseline BASELINE_NAME=gcn  # BASELINE_NAME=[cluster_gcn, gcn, gcnii, gin, sgcn]
```
Similarly, one can perform baseline model trainings for other tasks.



## Anisotropic nonlinear heat equation dataset

### Run whole process with small data to check the process (Optional)

It generates a small dataset to simulate the whole process of data generation, preprocessing, training, and inference. This process requires either FrontISTR installed locally or Docker image.

```
make small_heat_nl_tensor_pipeline
```

### Dataset download

The dataset containing finite element analysis results is generated from the [ABC dataset](https://deep-geometry.github.io/abc-dataset/) using [gmsh](https://gmsh.info/) for meshing and [FrontISTR](https://github.com/FrontISTR/FrontISTR) for analysis.

Please download the dataset you need. (Note: To perform only training, you need only 'preprocessed' data.) The dataset can be downloaded via:

* Raw data (FrontISTR analysis files)
  * Train dataset
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/raw/train_50.tar.gz
  * Validation dataset
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/raw/validation_16.tar.gz
  * Test dataset
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/raw/test_16.tar.gz
* Interim data (.npy and .npz files before standardization)
  * Train dataset (splitted due to its large data size)
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/train_50.tar.gz.partaa
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/train_50.tar.gz.partab
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/train_50.tar.gz.partac
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/train_50.tar.gz.partad
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/train_50.tar.gz.partae
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/train_50.tar.gz.partaf
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/train_50.tar.gz.partag
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/train_50.tar.gz.partah
  * Validation dataset
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/validation_16.tar.gz
  * Test dataset
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/interim/test_16.tar.gz
* Preprocessed data (.npy and .npz files after standardization)
  * Train dataset (splitted due to its large data size)
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/preprocessed/train_50.tar.gz.partaa
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/preprocessed/train_50.tar.gz.partab
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/preprocessed/train_50.tar.gz.partac
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/preprocessed/train_50.tar.gz.partad
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/preprocessed/train_50.tar.gz.partae
  * Validation dataset
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/preprocessed/validation_16.tar.gz
  * Test dataset
    * https://savanna.ritc.jp/~horiem/isogcn_iclr2021/data/heat_nl_tensor/preprocessed/test_16.tar.gz

After download finished, please merge the split archives with:

```
cat train_50.tar.gz.parta* > train.tar.gz
```

, extract them with `tar xvf *.tar.gz`, then place them in the corresponding `data/heat_nl_tensor/(raw|interim|preprocessed)` directory.

### Training IsoGCN

```
make heat_nl_tensor
```

### Training baseline models

```
make heat_nl_tensor_baseline BASELINE_NAME=gcn  # BASELINE_NAME=[cluster_gcn, gcn, gcnii, gin, sgcn]
```



## IsoGCN core implementation

The core implementation of the IsoGCN layer is separated in the library [SiML](https://github.com/ricosjp/siml) and can be found [here](https://github.com/ricosjp/siml/blob/master/siml/networks/iso_gcn.py). Also, the code to generate IsoAMs is separated in the library [Femio](https://github.com/ricosjp/femio) and can be found [here](https://github.com/ricosjp/femio/blob/ddda7ba18565e2ce52044be546d83f819a9cce27/femio/signal_processor.py#L807).



## License

[Apache License 2.0](./LICENSE).


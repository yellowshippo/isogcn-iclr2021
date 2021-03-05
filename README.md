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

We are using poetry, thus first install it following the instruction on https://python-poetry.org/docs/ .

For GPU environment,

```
make poetry
poetry install
make install_pyg_gpu
```

For CPU environment,

```
make poetry
poetry install
make install_pyg_cpu
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

Please download the preprocessed dataset via:

- Train dataset (splitted due to its large data size):
  - https://drive.google.com/file/d/1yBKWMTYjdbOEWLf-MAvcIKeuyY9riPtB/view?usp=sharing (5.0 GB)
  - https://drive.google.com/file/d/1p8TJkhSkGE02Yve8Ub35f8irKzB7TPrZ/view?usp=sharing (5.0 GB)
  - https://drive.google.com/file/d/1lYt02gGAxAoclIGE_SGf6EnspOF_5-fu/view?usp=sharing (5.0 GB)
  - https://drive.google.com/file/d/1VQaUagwpbbsY4qT_QUHM3Uv03k-GZ8lY/view?usp=sharing (5.0 GB)
  - https://drive.google.com/file/d/1vUFGrWsu0X7R8WK3n_7btn92WJbgxoRa/view?usp=sharing (5.0 GB)
- Validation dataset:
  - https://drive.google.com/file/d/1yhOQ8x3O02xU18ZtS74uRIfOybjvzs0J/view?usp=sharing (6.6 GB)
- Test dataset:
  - https://drive.google.com/file/d/1F7kPoUqxGmlkg9ROW65AQTrTgp8vGupr/view?usp=sharing (8.4 GB)

After download finished, please merge the training dataset archive with:

```
cat train_50.tar.gz.parta* > train.tar.gz
```

, extract them with `tar xvf *.tar.gz`, then place them in the `data/heat_nl_tensor/preprocessed` directory.

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


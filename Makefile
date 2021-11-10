.PHONY: clean data

# ===== User settings ===== #
PYTHON ?= python3.7  # Set 'python3.7' for the docker, 'poetry run python3' for local
RUN ?= poetry run
PIP ?= pip3
CUDA ?= cu101  # CUDA version 10.1
GPU_ID ?= 0  # Set -1 to use CPU

# Data generation setting
N_GRID_TRAIN ?= 100  # The number of training dataset size for the grid dataset
N_GRID_VALIDATION ?= 10  # The number of validation dataset size for the grid dataset
N_GRID_TEST ?= 10  # The number of test dataset size for the grid dataset


# Experiment setting
N_EPOCH ?= 1000
## The number of hops. It should be 2 or 5.
ADJ ?= 2
## Please set _w_node to include vertex positions in the input vertex features for baseline models
INPUT ?=
## Select from [cluster_gcn, gcn, gcnii, gin, sgcn]
BASELINE_NAME ?= gcn

# == End of User settings == #


RM = rm -rf
DOCKER_IMAGE  ?= isogcn


# Log in to the docker
in:
	docker load -i images/$(DOCKER_IMAGE).tar
	docker run -w /src -it --gpus all -v${PWD}:/src --rm $(DOCKER_IMAGE) /bin/bash

# Log in to the docker (CPU)
in_cpu:
	docker load -i images/$(DOCKER_IMAGE)
	docker run -w /src -it -v${PWD}:/src --rm $(DOCKER_IMAGE):$(IMAGE_TAG) /bin/bash


# Installation
## Install local libraries
requirements: poetry
	$(PYTHON) -m pip install pysiml==0.2.4
	touch requirements

## Install PyTorch geometric with CPU
install_pyg_cpu: poetry
	$(PYTHON) -m pip install torch==1.6.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html \
  && $(PYTHON) -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && $(PYTHON) -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && $(PYTHON) -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && $(PYTHON) -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && $(PYTHON) -m pip install torch-geometric

## Install PyTorch geometric with GPU
install_pyg_gpu: poetry
	$(PYTHON) -m pip install torch==1.6.0+$(CUDA) torchvision==0.7.0+$(CUDA) -f https://download.pytorch.org/whl/torch_stable.html \
  && $(PYTHON) -m pip install torch-scatter==2.0.5+$(CUDA) -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && $(PYTHON) -m pip install torch-sparse==0.6.7+$(CUDA) -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && $(PYTHON) -m pip install torch-cluster==1.5.7+$(CUDA) -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && $(PYTHON) -m pip install torch-spline-conv==1.2.0+$(CUDA) -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && $(PYTHON) -m pip install torch-geometric

poetry:
	$(PIP) install pip --upgrade
	poetry config virtualenvs.in-project true
	-$(PYTHON) -m pip uninstall -y femio siml pysiml
	touch poetry


# Differential operator dataset
scalar2grad: requirements
	$(PYTHON) src/train.py inputs/differential/scalar2grad/iso_gcn_adj$(ADJ).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/differential/scalar2grad/iso_gcn_adj$(ADJ) \
		data/differential/preprocessed/test \
		-p data/differential/preprocessed/preprocessors.pkl

scalar2grad_baseline: requirements
	$(PYTHON) src/train.py inputs/differential/scalar2grad/$(BASELINE_NAME)_adj$(ADJ)$(INPUT).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/differential/scalar2grad/$(BASELINE_NAME)_adj$(ADJ)$(INPUT) \
		data/differential/preprocessed/test \
		-p data/differential/preprocessed/preprocessors.pkl


scalar2hessian: requirements
	$(PYTHON) src/train.py inputs/differential/scalar2hessian/iso_gcn_adj$(ADJ).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/differential/scalar2hessian/iso_gcn_adj$(ADJ) \
		data/differential/preprocessed/test \
		-p data/differential/preprocessed/preprocessors.pkl

scalar2hessian_baseline: requirements
	$(PYTHON) src/train.py inputs/differential/scalar2hessian/$(BASELINE_NAME)_adj$(ADJ)$(INPUT).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/differential/scalar2hessian/$(BASELINE_NAME)_adj$(ADJ)$(INPUT) \
		data/differential/preprocessed/test \
		-p data/differential/preprocessed/preprocessors.pkl


grad2laplacian: requirements
	$(PYTHON) src/train.py inputs/differential/grad2laplacian/iso_gcn_adj$(ADJ).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/differential/grad2laplacian/iso_gcn_adj$(ADJ) \
		data/differential/preprocessed/test \
		-p data/differential/preprocessed/preprocessors.pkl

grad2laplacian_baseline: requirements
	$(PYTHON) src/train.py inputs/differential/grad2laplacian/$(BASELINE_NAME)_adj$(ADJ)$(INPUT).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/differential/grad2laplacian/$(BASELINE_NAME)_adj$(ADJ)$(INPUT) \
		data/differential/preprocessed/test \
		-p data/differential/preprocessed/preprocessors.pkl


grad2hessian: requirements
	$(PYTHON) src/train.py inputs/differential/grad2hessian/iso_gcn_adj$(ADJ).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/differential/grad2hessian/iso_gcn_adj$(ADJ) \
		data/differential/preprocessed/test \
		-p data/differential/preprocessed/preprocessors.pkl

grad2hessian_baseline: requirements
	$(PYTHON) src/train.py inputs/differential/grad2hessian/$(BASELINE_NAME)_adj$(ADJ)$(INPUT).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/differential/grad2hessian/$(BASELINE_NAME)_adj$(ADJ)$(INPUT) \
		data/differential/preprocessed/test \
		-p data/differential/preprocessed/preprocessors.pkl


## Data generation for the differential operator dataset
differential_data: requirements
	$(PYTHON) src/generate_grid_dataset.py data/differential/interim/train -n $(N_GRID_TRAIN) -s 1
	$(PYTHON) src/generate_grid_dataset.py data/differential/interim/validation -n $(N_GRID_VALIDATION) -s 2  # Change random seed
	$(PYTHON) src/generate_grid_dataset.py data/differential/interim/test -n $(N_GRID_TEST) -s 3  # Change random seed
	$(PYTHON) src/preprocess_interim_data.py inputs/differential/data.yml
	touch differential_data



# Anisotropic nonlinear heat equation dataset operator dataset
heat_nl_tensor: requirements
	$(PYTHON) src/train.py inputs/heat_nl_tensor/isogcn_adj$(ADJ).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/heat_nl_tensor/isogcn_adj$(ADJ) \
		data/heat_nl_tensor/preprocessed/test_16 \
		-p data/heat_nl_tensor/preprocessed/preprocessors.pkl

heat_nl_tensor_baseline: requirements
	$(PYTHON) src/train.py inputs/heat_nl_tensor/$(BASELINE_NAME)_adj$(ADJ)$(INPUT).yml -g $(GPU_ID) -n $(N_EPOCH)
	$(PYTHON) src/infer.py \
		models/heat_nl_tensor/$(BASELINE_NAME)_adj$(ADJ)$(INPUT) \
		data/heat_nl_tensor/preprocessed/test_16 \
		-p data/heat_nl_tensor/preprocessed/preprocessors.pkl

heat_nl_tensor_data: requirements
	$(PYTHON) src/convert_raw_data.py inputs/heat_nl_tensor/data.yml -w true
	$(PYTHON) src/preprocess_interim_data.py inputs/heat_nl_tensor/data.yml
	$(PYTHON) src/calculate_scale_genam.py data/heat_nl_tensor/preprocessed


## Sample process for the anisotropic nonlinear heat dataset
small_heat_nl_tensor_pipeline: requirements
	$(RM) tests/data/simple/nl_tensor/raw tests/data/simple/nl_tensor/interim tests/data/simple/nl_tensor/preprocessed
	$(RM) tests/data/models/nl_tensor/ci_train_iso_gcn_simple
	$(RM) tests/data/simple/nl_tensor/iso_gcn.yml.scaled.yml
	$(RM) tests/data/simple/nl_tensor/iso_gcn.scaled
	$(PYTHON) src/generate_data.py tests/data/simple/external \
		-o tests/data/simple/nl_tensor/raw -s 1. -n 2 -l false -m tensor
	bash ./tests/run_fistr.sh tests/data/simple/nl_tensor/raw
	$(PYTHON) src/convert_raw_data.py tests/data/simple/nl_tensor/data.yml --recursive true -w true
	$(PYTHON) src/preprocess_interim_data.py tests/data/simple/nl_tensor/data.yml
	$(PYTHON) src/calculate_scale_genam.py tests/data/simple/nl_tensor/preprocessed
	./tests/rewrite_input_yml.sh tests/data/simple/nl_tensor/iso_gcn.yml \
		tests/data/simple/nl_tensor/preprocessed/grad_stats.yml nodal_grad_2
	$(PYTHON) src/train.py tests/data/simple/nl_tensor/iso_gcn.yml.scaled.yml -g $(GPU_ID)
	$(PYTHON) src/infer.py \
		tests/data/simple/nl_tensor/iso_gcn.scaled \
		tests/data/simple/nl_tensor/preprocessed \
		-p tests/data/simple/nl_tensor/preprocessed/preprocessors.pkl



# Other
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete all data
delete_all_data: clean
	$(RM) data/grid/*
	$(RM) models/*

# Self-Supervised Learning for Fine Grained Image Categorization

The repository contains the implementation of adding self-supervision as an auxiliary task to the baseline model for  fine-grained visual categorization (FGVC) task. 
Specifically, it provides the implementation for rotation, pretext invariant representation learning (PIRL) and destruction and construction learning (DCL) 
as auxiliary tasks for the baseline model.

## Available Models
The list of implemented model architectures can be found at [here](model/README.md).

## Dependencies
* Ubuntu based machine with NVIDIA GPU is required to run the training and evaluation. The code has been developed on a machine having Ubuntu 18.04 LTS distribution with one 24GB Quadro RTX 6000 GPU. 
* Python 3.8.
* Pytorch 1.7.1 and corresponding torchvision version.

## Installation
It is recommended to create a new conda environment for this project. The installation steps are as follows:
1. Create new conda environment and activate it.
```bash
$ conda create env --name=ssl_for_fgvc python=3.8
$ conda activate ssl_for_fgvc
```
2. Install requirements as,
```bash
$ pip install -r requirements.txt
```

## Evaluation of Pretrained Models
All the pretrained models can be found at [here](). In order to evaluate a model, download the model 
checkpoints from the link and use `scripts/evaluate.py` script for evaluating the model on the test set.

```bash
$ cd scripts
$ python evaluate.py --config_path=<path to the corresponding configuration '.yml' file.> \
--model_checkpoints=<path to the downloaded model checkpoints> \
--root_dataset_path=<path to the dataset root directory>
```
If the `--root_dataset_path` command line parameter has not been provided to `evaluate.py` script, it download the dataset 
and perform the testing. The downloading of data may take some time based on the network stability and speed.


Follow the following commands to run the training,

1. Install docker dependencies using [install_docker_dependencies.sh](scripts/install_docker_dependencies.sh).
```bazaar
bash install_docker_dependencies.sh
```
2. Login to docker.
```bazaar
docker login --username=USERNAME
```
3. Update the configuration [config.yml](config.yml).
4. Update the [docker-compose.yml](docker-compose.yml) if needed.
5. Run the following command,
```bazaar
docker-compose up -d
```

In order to build and push the docker image on docker hub, run the script [build_and_push_docker_image.sh](scripts/build_and_push_docker_image.sh).
```bazaar
bash scripts/build_and_push_docker_image.sh IMAGE_TAG
```
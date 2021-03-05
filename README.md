# Self-Supervised Pretraining for Fine Grained Image Classification

Follow the following commands to run the training,

1. Install docker dependencies using [install_docker_dependencies.sh](scripts/install_docker_dependencies.sh).
```bazaar
bash install_docker_dependencies.sh
```
2. Login to docker.
```bazaar
docker login --username=USERNAME
```
3. Update the configuration [config.yml](fine_grained_classification/config.yml).
4. Update the [docker-compose.yml](docker-compose.yml) if needed.
5. Run the following command,
```bazaar
docker-compose up -d
```

In order to build and push the docker image on docker hub, run the script [build_and_push_docker_image.sh](scripts/build_and_push_docker_image.sh).
```bazaar
bash scripts/build_and_push_docker_image.sh IMAGE_TAG
```
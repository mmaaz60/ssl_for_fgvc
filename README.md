# Self-Supervised Pretraining for Fine Grained Image Classification

Follow the following commands to run the training,

1. Install docker dependencies using [install_docker_dependencies.sh](scripts/install_docker_dependencies.sh).
```bazaar
bash install_docker_dependencies.sh
```
2. Update the configuration [config.yml](fine_grained_classification/config.yml).
3. Run the following command,
```bazaar
docker-compose up -d
```
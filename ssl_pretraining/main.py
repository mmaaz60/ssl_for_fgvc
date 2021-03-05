import sys
import os
import logging
import shutil

from fine_grained_classification.config.config import Configuration as config
from dataloader.common import Dataloader
from ssl_pretraining.models.ssl_model_selection import SSLModel
from ssl_pretraining.train.ssl_train import SSLtrainer


sys.path.append(f"{'/'.join(os.getcwd().split('/')[:-1])}")

def main():
    # Config path
    config_path = "./config_ssl.yml"
    # Load the configuration file
    config.load_config(config_path)
    # Read the general configuration parameters
    output_directory = config.cfg["general"]["output_directory"]
    experiment_id = config.cfg["general"]["experiment_id"]
    model_checkpoints_directory_name = config.cfg["general"]["model_checkpoints_directory_name"]

    # Create the output and experiment directory
    if not os.path.exists(f"{output_directory}/{experiment_id}/{model_checkpoints_directory_name}"):
        os.makedirs(f"{output_directory}/{experiment_id}/{model_checkpoints_directory_name}")
    else:
        print(f"The directory {output_directory}/{experiment_id} already exits. Please delete the directory or change "
              f"the experiment_id in the configuration file.")
        sys.exit(1)

    # Copy the configuration file to experiment directory
    shutil.copyfile(config_path, f"{output_directory}/{experiment_id}/{config_path.split('/')[-1]}")
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)-s: %(levelname)-s: %(message)s",
                        datefmt="%m-%d %H:%M",
                        filename=f"{output_directory}/{experiment_id}/{experiment_id}.log",
                        filemode="w")

    # Define a Handler which writes INFO messages
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # Set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-s: %(levelname)-s: %(message)s')
    console.setFormatter(formatter)  # Tell the handler to use this format
    logging.getLogger().addHandler(console)  # Add the handler to the root logger

    # Create the dataloaders
    dataloader = Dataloader(config=config)
    train_loader, test_loader = dataloader.get_loader()

    # Create the base feature extractor model
    model = SSLModel(config=config).get_model()
    # feature_extractor = SSLModel(config=config).get_feature_extractor()
    # Create the trainer and run training
    warm_up_epochs = config.cfg["ssl_train"]["warm_up_epochs"]
    if warm_up_epochs > 0:
        logging.info(f"Starting warp-up training loop using "
                     f"{config.cfg['ssl_train']['warm_up_loss_function_path'].split('.')[-1]} for {warm_up_epochs} "
                     f"epochs.")
        trainer = SSLtrainer(config=config, model=model, dataloader=train_loader, val_dataloader=test_loader,
                          warm_up=False).get_trainer()
        trainer.train_and_validate(start_epoch=1, end_epoch=warm_up_epochs)
    logging.info(f"Staring fgc training loop "
                 f"using {config.cfg['ssl_train']['fgc_loss_function_path'].split('.')[-1]} with rotation using"
                 f"{config.cfg['ssl_train']['rot_loss_function_path'].split('.')[-1]} "
                 f"for {config.cfg['ssl_train']['epochs'] - warm_up_epochs} epochs.")
    trainer = SSLtrainer(config=config, model=model, dataloader=train_loader, val_dataloader=test_loader).get_trainer()
    trainer.train_and_validate(start_epoch=warm_up_epochs)


if __name__ == "__main__":
    main()
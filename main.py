import sys
import os
import logging
import shutil
from config.config import Configuration as config
from dataloader.common import Dataloader
from model.common import Model
from train.common import Trainer
from utils.util import load_vissl_weights

if __name__ == "__main__":
    # Config path
    config_path = "config.yml"
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
    # Configure the logger
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
    # Create the model
    model = Model(config=config).get_model()
    # Load pre-trained weights if required
    try:
        checkpoints_path = config.cfg["model"]["checkpoints_path"]
        model = load_vissl_weights(model, checkpoints_path)
    except KeyError:
        pass
    # Create the trainer and run training
    warm_up_epochs = config.cfg["train"]["warm_up_epochs"]
    if warm_up_epochs > 0:
        logging.info(f"Starting warp-up training loop using "
                     f"{config.cfg['train']['warm_up_loss_function_path'].split('.')[-1]} for {warm_up_epochs} epochs.")
        trainer = Trainer(config=config, model=model, dataloader=train_loader, val_dataloader=test_loader,
                          warm_up=True).get_trainer()
        trainer.train_and_validate(start_epoch=1, end_epoch=warm_up_epochs)
    logging.info(f"Staring main training loop using {config.cfg['train']['class_loss_function_path'].split('.')[-1]} "
                 f"for {config.cfg['train']['epochs'] - warm_up_epochs} epochs.")
    trainer = Trainer(config=config, model=model, dataloader=train_loader, val_dataloader=test_loader).get_trainer()
    trainer.train_and_validate(start_epoch=warm_up_epochs + 1)

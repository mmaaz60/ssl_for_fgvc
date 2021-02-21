import sys
import os
import logging

# Add the root folder (Visitor Tracking Utils) as the path to modules.
sys.path.append(f"{'/'.join(os.getcwd().split('/')[:-1])}")

from fine_grained_classification.config.config import Configuration as config
from fine_grained_classification.dataloader.common import Dataloader
from fine_grained_classification.model.common import Model
from fine_grained_classification.train.common import Trainer

if __name__ == "__main__":
    # Load the configuration file
    config.load_config("./config.yml")
    # Read the general configuration parameters
    output_directory = config.cfg["general"]["output_directory"]
    experiment_id = config.cfg["general"]["experiment_id"]
    # Create the output and experiment directory
    if not os.path.exists(f"{output_directory}/{experiment_id}"):
        os.makedirs(f"{output_directory}/{experiment_id}")
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
    # Create the trainer and run training
    warm_up_epochs = config.cfg["train"]["warm_up_epochs"]
    if warm_up_epochs > 0:
        logging.info(f"Starting warp-up training loop using "
                     f"{config.cfg['train']['warm_up_loss_function_path'].split('.')[-1]} for {warm_up_epochs} epochs.")
        trainer = Trainer(config=config, model=model, dataloader=train_loader, val_dataloader=test_loader,
                          warm_up=True).get_trainer()
        trainer.train_and_validate(start_epoch=1, end_epoch=warm_up_epochs)
    logging.info(f"Staring main training loop using {config.cfg['train']['loss_function_path'].split('.')[-1]} "
                 f"for {config.cfg['train']['epochs'] - warm_up_epochs} epochs.")
    trainer = Trainer(config=config, model=model, dataloader=train_loader, val_dataloader=test_loader).get_trainer()
    trainer.train_and_validate(start_epoch=warm_up_epochs)

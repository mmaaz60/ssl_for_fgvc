import sys
from fine_grained_classification.utils.utils import get_object_from_path
from torch.optim.lr_scheduler import StepLR as LRScheduler


class Trainer:
    """
    This class initiates the specified trainer object
    """
    def __init__(self, config, model, dataloader, val_dataloader=None):
        # Initialize the selected trainer
        self.trainer = None
        # Select the correct model
        if config.cfg["train"]["name"] == "base_trainer":
            self.trainer = self.__get_base_trainer(config.cfg, model, dataloader, val_dataloader)
        else:
            print(f"Please provide correct trainer to use in configuration. "
                  f"Available options are ['base_trainer']")
            sys.exit(1)

    @staticmethod
    def __get_base_trainer(config, model, dataloader, val_dataloader=None):
        """
        Create and return the base trainer object
        """
        # Import the base trainer class
        from fine_grained_classification.train.base_trainer import BaseTrainer as Trainer
        # Parse the config
        loss_func = get_object_from_path(config["train"]["loss_function_path"])
        optimizer_func = get_object_from_path(config["train"]["optimizer_path"])
        optimizer_param = config["train"]["optimizer_param"]
        optimizer = optimizer_func(optimizer_param)
        epochs = config["train"]["epochs"]
        lr_scheduler = LRScheduler(optimizer, config["train"]["lr_scheduler"])
        # Create and return the trainer object
        return Trainer(model=model, dataloader=dataloader, loss_function=loss_func, optimizer=optimizer, epochs=epochs,
                       lr_scheduler=lr_scheduler, val_dataloader=val_dataloader)

    def get_trainer(self):
        """
        This function returns the selected trainer
        """
        return self.trainer

import sys
from utils.util import get_object_from_path
from torch.optim.lr_scheduler import StepLR as LRScheduler
import logging

logger = logging.getLogger(f"train/common.py")


class Trainer:
    """
    This class initiates the specified trainer object
    """

    def __init__(self, config, model, dataloader, val_dataloader=None, warm_up=False):
        # Initialize the selected trainer
        self.trainer = None
        # Select the correct model
        if config.cfg["train"]["name"] == "base_trainer":
            self.trainer = self.__get_base_trainer(config.cfg, model, dataloader, val_dataloader, warm_up)
        elif config.cfg["train"]["name"] == "ssl_rot_trainer":
            self.trainer = self.__get_ssl_rot_trainer(config.cfg, model, dataloader, val_dataloader, warm_up)
        else:
            logger.info(f"Please provide correct trainer to use in configuration. "
                        f"Available options are ['base_trainer', 'ssl_rot_trainer']")
            sys.exit(1)

    @staticmethod
    def __get_base_trainer(config, model, dataloader, val_dataloader=None, warm_up=False):
        """
        Create and return the base trainer object
        """
        # Import the trainer
        from train.base_trainer import BaseTrainer as Trainer
        # Import the base trainer class
        # Parse the config
        if warm_up:
            loss_func = get_object_from_path(config["train"]["warm_up_loss_function_path"])
        else:
            loss_func = get_object_from_path(config["train"]["class_loss_function_path"])
        optimizer_func = get_object_from_path(config["train"]["optimizer_path"])
        optimizer_param = config["train"]["optimizer_param"]
        epochs = config["train"]["epochs"]
        output_directory = config["general"]["output_directory"]
        experiment_id = config["general"]["experiment_id"]
        model_checkpoints_directory_name = config["general"]["model_checkpoints_directory_name"]
        params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value]}]
        optimizer = optimizer_func(params=params, lr=optimizer_param["lr"], momentum=optimizer_param["momentum"],
                                   weight_decay=optimizer_param["weight_decay"])
        lr_scheduler = LRScheduler(optimizer, step_size=config["train"]["lr_scheduler"]["step_size"],
                                   gamma=config["train"]["lr_scheduler"]["gamma"])
        # Create and return the trainer object
        return Trainer(model=model, dataloader=dataloader, loss_function=loss_func, optimizer=optimizer,
                       epochs=epochs, lr_scheduler=lr_scheduler, val_dataloader=val_dataloader,
                       checkpoints_dir_path=f"{output_directory}/{experiment_id}/"
                                            f"{model_checkpoints_directory_name}")

    @staticmethod
    def __get_ssl_rot_trainer(config, model, dataloader, val_dataloader=None, warm_up=False):
        """
        Create and return the ssl rotation trainer object
        """
        # Import the trainer class
        from train.ssl_rot_trainer import SSLROTTrainer as Trainer
        # Parse the config
        if warm_up:
            class_loss_func = get_object_from_path(config["train"]["warm_up_loss_function_path"])
            rot_loss_func = get_object_from_path(config["train"]["warm_up_loss_function_path"])
        else:
            class_loss_func = get_object_from_path(config["train"]["class_loss_function_path"])
            rot_loss_func = get_object_from_path(config["train"]["rotation_loss_function_path"])
        rotation_loss_weight = config["train"]["rotation_loss_weight"]
        optimizer_func = get_object_from_path(config["train"]["optimizer_path"])
        optimizer_param = config["train"]["optimizer_param"]
        epochs = config["train"]["epochs"]
        output_directory = config["general"]["output_directory"]
        experiment_id = config["general"]["experiment_id"]
        model_checkpoints_directory_name = config["general"]["model_checkpoints_directory_name"]
        params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value]}]
        optimizer = optimizer_func(params=params, lr=optimizer_param["lr"], momentum=optimizer_param["momentum"],
                                   weight_decay=optimizer_param["weight_decay"])
        lr_scheduler = LRScheduler(optimizer, step_size=config["train"]["lr_scheduler"]["step_size"],
                                   gamma=config["train"]["lr_scheduler"]["gamma"])
        # Create and return the trainer object
        return Trainer(model=model, dataloader=dataloader, class_loss_function=class_loss_func,
                       rot_loss_function=rot_loss_func, rotation_loss_weight=rotation_loss_weight,
                       optimizer=optimizer, epochs=epochs, lr_scheduler=lr_scheduler, val_dataloader=val_dataloader,
                       checkpoints_dir_path=f"{output_directory}/{experiment_id}/{model_checkpoints_directory_name}")

    @staticmethod
    def __get_ssl_bt_trainer(config, model, dataloader, val_dataloader=None, warm_up=False):
        """
        Create and return the base trainer object
        """
        # Import the trainer
        from train.ssl_barlow_twins_trainer import SSLBarlowTwinsTrainer as Trainer
        # Import the base trainer class
        # Parse the config
        if warm_up:
            loss_func = get_object_from_path(config["train"]["warm_up_loss_function_path"])
        else:
            loss_func = get_object_from_path(config["train"]["class_loss_function_path"])
        optimizer_func = get_object_from_path(config["train"]["optimizer_path"])
        optimizer_param = config["train"]["optimizer_param"]
        epochs = config["train"]["epochs"]
        output_directory = config["general"]["output_directory"]
        experiment_id = config["general"]["experiment_id"]
        model_checkpoints_directory_name = config["general"]["model_checkpoints_directory_name"]
        params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value]}]
        optimizer = optimizer_func(params=params, lr=optimizer_param["lr"], momentum=optimizer_param["momentum"],
                                   weight_decay=optimizer_param["weight_decay"])
        lr_scheduler = LRScheduler(optimizer, step_size=config["train"]["lr_scheduler"]["step_size"],
                                   gamma=config["train"]["lr_scheduler"]["gamma"])
        # Create and return the trainer object
        return Trainer(model=model, dataloader=dataloader, class_loss_function=loss_func, optimizer=optimizer,
                       epochs=epochs, lr_scheduler=lr_scheduler, val_dataloader=val_dataloader,
                       checkpoints_dir_path=f"{output_directory}/{experiment_id}/{model_checkpoints_directory_name}")

    def get_trainer(self):
        """
        This function returns the selected trainer
        """
        return self.trainer

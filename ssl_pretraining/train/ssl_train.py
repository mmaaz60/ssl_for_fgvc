import sys
from fine_grained_classification.utils.utils import get_object_from_path
from torch.optim.lr_scheduler import StepLR as LRScheduler
import logging
logger = logging.getLogger(f"train/ssl_train.py")


class SSLtrainer:
    """
    This class initiates the specified trainer object
    """

    def __init__(self, config, model, dataloader, val_dataloader=None, warm_up=False):
        # Initialize the selected trainer
        self.trainer = None
        # Select the correct model
        if config.cfg["ssl_train"]["name"] == "ssl_trainer":
            from ssl_pretraining.train.ssl_rot_trainer import SSLrotTrainer
        else:
            logger.info(f"Please provide correct trainer to use in configuration. "
                        f"Available options are ['ssl_trainer']")
            sys.exit(1)
        self.trainer = self.__get_trainer(SSLrotTrainer, config.cfg, model, dataloader, val_dataloader, warm_up)

    @staticmethod
    def __get_trainer(trainer_cls, config, model, dataloader, val_dataloader=None, warm_up=False):
        """
        Create and return the base trainer object depending on basic or FGC
        """
        # Import the base trainer class
        # Parse the config
        print(config)
        if config["pipeline"]["name"] == "classification":
            if warm_up:
                class_loss_func = get_object_from_path(config["ssl_train"]["warm_up_loss_function_path"])
                rot_loss_func = get_object_from_path(config["ssl_train"]["warm_up_loss_function_path"])
            else:
                class_loss_func = get_object_from_path(config["ssl_train"]["class_loss_function_path"])
                rot_loss_func = get_object_from_path(config["ssl_train"]["rot_loss_function_path"])
            optimizer_func = get_object_from_path(config["ssl_train"]["optimizer_path"])
            optimizer_param = config["ssl_train"]["optimizer_param"]
            epochs = config["ssl_train"]["epochs"]
            output_directory = config["general"]["output_directory"]
            experiment_id = config["general"]["experiment_id"]
            model_checkpoints_directory_name = config["general"]["model_checkpoints_directory_name"]
            params = []
            for key, value in dict(model.named_parameters()).items():
                if value.requires_grad:
                    params += [{'params': [value]}]
            optimizer = optimizer_func(params=params, lr=optimizer_param["lr"], momentum=optimizer_param["momentum"],
                                       weight_decay=optimizer_param["weight_decay"])
            lr_scheduler = LRScheduler(optimizer, step_size=config["ssl_train"]["lr_scheduler"]["step_size"],
                                       gamma=config["ssl_train"]["lr_scheduler"]["gamma"])
        else:
            # Allows to switch between classification and FGC (future work)
            print(f"Choose a correct pipeline"
                  f"Available options are ['classification']")
            sys.exit(1)
        # Create and return the trainer object
        return trainer_cls(model=model, dataloader=dataloader, class_loss_function=class_loss_func,
                           rot_loss_function=rot_loss_func, optimizer=optimizer, epochs=epochs,
                           lr_scheduler=lr_scheduler, val_dataloader=val_dataloader,
                           checkpoints_dir_path=f"{output_directory}/{experiment_id}/"
                                                f"{model_checkpoints_directory_name}")

    def get_trainer(self):
        """
        This function returns the selected trainer
        """
        return self.trainer

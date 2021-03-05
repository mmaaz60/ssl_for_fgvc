import sys
import torch.nn as nn


class SSLModel:
    """
    This class initiates the ssl model by initiating the base classifier and rotation
    classifier
    """
    def __init__(self, config):
        # Select base classifier or fgc based on config
        if config.cfg["pipeline"]["name"] == "classification":
            # Chooses basic classifier from base_classification
            from ssl_pretraining.models.ssl_custom_model import SSLCustomModel
        elif config.cfg["pipeline"]["name"] == "fine_grained_classification":
            # Choose fine grained classifier from fine_grained_classification dir
            # Would later be changed to have a switch between simple classifiers and custom FGC model
            pass
        else:
            print(f"Please provide correct model to use in configuration. "
                  f"Available options are ['torchvision']")
            sys.exit(1)
        # select basic classifier or fgc as base model
        self.model = SSLCustomModel(config)

    def get_model(self):
        """
        This function returns the selected model
        """
        return self.model

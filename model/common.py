import sys


class Model:
    """
    This class initiates the specified model
    """
    def __init__(self, config):
        # Select the correct model
        if config.cfg["model"]["name"] == "torchvision":
            from model.torchvision import TorchVision as Model
        elif config.cfg["model"]["name"] == "fgvc_resnet":
            from model.fgvc_resnet import FGVCResnet as Model
        else:
            print(f"Please provide correct model to use in configuration. "
                  f"Available options are ['torchvision', fgvc_resnet]")
            sys.exit(1)
        # Initialize the selected DataLoader
        self.model = Model(config)

    def get_model(self):
        """
        This function returns the selected model
        """
        return self.model

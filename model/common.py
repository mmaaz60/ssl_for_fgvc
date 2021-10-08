import sys


class Model:
    """
    This class initiates the specified model.
    """
    def __init__(self, config):
        """
        Constructor, select the model specified in the configuration file

        :param config: Configuration class object
        """
        # Select the correct model
        if config.cfg["model"]["name"] == "torchvision":
            from model.torchvision import TorchVision as Model
        elif config.cfg["model"]["name"] == "fgvc_resnet":
            from model.fgvc_resnet import FGVCResnet as Model
        elif config.cfg["model"]["name"] == "torchvision_ssl_rotation":
            from model.torchvision_ssl_rotation import TorchvisionSSLRotation as Model
        elif config.cfg["model"]["name"] == "fgvc_ssl_rotation":
            from model.fgvc_ssl_rotation import FGVCSSLRotation as Model
        elif config.cfg["model"]["name"] == "torchvision_ssl_pirl":
            from model.torchvision_ssl_pirl import TorchVisionSSLPIRL as Model
        elif config.cfg["model"]["name"] == "dcl":
            from model.torchvision_ssl_dcl import TorchVisionSSLDCL as Model
        elif config.cfg["model"]["name"] == "fgvc_bb_supervision":
            from model.fgvc_bb_supervision import AttentionResnet as Model
        else:
            print(f"Please provide correct model to use in configuration. "
                  f"Available options are ['torchvision', 'fgvc_resnet', "
                  f"'torchvision_ssl_rotation', 'fgvc_ssl_rotation', 'torchvision_ssl_pirl', 'dcl']")
            sys.exit(1)
        # Initialize the selected DataLoader
        self.model = Model(config)

    def get_model(self):
        """
        This function returns the selected model
        """
        return self.model

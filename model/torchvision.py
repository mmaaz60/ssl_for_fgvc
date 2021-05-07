import torch.nn as nn
from utils.util import get_object_from_path


class TorchVision(nn.Module):
    """
    The class defines a specified torchvision model.
    """

    def __init__(self, config):
        """
        Constructor, the function parse the config and initialize the layers of the corresponding model,

        :param config: Configuration class object
        """
        super(TorchVision, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        # Load the model
        self.model = self.model_function(pretrained=self.pretrained)
        # Alter the classification layer as per the specified number of classes
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes,
                                  bias=(self.model.fc.bias is not None))

    def forward(self, x, train=False):
        """
        The function implements the forward pass of the model.

        :param x: Input image tensor
        :param train: Flag to specify either train or test mode
        """
        out = self.model(x)
        return out

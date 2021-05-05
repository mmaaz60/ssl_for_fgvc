import torch.nn as nn
from utils.util import get_object_from_path


class TorchvisionSSLRotation(nn.Module):
    """
    The class adds rotation as an auxiliary task to the standard torchvision network.
    """

    def __init__(self, config):
        """
        Constructor, the function parse the config and initialize the layers of the corresponding model.

        :param config: Configuration class object
        """
        super(TorchvisionSSLRotation, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes_classification = config.cfg["model"]["classes_count"]  # No. of classes for classification
        self.num_classes_rot = config.cfg["model"]["rotation_classes_count"]  # No. of classes for rotation head
        # Load the model
        self.model = self.model_function(pretrained=self.pretrained)
        net_list = list(self.model.children())
        self.feature_extractor = nn.Sequential(*net_list[:-1])  # Feature extractor
        self.flatten = nn.Flatten()  # Flatten layer
        # CUB classification head
        self.classification_head = nn.Linear(in_features=self.model.fc.in_features,
                                             out_features=self.num_classes_classification,
                                             bias=(self.model.fc.bias is not None))
        # Rotation classification head
        self.rotation_head = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes_rot,
                                       bias=(self.model.fc.bias is not None))

    def forward(self, x, train=False):
        """
        The function implements the forward pass of the model.

        :param x: Input image tensor
        :param train: Flag to specify either train or test mode
        """
        features = self.feature_extractor(x)  # Feature extraction
        features = self.flatten(features)  # Flatten the features
        y_classification = self.classification_head(features)  # CUB Classification
        if train:
            # Rotation classification if train
            y_rotation = self.rotation_head(features)
            return y_classification, y_rotation
        else:
            return y_classification

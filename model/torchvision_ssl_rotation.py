import torch.nn as nn
from utils.util import get_object_from_path


class TorchvisionSSLRotation(nn.Module):
    """
    This class inherits from nn.Module class
    """

    def __init__(self, config):
        """
        The function parse the config and initialize the layers of the corresponding model
        :param config: YML configuration file to parse the parameters from
        """
        super(TorchvisionSSLRotation, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes_classification = config.cfg["model"]["classes_count"]  # No. of classes for classification
        self.num_classes_rot = config.cfg["model"]["rotation_classes_count"]  # No. of classes for rotation head
        self.feature_embedding = config.cfg["model"]["rotation_feature_embedding"]  # Rotation feature embedding
        # Load the model
        self.model = self.model_function(pretrained=self.pretrained)
        net_list = list(self.model.children())
        self.feature_extractor = nn.Sequential(*net_list[:-1])
        self.flatten = nn.Flatten()
        self.classification_head = net_list[-1]
        self.rotation_head = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes_rot,
                                       bias=(self.model.fc.bias is not None))

    def forward(self, x):
        """
        The function implements the forward pass of the network/model
        :param x: Batch of inputs (images)
        :return:
        """

        # Model without diverse block
        features = self.feature_extractor(x)
        features = self.flatten(features)
        y_classification = self.classification_head(features)
        y_rotation = self.rotation_head(features)
        return y_classification, y_rotation

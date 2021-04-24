import torch.nn as nn
import torch
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
        # Load the model
        model = self.model_function(pretrained=self.pretrained)
        net_list = list(model.children())
        self.feature_extractor = nn.Sequential(*net_list[:-1])
        self.flatten = nn.Flatten()
        self.classification_head = nn.Linear(in_features=model.fc.in_features,
                                             out_features=self.num_classes_classification,
                                             bias=(model.fc.bias is not None))
        self.rotation_head = nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes_rot,
                                       bias=(model.fc.bias is not None))

    def forward(self, x, train=True):
        """
        The function implements the forward pass of the network/model
        :param train:
        :param x: Batch of inputs (images)
        :return:
        """

        # Model without diverse block
        features = self.feature_extractor(x)
        features = self.flatten(features)
        y_classification = self.classification_head(features)
        y_rotation = self.rotation_head(features)
        return y_classification, y_rotation

    def get_cam(self, x):
        net_list = list(self.feature_extractor.children())
        feature_extractor = nn.Sequential(*net_list[:-1])
        feature_map = feature_extractor(x)
        b, c, h, w = feature_map.size()
        feature_map = feature_map.view(b, c, h * w).transpose(1, 2)
        cam = torch.bmm(feature_map,
                        torch.repeat_interleave(self.classification_head.weight.t().unsqueeze(0),
                                                b, dim=0)).transpose(1, 2)
        out = torch.reshape(cam, [b, self.num_classes_classification, h, w])

        return out

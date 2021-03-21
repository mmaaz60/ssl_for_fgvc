import torch.nn as nn
from utils.util import get_object_from_path
import torch


class TorchvisionSSLBarlowTwins(nn.Module):
    """
    This class inherits from nn.Module class
    """

    def __init__(self, config):
        """
        The function parse the config and initialize the layers of the corresponding model
        :param config: YML configuration file to parse the parameters from
        """
        super(TorchvisionSSLBarlowTwins, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes_classification = config.cfg["model"]["classes_count"]  # No. of classes for classification
        self.num_classes_rot = config.cfg["model"]["rotation_classes_count"]  # No. of classes for rotation head
        # Load the model and initialize layers related to original classification task
        self.model = self.model_function(pretrained=self.pretrained)
        net_list = list(self.model.children())
        self.feature_extractor = nn.Sequential(*net_list[:-1])
        self.flatten = nn.Flatten()
        self.classification_head = nn.Linear(in_features=self.model.fc.in_features,
                                             out_features=self.num_classes_classification,
                                             bias=(self.model.fc.bias is not None))
        # Barlow twins
        projector = "8192-8192-8192"
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.scale_loss = 1 / 32
        self.lambd = 3.9e-3

    def forward(self, x, t1, t2, train=True):
        """
        The function implements the forward pass of the network/model
        :param t2:
        :param t1:
        :param train:
        :param x: Batch of inputs (images)
        :return:
        """

        # Perform original classification task
        features = self.feature_extractor(x)
        features = self.flatten(features)
        y_classification = self.classification_head(features)
        # Barlow twins
        bt_loss = None
        if train:
            z_1 = self.projector(self.feature_extractor(t1))
            z_2 = self.projector(self.feature_extractor(t2))
            c = self.bn(z_1).T @ self.bn(z_2)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
            off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
            bt_loss = on_diag + self.lambd * off_diag

        return y_classification, bt_loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

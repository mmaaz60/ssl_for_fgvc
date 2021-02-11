import torch.nn as nn
from fine_grained_classification.utils.utils import get_object_from_path


class TorchVision(nn.Module):
    """
    This class inherits from nn.Module class
    """

    def __init__(self, config):
        """
        The function parse the config and initialize the layers of the corresponding model
        :param config: YML configuration file to parse the parameters from
        """
        super(TorchVision, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        # Load the model
        self.model = self.model_function(pretrained=self.pretrained)
        # Remove the last (classification) layer of the model
        modules = list(self.model.children())[:-1]
        self.features = nn.Sequential(*modules)
        # Add linear layer on top of feature embeddings
        self.linear = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax

    def forward(self, x):
        """
        The function implements the forward pass of the network/model
        :param x: Batch of inputs (images)
        :return: The model output (class logits)
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.softmax(out)
        return out

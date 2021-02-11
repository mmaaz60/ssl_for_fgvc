import torch.nn as nn
import torchvision.models as models


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
        self.type = config.cfg["model"]["type"]  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        # Select the model
        self.model = self.select_model()
        # Remove the last (classification) layer of the model
        modules = list(self.model.children())[:-1]
        self.features = nn.Sequential(*modules)
        # Add linear layer on top of feature embeddings
        self.linear = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax

    def select_model(self):
        """
        The function selects the model as per the configuration parameters specified in the configuration file
        :return: The selected model (nn.Module class object)
        """
        if self.type == "resnet50":
            return models.resnet50(pretrained=self.pretrained, progress=True)
        else:
            raise RuntimeError(f"The requested model type is not support. The supported model types are ['resnet50']")

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

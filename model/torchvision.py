import torch.nn as nn
import torch
from utils.util import get_object_from_path


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
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        # Load the model
        self.model = self.model_function(pretrained=self.pretrained)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes,
                                  bias=(self.model.fc.bias is not None))

    def forward(self, x, train=True):
        """
        The function implements the forward pass of the network/model
        :param train:
        :param x: Batch of inputs (images)
        :return: The model output (class logits)
        """
        out = self.model(x)
        return out

    def get_cam(self, x, topk):
        net_list = list(self.model.children())
        feature_extractor = nn.Sequential(*net_list[:-2])
        feature_map = feature_extractor(x)
        b, c, h, w = feature_map.size()
        feature_map = feature_map.view(b, c, h * w).transpose(1, 2)
        cam = torch.bmm(feature_map,
                        torch.repeat_interleave(self.model.fc.weight.t().unsqueeze(0), b, dim=0)).transpose(1, 2)
        out = torch.reshape(cam, [b, self.num_classes, h, w])
        predictions = self.model(x)
        _, preds = torch.sort(predictions, dim=1, descending=True)
        topk_pred = preds.squeeze().tolist()[:topk]
        return out, topk_pred

import torch.nn as nn
from utils.util import get_object_from_path
from model.fgvc_resnet import CAM
from layers.diversification_block import DiversificationBlock


class FGVCSSLRotation(nn.Module):
    """
    This class inherits from nn.Module class
    """

    def __init__(self, config):
        """
        The function parse the config and initialize the layers of the corresponding model
        :param config: YML configuration file to parse the parameters from
        """
        super(FGVCSSLRotation, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes_classification = config.cfg["model"]["classes_count"]   # No. of classes for classification
        self.num_classes_rot = config.cfg["model"]["rotation_classes_count"]  # No. of classes for rotation head
        self.kernel_size = config.cfg["diversification_block"]["patch_size"]  # Patch size to be suppressed
        self.alpha = config.cfg["diversification_block"]["alpha"]  # Suppression factor]
        self.p_peak = config.cfg["diversification_block"]["p_peak"]  # Probability for peak selection
        self.p_patch = config.cfg["diversification_block"]["p_patch"]  # Probability for peak selection
        # Load the model
        self.cam = CAM(self.model_function, self.num_classes_classification, self.pretrained)
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(3)
        self.flatten = nn.Flatten()
        self.rotation_head = nn.Linear(self.num_classes_classification * 3 * 3, self.num_classes_rot)
        self.diversification_block = DiversificationBlock(self.kernel_size, self.alpha, self.p_peak, self.p_patch)

    def forward(self, x, db_flag=True):
        """
        The function implements the forward pass of the network/model
        :param db_flag:
        :param x: Batch of inputs (images)
        :return:
        """
        out = self.cam(x)
        if db_flag:
            out = self.diversification_block(out)
        y_classification = out.mean([2, 3])
        out = self.adaptive_pooling(out)
        out = self.flatten(out)
        y_rotation = self.rotation_head(out)
        return y_classification, y_rotation

import torch.nn as nn
from utils.util import get_object_from_path
from model.fgvc_resnet import CAM
from layers.diversification_block import DiversificationBlock


class FGVCSSLRotation(nn.Module):
    """
    The class adds the rotation as an auxiliary task to the FGVC model from
    (http://arxiv.org/abs/1912.06842).
    """

    def __init__(self, config):
        """
        Constructor, The function parse the config and initialize the layers of the corresponding model.

        :param config: Configuration class object
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
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(3)  # Adaptive average pooling for classification prediction
        self.flatten = nn.Flatten()  # Flatten the features
        # Adds a classification head for rotation prediction
        self.rotation_head = nn.Linear(self.num_classes_classification * 3 * 3, self.num_classes_rot)
        self.diversification_block = DiversificationBlock(self.kernel_size, self.alpha, self.p_peak, self.p_patch)

    def forward(self, x, train=False):
        """
        The function implements the forward pass of the model.

        :param x: Input image tensor
        :param train: Flag to specify either train or test mode
        """
        out = self.cam(x)  # Calculate the CAMs
        if train:
            # Diversification block is only used during training
            out = self.diversification_block(out)
        y_classification = out.mean([2, 3])  # Get the classification scores
        if train:
            # SSL rotation prediction part, only during training
            out = self.adaptive_pooling(out)
            out = self.flatten(out)
            y_rotation = self.rotation_head(out)  # Classification head predicts rotation augmentation applied
            return y_classification, y_rotation
        else:
            return y_classification

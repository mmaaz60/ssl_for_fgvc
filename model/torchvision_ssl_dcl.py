"""
Credits: This implementation is inspired from https://github.com/JDAI-CV/DCL
"""

import torch.nn as nn
from utils.util import get_object_from_path


class TorchVisionSSLDCL(nn.Module):
    """
    The class adds the DCL-SSL task to the standard specified torchvision model.
    """
    def __init__(self, config):
        """
        Constructor, The function parse the config and initialize the layers of the corresponding model.

        :param config: Configuration class object
        """
        super(TorchVisionSSLDCL, self).__init__()  # Call the constructor of the parent class
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        self.prediction_type = config.cfg["model"]["prediction_type"]  # Jigsaw permutation prediction type
        jigsaw_size = config.cfg["dataloader"]["transforms"]["jigsaw"]["t_1"]["param"]["size"]  # Jigsaw patch size
        self.jigsaw_class = jigsaw_size[0] * jigsaw_size[1]
        # Load the model
        net = self.model_function(pretrained=self.pretrained)
        net_list = list(net.children())
        self.feature_extractor = nn.Sequential(*net_list[:-2])  # Feature extractor
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)  # Adaptive average pooling
        # CLS classifier
        self.cls_classifier = nn.Linear(in_features=net.fc.in_features, out_features=self.num_classes,
                                        bias=(net.fc.bias is not None))
        # Adversarial classifier
        self.adv_classifier = nn.Linear(in_features=net.fc.in_features, out_features=2 * self.num_classes,
                                        bias=(net.fc.bias is not None))
        # Convolutional for jigsaw permutation prediction
        self.conv_mask = nn.Conv2d(in_channels=net.fc.in_features, out_channels=1, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.conv_mask_cls = nn.Conv2d(in_channels=net.fc.in_features, out_channels=self.jigsaw_class, kernel_size=1,
                                       stride=1, padding=0, bias=True)
        self.avg_pool_1 = nn.AvgPool2d(2, stride=2)
        # Jigsaw permutation prediction head
        self.jigsaw_cls_classifier = nn.Linear(in_features=49, out_features=self.jigsaw_class, bias=True)
        self.flatten = nn.Flatten()  # Flatten layer
        self.tan_h = nn.Tanh()  # Tanh activation
        self.relu = nn.ReLU()  # ReLU activation
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

    def forward(self, x, train=False):
        """
        The function implements the forward pass of the model.

        :param x: Input image tensor
        :param train: Flag to specify either train or test mode
        """
        feat = self.feature_extractor(x)  # Feature extraction
        classifier = self.avg_pool(feat)  # Adaptive average pooling
        classifier = self.flatten(classifier)  # Flatten the features
        cls_classifier = self.cls_classifier(classifier)  # Cls classifier
        if train:
            # Calculate Adv classifier and jigsaw permutation head outputs if train
            adv_classifier = self.adv_classifier(classifier)
            jigsaw_mask = self.conv_mask(feat)
            if self.prediction_type == "regression":
                jigsaw_mask = self.avg_pool_1(jigsaw_mask)
                jigsaw_mask = self.tan_h(jigsaw_mask)
                jigsaw_mask = self.flatten(jigsaw_mask)
            else:
                jigsaw_mask = self.relu(jigsaw_mask)
                jigsaw_mask = self.avg_pool_1(jigsaw_mask)
                jigsaw_mask = self.flatten(jigsaw_mask)
                jigsaw_mask = self.jigsaw_cls_classifier(jigsaw_mask)
                jigsaw_mask = self.sigmoid(jigsaw_mask)

            return [cls_classifier, adv_classifier, jigsaw_mask]
        else:
            return cls_classifier

import torch.nn as nn
from utils.util import get_object_from_path


class TorchVisionSSLDCL(nn.Module):
    def __init__(self, config):
        super(TorchVisionSSLDCL, self).__init__()
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        # Load the model
        net = self.model_function(pretrained=self.pretrained)
        net_list = list(net.children())
        self.feature_extractor = nn.Sequential(*net_list[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.cls_classifier = nn.Linear(in_features=net.fc.in_features, out_features=self.num_classes,
                                        bias=(net.fc.bias is not None))
        self.adv_classifier = nn.Linear(in_features=net.fc.in_features, out_features=2 * self.num_classes,
                                        bias=(net.fc.bias is not None))
        self.conv_mask = nn.Conv2d(in_channels=net.fc.in_features, out_channels=1, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.avg_pool_1 = nn.AvgPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.tan_h = nn.Tanh()

    def forward(self, x, train=False):
        feat = self.feature_extractor(x)
        classifier = self.avg_pool(feat)
        classifier = self.flatten(classifier)
        cls_classifier = self.cls_classifier(classifier)
        if train:
            adv_classifier = self.adv_classifier(classifier)
            jigsaw_mask = self.conv_mask(feat)
            jigsaw_mask = self.avg_pool_1(jigsaw_mask)
            jigsaw_mask = self.tan_h(jigsaw_mask)
            jigsaw_mask = self.flatten(jigsaw_mask)

            return [cls_classifier, adv_classifier, jigsaw_mask]
        else:
            return cls_classifier

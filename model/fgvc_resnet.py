import torch
import torch.nn as nn
from layers.diversification_block import DiversificationBlock
from utils.util import get_object_from_path


class FGVCResnet(nn.Module):
    def __init__(self, config):
        super(FGVCResnet, self).__init__()
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        self.kernel_size = config.cfg["diversification_block"]["patch_size"]  # Patch size to be suppressed
        self.alpha = config.cfg["diversification_block"]["alpha"]  # Suppression factor
        self.p_peak = config.cfg["diversification_block"]["p_peak"]
        self.p_patch = config.cfg["diversification_block"]["p_patch"]
        self.cam = CAM(self.model_function, self.num_classes, self.pretrained)
        self.diversification_block = DiversificationBlock(self.kernel_size, self.alpha, self.p_peak, self.p_patch)

    def forward(self, x, db_flag=True):
        out = self.cam(x)
        if db_flag:
            out = self.diversification_block(out)
        out = out.mean([2, 3])

        return out


class CAM(nn.Module):
    def __init__(self, model_function, num_classes, pretrained):
        super(CAM, self).__init__()
        self.num_classes = num_classes
        self.network = ResNet(model_function, self.num_classes, pretrained)

    def forward(self, x):
        feature_map, _ = self.network(x)
        # Generate class activation map
        b, c, h, w = feature_map.size()
        feature_map = feature_map.view(b, c, h * w).transpose(1, 2)
        cam = torch.bmm(feature_map, torch.repeat_interleave(self.network.fc_weight, b, dim=0)).transpose(1, 2)
        cam = torch.reshape(cam, [b, self.num_classes, h, w])
        return cam


class ResNet(nn.Module):
    def __init__(self, model_function, num_classes, pretrained=True):
        super(ResNet, self).__init__()
        net = model_function(pretrained=pretrained)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=num_classes, bias=(net.fc.bias is not None))
        net_list = list(net.children())

        self.feature_extractor = nn.Sequential(*net_list[:-2])
        self.fc_layer = net_list[-1]
        self.fc_weight = nn.Parameter(self.fc_layer.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = self.fc_layer(feature_map.mean([2, 3]))
        return feature_map, output

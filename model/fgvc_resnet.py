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
        self.p_peak = config.cfg["diversification_block"]["p_peak"]  # Probability for peak selection
        self.p_patch = config.cfg["diversification_block"]["p_patch"]  # Probability for patch selection
        self.cam = CAM(self.model_function, self.num_classes, self.pretrained)
        self.diversification_block = DiversificationBlock(self.kernel_size, self.alpha, self.p_peak, self.p_patch)

    def forward(self, x, db_flag=True):
        out = self.cam(x)
        if db_flag:
            out = self.diversification_block(out)
        out = out.mean([2, 3])

        return out


class CAM(nn.Module):
    def __init__(self, model_function, num_classes, pretrained=True):
        super(CAM, self).__init__()
        net = model_function(pretrained=pretrained)
        net_list = list(net.children())

        self.feature_extractor = nn.Sequential(*net_list[:-2])
        self.conv = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        conv_out = self.conv(feature_map)
        return conv_out

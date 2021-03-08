import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import get_object_from_path


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
        self.feature_embedding = config.cfg["model"]["rotation_feature_embedding"]  # Rotation feature embedding
        # Load the model
        self.cam = CAM(self.model_function, self.num_classes_classification, self.pretrained)
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(3)
        self.flatten = nn.Flatten()
        self.rotation_head = nn.Linear(self.num_classes_classification * 3 * 3, self.num_classes_rot)
        self.kernel_size = config.cfg["diversification_block"]["patch_size"]
        self.alpha = config.cfg["diversification_block"]["alpha"]

    @staticmethod
    def __diversification_block(activation, kernel_size, alpha):
        activation = activation.detach().clone()
        p_peak = torch.max(torch.max(activation, 3).values, 2).values
        rc = torch.bernoulli(p_peak)  # Bernoulli prob for P peak: 0 or 1 randomly for c classes
        b, c, m, n = activation.shape

        # Peak Suppression
        pc = torch.zeros_like(activation)  # Mask for peaks for each class
        pc[activation == torch.unsqueeze(torch.unsqueeze(p_peak, 2), 3)] = 1
        bc_dash = torch.mul(torch.unsqueeze(torch.unsqueeze(rc, 2), 3), pc)  # Peak suppression mask

        # Patch suppression
        # patching image to G*G patches
        kernel_size = kernel_size  # G
        stride = kernel_size  # G*G patch
        patches = activation.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
        l, k = patches.shape[2], patches.shape[3]

        max_patch = torch.max(torch.max(patches, 5).values, 4).values
        p_patch = torch.bernoulli(max_patch)
        bc_dd = torch.zeros_like(patches)  # Mask for peaks for each class
        bc_dd[p_patch == 1] = 1
        bc_dd = (bc_dd.reshape(b, c, l, k, kernel_size * kernel_size)).permute(0, 1, 4, 2, 3)
        bc_dd = bc_dd.reshape(b, c, kernel_size * kernel_size, -1)
        bc_dd_batch = torch.zeros_like(activation)
        for i in range(b):
            bc_dd_batch[i] = F.fold(bc_dd[i], (m, n), kernel_size=kernel_size, stride=stride).squeeze(1)
        bc_dd_batch[activation == torch.unsqueeze(torch.unsqueeze(p_peak, 2), 3)] = 0
        bc = bc_dash + bc_dd_batch

        # Activation Suppression Factor
        suppress_ind = ((bc == 1).nonzero(as_tuple=False))
        activation[suppress_ind[:, 0], suppress_ind[:, 1]] *= alpha

        return activation

    def forward(self, x):
        """
        The function implements the forward pass of the network/model
        :param x: Batch of inputs (images)
        :return:
        """
        out = self.cam(x)
        out = self.__diversification_block(out, self.kernel_size, self.alpha)
        y_classification = out.mean([2, 3])
        return mean


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
        min_val = torch.min(cam)
        cam -= min_val
        max_val = torch.max(cam)
        cam /= max_val
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli
from fine_grained_classification.utils.utils import get_object_from_path


class FGVCResnet(nn.Module):
    def __init__(self, config):
        super(FGVCResnet, self).__init__()
        # Parse the configuration parameters
        self.model_function = get_object_from_path(config.cfg["model"]["model_function_path"])  # Model type
        self.pretrained = config.cfg["model"]["pretrained"]  # Either to load weights from pretrained model or not
        self.num_classes = config.cfg["model"]["classes_count"]  # Number of classes
        self.cam = CAM(self.model_function, self.num_classes, self.pretrained)

    @staticmethod
    def __diversification_block(cams, kernel_size, alpha):
        num_classes = len(cams)
        activation_all_classes = []
        for i in range(num_classes):
            activation = cams[i]
            rc = activation.max().item()
            p_peak = bernoulli.rvs(rc, size=1)  # Bernoulli prob for P peak: 0 or 1 randomly for c classes
            b, c, m, n = activation.shape
            bc = torch.zeros(m, n)  # Initializing Final suppression mask

            # Peak Suppression
            max_ind = ((activation == activation.max()).nonzero(as_tuple=False))
            pc = bc
            pc[max_ind[:, 0], max_ind[:, 1]] = 1
            bc_dash = p_peak * pc  # Peak suppression mask

            # Patch suppression
            # patching image to G*G patches
            kernel_size = kernel_size  # G
            stride = kernel_size  # G*G patch
            patches = activation.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
            # num_patches = patches.shape[2] * patches.shape[3]
            l, m = patches.shape[2], patches.shape[3]

            # Patching mask bc''(Bc_dd) to G*G patches (Bc_patch)
            # p_patch by probability of patch wise maximums
            max_vals = torch.max(patches[0][0], 3)
            max_vals = torch.max(max_vals.values, 2)
            p_patch = max_vals.values.cpu().apply_(lambda x: (bernoulli.rvs(x, size=1)))
            p_patch_ind = torch.nonzero(torch.tensor(p_patch) == 1, as_tuple=False)
            bc_dd = bc  # Bc double dash allocating size with zeros
            bc_patch = bc_dd.unsqueeze(0).unsqueeze(0).unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
            # bc_patch[0][0][P_patch_ind[:, 0]][P_patch_ind[:, 1]]=1 # Implementation of bc'' without for loop
            bc_patch[0][0] = 0
            for i in range(len(p_patch_ind)):
                bc_patch[0][0][p_patch_ind[i][0]][p_patch_ind[i][1]] = 1

            # bc_patch folding back Bc''(Bc_dd)
            bc_patch = (bc_patch.reshape(1, 1, l, m, kernel_size * kernel_size)).permute(0, 1, 4, 2, 3).squeeze(
                0).squeeze(
                0)
            bc_patch = bc_patch.view(kernel_size * kernel_size, -1)
            bc_dd = F.fold(bc_patch.unsqueeze(0), (m, n), kernel_size=kernel_size, stride=stride)
            bc_dd = bc_dd.squeeze(0).squeeze(0)

            # bc'' not supressing peak(peak=0)
            bc_dd[max_ind[:, 0], max_ind[:, 1]] = 0

            # Final Supression Mask Bc
            bc = bc_dash + bc_dd

            # Activation Supression Factor
            supress_ind = ((bc == 1).nonzero(as_tuple=False))
            activation[0][supress_ind[:, 0], supress_ind[:, 1]] *= alpha
            activation_all_classes = torch.stack(activation.unsqueeze(1), dim=0)

        return activation_all_classes

    def forward(self, x):
        out = self.cam(x)
        out = self.__diversification_block(out[0], 20, 0.1)
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

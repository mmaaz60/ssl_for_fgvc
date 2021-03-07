
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from gradcam import GradCAM
from scipy.stats import bernoulli

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GradCAM_un_norm(GradCAM):
    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        # saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        # saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        saliency_map = saliency_map.data

        return saliency_map, logit


def get_CAM(normed_torch_img, model, num_classes):
    resnet = model
    # Considering architecture selected is resnet(18, 50, 101,..)
    configs = [dict(model_type='resnet', arch=resnet, layer_name='layer4'), ]
    for config in configs:
        config['arch'].to(device).eval()
    gradcam = GradCAM_un_norm.from_config(**config)  # Un-Normalized mask
    # gradcam = GradCAM.from_config(**config) # Normalized gradcam
    class_specific_maps = []
    for i in range(num_classes):
        mask, logits = gradcam(normed_torch_img, class_idx=i)
        class_specific_maps.extend([mask])
    return class_specific_maps


def diverse_block(model, class_specific_maps, kernel_size, alpha):
    num_classes = len(class_specific_maps)
    gap_all_classes = []
    activation_all_classes = []
    for i in range(num_classes):
        activation = class_specific_maps[i]
        rc = activation.max().item()
        P_peak = bernoulli.rvs(rc, size=1)  # Bernoulli prob for P peak: 0 or 1 randomly for c classes
        B, C, M, N = activation.shape
        Bc = torch.zeros(M, N)  # Initializing Final suppression mask

        # Peak Suppression
        max_ind = ((activation == activation.max()).nonzero(as_tuple=False))
        Pc = Bc
        Pc[max_ind[:, 0], max_ind[:, 1]] = 1
        Bc_dash = P_peak * Pc  # Peak suppression mask

        ## Patch Supression
        # patching image to G*G patches
        kernel_size = kernel_size  # G
        stride = kernel_size  # G*G patch
        patches = activation.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
        # num_patches = patches.shape[2] * patches.shape[3]
        l, m = patches.shape[2], patches.shape[3]

        # patching mask bc''(Bc_dd) to G*G patches (Bc_patch)
        # Ppatch by probability of patch wise maximums
        maxvals = torch.max(patches[0][0], 3)
        maxvals = torch.max(maxvals.values, 2)
        P_patch = maxvals.values.cpu().apply_(lambda x: (bernoulli.rvs(x, size=1)))
        P_patch_ind = torch.nonzero(torch.tensor(P_patch) == 1, as_tuple=False)
        Bc_dd = Bc  # Bc double dash allocating size with zeros
        Bc_patch = Bc_dd.unsqueeze(0).unsqueeze(0).unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
        # Bc_patch[0][0][P_patch_ind[:, 0]][P_patch_ind[:, 1]]=1 # Implementation of bc'' without for loop
        Bc_patch[0][0] = 0
        for i in range(len(P_patch_ind)):
            Bc_patch[0][0][P_patch_ind[i][0]][P_patch_ind[i][1]] = 1

        # Bc_patch folding back Bc''(Bc_dd)
        Bc_patch = (Bc_patch.reshape(1, 1, l, m, kernel_size * kernel_size)).permute(0, 1, 4, 2, 3).squeeze(0).squeeze(
            0)
        Bc_patch = Bc_patch.view(kernel_size * kernel_size, -1)
        Bc_dd = F.fold(Bc_patch.unsqueeze(0), (M, N), kernel_size=kernel_size, stride=stride)
        Bc_dd = Bc_dd.squeeze(0).squeeze(0)

        # Bc'' not supressing peak(peak=0)
        Bc_dd[max_ind[:, 0], max_ind[:, 1]] = 0

        # Final Supression Mask Bc
        Bc = Bc_dash + Bc_dd

        # Activation Supression Factor
        supress_ind = ((Bc == 1).nonzero(as_tuple=False))
        activation[0][supress_ind[:, 0], supress_ind[:, 1]] *= alpha
        activation_all_classes = torch.stack(activation.unsqueeze(1), dim=0)

        # Global Average Pool
        gap_activation = model.avgpool(activation)
        gap_all_classes = torch.stack(gap_activation, dim=1)

    return (gap_all_classes, activation_all_classes)

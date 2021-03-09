import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversificationBlock(nn.Module):
    def __init__(self, kernel_size, alpha):
        super(DiversificationBlock, self).__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha

    def forward(self, cam):
        activation = cam.clone().detach()
        min_val = activation.min(-1)[0].min(-1)[0]
        max_val = activation.max(-1)[0].max(-1)[0]
        activation = (activation - min_val[:, :, None, None]) / (max_val[:, :, None, None] - min_val[:, :, None, None])
        p_peak = torch.max(torch.max(activation, 3).values, 2).values
        rc = torch.bernoulli(p_peak)  # Bernoulli prob for P peak: 0 or 1 randomly for c classes
        b, c, m, n = activation.shape
        # Peak Suppression
        pc = torch.zeros_like(activation)  # Mask for peaks for each class
        pc[activation == torch.unsqueeze(torch.unsqueeze(p_peak, 2), 3)] = 1
        bc_dash = torch.mul(torch.unsqueeze(torch.unsqueeze(rc, 2), 3), pc)  # Peak suppression mask
        # Patch suppression
        # patching image to G*G patches
        kernel_size = self.kernel_size  # G
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
        cam[suppress_ind[:, 0], suppress_ind[:, 1]] *= self.alpha
        return cam

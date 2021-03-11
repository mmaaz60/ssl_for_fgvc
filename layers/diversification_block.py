import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversificationBlock(nn.Module):
    def __init__(self, kernel_size=3, alpha=0.1, p_peak=0.5, p_patch=0.5, device="cuda"):
        super(DiversificationBlock, self).__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.p_peak = p_peak
        self.p_patch = p_patch
        self.device = device

    def forward(self, activation):
        peak = torch.max(torch.max(activation, 3).values, 2).values
        # Bernoulli prob for P peak: 0 or 1 randomly for c classes
        rc = torch.bernoulli(torch.mul(torch.ones(activation.size(), device=self.device), torch.tensor(self.p_peak)))
        b, c, m, n = activation.shape
        # Peak Suppression
        pc = torch.zeros_like(activation)  # Mask for peaks for each class
        pc[activation == torch.unsqueeze(torch.unsqueeze(peak, 2), 3)] = 1
        bc_dash = torch.mul(rc, pc)  # Peak suppression mask
        # Patch suppression
        # patching image to G*G patches
        kernel_size = self.kernel_size  # G
        stride = kernel_size  # G*G patch
        patches = activation.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
        l, k = patches.shape[2], patches.shape[3]
        p_patch = torch.bernoulli(torch.mul(torch.ones(patches.size()[:-2],
                                                       device=self.device), torch.tensor(self.p_patch)))
        bc_dd = torch.zeros_like(patches)  # Mask for peaks for each class
        bc_dd[p_patch == 1] = 1
        bc_dd = (bc_dd.reshape(b, c, l, k, kernel_size * kernel_size)).permute(0, 1, 4, 2, 3)
        bc_dd = bc_dd.reshape(b, c, kernel_size * kernel_size, -1)
        bc_dd_batch = torch.zeros_like(activation)
        for i in range(b):
            bc_dd_batch[i] = F.fold(bc_dd[i], (m, n), kernel_size=kernel_size, stride=stride).squeeze(1)
        bc_dd_batch[activation == torch.unsqueeze(torch.unsqueeze(peak, 2), 3)] = 0
        bc = bc_dash + bc_dd_batch
        # Activation Suppression Factor
        activation[bc >= 1] *= self.alpha
        return activation

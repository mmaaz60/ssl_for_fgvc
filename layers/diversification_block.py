import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversificationBlock(nn.Module):
    """
    The class implements the Diversification Block (DB) introduced in the paper
    "Fine-grained Recognition: Accounting for Subtle Differences between Similar Classes".
    (http://arxiv.org/abs/1912.06842).
    """
    def __init__(self, kernel_size, alpha, p_peak=0.5, p_patch=0.5, device="cuda"):
        """
        Constructor, the function initializes the DB parameters provided by the user.

        :param kernel_size: The kernel size patch suppression
        :param alpha: Suppression factor
        :param p_peak: Probability for peak suppression
        :param p_patch: Probability for patch suppression
        :param device: Device of execution
        """
        # Call the parent constructor
        super(DiversificationBlock, self).__init__()
        # Initialize the parameters
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.p_peak = p_peak
        self.p_patch = p_patch
        self.device = device

    def forward(self, activation):
        """
        The function implements the forward pass for the DB.

        :param activation: The class activation maps (CAMs) to apply the suppression on.
        """
        peak = torch.max(torch.max(activation, 3).values, 2).values  # Find the peak location in CAMs
        # Bernoulli prob for p_peak: 0 or 1 randomly for c classes
        rc = torch.bernoulli(torch.mul(torch.ones(activation.size(), device=self.device), torch.tensor(self.p_peak)))
        b, c, m, n = activation.shape
        # Peak Suppression
        pc = torch.zeros_like(activation)  # Mask for peaks for each class
        pc[activation == torch.unsqueeze(torch.unsqueeze(peak, 2), 3)] = 1
        bc_dash = torch.mul(rc, pc)  # Peak suppression mask

        # Patch suppression
        # Patching image to 'kernel_size x kernel_size' patches
        stride = self.kernel_size  # Kernel size
        # Extract patches
        patches = activation.unfold(2, self.kernel_size, stride).unfold(3, self.kernel_size, stride)
        l, k = patches.shape[2], patches.shape[3]
        # Bernoulli prob for p_patch
        p_patch = torch.bernoulli(torch.mul(torch.ones(patches.size()[:-2],
                                                       device=self.device), torch.tensor(self.p_patch)))
        bc_dd = torch.zeros_like(patches)  # Mask initialization for peaks for each patch
        bc_dd[p_patch == 1] = 1  # Sets value 1 to suppress at random peak locations
        # Combines mask patches to single mask
        bc_dd = (bc_dd.reshape(b, c, l, k, self.kernel_size * self.kernel_size)).permute(0, 1, 4, 2, 3)
        bc_dd = bc_dd.reshape(b, c, self.kernel_size * self.kernel_size, -1)
        bc_dd_batch = torch.zeros_like(activation)
        for i in range(b):
            bc_dd_batch[i] = F.fold(bc_dd[i], (m, n), kernel_size=self.kernel_size, stride=stride).squeeze(1)
        bc_dd_batch[activation == torch.unsqueeze(torch.unsqueeze(peak, 2), 3)] = 0
        # Mask for total suppression
        bc = bc_dash + bc_dd_batch
        # Suppress the activations using activation suppression factor called alpha
        activation[bc >= 1] *= self.alpha

        return activation

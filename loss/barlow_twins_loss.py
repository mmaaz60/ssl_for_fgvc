"""
Credits: The code has been taken from https://github.com/IgorSusmelj/barlowtwins
"""

import torch
import torch.nn as nn


class BarlowTwinsLoss(nn.Module):
    """
    The class implements the Barlow Twins loss introduced in the paper
    "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (https://arxiv.org/abs/2103.03230).
    """

    def __init__(self, device="cuda", lambda_param=5e-3):
        """
        Constructor, the function initialize the parameters.

        :param device: Device of execution
        :param lambda_param: The loss scaling factor
        """

        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a, z_b):
        """
        The function implements the forward pass for the loss.

        :param z_a: The representation from the first view of the original image
        :param z_b: The representation from the second view of the original image
        """

        # Normalize the representations along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # Calculate the cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD
        # Calculate the loss
        c_diff = (c - torch.eye(D, device=self.device)).pow(2)  # DxD
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param  # Multiply off-diagonal elements of c_diff by lambda
        loss = c_diff.sum()  # Sum the elements to calculate the loss

        return loss

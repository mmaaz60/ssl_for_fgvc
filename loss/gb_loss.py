import torch
import torch.nn as nn


class GBLoss(torch.nn.Module):
    """
    The class implements the gradient-boosting (GB) loss introduced in
    "Fine-grained Recognition: Accounting for Subtle Differences between Similar Classes".
    (http://arxiv.org/abs/1912.06842).
    """
    def __init__(self):
        """
        Constructor, initialize the base cross-entropy loss
        """
        # Call the parent constructor
        super(GBLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()  # Cross entropy loss

    def forward(self, x, y):
        """
        The function implements the forward pass for the GB loss.

        :param x: Predictions
        :param y: Ground truth labels
        """
        x1 = x.clone()
        x1[range(x1.size(0)), y] = -float("Inf")
        x_gt = x[range(x.size(0)), y].unsqueeze(1)
        x_topk = torch.topk(x1, 15, dim=1)[0]  # 15 Negative classes to focus on, its a hyperparameter
        x_new = torch.cat([x_gt, x_topk], dim=1)

        return self.ce(x_new, torch.zeros(x_new.size(0)).cuda().long())

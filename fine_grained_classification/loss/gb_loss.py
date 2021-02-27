import torch
import torch.nn as nn


class GBLoss(torch.nn.Module):
    # selected cross_entropy
    def __init__(self):
        super(GBLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x1 = x.clone()
        x1[range(x1.size(0)), y] = -float("Inf")
        x_gt = x[range(x.size(0)), y].unsqueeze(1)
        x_topk = torch.topk(x1, 15, dim=1)[0]  # 15 negative classes to focus on, hyperparameter
        x_new = torch.cat([x_gt, x_topk], dim=1)

        return self.ce(x_new, torch.zeros(x_new.size(0)).cuda().long())

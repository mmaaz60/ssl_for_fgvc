"""
Credits: The code has been taken from https://github.com/HobbitLong/PyContrast/tree/master/pycontrast1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .alias_multinomial import AliasMethod


class BaseMem(nn.Module):
    """Base Memory Class"""
    def __init__(self, K=1024, T=0.07, m=0.5):
        super(BaseMem, self).__init__()
        self.K = K
        self.T = T
        self.m = m

    def _update_memory(self, memory, x, y):
        """
        Args:
          memory: memory buffer
          x: features
          y: index of updating position
        """
        with torch.no_grad():
            x = x.detach()
            w_pos = torch.index_select(memory, 0, y.view(-1))
            w_pos.mul_(self.m)
            w_pos.add_(torch.mul(x, 1 - self.m))
            updated_weight = F.normalize(w_pos)
            memory.index_copy_(0, y, updated_weight)

    def _compute_logit(self, x, w):
        """
        Args:
          x: feat, shape [bsz, n_dim]
          w: softmax weight, shape [bsz, self.K + 1, n_dim]
        """
        x = x.unsqueeze(2)
        out = torch.bmm(w, x)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()
        return out


class RGBMem(BaseMem):
    """Memory bank for single modality"""
    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5):
        super(RGBMem, self).__init__(K, T, m)
        # create sampler
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial.cuda()

        # create memory bank
        self.register_buffer('memory', torch.randn(n_data, n_dim))
        self.memory = F.normalize(self.memory)

    def forward(self, x, y, x_jig=None, all_x=None, all_y=None):
        """
        Args:
          x: feat on current node
          y: index on current node
          x_jig: jigsaw feat on current node
          all_x: gather of feats across nodes; otherwise use x
          all_y: gather of index across nodes; otherwise use y
        """
        bsz = x.size(0)
        n_dim = x.size(1)

        # sample negative features
        idx = self.multinomial.draw(bsz * (self.K + 1)).view(bsz, -1)
        idx.select(1, 0).copy_(y.data)
        w = torch.index_select(self.memory, 0, idx.view(-1))
        w = w.view(bsz, self.K + 1, n_dim)

        # compute logits
        logits = self._compute_logit(x, w)
        if x_jig is not None:
            logits_jig = self._compute_logit(x_jig, w)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        if (all_x is not None) and (all_y is not None):
            self._update_memory(self.memory, all_x, all_y)
        else:
            self._update_memory(self.memory, x, y)

        if x_jig is not None:
            return logits, logits_jig, labels
        else:
            return logits, labels

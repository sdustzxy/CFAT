import torch
import torch.nn as nn
from torch.nn import functional as F


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 256, 1, 1)

    def forward(self, x):
        x = torch.einsum('nhwc->nchw', x)
        up = F.interpolate(x, scale_factor=16, mode='nearest')
        out = self.layer(up)
        return out

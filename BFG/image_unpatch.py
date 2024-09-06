import torch
import torch.nn as nn


class Unpatch(nn.Module):
    def __init__(self, patch_height=16, patch_weight=8, patch_size=16, in_channel=3):
        super(Unpatch, self).__init__()
        self.patch_height = patch_height
        self.patch_weight = patch_weight
        self.patch_size = patch_size
        self.in_channel = in_channel

    def forward(self, x):
        # x=[64, 129, 768]
        x = x[:, 1:, :]  # [64, 128, 768]
        x = x.reshape(shape=(x.shape[0], self.patch_height, self.patch_weight, self.patch_size, self.patch_size, self.in_channel))  # [64, 16, 8, 16, 16, 3]
        x = torch.einsum('nhwpqc->nchpwq', x)  # [64, 3, 16, 16, 8, 16]
        x = x.reshape(x.shape[0], self.in_channel, self.patch_height*self.patch_size, self.patch_weight*self.patch_size)  # [64, 3, 256, 128]
        return x

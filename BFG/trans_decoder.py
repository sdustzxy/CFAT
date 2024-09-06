import torch
import torch.nn as nn


class trans_decoder(nn.Module):
    def __init__(self):
        super(trans_decoder, self).__init__()
        self.transpose = nn.ConvTranspose2d(768, 3, kernel_size=16, stride=16)

    def forward(self, x):
        x = torch.einsum('nhwc->nchw', x)
        x = self.transpose(x)  # [64, 3, 256, 128]
        return x

import torch
import torch.nn as nn


class micro(nn.Module):
    def __init__(self):
        super(micro, self).__init__()

    def forward(self, feat_clear, feat_cor, t):
        for i in range(feat_clear.shape[0]):
            if i == 0:
                numerator = t * feat_cor[i] + (1 - t) * feat_clear[i]
                denominator = torch.sqrt(t ** 2 + (1 - t) ** 2 + 2 * t * (1 - t) * torch.dot(feat_clear[i], feat_cor[i]))
                result = numerator / denominator
                all_result = result.unsqueeze(0)
            else:
                numerator = t * feat_cor[i] + (1 - t) * feat_clear[i]
                denominator = torch.sqrt(t ** 2 + (1 - t) ** 2 + 2 * t * (1 - t) * torch.dot(feat_clear[i], feat_cor[i]))
                result = numerator / denominator
                all_result = torch.cat([all_result, result.unsqueeze(0)], dim=0)
        return all_result

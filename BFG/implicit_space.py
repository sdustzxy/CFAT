import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_implicit(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class implicit_feat(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, hidden_dim=256):
        super(implicit_feat, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias=False)

        self.linear1.apply(weights_init_implicit)
        self.linear2.apply(weights_init_implicit)
        self.linear3.apply(weights_init_implicit)

    def forward(self, cor_feat):
        imp_feat = self.linear1(cor_feat)
        imp_feat = self.linear2(imp_feat)
        imp_feat = self.linear3(imp_feat)
        return imp_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained implicit_model from {}'.format(trained_path))


class micro(nn.Module):
    def __init__(self):
        super(micro, self).__init__()

    def forward(self, feat_clear, feat_cor, t):
        numerator = t * feat_cor + (1 - t) * feat_clear
        denominator = torch.sqrt(t ** 2 + (1 - t) ** 2 + 2 * t * (1 - t) * torch.dot(feat_clear, feat_cor))

        result = numerator / denominator
        return result


class BrownianBridgeLoss(nn.Module):
    def __init__(self):
        super(BrownianBridgeLoss, self).__init__()
        self.micro = micro()
        self.mse = nn.MSELoss()

    def forward(self, feat_clear, feat_cor, feat_mid, t):
        feat_cl = F.normalize(feat_clear, dim=1)
        feat_co = F.normalize(feat_cor, dim=1)
        feat_mi = F.normalize(feat_mid, dim=1)
        loss = 0.0
        for i in range(feat_mid.shape[0]):
            micro_feat = self.micro(feat_cl[i], feat_co[i], t)
            sim_cm1 = F.cosine_similarity(feat_cl[i].unsqueeze(0), micro_feat.unsqueeze(0), dim=1)
            sim_cm2 = F.cosine_similarity(feat_co[i].unsqueeze(0), micro_feat.unsqueeze(0), dim=1)
        loss += brownian_bridge_loss / feat_cor.shape[0]
        return loss * 1e6

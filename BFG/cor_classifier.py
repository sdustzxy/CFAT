import torch
import torch.nn as nn


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class classifier(nn.Module):
    def __init__(self, id_feature, cor_feature, num_classes, cor_classes):
        super(classifier, self).__init__()
        self.id_feature = id_feature
        self.num_classes = num_classes
        self.cor_feature = cor_feature
        self.cor_classes = cor_classes
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.bn_num = nn.BatchNorm1d(num_features=768)
        self.bn_cor = nn.BatchNorm1d(num_features=768)
        self.id_classifier = nn.Linear(self.id_feature,
                                       self.num_classes,
                                       bias=False)
        self.cor_classifier = nn.Linear(self.cor_feature,
                                        self.cor_classes,
                                        bias=False)
        self.id_classifier.apply(weights_init_classifier)
        self.cor_classifier.apply(weights_init_classifier)

    def forward(self, id_feat, cor_feat):
        # id_feat[64, 16, 8, 768], cor_feat[64, 16, 8, 768]
        id_feat = torch.einsum('nhwc->nchw', id_feat)
        id_feat = self.global_pooling(id_feat)
        id_feat = id_feat.view(id_feat.size(0), -1)
        id_feat_bn = self.bn_num(id_feat)

        cor_feat = torch.einsum('nhwc->nchw', cor_feat)
        cor_feat = self.global_pooling(cor_feat)
        cor_feat = cor_feat.view(cor_feat.size(0), -1)
        cor_feat_bn = self.bn_num(cor_feat)

        id_score = self.id_classifier(id_feat_bn)
        cor_score = self.cor_classifier(cor_feat)

        return id_score, cor_score

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained classifier model from {}'.format(trained_path))

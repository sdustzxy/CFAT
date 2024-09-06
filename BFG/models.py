import torch
import torch.nn as nn


class make_models(nn.Module):
    def __init__(self, model, feat_sep, classifier, decoder):
        super(make_models, self).__init__()
        self.model = model
        self.feat_sep = feat_sep
        self.classifier = classifier
        self.decoder = decoder

    def forward(self, x, camids, target_view):
        feat, feat_ori = self.model(x,
                                    cam_label=camids,
                                    view_label=target_view)
        img_clear, img_corrupt = self.feat_sep(feat_ori)
        id_score, cor_score = self.classifier(img_clear, img_corrupt)
        rec_clean_image, rec_feat = self.decoder(img_clear, feat)
        # models = nn.Sequential(self.model, self.feat_sep, self.classifier, self.decoder)
        return rec_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

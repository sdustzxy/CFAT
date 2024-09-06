import torch
import torch.nn as nn
import torch.nn.functional as F


class KL_Loss1(nn.Module):
    def __init__(self):
        super(KL_Loss1, self).__init__()

    def forward(self, mid_feat, re_feat):
        """
        p1 = N(mu1, sigma1 ** 2), p2 = N(mu2, sigma2 ** 2)
        KL(p1||p2) = log(sigma2 / sigma1) + 1 / (2 * sigma2 ** 2) * (sigma1 ** 2 + (mu1 - mu2) ** 2) - 1 / 2
        """
        # loss_kl = F.kl_div(torch.log(mid_feat), re_feat)
        loss_kl = F.kl_div(mid_feat, re_feat)
        return loss_kl


class KL_Loss2(nn.Module):
    def __init__(self):
        super(KL_Loss2, self).__init__()

    def forward(self, mu_re, sigma_re, mu_target, sigma_target):
        """
        p1 = N(mu1, sigma1 ** 2), p2 = N(mu2, sigma2 ** 2)
        KL(p1||p2) = log(sigma2 / sigma1) + 1 / (2 * sigma2 ** 2) * (sigma1 ** 2 + (mu1 - mu2) ** 2) - 1 / 2
        """
        mu_re = F.normalize(mu_re)
        sigma_re = F.normalize(sigma_re)
        mu_target = F.normalize(mu_target)
        sigma_target = F.normalize(sigma_target)
        # with open('./mu_re.txt', 'a') as a:
        #     print(mu_re, file=a)
        # with open('./sigma_re.txt', 'a') as b:
        #     print(sigma_re, file=b)
        # with open('./mu_target.txt', 'a') as c:
        #     print(mu_target, file=c)
        # with open('./sigma_target.txt', 'a') as d:
        #     print(sigma_target, file=d)
        # mid_pro1 = torch.sum(torch.log(sigma_target / sigma_re))
        # mid_pro2 = torch.sum(1 / (2 * sigma_target ** 2))
        # mid_pro3 = torch.sum(sigma_re ** 2 + (mu_re - mu_target) ** 2)
        # mid_pro4 = torch.sum((1 / (2 * sigma_target ** 2)) * (sigma_re ** 2 + (mu_re - mu_target) ** 2))
        # with open('./log(sigma2_div_sigma1).txt', 'a') as e:
        #     print(mid_pro1, file=e)
        # with open('./1_div_(2_mut_sigma2_pow.txt', 'a') as f:
        #     print(mid_pro2, file=f)
        # with open('./(sigma1_pow_plus_(mu1_sub_mu2)_pow).txt', 'a') as g:
        #     print(mid_pro3, file=g)
        # with open('./(1_div_(2_mut_sigma2_pow)_mul_(sigma1_pow_plus_(mu1_sub_mu2)_pow).txt', 'a') as h:
        #     print(mid_pro4, file=h)
        loss_kl = torch.sum(torch.log(sigma_target / sigma_re) + (1 / (2 * sigma_target ** 2))
                            * (sigma_re ** 2 + (mu_re - mu_target) ** 2) - 0.5)
        return loss_kl


class KL_Loss3(nn.Module):
    def __init__(self):
        super(KL_Loss3, self).__init__()

    def forward(self, mu_re, sigma_re):
        """
        p1 = N(mu1, sigma1 ** 2), p2 = N(mu2, sigma2 ** 2)
        KL(p1||p2) = log(sigma2 / sigma1) + 1 / (2 * sigma2 ** 2) * (sigma1 ** 2 + (mu1 - mu2) ** 2) - 1 / 2
        """
        mu_re = F.normalize(mu_re)
        sigma_re = F.normalize(sigma_re)
        # loss_kl = lambda mu, sigma: -0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu.pow(2) - sigma ** 2)
        loss_kl = -0.5 * torch.mean(1 + torch.log(sigma_re ** 2) - mu_re.pow(2) - sigma_re ** 2)
        return loss_kl

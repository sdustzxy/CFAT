import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)

        # self.linear.apply(weights_init_vae)
        # self.mu.apply(weights_init_vae)
        # self.sigma.apply(weights_init_vae)

    def forward(self, x):
        x = F.relu(self.linear(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        # self.linear1.apply(weights_init_vae)
        # self.linear2.apply(weights_init_vae)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


class BFG(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim):
        super(BFG, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(hidden_dim, latent_dim, output_dim)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        re_x = self.decoder(z)
        return re_x, mu, sigma

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained implicit_model from {}'.format(trained_path))

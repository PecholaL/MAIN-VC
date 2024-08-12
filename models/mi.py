"""MI module
    Modified from: https://github.com/Linear95/CLUB
"""

import torch
import numpy
import torch.nn as nn


class CLUBSample_group(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample_group, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        mu = (
            mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, mu.shape[-1])
        )  # (bs, y_dim) -> (bs, 1, y_dim) -> (bs, T, y_dim) -> (bs*T, y_dim)
        logvar = (
            logvar.unsqueeze(1)
            .expand(-1, y_samples.shape[1], -1)
            .reshape(-1, logvar.shape[-1])
        )
        y_samples = y_samples.reshape(
            -1, y_samples.shape[-1]
        )  # (bs, T, y_dim) -> (bs*T, y_dim)
        return (-((mu - y_samples) ** 2) / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def mi_est(self, x_samples, y_samples):
        x_samples = (
            x_samples.unsqueeze(1)
            .expand(-1, y_samples.shape[1], -1)
            .reshape(-1, x_samples.shape[-1])
        )
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])

        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = -((mu - y_samples) ** 2) / logvar.exp()
        negative = -((mu - y_samples[random_index]) ** 2) / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() / 2


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_samples, y_samples):
        x_samples = (
            x_samples.unsqueeze(1)
            .expand(-1, y_samples.shape[1], -1)
            .reshape(-1, x_samples.shape[-1])
        )
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])
        sample_size = y_samples.shape[0]
        tiled_x = torch.cat([x_samples, x_samples], dim=0)
        random_index = torch.randperm(sample_size)
        y_shuffle = y_samples[random_index]
        concat_y = torch.cat([y_samples, y_shuffle], dim=0)
        inputs = torch.cat([tiled_x, concat_y])
        logits = self.T_func(inputs)
        tmp_1 = logits[:sample_size]
        tmp_2 = logits[sample_size:]
        lower_bound = numpy.log2(numpy.exp(1)) * (
            torch.mean(tmp_1) - torch.log(torch.mean(torch.exp(tmp_2)))
        )

        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


"""
# __________test__________
import yaml
with open("../config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
mine_size = config["CMI"]["mine"]
club_size = config["CMI"]["club"]
mi_club = CLUBSample_group(club_size, club_size, club_size).to("cpu")
mi_mine = MINE(mine_size//2, mine_size//2, mine_size).to("cpu")

mu = torch.rand(1, 64, 16)
emb = torch.rand(1, 64)

mu = mu.transpose(1, 2)
club_loss = -mi_club.loglikeli(emb, mu)
mine_loss = mi_mine.learning_loss(emb, mu)
club_mi_est = mi_club.mi_est(emb, mu)
mine_mi_est = mi_mine(emb, mu)
print(club_loss)
print(mine_loss)
print(club_mi_est)
print(mine_mi_est)

"""

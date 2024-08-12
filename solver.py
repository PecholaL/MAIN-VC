"""training strategy of MAIN-VC
"""

import os
import torch
import torch.nn as nn
import yaml
import random
import itertools

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from data.PickleDataset import PickleDataset, get_data_loader
from models.mi import CLUBSample_group, MINE
from models.model import MAINVC
from models.tools import *


class Solver(object):
    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.logger = SummaryWriter(self.args.log_dir)

        self.get_data_loaders()

        self.build_model()
        self.save_config()

        if args.load_model:
            self.load_model()

    def time_shuffle(self, data):
        seg_list = list(torch.split(data, 20, dim=2))
        random.shuffle(seg_list)
        return torch.cat(seg_list, dim=2)

    def save_model(self, iteration):
        torch.save(self.model.state_dict(), f"{self.args.store_model_path}.ckpt")
        torch.save(self.opt.state_dict(), f"{self.args.store_model_path}.opt")

    def save_config(self):
        with open(f"{self.args.store_model_path}.config.yaml", "w") as f:
            yaml.dump(self.config, f)
        with open(f"{self.args.store_model_path}.args.yaml", "w") as f:
            yaml.dump(vars(self.args), f)
        return

    def load_model(self):
        print(f"[MAIN-VC]load model from {self.args.load_model_path}")
        self.model.load_state_dict(torch.load(f"{self.args.load_model_path}.ckpt"))
        self.opt.load_state_dict(torch.load(f"{self.args.load_model_path}.opt"))
        return

    def get_data_loaders(self):
        data_dir = self.args.data_dir
        self.train_dataset = PickleDataset(
            os.path.join(data_dir, f"{self.args.train_set}.pkl"),
            os.path.join(data_dir, self.args.train_index_file),
            segment_size=self.config["data_loader"]["segment_size"],
        )
        self.train_loader = get_data_loader(
            self.train_dataset,
            frame_size=self.config["data_loader"]["frame_size"],
            batch_size=self.config["data_loader"]["batch_size"],
            shuffle=self.config["data_loader"]["shuffle"],
            num_workers=4,
            drop_last=False,
        )
        self.train_iter = infinite_iter(self.train_loader)
        return

    def build_model(self):
        self.model = cc(MAINVC(self.config))
        print("[MAIN-VC]model built")
        print(
            "[MAIN-VC]total parameter count: {}".format(
                sum(x.numel() for x in self.model.parameters())
            )
        )
        optimizer = self.config["optimizer"]
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=optimizer["lr"],
            betas=(optimizer["beta1"], optimizer["beta2"]),
            amsgrad=optimizer["amsgrad"],
            weight_decay=optimizer["weight_decay"],
        )
        scheduler = self.config["scheduler"]
        self.sched = StepLR(
            self.opt, step_size=scheduler["step_size"], gamma=scheduler["gamma"]
        )
        print("[MAIN-VC]optimizer built")
        mine_size = self.config["CMI"]["mine"]
        club_size = self.config["CMI"]["club"]
        self.mi_club = CLUBSample_group(club_size, club_size, club_size).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mi_mine = MINE(mine_size // 2, mine_size // 2, mine_size).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sia_activate = self.config["sia_activate"]
        self.cmi_activate = self.config["cmi_activate"]
        self.cmi_steps = self.config["cmi_steps"]
        print("[MAIN-VC]MI net built")
        self.mi_opt = torch.optim.Adam(
            itertools.chain(self.mi_club.parameters(), self.mi_mine.parameters()),
            lr=1e-4,
        )
        print("[MAIN-VC]MI optimizer built")
        return

    def mainvc_step(self, data1, data2, lambda_kl, lambda_mi):
        data1 = data1.float()
        data2 = data2.float()
        x = cc(data1)
        x_sf = cc(self.time_shuffle(data1))
        x_ = cc(data2)
        mu, log_sigma, emb, emb_, dec = self.model(x, x_sf, x_)

        # loss
        criterion = nn.L1Loss()
        cos = nn.CosineEmbeddingLoss(reduction="mean")
        loss_flag = torch.ones([emb.shape[0]]).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        emb = emb.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        emb_ = emb_.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # reconstruction loss
        loss_rec = criterion(dec, x)
        # KL loss
        loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu**2 - 1 - log_sigma)
        # siamese loss
        loss_sia = cos(emb, emb_, loss_flag)

        # CMI first forward
        if self.cmi_activate:
            for _ in range(self.cmi_steps):
                self.mi_opt.zero_grad()
                mu_tmp = mu.transpose(1, 2)
                emb_tmp = emb
                mu_tmp = mu_tmp.detach()
                emb_tmp = emb_tmp.detach()
                self.mi_club.train()
                self.mi_mine.train()
                # jointly train CLUB and MINE
                self.club_loss = -self.mi_club.loglikeli(emb_tmp, mu_tmp)
                self.mine_loss = self.mi_mine.learning_loss(emb_tmp, mu_tmp)
                delta = self.mi_club.mi_est(emb_tmp, mu_tmp) - self.mi_mine(
                    emb_tmp, mu_tmp
                )
                gap_loss = delta if delta > 0 else 0
                mimodule_loss = self.club_loss + self.mine_loss + gap_loss
                mimodule_loss.backward(retain_graph=True)
                self.mi_opt.step()

        # CMI second forward
        # MI loss
        loss_mi = self.mi_club.mi_est(emb, mu.transpose(1, 2))

        # total loss
        lambda_sia = self.config["lambda"]["lambda_sia"] if self.sia_activate else 0
        lambda_mi = lambda_mi if self.cmi_activate else 0
        loss = (
            self.config["lambda"]["lambda_rec"] * loss_rec
            + lambda_kl * loss_kl
            + lambda_sia * loss_sia
            + lambda_mi * loss_mi
        )
        # backward
        self.opt.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config["optimizer"]["grad_norm"]
        )
        self.opt.step()
        self.sched.step()
        meta = {
            "loss_rec": loss_rec.item(),
            "loss_kl": loss_kl.item(),
            "loss_sia": loss_sia.item() if self.sia_activate else 0,
            "loss_mi": loss_mi.item(),
            "club_loss": self.club_loss.item() if self.cmi_activate else 0,
            "mine_loss": self.mine_loss.item() if self.cmi_activate else 0,
            "grad_norm": grad_norm,
        }
        return meta

    def train(self, n_iterations):
        for iteration in range(n_iterations):
            if iteration >= self.config["annealing_iters"]:
                lambda_kl = self.config["lambda"]["lambda_kl"]
                lambda_mi = self.config["lambda"]["lambda_mi"]
            else:
                lambda_kl = (
                    self.config["lambda"]["lambda_kl"]
                    * (iteration + 1)
                    / self.config["annealing_iters"]
                )
                lambda_mi = (
                    self.config["lambda"]["lambda_mi"]
                    * (iteration + 1)
                    / self.config["annealing_iters"]
                )
            data = next(self.train_iter)

            # log-tensorboard
            meta = self.mainvc_step(*data, lambda_kl, lambda_mi)
            if iteration % self.args.summary_steps == 0:
                self.logger.add_scalars(
                    f"{self.args.tag}/mainvc_train", meta, iteration
                )
            # log-terminal
            loss_rec = meta["loss_rec"]
            loss_kl = meta["loss_kl"]
            loss_sia = meta["loss_sia"]
            loss_mi = meta["loss_mi"]
            club_loss = meta["club_loss"]
            mine_loss = meta["mine_loss"]
            print(
                f"[MAIN-VC]:[{iteration + 1}/{n_iterations}]",
                f"loss_rec={loss_rec:.6f}",
                f"loss_kl={loss_kl:.6f}",
                f"loss_sia={loss_sia:.6f}",
                f"loss_mi={loss_mi:.6f}",
                f"club_loss={club_loss:.6f}",
                f"mine_loss={mine_loss:.6f}",
                end="\r",
            )
            # autosave
            if (
                iteration + 1
            ) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()
        self.logger.close()
        return

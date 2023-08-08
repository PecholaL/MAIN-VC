"""training strategy of MAIN-VC
"""

import torch
import os 
import torch.nn as nn
import yaml
import sys
import random

sys.path.append("..")
from data.PickleDataset import PickleDataset, get_data_loader
from models.club import CLUBSample_group
from models.model import MAINVC
from models.tools import *


class Solver(object):
    def __init__(self, config, args):
        self.config = config

        self.args = args

        self.logger = Logger(self.args.logdir)

        self.get_data_loaders()

        self.build_model()
        self.save_config()

        if args.load_model:
            self.load_model()


    def time_shuffle(self, data):
        seg_list = list(torch.split(data, 5, dim=2))
        random.shuffle(seg_list)
        return torch.cat(seg_list, dim=2)


    def save_model(self, iteration):
        torch.save(self.model.state_dict(), f'{self.args.store_model_path}.ckpt')
        torch.save(self.opt.state_dict(), f'{self.args.store_model_path}.opt')


    def save_config(self):
        with open(f'{self.args.store_model_path}.config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}.args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return


    def load_model(self):
        print(f'[MAIN-VC]load model from {self.args.load_model_path}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}.ckpt'))
        self.opt.load_state_dict(torch.load(f'{self.args.load_model_path}.opt'))
        return


    def get_data_loaders(self):
        data_dir = self.args.data_dir
        self.train_dataset = PickleDataset(
            os.path.join(data_dir, f'{self.args.train_set}.pkl'), 
            os.path.join(data_dir, self.args.train_index_file), 
            segment_size=self.config['data_loader']['segment_size']
        )
        self.train_loader = get_data_loader(
            self.train_dataset,
            frame_size=self.config['data_loader']['frame_size'],
            batch_size=self.config['data_loader']['batch_size'], 
            shuffle=self.config['data_loader']['shuffle'], 
            num_workers=4, drop_last=False
        )
        self.train_iter = infinite_iter(self.train_loader)
        return


    def build_model(self): 
        self.model = cc(MAINVC(self.config))
        print('[MAIN-VC]model built')
        print('[MAIN-VC]total parameter count: {}'.format(sum(x.numel() for x in self.model.parameters())))
        optimizer = self.config['optimizer']
        self.opt = torch.optim.Adam(self.model.parameters(), 
            lr=optimizer['lr'], betas=(optimizer['beta1'], optimizer['beta2']), 
            amsgrad=optimizer['amsgrad'], weight_decay=optimizer['weight_decay'])
        print('[MAIN-VC]optimizer built')
        self.mi_net = CLUBSample_group(64,64,64).to('cuda' if torch.cuda.is_available() else 'cpu')
        print('[MAIN-VC]MI net built')
        self.mi_opt = torch.optim.Adam(self.mi_net.parameters(), lr=1e-4)
        print('[MAIN-VC]MI optimizer built')
        return


    def mainvc_step(self, data1, data2, lambda_kl):
        data1 = data1.float()
        data2 = data2.float()
        x = cc(data1)
        x_sf = cc(self.time_shuffle(data1))
        x_ = cc(data2)
        mu, log_sigma, emb, dec = self.model(x, x_sf)
        
        # loss
        criterion = nn.L1Loss()
        cos = nn.CosineEmbeddingLoss(reduction='mean')
        loss_flag = torch.ones([emb.shape[0]]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # reconstruction loss
        loss_rec = criterion(dec, x)
        # KL loss
        loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
        emb = emb.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        emb_ = emb_.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # siamese loss
        loss_sia = cos(emb, emb_, loss_flag)
        
        # MI first forward
        for _ in range(3):
            self.mi_opt.zero_grad()
            mu_tmp = mu.transpose(1,2)
            emb_tmp = emb
            mu_tmp = mu_tmp.detach()
            emb_tmp = emb_tmp.detach()
            self.mi_net.train()
            mi_loss = -self.mi_net.loglikeli(emb_tmp, mu_tmp)
            self.mi_opt.step()

        # MI second forward
        # MI loss
        loss_mi = -self.mi_net.mi_est(emb, mu.transpose(1,2))
        # total loss
        loss = self.config['lambda']['lambda_rec'] * loss_rec + lambda_kl * loss_kl + loss_sia + 0.01 * loss_mi
        # backward 
        self.opt.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                max_norm=self.config['optimizer']['grad_norm'])
        self.opt.step()
        meta = {'loss_rec': loss_rec.item(),
                'loss_kl': loss_kl.item(),
                'loss_sia': loss_sia.item(),
                'loss_mi': loss_mi.item(),
                'grad_norm': grad_norm}
        return meta


    def train(self, n_iterations):
        for iteration in range(n_iterations):
            if iteration >= self.config['annealing_iters']:
                lambda_kl = self.config['lambda']['lambda_kl']
            else:
                lambda_kl = self.config['lambda']['lambda_kl'] * (iteration + 1) / self.config['annealing_iters'] 
            data = next(self.train_iter)

            # logger
            meta = self.ae_step(data, lambda_kl)
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/ae_train', meta, iteration)
            loss_rec = meta['loss_rec']
            loss_kl = meta['loss_kl']
            loss_sia = meta['loss_sia']
            loss_mi = meta['loss_mi']
            print(f'[MAIN-VC]:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.6f}, '
                f'loss_kl={loss_kl:.6f}', f'loss_sia={loss_sia:.6f}', f'loss_mi={loss_mi:.6f}', end='\r')
            # autosave
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()
        self.logger.writer.close()
        return


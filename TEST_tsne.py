import matplotlib.pyplot as pyplot
import matplotlib.patheffects as pe
import numpy
import os
import pickle
import seaborn
import torch
import torch.nn.functional as functional
import yaml

from functools import reduce
from sklearn.manifold import TSNE

from models.model import MAINVC
from models.tools import *
from data.utils import *
from data.preprocess import HyperParameters as hp


class TSNE(object):
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.build_load()

        self.spk_path = []
        self.emb_list = []

        # mean, std
        with open(self.args.attr, "rb") as f:
            self.attr = pickle.load(f)
            self.m, self.s = self.attr["mean"], self.attr["std"]

    def build_load(self):
        self.model = cc(MAINVC(self.config))
        self.model.eval()
        self.model.load_state_dict(torch.load(f"{self.args.model}"))
        return

    def normalize(self, x):
        ret = (x - self.m) / self.s
        return ret

    def denormalize(self, x):
        ret = x * self.s + self.m
        return ret

    def utt_make_frame(self, x):
        frame_size = self.config["data_loader"]["frame_size"]
        remains = x.size(0) % frame_size
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, x.size(1) * frame_size).transpose(1, 2)
        return out
    
    def get_spk_embeddings(self, spk_path):
        

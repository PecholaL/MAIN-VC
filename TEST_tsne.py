import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy
import os
import pickle
import seaborn
import torch
import yaml

from argparse import ArgumentParser
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

    def get_spk_embeddings(self, spk_path, num):
        file_list = os.listdir(spk_path)
        for i in range(num):
            file_path = os.path.join(spk_path, file_list[i])
            wav = load_wav(file_path, hp.sr)
            mel = log_mel_spectrogram(
                wav,
                hp.preemph,
                hp.sr,
                hp.n_mels,
                hp.n_fft,
                hp.hop_len,
                hp.win_len,
                hp.f_min,
            )
            mel = torch.from_numpy(self.normalize(mel)).float().cuda()
            x = self.utt_make_frame(mel)
            emb = self.model.get_speaker_embedding(x)
            self.emb_list.append(emb.cpu().detach().numpy())
        return

    def get_spk_paths(self):
        self.spk_path.append(self.args.spk1)
        self.spk_path.append(self.args.spk2)
        self.spk_path.append(self.args.spk3)
        self.spk_path.append(self.args.spk4)
        self.spk_path.append(self.args.spk5)
        self.spk_path.append(self.args.spk6)
        self.spk_path.append(self.args.spk7)
        self.spk_path.append(self.args.spk8)
        return

    def set_Y(self, num):
        list_Y = []
        for i in range(8):
            for i in range(num):
                list_Y.append(i)
        self.Y = numpy.hstack(list_Y)

    def plot(self, x, colors):
        palette = numpy.array(seaborn.color_palette("pastel", 8))
        f = plt.figure(figsize=(6, 6))
        ax = plt.subplot(aspect="equal")
        sc = ax.scatter(
            x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(numpy.int8)]
        )
        plt.axis("equal")
        txts = []
        for i in range(8):
            xtext, ytext = numpy.median(x[colors == i, :], axis=0)
            spk_name = self.spk_path[i].split("/")[-1]
            txt = ax.text(xtext, ytext, spk_name, fontsize=10)
            txt.set_path_effects(
                [pe.Stroke(linewidth=0.5, foreground="w"), pe.Normal()]
            )
            txts.append(txt)
        plt.savefig("tsne.pdf")
        print("save pdf")
        return f, ax, txts

    def plot_all(self, num):
        for i in range(8):
            self.get_spk_embeddings(self.spk_path[i], num)
        self.x = numpy.vstack(self.emb_list)
        self.set_Y(num)
        print(self.X.shape)
        print(self.Y.shape)
        X_final = TSNE(perplexity=15).fit_transform(self.X)
        self.plot(X_final, self.Y)
        return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-attr", "-a", help="attr file path")
    parser.add_argument("-config", "-c", help="config file path")
    parser.add_argument("-model", "-m", help="model ckpt path")
    parser.add_argument("-spk1")
    parser.add_argument("-spk2")
    parser.add_argument("-spk3")
    parser.add_argument("-spk4")
    parser.add_argument("-spk5")
    parser.add_argument("-spk6")
    parser.add_argument("-spk7")
    parser.add_argument("-spk8")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tsne = TSNE(config=config, args=args)
    tsne.get_spk_paths()
    tsne.plot_all(100)

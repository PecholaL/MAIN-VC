"""inference(voice conversion)
    Please prepare the pretrained vocoder and set the correct path.
"""

import torch
import torch.nn.functional as F
import yaml
import pickle
from models.model import MAINVC
from models.tools import *
from data.utils import *
from data.preprocess import HyperParameters as hp
from argparse import ArgumentParser
from scipy.io.wavfile import write


class Inferencer(object):
    def __init__(self, config, args):
        self.config = config
        print(config)

        self.args = args
        print(self.args)

        self.build_model()
        self.load_model()

        # mean, stdev
        with open(self.args.attr, "rb") as f:
            self.attr = pickle.load(f)

    def build_model(self):
        self.model = cc(MAINVC(self.config))
        print(self.model)
        self.model.eval()
        return

    def load_model(self):
        print(f"[MAIN-VC]load model from {self.args.model}")
        self.model.load_state_dict(torch.load(f"{self.args.model}"))
        return

    def load_vocoder(self):
        print("[MAIN-VC]load vocoder from {self.args.vocoder}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocoder = torch.jit.load(f"{self.args.vocoder}").to(device).eval()

    def toWav(self, mel):
        print(f"convert mel-spect shapes {mel.shape} into wav")
        with torch.no_grad():
            wav = self.vocoder.generate([cc(torch.from_numpy(mel).float())])[0]
            wav = wav.cpu().numpy()
            print("generate wav")
            return wav

    def utt_make_frames(self, x):
        frame_size = self.config["data_loader"]["frame_size"]
        # x.shape (n_frames, n_mels)
        remains = x.size(0) % frame_size
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, x.size(1) * frame_size).transpose(1, 2)
        # out.shape ((B)1, (C)n_mels, (W)n_frames)
        return out

    def inference_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)  # (batch_size(1), n_mels, n_frames)
        x_cond = self.utt_make_frames(x_cond)  # (batch_size(1), n_mels, n_frames)
        dec = self.model.inference(x, x_cond)  # (batch_size(1), n_mels, n_frames)
        dec = dec.transpose(1, 2).squeeze(0)  # (n_frames, n_mels)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)
        wav_data = self.toWav(dec)
        return wav_data, dec

    def denormalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = (x - m) / s
        return ret

    def inference_from_path(self):
        src_wav, _ = load_wav(self.args.source, hp.sr)
        tar_wav, _ = load_wav(self.args.target, hp.sr)
        src_mel = log_mel_spectrogram(
            src_wav,
            hp.preemph,
            hp.sr,
            hp.n_mels,
            hp.n_fft,
            hp.hop_len,
            hp.win_len,
            hp.f_min,
        )
        tar_mel = log_mel_spectrogram(
            tar_wav,
            hp.preemph,
            hp.sr,
            hp.n_mels,
            hp.n_fft,
            hp.hop_len,
            hp.win_len,
            hp.f_min,
        )
        src_mel = torch.from_numpy(self.normalize(src_mel)).float().cuda()
        tar_mel = torch.from_numpy(self.normalize(tar_mel)).float().cuda()
        conv_wav, _ = self.inference_one_utterance(src_mel, tar_mel)
        src_info = self.args.source.split("/")[-1][:-4]
        tar_info = self.args.target.split("/")[-1][:-4]
        write(
            f"{self.arg.output}/{src_info}_{tar_info}.wav",
            rate=self.args.sample_rate,
            data=conv_wav,
        )
        return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--attr",
        "-a",
        help="attr file path",
        default="/Users/pecholalee/Coding/VC/mainVc_data/attr.pkl",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="config file path",
        default="/Users/pecholalee/Coding/VC/MAIN-VC/config.yaml",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="model path",
        default="/Users/pecholalee/Coding/VC/mainVc_data/save/mainVcModel.ckpt",
    )
    parser.add_argument(
        "--vocoder",
        "-v",
        help="vocoder path",
        default="/Users/pecholalee/Coding/VC/mainVc_data/vocoder/vocoder.pt",
    )
    parser.add_argument("-source", "-s", help="source wav path")
    parser.add_argument("-target", "-t", help="target wav path")
    parser.add_argument("-output", "-o", help="output wav path")
    parser.add_argument(
        "--sample_rate", "-r", help="sample rate", default=16000, type=int
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    inferencer = Inferencer(config=config, args=args)
    inferencer.inference_from_path()

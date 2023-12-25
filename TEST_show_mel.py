import librosa
import matplotlib.pyplot as plt
import numpy
import torch

from data.utils import *
from data.preprocess import HyperParameters as hp

path = "../VC/miniVCTK/p225/p225_003.wav"
outpath = "../VC/mainvc_data/mel.pdf"


# ---------- ---------- ---------- ---------- ----------

if __name__ == "__main__":
    wav = load_wav(path, hp.sr)
    mel = log_mel_spectrogram(
        wav, hp.preemph, hp.sr, 512, hp.n_fft, hp.hop_len, hp.win_len, hp.f_min
    )
    mel = torch.from_numpy(mel).float()
    mel = mel.squeeze().numpy()
    mel = numpy.clip((mel - 20 + 100) / 100, 1e-8, 1)
    mel_db = librosa.power_to_db(mel, ref=numpy.max).swapaxes(0, 1)

    plt.figure(dpi=100, figsize=(24, 8))
    librosa.display.specshow(mel_db, sr=hp.sr, hop_length=hp.hop_len)
    plt.savefig(outpath)

"""utilities for data process
"""

import os
import re
import glob
import librosa
import numpy as np
import matplotlib

from scipy.signal import lfilter
from collections import defaultdict
from typing import Union
from pathlib import Path


def load_wav(
    audio_path: Union[str, Path], sample_rate: int, trim: bool = False
) -> np.ndarray:
    """Load and preprocess waveform."""
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    wav = wav / (np.abs(wav).max() + 1e-6)
    if trim:
        _, (start_frame, end_frame) = librosa.effects.trim(
            wav, top_db=25, frame_length=512, hop_length=128
        )
        start_frame = max(0, start_frame - 0.1 * sample_rate)
        end_frame = min(len(wav), end_frame + 0.1 * sample_rate)

        start = int(start_frame)
        end = int(end_frame)
        if end - start > 1000:
            wav = wav[start:end]

    return wav


def log_mel_spectrogram(
    x: np.ndarray,
    preemph: float,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    f_min: int,
) -> np.ndarray:
    """Create a log Mel spectrogram from a raw audio signal."""
    x = lfilter([1, -preemph], [1], x)
    magnitude = np.abs(
        librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min)
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel_spec = np.log(mel_spec + 1e-9)
    return log_mel_spec.T  # shape(T, n_mels)


def read_speaker_info(speaker_info_path):
    speaker_ids = []
    with open(speaker_info_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            speaker_id = "p" + line.strip().split()[0]
            speaker_ids.append(speaker_id)
    return speaker_ids


def speaker_file_paths(root_dir):
    speaker2filenames = defaultdict(lambda: [])
    for path in sorted(glob.glob(os.path.join(root_dir, "*/*"))):
        filename = path.strip().split("\\")[-1]  # "\\" for Windows, "/" for Linux
        speaker_id = get_speaker_id(filename)
        speaker2filenames[speaker_id].append(path)
    return speaker2filenames


def get_speaker_id(filename):
    pattern = r'^p\d{3}_(\d{3})\.wav$'
    match = re.search(pattern, filename)
    if match:
        speaker_id = filename.split('_')[0]
    return speaker_id

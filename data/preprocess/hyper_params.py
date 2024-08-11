"""Hyperparameters in data processing,
    including stft transformation and mel spectrogram related parameters
"""


class HyperParameters:
    # sampling rate
    sr = 16000

    # preemphsis
    preemph = 0.97

    # fft points
    n_fft = 2048

    # mel banks
    n_mels = 80

    # hop length
    hop_len = 300

    # window length
    win_len = 1200

    # f_min
    f_min = 80

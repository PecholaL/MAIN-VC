"""make datasets
    files generated:
    attr.pkl(mean and std of train set)
        format: {mean:, std:}
    train.pkl, in_test.pkl, out_test_pkl(train and test sets)
        format: {key: filename, val: mel}
"""

import os
import sys
import random
import pickle
import numpy as np

sys.path.append("..")
from hyper_params import HyperParameters as hp
from utils import *


if __name__ == "__main__":
    data_dir = sys.argv[1]
    speaker_info_path = sys.argv[2]
    output_dir = sys.argv[3]
    test_speakers = int(sys.argv[4])
    test_proportion = float(sys.argv[5])
    n_utts_attr = int(sys.argv[6])

    sample_rate = hp.sr
    preemph = hp.preemph
    n_fft = hp.n_fft
    n_mels = hp.n_mels
    hop_len = hp.hop_len
    win_len = hp.win_len
    f_min = hp.f_min

    speaker_ids = read_speaker_info(speaker_info_path)
    print(f"[MAIN-VC](make_datasets) got {len(speaker_ids)} speakers' ids")
    random.shuffle(speaker_ids)

    train_speaker_ids = speaker_ids[:-test_speakers]
    test_speaker_ids = speaker_ids[-test_speakers:]

    with open(os.path.join(output_dir, "unseen_speaker_ids.txt"), "w") as f:
        for id in test_speaker_ids:
            f.write(f"{id}\n")
    with open(os.path.join(output_dir, "seen_speaker_ids.txt"), "w") as f:
        for id in train_speaker_ids:
            f.write(f"{id}\n")

    print(
        f"[MAIN-VC](make_datasets) {len(train_speaker_ids)} train speakers, {len(test_speaker_ids)} test speakers"
    )

    speaker2filepaths = speaker_file_paths(data_dir)

    train_path_list, in_test_path_list, out_test_path_list = [], [], []
    train_speaker2filenames = {}

    # Divide the data from train_speaker into training and test data
    # (in_test means seen speaker)
    for speaker in train_speaker_ids:
        path_list = speaker2filepaths[speaker]
        random.shuffle(path_list)
        test_data_size = int(len(path_list) * test_proportion)
        train_speaker2filenames[speaker] = path_list[:-test_data_size]
        train_path_list += path_list[:-test_data_size]
        in_test_path_list += path_list[-test_data_size:]

    with open(os.path.join(output_dir, "in_test_files.txt"), "w") as f:
        for path in in_test_path_list:
            f.write(f"{path}\n")
    with open(os.path.join(output_dir, "speaker2filenames.pkl"), "wb") as f:
        pickle.dump(train_speaker2filenames, f)

    # add paths of test_speakers' speech to out_test
    # (out_test means unseen speaker)
    for speaker in test_speaker_ids:
        path_list = speaker2filepaths[speaker]
        out_test_path_list += path_list

    with open(os.path.join(output_dir, "out_test_files.txt"), "w") as f:
        for path in out_test_path_list:
            f.write(f"{path}\n")

    for dataset_type, path_list in zip(
        ["train", "in_test", "out_test"],
        [train_path_list, in_test_path_list, out_test_path_list],
    ):
        print(f"[MAIN-VC](make_datasets) processed {dataset_type} set, {len(path_list)} files")
        data = {}
        output_path = os.path.join(output_dir, f"{dataset_type}.pkl")
        all_train_data = []

        for i, path in enumerate(sorted(path_list)):
            if i % 1000 == 0 or i == len(path_list) - 1:
                print(f"[MAIN-VC](make_datasets) processed {i} file of {dataset_type} set")
            filename = path.strip().split("/")[-1]
            wav = load_wav(path, sample_rate)
            mel = log_mel_spectrogram(
                wav,
                preemph,
                sample_rate,
                n_mels,
                n_fft,
                hop_len,
                win_len,
                f_min,
            )  # mel-spec shape: (T, n_mels)
            data[filename] = mel

            if dataset_type == "train" and i < n_utts_attr:
                all_train_data.append(mel)

        # get mean and stdev of train set
        if dataset_type == "train":
            all_train_data = np.concatenate(all_train_data)
            mean = np.mean(all_train_data, axis=0)
            std = np.std(all_train_data, axis=0)
            attr = {"mean": mean, "std": std}
            with open(os.path.join(output_dir, "attr.pkl"), "wb") as f:
                pickle.dump(attr, f)

        # normalization
        for key, val in data.items():
            val = (val - mean) / std
            data[key] = val
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

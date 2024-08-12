"""random sample training data
    get tuples:
        (filename_1, filename_2, cliptime_1, cliptime_2)
    then write into json
"""

import sys
import json
import pickle
import random

sys.path.append("..")
from utils import get_speaker_id

if __name__ == "__main__":
    pickle_path = sys.argv[1]
    s2f_path = sys.argv[2]
    sample_path = sys.argv[3]
    n_samples = int(sys.argv[4])
    segment_size = int(sys.argv[5])

    # load from train.pkl
    # data: {filepath=mel}
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    with open(s2f_path, "rb") as f:
        speaker2filepaths = pickle.load(f)

    # samples: (utt_id(filepath), timestep)
    samples = []

    filepath_list = [key for key in data]

    # filter out data (keys) whose length is less than segment_size
    print(len(filepath_list))
    filepath_list = sorted(
        list(filter(lambda u: len(data[u]) > segment_size, filepath_list))
    )
    print(len(filepath_list))
    print(f"[MAIN-VC](sample_dataset) {len(filepath_list)} utterances")

    # sample
    # though a piece of voice data may be extracted multiple times, training only requires segments of segment_size length of it
    # perform random clipping, so that the same data can get multiple different training data
    sample_utt_index_list = random.choices(range(len(filepath_list)), k=n_samples)

    for i, utt_index in enumerate(sample_utt_index_list):
        if i % 10000 == 0 or i == len(sample_utt_index_list) - 1:
            print(f"[MAIN-VC](sample_dataset) sample {i} samples")

        filepath_1 = filepath_list[utt_index]
        speaker_id = get_speaker_id(
            filepath_1.strip().split("\\")[-1]
        )  # split("/") in Linux
        filepath_2 = random.choice(speaker2filepaths[speaker_id])
        while len(data[filepath_2]) < segment_size:
            filepath_2 = random.choice(speaker2filepaths[speaker_id])

        # t is the start point of clipping
        t_1 = random.randint(0, len(data[filepath_1]) - segment_size)
        t_2 = random.randint(0, len(data[filepath_2]) - segment_size)
        samples.append((filepath_1, filepath_2, t_1, t_2))

    with open(sample_path, "w") as f:
        json.dump(samples, f)

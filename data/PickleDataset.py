"""make Dataset and Dataloader
    DataLoader provides two different segments 
    from the same speaker's different utterances each time
"""

import torch
import pickle
import json
import numpy as np

from torch.utils.data import Dataset, DataLoader


class CollateFn(object):
    def __init__(self, frame_size):
        self.frame_size = frame_size

    # numpy->tensor then make_frames
    def __call__(self, batch):
        data_1, data_2 = zip(*batch)
        tensor_1 = torch.from_numpy(np.array(data_1)).transpose(1, 2)
        tensor_2 = torch.from_numpy(np.array(data_2)).transpose(1, 2)
        # shape: [batch_size, n_mels, T]
        return tensor_1, tensor_2


def get_data_loader(
    dataset, batch_size, frame_size, shuffle=True, num_workers=4, drop_last=False
):
    _collate_fn = CollateFn(frame_size=frame_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader


class PickleDataset(Dataset):
    def __init__(self, pickle_path, sample_index_path, segment_size):
        # {filename: mel}
        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)
        # [(filename1/2, timestamp1/2)]
        with open(sample_index_path, "r") as f:
            self.indexes = json.load(f)

        self.segment_size = segment_size

    def __getitem__(self, ind):
        filename_1, filename_2, t_1, t_2 = self.indexes[ind]
        # clip from t (get [t : t+segment_size])
        segment_1 = self.data[filename_1][t_1 : t_1 + self.segment_size]
        segment_2 = self.data[filename_2][t_2 : t_2 + self.segment_size]
        return segment_1, segment_2

    def __len__(self):
        return len(self.indexes)

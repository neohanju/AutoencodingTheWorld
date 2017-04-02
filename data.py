import os
import torch
import torch.utils.data

class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root):
        super().__init__()
        self.path = dataset_root
        self.num_samples = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # TODO: load motion data



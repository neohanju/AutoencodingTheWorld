import os
import glob
import torch.utils.data


class VideoClipSets(torch.utils.data.Dataset):
    def __init__(self, paths):
        # paths can be a single string ar an array of strings
        # about paths that contain data samples right below
        super().__init__()

        self.paths = paths
        if not isinstance(self.paths, list):
            self.paths = [self.paths]
        self.num_dataset = len(self.paths)
        self.filelist = []
        for path in self.paths:
            assert os.path.exists(path)
            self.filelist += glob.glob(path + '/*.t7')
        self.filelist = sorted(self.filelist, key=lambda file: (os.path.dirname(file), os.path.basename(file)))
        self.num_samples = len(self.filelist)
        assert self.num_samples > 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return torch.load(self.filelist[item])


# ()()
# ('')HAANJU.YOO

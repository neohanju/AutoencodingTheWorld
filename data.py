import glob
import torch
import torch.utils.data


class VideoClipSets(torch.utils.data.Dataset):
    def __init__(self, paths):
        # paths can be a single string ar an array of strings
        # about paths that contain data samples right below
        super().__init__()
        self.paths = paths
        if not isinstance(self.paths, list):
            self.paths = [self.paths]
        self.num_videos = len(self.paths)
        self.filelist = []
        for path in paths:
            self.filelist.append(glob.glob(path + '/*.pt').sort())
        self.num_samples = len(self.filelist)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # TODO: load motion data
        return torch.load(self.filelist[item])


# ()()
# ('')HAANJU.YOO

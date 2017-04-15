import os
import glob
import torch.utils.data
import numpy as np


class VideoClipSets(torch.utils.data.Dataset):
    def __init__(self, paths, centered=True, num_input_channel=10):
        # paths can be a single string ar an array of strings about paths that contain data samples right below
        super().__init__()

        self.centered = centered

        # parsing paths of multiple dataset
        self.paths = paths
        if not isinstance(self.paths, list):
            self.paths = [self.paths]
        self.num_dataset = len(self.paths)

        # get file paths
        self.file_paths = []
        self.set_namelist = []
        self.mean_images = {}  # save negative mean images for saving computations
        for path in self.paths:
            assert os.path.exists(path)

            cur_file_paths = glob.glob(path + '/*.npy')
            self.file_paths += cur_file_paths

            # expect '.../[name]/train' format as path
            cur_set_name = os.path.basename(os.path.dirname(path))
            self.set_namelist += len(cur_file_paths) * [cur_set_name]

            if not self.centered:
                # make cube with mean image
                mean_image = np.load(os.path.join(os.path.dirname(path), 'mean_image.npy'))
                mean_image_cube = mean_image[np.newaxis, :, :].repeat(num_input_channel, axis=0)
                self.mean_images[cur_set_name] = torch.FloatTensor(mean_image_cube)

        self.file_paths = sorted(self.file_paths, key=lambda file: (os.path.dirname(file), os.path.basename(file)))

        # count samples
        self.num_samples = len(self.file_paths)
        assert self.num_samples > 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        if self.centered:
            data = torch.FloatTensor(np.load(self.file_paths[item]))
        else:
            data = torch.ByteTensor(np.load(self.file_paths[item])).float()
            data = data - self.mean_images[self.set_namelist[item]]
            data.div_(255)
            # loaded_data = torch.ByteTensor(np.load(self.file_list[item])).float()
            # data = torch.FloatTensor(loaded_data.size())
            # data.copy_(loaded_data)
        return data, self.set_namelist[item]


# ()()
# ('')HAANJU.YOO

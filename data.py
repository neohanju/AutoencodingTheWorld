import os
import glob
import torch.utils.data
import numpy as np


class VideoClipSets(torch.utils.data.Dataset):
    def __init__(self, paths, centered=False, num_input_channel=10, video_ids=None):
        # paths can be a single string ar an array of strings about paths that contain data samples right below
        super().__init__()

        if video_ids is not None:
            assert(isinstance(video_ids, list))

        self.centered = centered

        # parsing paths of multiple dataset
        self.paths = paths
        if not isinstance(self.paths, list):
            self.paths = [self.paths]
        self.num_dataset = len(self.paths)
        self.num_input_channel = num_input_channel
        self.video_ids = video_ids

        self.file_paths = []
        self.dataset_names = []
        self.video_names = []
        self.mean_images = {}
        self.num_samples = 0

        self.refresh_sample_info()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        if self.centered:
            data = torch.FloatTensor(np.load(self.file_paths[item]))
        else:
            data = torch.ByteTensor(np.load(self.file_paths[item])).float()
            data = data - self.mean_images[self.dataset_names[item]]
            data.div_(255)
        return data, self.dataset_names[item], self.video_names[item], os.path.basename(self.file_paths[item])

    def add_path(self, path):
        # check path in path list
        assert os.path.exists(path)
        self.paths.append(path)
        self.refresh_sample_info()

    def remove_path(self, path):
        if path in self.paths:
            self.paths.remove(path)
            self.refresh_sample_info()

    def refresh_sample_info(self):
        # get file paths
        self.file_paths = []
        self.dataset_names = []
        self.video_names = []
        self.mean_images = {}  # save negative mean images for saving computations
        include_file_path = []
        for path in self.paths:
            assert os.path.exists(path)

            cur_file_paths = glob.glob(path + '/*.npy')
            self.file_paths += cur_file_paths

            cur_dataset_name = ''  # to use in loading mean image
            for file_path in cur_file_paths:
                # expect naming format 'avenue_video_01_frame_interval_1_stride_2_000000.npy'
                name_splits = os.path.basename(file_path).split('_')
                cur_dataset_name, cur_video_name = name_splits[0], name_splits[2]

                if self.video_ids is None or int(cur_video_name) in self.video_ids:
                    include_file_path += [file_path]
                    self.dataset_names += [cur_dataset_name]
                    self.video_names += [cur_video_name]

            if not self.centered:
                # make cube with mean image
                mean_image = np.load(os.path.join(os.path.dirname(path), 'mean_image.npy'))
                mean_image_cube = mean_image[np.newaxis, :, :].repeat(self.num_input_channel, axis=0)
                self.mean_images[cur_dataset_name] = torch.FloatTensor(mean_image_cube)
        self.file_paths = include_file_path

        # sort samples by file names
        idx_path_pairs = [pair for pair in enumerate(self.file_paths, 0)]
        idx_path_pairs = sorted(idx_path_pairs, key=lambda pair: (os.path.dirname(pair[1]), os.path.basename(pair[1])))
        self.file_paths = [pair[1] for pair in idx_path_pairs]
        self.dataset_names = [self.dataset_names[pair[0]] for pair in idx_path_pairs]
        self.video_names = [self.video_names[pair[0]] for pair in idx_path_pairs]

        # count samples
        self.num_samples = len(self.file_paths)
        assert self.num_samples > 0



# ()()
# ('')HAANJU.YOO

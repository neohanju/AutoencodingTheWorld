import os
import glob
import torch.utils.data
import numpy as np


class RGBImageSet_augmented(torch.utils.data.Dataset):
    def __init__(self, path, type = 'train', centered=False, fold_number = None):
        super().__init__()
        self.centered = centered
        self.add_string = lambda a, b: a + b

        assert os.path.exists(path)
        self.base_path = path

        self.mean_image = self.get_mean_image()

        if fold_number is not None:
            self.fold_number = fold_number
            if type == 'train':
                cur_file_paths = list(
                    np.load(os.path.join(os.path.split(path)[0], '10fold_%d_train.npy' % self.fold_number)))
            elif type == 'test':
                cur_file_paths = list(
                    np.load(os.path.join(os.path.split(path)[0], '10fold_%d_test.npy' % self.fold_number)))
        else:
            cur_file_paths = glob.glob(self.base_path + '/*.npy')

        cur_file_paths.sort()
        self.file_paths = cur_file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        data = np.load(self.file_paths[item])
        data_name = self.file_paths[item].split('/')[-1]
        mask = np.load(os.path.join(os.path.split(os.path.dirname(self.file_paths[item]))[0], 'Ground_Truth_augmented',
                                    os.path.split(self.file_paths[item])[1]))
        mask = (np.ones(mask.shape) - mask / 255)

        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        if mask.shape[0] != 1 and mask.shape[0] != 3:
            mask = np.transpose(mask, (2, 0, 1))
        if data.shape[0] != 1 and data.shape[0] != 3:
            data = np.transpose(data, (2, 0, 1))
        # (h w c) => (c h w)

        if data.dtype.name == 'uint8':
            data = data.astype(float)

        if self.centered:
            data = torch.FloatTensor(data)
        else:
            data = torch.FloatTensor(data)

            h = self.mean_image.shape[0]
            w = self.mean_image.shape[1]
            image_anchor = [[int(h / 4), int(w / 4)], [0, int(w / 4)],
                            [int(h / 4), 0], [int(h / 4), int(w / 2)],
                            [int(h / 2), int(w / 4)]]

            anchor_index = int(os.path.basename(self.file_paths[item]).split('.')[0].split('_')[1])
            flip_index = int(os.path.basename(self.file_paths[item]).split('.')[0].split('_')[2])
            mean_patch = self.mean_image[image_anchor[anchor_index][0]:image_anchor[anchor_index][0] + int(h/2), image_anchor[anchor_index][1]:image_anchor[anchor_index][1] + int(w/2),:]
            if flip_index == 0:
                 mean_patch = mean_patch
            elif flip_index == 1:
                mean_patch = np.flip(mean_patch, 0)
            elif flip_index == 2:
                mean_patch = np.flip(mean_patch, 1)
            elif flip_index == 3:
                mean_patch = np.flip(np.flip(mean_patch, 0), 1)
            mean_patch = np.transpose(mean_patch, (2, 0, 1))
            mean_patch = torch.FloatTensor(mean_patch)
            data = data - mean_patch
            data.div_(255)
        return data, mask, data_name

    def get_decenterd_data(self, centered_data):
        result = centered_data.mul_(255) + self.mean_image
        result = result.byte()
        return result

    def get_mean_image(self):
        mean_image = np.load(os.path.join(os.path.dirname(self.base_path), "mean_image.npy"))
        return mean_image
import os
import glob
import torch.utils.data
import numpy as np
import numpy.ma as ma

class RGBImageSet_masked_loss(torch.utils.data.Dataset):
    def __init__(self, path, type= 'train', centered=False, fold_number=None):
        super().__init__()
        self.centered = centered
        self.add_string = lambda a, b: a + b


        assert os.path.exists(path)
        self.base_path = path

        self.mean_image = self.get_mean_image()

        if fold_number is not None:
            self.fold_number = fold_number
            if type == 'train':
                cur_file_paths = list(np.load(os.path.join(os.path.split(path)[0], '10fold_%d_train.npy' % self.fold_number)))
            elif type == 'test':
                cur_file_paths = list(np.load(os.path.join(os.path.split(path)[0], '10fold_%d_test.npy' % self.fold_number)))
        else:
            cur_file_paths = glob.glob(self.base_path + '/*.npy')
        cur_file_paths.sort()

        self.file_paths = cur_file_paths


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        data = np.load(self.file_paths[item])
        mask = np.load(os.path.join(os.path.split(os.path.dirname(self.file_paths[item]))[0], 'Ground_Truth', os.path.split(self.file_paths[item])[1]))
        mask = (np.ones(mask.shape) - mask/255)

        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask = np.transpose(mask, (2, 0, 1))
        data = np.transpose(data, (2, 0, 1))
        # (h w c) => (c h w)
        if data.dtype.name == 'uint8':
            data = data.astype(float)

        if self.centered:
            data = torch.FloatTensor(data)
        else:
            data = torch.FloatTensor(data)
            data = data - self.mean_image
            data.div_(255)
        return data, mask

    def get_decenterd_data(self, centered_data):
        result = centered_data.mul_(255) + self.mean_image
        result = result.byte()
        return result

    def get_mean_image(self):
        mean_image = np.load(os.path.join(os.path.dirname(self.base_path), "mean_image.npy"))
        mean_image = np.transpose(mean_image, (2, 0, 1))
        mean_image = torch.from_numpy(mean_image).float()
        return mean_image
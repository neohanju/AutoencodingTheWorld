import signal
import os
import numpy as np
import json
import torch
from torch.autograd import Variable
from visdom import Visdom
from time import gmtime, strftime

viz = Visdom()

target_sample_index = 0
target_frame_index = 5
mean_images = {}
mean_cubes = {}
optical_flow = False

# =============================================================================
# SYSTEM
# =============================================================================
def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print('ERROR: Cannot make saving folder')
    return


def sort_file_paths(path_list):
    return sorted(path_list, key=lambda file: (os.path.dirname(file), os.path.basename(file)))


# critical code section
# http://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py
class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


# =============================================================================
# VISUALIZATION
# =============================================================================
def pick_frame_from_batch(batch_data, RGB=False, sample_index=target_sample_index, frame_index=target_frame_index):
    if isinstance(batch_data, Variable):
        samples = batch_data.data
    else:
        samples = batch_data
    assert isinstance(samples, torch.FloatTensor) or isinstance(samples, torch.cuda.FloatTensor)
    return samples[sample_index, frame_index].cpu().numpy()


def gray_single_to_image(image):
    return np.uint8(image[np.newaxis, :, :].repeat(3, axis=0))


def sample_batch_to_image(batch_data):
    single_image = ((pick_frame_from_batch(batch_data) * 0.5) + 0.5) * 255  # [-1, +1] to [0, 255]
    # un-normalize
    return gray_single_to_image(single_image)


def decentering(image, mean_image):

    if mean_image.ndim == 3:
        return image * 255 + mean_image
    elif mean_image.ndim > 3:
        mean_image = mean_image[5]
    return gray_single_to_image(image * 255 + mean_image)


def find_best_sample_indexes(loss_per_batch, nindex=4):
    indexes = []
    indexes_tmp = []
    for i in range(0, len(loss_per_batch)):
        if nindex > len(indexes_tmp):
            indexes_tmp.append([loss_per_batch[i].data.max(), i])
        else:
            indexes_tmp.sort()
            if loss_per_batch[i].data.max() > indexes_tmp[0][0]:
                indexes_tmp[0][0] = loss_per_batch[i].data.max()
                indexes_tmp[0][1] = i
    indexes_tmp.sort()
    for i in range(0, nindex):
        indexes.append(indexes_tmp[i][1])

    return indexes


def draw_images(win_dict, input_batch, recon_batch, setnames):
    # visualize input / reconstruction pair
    input_data = pick_frame_from_batch(input_batch)
    recon_data = pick_frame_from_batch(recon_batch)
    if optical_flow:
        viz_input_frame = decentering(input_data, mean_cubes[setnames[target_sample_index]])
    else:
        viz_input_frame = decentering(input_data, mean_images[setnames[target_sample_index]])
    viz_input_data = sample_batch_to_image(input_batch)
    viz_recon_data = sample_batch_to_image(recon_batch)
    # viz_recon_frame = decentering(recon_data, mean_images[setnames[target_sample_index]])
    # viz_recon_error = np.flip(np.abs(input_data - recon_data), 0)  # for reverse y-axis in heat map
    viz_recon_error = gray_single_to_image(np.abs(input_data - recon_data) * 127.5)
    if not win_dict['exist']:
        win_dict['exist'] = True
        win_dict['input_frame'] = viz.image(viz_input_frame, opts=dict(title='Input'))
        win_dict['input_data'] = viz.image(viz_input_data, opts=dict(title='Input'))
        win_dict['recon_data'] = viz.image(viz_recon_data, opts=dict(title='Reconstruction'))
        # win_dict['recon_frame'] = viz.image(viz_recon_frame, opts=dict(title='Reconstructed video frame'))
        # win_dict['recon_error'] = viz.heatmap(X=viz_recon_error,
        #                                       opts=dict(title='Reconstruction error', xmin=0, xmax=2))
        win_dict['recon_error'] = viz.image(viz_recon_error, opts=dict(title='Reconstruction error'))
    else:
        viz.image(viz_input_frame, win=win_dict['input_frame'])
        viz.image(viz_input_data, win=win_dict['input_data'])
        viz.image(viz_recon_data, win=win_dict['recon_data'])
        # viz.image(viz_recon_frame, win=win_dict['recon_frame'])
        # viz.heatmap(X=viz_recon_error, win=win_dict['recon_error'])
        viz.image(viz_recon_error, win=win_dict['recon_error'])
    return win_dict

def draw_images(win_dict, input_batch, recon_batch, mean_image):
    # visualize input / reconstruction pair
    input_data = pick_frame_from_batch(input_batch)
    recon_data = pick_frame_from_batch(recon_batch)
    if optical_flow:
        viz_input_frame = decentering(input_data, mean_image)
    else:
        viz_input_frame = decentering(input_data, mean_image)
    viz_input_data = sample_batch_to_image(input_batch)
    viz_recon_data = sample_batch_to_image(recon_batch)
    # viz_recon_frame = decentering(recon_data, mean_images[setnames[target_sample_index]])
    # viz_recon_error = np.flip(np.abs(input_data - recon_data), 0)  # for reverse y-axis in heat map
    viz_recon_error = gray_single_to_image(np.abs(input_data - recon_data) * 127.5)
    if not win_dict['exist']:
        win_dict['exist'] = True
        win_dict['input_frame'] = viz.image(viz_input_frame, opts=dict(title='Input'))
        win_dict['input_data'] = viz.image(viz_input_data, opts=dict(title='Input'))
        win_dict['recon_data'] = viz.image(viz_recon_data, opts=dict(title='Reconstruction'))
        # win_dict['recon_frame'] = viz.image(viz_recon_frame, opts=dict(title='Reconstructed video frame'))
        # win_dict['recon_error'] = viz.heatmap(X=viz_recon_error,
        #                                       opts=dict(title='Reconstruction error', xmin=0, xmax=2))
        win_dict['recon_error'] = viz.image(viz_recon_error, opts=dict(title='Reconstruction error'))
    else:
        viz.image(viz_input_frame, win=win_dict['input_frame'])
        viz.image(viz_input_data, win=win_dict['input_data'])
        viz.image(viz_recon_data, win=win_dict['recon_data'])
        # viz.image(viz_recon_frame, win=win_dict['recon_frame'])
        # viz.heatmap(X=viz_recon_error, win=win_dict['recon_error'])
        viz.image(viz_recon_error, win=win_dict['recon_error'])


def draw_images_GAN(win_dict, fake, best_indexes):
    # visualize for GAN (top4 batch)
    sample_1 = gray_single_to_image(((pick_frame_from_batch(fake, best_indexes[0])*0.5)+0.5)*255)
    sample_2 = gray_single_to_image(((pick_frame_from_batch(fake, best_indexes[1])*0.5)+0.5)*255)
    sample_3 = gray_single_to_image(((pick_frame_from_batch(fake, best_indexes[2])*0.5)+0.5)*255)
    sample_4 = gray_single_to_image(((pick_frame_from_batch(fake, best_indexes[3])*0.5)+0.5)*255)

    if not win_dict['exist']:
        win_dict['exist'] = True
        win_dict['1'] = viz.image(sample_1, opts=dict(title='1'))
        win_dict['2'] = viz.image(sample_2, opts=dict(title='2'))
        win_dict['3'] = viz.image(sample_3, opts=dict(title='3'))
        win_dict['4'] = viz.image(sample_4, opts=dict(title='4'))
    else:
        viz.image(sample_1, win=win_dict['1'])
        viz.image(sample_2, win=win_dict['2'])
        viz.image(sample_3, win=win_dict['3'])
        viz.image(sample_4, win=win_dict['4'])
    return win_dict


def viz_append_line_points(win, lines_dict, x_pos, title='losses at each iteration', ylabel='loss',
                           xlabel='number of epochs'):
    y_len = len(lines_dict.keys())
    assert y_len > 0
    if 1 == y_len:
        dict_values = [val for val in lines_dict.values()]
        x_values = np.array([x_pos])
        y_values = np.array([dict_values[0]])
    else:
        x_values = np.ones([1, y_len]) * x_pos
        y_values = np.zeros([1, y_len])
        for i, value in enumerate(lines_dict.values()):
            y_values[0][i] = value

    if win is None:
        legends = []
        for key in lines_dict.keys():
            legends.append(key)
        win = viz.line(X=x_values, Y=y_values,
                       opts=dict(
                           title=title,
                           xlabel=xlabel,
                           ylabel=ylabel,
                           xtype='linear',
                           ytype='linear',
                           legend=legends,
                           makers=False
                       )
                       )
    else:
        viz.line(X=x_values, Y=y_values, win=win, update='append')

    return win


def get_loss_string(losses):
    str_losses = 'Total: %.4f' % (losses['total'])
    if 'recon' in losses:
        str_losses += ' Recon: %.4f' % (losses['recon'])
    if 'variational' in losses:
        str_losses += ' Var: %.4f' % (losses['variational'])
    if 'l1_reg' in losses:
        str_losses += ' L1: %.4f' % (losses['l1_reg'])
    if 'l2_reg' in losses:
        str_losses += ' L2: %.4f' % (losses['l2_reg'])
    return str_losses


# =============================================================================
# DICTIONARY
# =============================================================================
def save_dict_as_json_file(path, in_dict):
    assert isinstance(in_dict, dict)
    with open(path, 'w') as outfile:
        json.dump(in_dict, outfile)


def load_dict_from_json_file(path):
    print(path)
    assert os.path.exists(path)
    with open(path, 'r') as infile:
        return json.load(infile)


def add_dict(dict1, dict2):
    # merge two dictionaries with adding values in common key
    assert isinstance(dict1, dict) and isinstance(dict2, dict)
    result_dict = dict1.copy()
    for key2, value2 in dict2.items():
        key_found = False
        for key1, value1 in dict1.items():
            if key1 == key2:
                key_found = True
                result_dict[key1] = value1 + value2
        if not key_found:
            result_dict[key2] = value2
    return result_dict


def namespace_to_dict(namespace):
    return vars(namespace)


class Bunch(object):
    def __init__(self, input_dict):
        self.__dict__.update(input_dict)


def dict_to_namespace(input_dict):
    assert isinstance(input_dict, dict)
    return Bunch(input_dict)


# =============================================================================
# FILE I/O
# =============================================================================
def file_print_list(path, out_list, overwrite=True):
    # path : file path
    # out_list : list of output values
    open_mode = "w" if overwrite else "a"
    fo = open(path, open_mode)
    for val in out_list:
        assert isinstance(val, float)
        fo.write('%.18e,' % val)
    if len(out_list) > 0:
        fo.write('\n')
    fo.close()
    return


def save_model(path, model_dict, metadata, console_print=False):
    assert isinstance(metadata, dict)
    assert isinstance(model_dict, dict)
    model_path = path if path.find('.pth') != -1 else path + '.pth'
    meta_path = model_path.replace('.pth', '.json')

    # save
    # to prevent data corruption by keyboard interrupt, using 'DelayedKeyboardInterrupt'
    with DelayedKeyboardInterrupt():
        torch.save(model_dict, model_path)
        save_dict_as_json_file(meta_path, metadata)
        if console_print:
            print('Model is saved at ' + os.path.basename(model_path))


def load_metadata(metadata_path, cur_options=None):
    # return metadata and options
    train_info = load_dict_from_json_file(metadata_path)
    loaded_options = dict_to_namespace(train_info['options'])
    if cur_options is None:
        return train_info, loaded_options
    result_options = cur_options
    # inheritate some options
    result_options.model = loaded_options.model
    result_options.nc = loaded_options.nc
    result_options.nf = loaded_options.nf
    result_options.nz = loaded_options.nz
    result_options.l1_coef = loaded_options.l1_coef
    result_options.l2_coef = loaded_options.l2_coef
    result_options.var_loss_coef = loaded_options.var_loss_coef

    result_options.image_size = loaded_options.image_size
    result_options.z_size = loaded_options.z_size

    return train_info, result_options, loaded_options


# =============================================================================
# MISCELLANEOUS
# =============================================================================
def get_dataset_paths_and_mean_images(datasets, root_path, type, optical_flow=False):
    if isinstance(datasets, str):  # legacy for the previous input option type
        datasets.replace(' ', '')  # remove white space
        datasets.replace("'", '')  # remove '
        if 'all' == datasets:
            datasets = 'avenue|ped1|ped2|enter|exit'
        dataset_names = datasets.split('|')
    else:
        dataset_names = datasets

    # findout paths
    dataset_paths = []
    mean_images = {}
    for name in dataset_names:
        if optical_flow:
            dataset_paths.append(os.path.join(root_path, name, 'optical_flow', type))
            mean_images[name] = np.load(os.path.join(root_path, name, 'optical_flow', 'mean_cube.npy'))
        else:
            dataset_paths.append(os.path.join(root_path, name, type))
            mean_images[name] = np.load(os.path.join(root_path, name, 'mean_image.npy'))
    return dataset_paths, mean_images


def formatted_time(time_sec):
    days, rem = divmod(time_sec, 86400)  # days
    hours, rem = divmod(rem, 3600)  # hours
    minutes, seconds = divmod(rem, 60)  # minutes

    if 0 < days:
        return '%02d:%02d:%06.3f' % (int(hours), int(minutes), seconds)
    else:
        return '%d-%02d:%02d:%06.3f' % (int(days), int(hours), int(minutes), seconds)


def now_to_string():
    return strftime("%Y%m%d-%H%M%S", gmtime())


def make_cube_with_single_frame(frame, nc):
    return frame[np.newaxis, :, :].repeat(nc, axis=0)

# ()()
# ('') HAANJU.YOO

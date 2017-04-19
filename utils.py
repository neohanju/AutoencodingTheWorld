import os
import numpy as np
import torch
from torch.autograd import Variable
from visdom import Visdom

viz = Visdom()
target_sample_index = 0
target_frame_index = 5
mean_images = {}


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


# =============================================================================
# VISUALIZATION
# =============================================================================
def pick_frame_from_batch(batch_data, sample_index=target_sample_index, frame_index=target_frame_index):
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
    return gray_single_to_image(image * 255 + mean_image)


def draw_images(win_dict, input_batch, recon_batch, setnames):
    # visualize input / reconstruction pair
    input_data = pick_frame_from_batch(input_batch)
    recon_data = pick_frame_from_batch(recon_batch)
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


def viz_append_line_points(win, lines_dict, x_pos, title='losses at each iteration', ylabel='loss', xlabel='iterations'):
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
    str_losses = 'Total: %.4f\tRecon: %.4f' % (losses['total'], losses['recon'])
    if 'variational' in losses:
        str_losses += ' Var: %.4f' % (losses['variational'])
    if 'l1_reg' in losses:
        str_losses += ' L1: %.4f' % (losses['l1_reg'])
    if 'l2_reg' in losses:
        str_losses += ' L2: %.4f' % (losses['l2_reg'])
    return str_losses


# =============================================================================
# FILE I/O
# =============================================================================
def file_print_recon_costs(path, costs):
    # path : file path
    # costs : list of reconstruction costs
    fo = open(path, "w")
    for cost in costs:
        assert isinstance(cost, float)
        fo.write('%.18e\n' % cost)
    fo.close()
    return


# =============================================================================
# MISCELLANEOUS
# =============================================================================
def get_dataset_paths_and_mean_images(str_dataset, root_path, type):
    str_dataset.replace(' ', '')  # remove white space
    str_dataset.replace("'", '')  # remove '
    dataset_paths = []
    mean_images = {}
    if 'all' == str_dataset:
        str_dataset = 'avenue|ped1|ped2|enter|exit'
    dataset_names = str_dataset.split('|')
    for name in dataset_names:
        dataset_paths.append(os.path.join(root_path, name, type))
        mean_images[name] = np.load(os.path.join(root_path, name, 'mean_image.npy'))
    return dataset_paths, mean_images


def formatted_time(time_sec):
    days, rem = divmod(time_sec, 86400) # days
    hours, rem = divmod(rem, 3600)      # hours
    minutes, seconds = divmod(rem, 60)  # minutes

    if 0 < days:
        return '%02d:%d:%.3f' % (int(hours), int(minutes), seconds)
    else:
        return '%d-%02d:%d:%.3f' % (int(days), int(hours), int(minutes), seconds)


# ()()
# ('') HAANJU.YOO

import argparse
import os
import random
import time
import torch.utils.data
from torch.autograd import Variable
from models import AE, VAE, AE_LTR, VAE_LTR, OurLoss
from data import VideoClipSets
import utils as util


def debug_print(arg):
    if not options.debug_print:
        return
    print(arg)


# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Detecting abnormal behavior in videos')

# model related ---------------------------------------------------------------
parser.add_argument('--model_path', type=str, required=True, help='path to trained model file')
# output related --------------------------------------------------------------
parser.add_argument('--output_type', type=str, default='recon_costs', help='type of output')
# data related ----------------------------------------------------------------
parser.add_argument('--dataset', type=str, required=True, nargs='+',
                    help="all | avenue | ped1 | ped2 | enter | exit. 'all' means using entire data")
parser.add_argument('--data_root', type=str, required=True, help='path to base folder of entire dataset')
# display related -------------------------------------------------------------
parser.add_argument('--display', action='store_true', default=False,
                    help='visualize things with visdom or not. default=False')
# GPU related -----------------------------------------------------------------
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use. default=1')
# ETC -------------------------------------------------------------------------
parser.add_argument('--random_seed', type=int, help='manual seed')
parser.add_argument('--debug_print', action='store_true', default=False, help='print debug information')
# -----------------------------------------------------------------------------

options = parser.parse_args()

# seed
if options.random_seed is None:
    options.random_seed = random.randint(1, 10000)

# load options from metadata
metadata_path = options.model_path.replace('.pth', '.json')
train_info, options, saved_options = util.load_metadata(metadata_path, options)

# print test options
options_dict = util.namespace_to_dict(options)
print('Options={')
for k, v in options_dict.items():
    print('\t' + k + ':', v)
print('}')

# print network options
saved_option_dict = util.namespace_to_dict(saved_options)
print('Restored options={')
for k, v in saved_option_dict.items():
    print('\t' + k + ':', v)
print('}')


# =============================================================================
# INITIALIZATION PROCESS
# =============================================================================
cuda_available = torch.cuda.is_available()
torch.manual_seed(options.random_seed)
if cuda_available:
    torch.cuda.manual_seed_all(options.random_seed)

# result saving
save_path = os.path.join(os.path.dirname(options.model_path), options.output_type)
util.make_dir(save_path)

# visualization
win_recon_cost = None
win_images = dict(
    exist=False,
    input_frame=None,
    input_data=None,
    recon_data=None,
    recon_frame=None,
    recon_error=None
)

# =============================================================================
# DATA PREPARATION
# =============================================================================
dataset_paths, mean_images = util.get_dataset_paths_and_mean_images(options.dataset, options.data_root, 'test')
dataset = VideoClipSets(dataset_paths)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                         num_workers=1, pin_memory=True)

# streaming buffer
tm_buffer_set = time.time()
input_batch = torch.FloatTensor(1, saved_options.nc, saved_options.image_size, saved_options.image_size)
recon_batch = torch.FloatTensor(1, saved_options.nc, saved_options.image_size, saved_options.image_size)
mu_batch = torch.FloatTensor(1, options.z_size[0], options.z_size[1], options.z_size[2])
logvar_batch = torch.FloatTensor(1, options.z_size[0], options.z_size[1], options.z_size[2])
debug_print('Stream buffers are set: %.3f sec elapsed' % (time.time() - tm_buffer_set))

if cuda_available:
    debug_print('Start transferring to CUDA')
    tm_gpu_start = time.time()
    input_batch = input_batch.cuda()
    recon_batch = recon_batch.cuda()
    mu_batch = mu_batch.cuda()
    logvar_batch = logvar_batch.cuda()
    debug_print('Transfer to GPU: %.3f sec elapsed' % (time.time() - tm_gpu_start))

tm_to_variable = time.time()
input_batch = Variable(input_batch)
recon_batch = Variable(recon_batch)
debug_print('To Variable for Autograd: %.3f sec elapsed' % (time.time() - tm_to_variable))

print('Data streaming is ready')

# for utility library
util.target_sample_index = 0
util.target_frame_index = int(saved_options.nc / 2)
util.mean_images = mean_images
debug_print('Utility library is ready')


# =============================================================================
# MODEL
# =============================================================================
# create model instance
if 'AE-LTR' == saved_options.model:
    model = AE_LTR(saved_options.nc)
elif 'VAE-LTR' == saved_options.model:
    model = VAE_LTR(saved_options.nc)
elif 'AE' == saved_options.model:
    model = AE(saved_options.nc, saved_options.z_size[0], saved_options.nf)
elif 'VAE' == saved_options.model:
    model = VAE(saved_options.nc, saved_options.z_size[0], saved_options.nf)
assert model
model.load_state_dict(torch.load(options.model_path))

print(saved_options.model + ' is loaded')
print(model)

# loss & criterions
our_loss = OurLoss(cuda_available)

# to gpu
if cuda_available:
    debug_print('Start transferring model to CUDA')
    tm_gpu_start = time.time()
    model.cuda()
    debug_print('Transfer to GPU: %.3f sec elapsed' % (time.time() - tm_gpu_start))


# =============================================================================
# TESTING
# =============================================================================
model.eval()
recon_cost_dict = dict(recon=0)
recon_costs = {}
sample_index = 0

print('Start testing...')
tm_test_start = time.time()

cost_file_path = ''
prev_dataset_name = ''
prev_video_name = ''
cnt_cost = 0


for i, (data, dataset_name, video_name) in enumerate(dataloader, 0):
    if prev_dataset_name != dataset_name or prev_video_name != video_name:
        # new video is started
        print("Testing on '%s' dataset video '%s'... " % (dataset_name, video_name))
        cost_file_path = os.path.join(save_path, '%s_video_%s_%s.txt'
                                      % (dataset_name[0], video_name[0], saved_options.model))
        util.file_print_recon_costs(cost_file_path, [], overwrite=True)
        cnt_cost = 0
        win_recon_cost = None
        prev_dataset_name, prev_video_name = dataset_name, video_name

    # data load
    input_batch.data.copy_(data)

    # forward
    recon_batch, mu_batch, logvar_batch = model(input_batch)
    loss, loss_detail = our_loss.calculate(recon_batch, input_batch, saved_options, mu_batch, logvar_batch)

    cur_cost = loss_detail['recon']
    cnt_cost += 1

    print('%s_video_%s:%04d, cost = %.3f' % (dataset_name[0], video_name[0], cnt_cost, cur_cost))

    # visualization
    if options.display:
        win_images = util.draw_images(win_images, input_batch, recon_batch.data, dataset_name)
        win_recon_cost = util.viz_append_line_points(win=win_recon_cost,
                                                     lines_dict=dict(recon=cur_cost, zero=0),
                                                     x_pos=cnt_cost,
                                                     title='%s: video_%s' % (dataset_name[0], video_name[0]),
                                                     ylabel='reconstruction cost', xlabel='sample index')
        time.sleep(0.005)  # for reliable drawing

    # save cost
    util.file_print_recon_costs(cost_file_path, [cur_cost], overwrite=True)


# for i, dataset_path in enumerate(dataset_paths, 1):
#
#     test_sample_lists = util.sort_file_paths(glob.glob(dataset_path + '/*.npy'))
#     dataset_name = os.path.basename(os.path.dirname(dataset_path))
#
#     recon_costs = {}
#     prev_video = None
#
#     sys.stdout.write("\tTesting on '%s'... [%d/%d] " % (dataset_name, i, len(dataset_paths)))
#     for j, sample_path in enumerate(test_sample_lists, 1):
#         sys.stdout.write("\r\tTesting on '%s'... [%d/%d] : %04d / %04d"
#                          % (dataset_name, i, len(dataset_paths), j, len(test_sample_lists)))
#
#         cur_video = os.path.basename(sample_path).split('_')[1]  # expect 'video_01_000000.t7' format
#         if prev_video != cur_video:
#             prev_video = cur_video
#             win_recon_cost = None
#
#         # data load
#         data = torch.from_numpy(np.load(sample_path))
#         input_batch.data.copy_(data)
#
#         # forward
#         tm_forward_start = time.time()
#         recon_batch, mu_batch, logvar_batch = model(input_batch)
#         loss, loss_detail = our_loss.calculate(recon_batch, input_batch, saved_options, mu_batch, logvar_batch)
#         tm_forward_consume = time.time() - tm_forward_start
#
#         # reconstruction cost
#         if cur_video in recon_costs:
#             recon_costs[cur_video].append(loss_detail['recon'])
#         else:
#             recon_costs[cur_video] = [loss_detail['recon']]
#
#         # visualization
#         if options.display:
#             win_images = util.draw_images(win_images, input_batch, recon_batch.data, [dataset_name])
#             win_recon_cost = util.viz_append_line_points(win=win_recon_cost,
#                                                          lines_dict=dict(recon=loss_detail['recon'], zero=0),
#                                                          x_pos=len(recon_costs[cur_video]),
#                                                          title='%s video_%s' % (dataset_name, cur_video),
#                                                          ylabel='reconstruction cost', xlabel='sample index')
#
#     print("\r\tTesting on '%s'... [%d/%d] : done" % (dataset_name, i, len(dataset_paths)))
#     print('\tSave cost files...')
#     for (video_name, costs) in recon_costs.items():
#         util.file_print_recon_costs(
#             os.path.join(save_path, '%s_video_%s_%s.txt' % (dataset_name, video_name, saved_options.model)), costs)


# ()()
# ('')HAANJU.YOO

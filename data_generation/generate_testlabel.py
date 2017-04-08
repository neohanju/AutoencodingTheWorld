import torch
from scipy import io
import numpy as np
import visdom
vis = visdom.Visdom()




file_PATH = '/home/leejeyeol/Documents/ground_truth_demo/testing_label_mask'
num_of_files = 21

for videos in range(1, num_of_files+1):
    volLabel = io.loadmat(file_PATH+'/%d_label.mat' % videos)['volLabel'].tolist()[0]
    data = [0 for j in range(len(volLabel)-1)]
    for frames in range(0, len(volLabel)-1):
        if(sum(sum(volLabel[frames])) > 0):
            data[frames] = 0
        else:
            data[frames] = 1
    torch.save(data, file_PATH+'/Ground_truth_%d.t7' % videos)



#torch.load(torch.load(file_PATH+'/Ground_truth_21.t7')
for videos in range(1,num_of_files+1):
    a = torch.IntTensor(torch.load(file_PATH+'/Ground_truth_%d.t7'%videos))
    print(a)

    vis.line(a,opts=dict(title='%d video' % videos, xlabel='frames',ylabel='anormaly'))


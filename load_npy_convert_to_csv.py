import numpy as np
import os
# load npy file and convert that to csv file.

npy_root_path="/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/recon_costs"
npy_file = "Kim Jun Hong_ground_truth.npy"
npy_file_name = npy_file.split('.')[0]
npy_file = os.path.join(npy_root_path, npy_file)
data = np.load(npy_file)

csv_file = npy_file_name + '.csv'
csv_file = os.path.join(npy_root_path, csv_file)
np.savetxt(csv_file, data, delimiter=',')


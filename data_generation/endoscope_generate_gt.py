import numpy as np

save_path = "/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/recon_costs"

# hh:mm:ss -> integer
def time_to_int(time):
    (h, m, s) = time.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# time integer -> frame number
def scale_to_frames(value, before, after):
    return int(float(value)*after/before)


# Enter variables corresponding to dataset.
start_time = "0:11:15"
end_time = "12:47:00"
num_of_frames = int(59630/2)
name = "Kim Jun Hong"
abnormal_list = ["1:42:54", "1:48:47",
                 "1:50:05", "1:57:18",
                 "1:58:09", "1:58:18",
                 "1:59:32", "1:59:39",
                 "2:00:37", "2:00:48",
                 "6:14:35", "6:14:53",
                 "6:20:19", "6:20:25",
                 "10:12:19", "10:44:18",
                 "11:26:48", "11:27:18",
                 "12:35:24", "12:35:54"]

# I created container to save ground truth.
grount_truth = np.zeros(num_of_frames, dtype=np.int)

end_time = time_to_int(end_time)
abnormal_list = [time_to_int(x) for x in abnormal_list]
start_time = time_to_int(start_time)


end_time = end_time - start_time
abnormal_list = [x - start_time for x in abnormal_list]
start_time = start_time - start_time


abnormal_list = [scale_to_frames(x, end_time, num_of_frames) for x in abnormal_list]
end_time = scale_to_frames(end_time, end_time, num_of_frames)

flag = True
gt_start = 0
gt_end = 0
for i in abnormal_list:
    if flag:
        gt_start = i
        flag = False
    else:
        gt_end = i
        grount_truth[gt_start:gt_end] = 1
        flag = True



np.save(save_path+"/%s_ground_truth.npy" % name, grount_truth)

# Je Yeol Lee \[T]/
# Jolly Co-operation


# load Recon cost and Ground truth.
# Compare the two and plot the results.
import numpy as np
import matplotlib.pyplot as plt
import os

cost_path = "/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/recon_costs"
ground_truth = np.load(os.path.join(cost_path, "Kim Jun Hong_ground_truth.npy"))
recon_cost = np.load(os.path.join(cost_path, "e_video_v_endoscope-BN.npy"))



#=======================================================================================================================
#   Visualize result
#=======================================================================================================================

# The maximum value and the minimum value of GT.
min_max_recon_cost = []
section_flag = True
for i in range(0, len(ground_truth)):
    if section_flag:
        if ground_truth[i] == 1:
            section_flag = False
            max_recon = recon_cost[i]
            min_recon = recon_cost[i]
    else:
        if ground_truth[i] == 0:
            min_max_recon_cost.append([max_recon, min_recon])
            section_flag = True
            continue
        if recon_cost[i] < min_recon:
            min_recon = recon_cost[i]
        if recon_cost[i] > max_recon:
            max_recon = recon_cost[i]

section_min = 999
for j in range(0, len(min_max_recon_cost)):
    if section_min > min_max_recon_cost[j][1]:
        section_min = min_max_recon_cost[j][1]
    print('%d section - Max : %.02f \t Min : %0.2f' % (j+1, min_max_recon_cost[j][0], min_max_recon_cost[j][1]))

num_of_frame_reduced = 0
for i in range(0, len(recon_cost)):
    if recon_cost[i] < section_min:
        num_of_frame_reduced = num_of_frame_reduced + 1
print("Number of frame reduced : %d \t Decreased Percent : %.02f" % (num_of_frame_reduced,num_of_frame_reduced/len(recon_cost)))


# Draw a Graph.
xaxis = range(0, len(ground_truth))
fig, ax1 = plt.subplots()
plt.suptitle("reconstruction cost and ground truth")
ax2 = ax1.twinx()
a, = ax1.plot(xaxis, ground_truth, color='blue', label='Groud Truth')
b, = ax2.plot(xaxis, recon_cost, color='red', label='Reconstruction Cost')
p = [a, b]
ax1.legend(p, [p_.get_label() for p_ in p],
           loc=9, fontsize='small', bbox_to_anchor=(0.5, 1.1))
plt.show()

# Je Yeol Lee \[T]/
# Jolly Co-operation

% set path and file
% set parameters (names and strides per dataset)
% save file (evaluation file)

% for in video list
%       set subparameter

% per video,
% call cost file 
%  run persistence 1d
%   make figure(optional)

% calcualte TP,FP

cost_base_path = '/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/recon_costs';
file_name = 'endoscope_full_test_endoscope-BN_grid_mse_list.csv';
ground_truth_name ='Kim Jun Hong_ground_truth.csv';
ground_truth = importdata(fullfile(cost_base_path,ground_truth_name));
threshold = 0.1;
    


cost_file_path = fullfile(cost_base_path,file_name);
data = importdata(cost_file_path);
mean_data = mean(data,2);
var_data = var(data,0,2);
min_data = min(data,[],2);
max_data = max(data,[],2);

%set data
data_for_evaluate = mean_data;

data_for_evaluate = (data_for_evaluate-min(data_for_evaluate))/max(data_for_evaluate);
data_for_evaluate = 1-data_for_evaluate

[minIndices, maxIndices, persistence, globalMinIndex, globalMinValue] = ...
		    run_persistence1d(single(data_for_evaluate)); 
		persistent_features = filter_features_by_persistence(...
		    minIndices, maxIndices, persistence, threshold); 
if ~isempty(persistent_features)
    minima_indices = [persistent_features(:,1); globalMinIndex];
else
    minima_indices = globalMinIndex;
end

abnormal_regs = combine_locals(length(data_for_evaluate), minima_indices, 100);
abnormal_regs = abnormal_regs';
% evaluate

tp = 0;
fp = 0;
fn = 0;

[det, gtg] = compute_overlaps(abnormal_regs, ground_truth);
tp = tp + sum(gtg==1);
fp = fp + sum(det==0);
fn = fn + sum(gtg==0);


fprintf('TP: %d\n', tp);
fprintf('FP: %d\n', fp);
fprintf('FN: %d\n', fn);
fprintf('Precision: %0.2f\n', tp/(tp+fp));
fprintf('Recall: %0.2f\n', tp/(tp+fn));
fprintf('----------------------------------------\n')

% mean_data = (mean_data-min(mean_data))/max(mean_data);
% var_data = (var_data-min(var_data))/max(var_data);
% min_data = (min_data-min(min_data))/max(min_data);
% max_data = (max_data-min(max_data))/max(max_data);

% x = linspace(0,size(data,1),size(data,1));
% plot(x,mean_data,'r',x,var_data,'b',x,min_data,'c',x,max_data,'g',x,ground_truth,'m')
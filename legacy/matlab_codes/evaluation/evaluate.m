% This code is based on Mahmudul Hasan's evaluation code. The original code
% is designed for evaluating with provided cost files. We append additional
% codes in order to evaluate various algorithm's cost files. We also fixed
% some bugs in the original code.
%
% 2017.04.16, Haanju Yoo, Jeyeol Lee.
%

function [] = evaluate(cost_base_path_)
    
    % model can be AE_TXT (provided) | AE | VAE | AE_LTR | VAE_LTR or any of
	% your models.
	print(pwd)    
	cost_base_path = cost_base_path_;
    datasets = {'avenue', 'ped1', 'ped2', 'enter', 'exit'};
	% datasets = {'avenue', 'enter', 'exit'};
	strides = [10, 10, 10, 10, 10];

	% visualize detections
	do_draw_figures = false;

	% if you want to save result figures, turn on this (do not use this on
	% Linux -> MATLAB will crash)
	do_save_figures = false;

	cost_files_dir = fullfile(cost_base_path, 'recon_costs');
	save_path = 'result_graphs';

	fprintf('----------------------------------------\n')
	fprintf(' EVALUATION\n')
	fprintf('----------------------------------------\n')
    
    fid = fopen(fullfile(cost_base_path, '../evaluation.csv'), 'a')
    fprintf(fid, '%s,,', model)
	for d = 1:length(datasets)    
	    dataset = datasets{d};
	    fprintf('Dataset: %s\n', dataset);
	    tp = 0;
	    fp = 0;
	    fn = 0;
	    if strcmp(dataset, 'avenue')
		load('gt_avenue.mat');
		threshold = 0.2;
		start_video = 1;
		num_video = 21;
		set_idx = 1;
		fig_pos = [30, 100, 1200, 500];
		stride = strides(set_idx);
	    elseif strcmp(dataset, 'ped1')
		load('gt_ped1.mat');
		threshold = 0.3;
		start_video = 1;
		num_video = 36;
		set_idx = 2;
		fig_pos = [50, 80, 1200, 500];
		stride = strides(set_idx);
	    elseif strcmp(dataset, 'ped2')
		load('gt_ped2.mat');
		threshold = 0.3;
		start_video = 1;
		num_video = 12;
		set_idx = 3;
		fig_pos = [70, 60, 1200, 500];
		stride = strides(set_idx);
	    elseif strcmp(dataset, 'enter')
		load('gt1_enter.mat');
		threshold = 0.2;
		start_video = 1;
		num_video = 5;
		set_idx = 4;
		fig_pos = [90, 40, 1200, 500];
		stride = strides(set_idx);
	    elseif strcmp(dataset, 'exit')
		load('gt1_exit.mat');
		threshold = 0.25;
		start_video = 1;
		num_video = 4;
		set_idx = 5;
		fig_pos = [110, 20, 1200, 500];
		stride = strides(set_idx);
	    end
	    
	    if do_draw_figures
		h = figure(set_idx); clf;
		set(h,'Position',[10, 50, 1200, 500]);        
	    end
	    
	    datalength = zeros(1, num_video);
	    if isempty(dir(fullfile(cost_files_dir, sprintf('%s_video_test_%s.txt', dataset ,model))))  
	        fprintf(fullfile(cost_files_dir, sprintf('%s_video_test_%s.txt\n', dataset, model)))
			fprintf('WARNING: There are no result files\n')
			fprintf('----------------------------------------\n')
	        fprintf(fid,',')
	        fprintf(fid,'-,-,-,-,-')
			continue
        end    
	
	    for i = start_video:start_video+num_video-1       
		
		agt = gt{i};
		
		if strcmp(model, 'AE_TXT')
		     cost_file_name = sprintf('%s_video_%02d_conv3_iter_150000.txt', dataset, i);
		else            
		    cost_file_name = sprintf('%s_video_test_%s.txt',dataset, model);
		end       
		try
		    cost_file = fullfile(cost_files_dir, cost_file_name);
		catch
		    fprintf('WARNING: There is no file %s\n', cost_file_name)
		    continue
		end        

		% computing regularity

        
		data  = importdata(cost_file);
		data  = data(data > 0);        
		
		numFrames = stride*size(data,1);
		
		ndata = imresize(data,[numFrames,1]); % interpolation (expand)
		ndata = ndata-min(ndata);
		ndata = 1-ndata/max(ndata);
	%         ndata = medfilt1(ndata, 20);

		% thresholding
		[minIndices, maxIndices, persistence, globalMinIndex, globalMinValue] = ...
		    run_persistence1d(single(ndata)); 
		persistent_features = filter_features_by_persistence(...
		    minIndices, maxIndices, persistence, threshold); 
		
		if ~isempty(persistent_features)
		    minima_indices = [persistent_features(:,1); globalMinIndex];
		else
		    minima_indices = globalMinIndex;
		end
	%         minima_indices = [persistent_features(:,1); globalMinIndex];
		
		min_cost = 0;
		max_cost = 1;
		
		if do_draw_figures
		    if ~do_save_figures
		        subplot(floor((num_video+1)/2), 2, i)
		    else
		        figure(h); clf;
		    end
		    l1 = plot(ndata,'LineWidth',2, 'Color', 'b');
		    hold on;
		    
		    markers = ndata(minima_indices);
		    s1 = scatter(minima_indices, markers, 100, 'm', 'fill');
		    
		    for k = 1:size(agt,2)
		        % draw ground truth
		        sframe = agt(1,k);
		        eframe = agt(2,k);

		        p1 = patch([sframe,sframe:eframe,eframe], ...
		            [min_cost,max_cost*ones(1,eframe-sframe+1),min_cost],'r');
		        set(p1,'FaceAlpha',0.3,'EdgeColor','r');       
		    end
		end       
		
		abnormal_regs = combine_locals(length(ndata), minima_indices, 100);
		abnormal_regs = abnormal_regs';
		
		if do_draw_figures
		    for k = 1:size(abnormal_regs,2)
		        % draw decision result
		        sframe = abnormal_regs(1,k);
		        eframe = abnormal_regs(2,k);
		        p2 = patch([sframe,sframe:eframe,eframe], ...
		            [min_cost,max_cost*ones(1,eframe-sframe+1),min_cost],'g');
		        set(p2,'FaceAlpha',0.3,'EdgeColor','g');       
		    end
		end
		
		% evaluate
		[det, gtg] = compute_overlaps(abnormal_regs, agt);
		tp = tp + sum(gtg==1);
		fp = fp + sum(det==0);
		fn = fn + sum(gtg==0);
		
		if do_draw_figures
		    xlim([0,length(ndata)]);
		    set(gca,'FontSize',14);
		    legend([l1,s1,p1,p2], 'Generalized Model', 'Local Mimimas', ...
		        'Ground Truth','Detection','Location','southoutside', ...
		        'Orientation','horizontal');
		    hold off;
		end
		
		if do_draw_figures && do_save_figures
		    figure(h);            
		    print(fullfile(save_path, sprintf('%s_%s_video_%02d', model, dataset, i)), '-dpng')
		    pause(max(0.2, length(ndata) * 2.0e-4)); % prevent from the crashing of MATLAB on Linux
		end
	    end   
	    
	    fprintf('TP: %d\n', tp);
	    fprintf('FP: %d\n', fp);
	    fprintf('FN: %d\n', fn);
	    fprintf('Precision: %0.2f\n', tp/(tp+fp));
	    fprintf('Recall: %0.2f\n', tp/(tp+fn));
	    fprintf('----------------------------------------\n')
        fprintf(fid,',')
        fprintf(fid,'%f,%f,%f,%f,%f',tp,fp,fn,tp/(tp+fp),tp/(tp+fn))
        
    end
    fprintf(fid,'\n')
    fclose(fid)
end


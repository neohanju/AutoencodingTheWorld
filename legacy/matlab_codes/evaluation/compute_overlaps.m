function [det_det, gt_det] = compute_overlaps(det, gt)

det_det = zeros(1, size(det,2));
gt_det = zeros(1, size(gt,2));

det_abnormal = cell(1, size(det,2));
gt_detected = cell(1, size(gt,2));
for i = 1:size(det,2)
    det_range = det(1,i):det(2,i);
    det_abnormal{i} = zeros(1, length(det_range));
end
for j = 1:size(gt,2)
    gt_range = gt(1,j):gt(2,j);
    gt_detected{j} = zeros(1, length(gt_range));
end

for i = 1:size(det,2)
    det_range = det(1,i):det(2,i);
    for j = 1:size(gt,2)        
        gt_range = gt(1,j):gt(2,j);        
        [~, det_idx, gt_idx] = intersect(det_range, gt_range);
        det_abnormal{i}(det_idx) = 1;
        gt_detected{j}(gt_idx) = 1;       
    end
    if sum(det_abnormal{i}) / length(det_abnormal{i}) > 0.5
        det_det(i) = 1;
    end
end
for j = 1:size(gt,2)
    if sum(gt_detected{j}) / length(gt_detected{j}) > 0.5
        gt_det(j) = 1;
    end    
end

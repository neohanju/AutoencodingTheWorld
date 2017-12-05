function [det_det, gt_det] = compute_overlaps_old(det, gt)

det_det = zeros(1, size(det,2));
gt_det = zeros(1, size(gt,2));

for i = 1:size(det,2)
    for j = 1:size(gt,2)
        det_range = det(1,i):det(2,i);
        gt_range = gt(1,j):gt(2,j);
        overlap = length(intersect(det_range,gt_range))/min(length(det_range),length(gt_range));
        if overlap > 0.5
            if gt_det(j) == 1
                det_det(i) = -1;
            else
                det_det(i) = 1;
                gt_det(j) = 1;
            end
        end
    end
end


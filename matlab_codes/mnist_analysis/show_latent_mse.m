clear;
% =========================================================================
input_path = '/home/mlpa/Workspace/github/VAE_regularization/data/mnist/perturb_100.txt';
% =========================================================================

data = importdata(input_path);
X = data(:,1);
Y = data(:,2);
labels = data(:,4);
mse = data(:, 3);

figure; clf;
grid on;
hold on;
cmap = distinguishable_colors(10);
legends = cell(1, 10);
for i = 0:9
    idx = find(labels == i);
    scatter3(X(idx), Y(idx), mse(idx), 20, cmap(i+1,:));
    legends{i+1} = sprintf('%d', i);
end
% gscatter(data(:,1), data(:,2), data(:,4));
legend(legends);
hold off;




%()()
%('')HAANJU.YOO

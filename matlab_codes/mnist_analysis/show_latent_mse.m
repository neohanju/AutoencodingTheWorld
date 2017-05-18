clear;
% =========================================================================
file_name = 'latent_perturb_1_random_0080.txt';
input_dir = '/home/mlpa/Workspace/github/VAE_regularization/data/mnist/';
% =========================================================================

input_path = fullfile(input_dir, file_name);
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
title(strrep(file_name(1:end-4), '_', ' '));
hold off;

figure; clf;
grid on;
hold on;
for i = 0:9
    subplot(2, 5, i+1)
    idx = find(labels == i);
    scatter3(X(idx), Y(idx), mse(idx), 20, cmap(i+1,:));
    view(90, 0)
end
title(strrep(file_name(1:end-4), '_', ' '));
hold off;

fprintf([file_name, '\n']);
fprintf('MSE: mean=%.3f, std=%.3f, max=%.3f\n', mean(mse), std(mse), max(mse));

pdSix = fitdist(mse, 'Kernel', 'BandWidth', 1);
x = 0:.1:150;
ySix = pdf(pdSix, x);

figure; clf;
hold on;
% histfit(mse);
plot(x,ySix, '-','LineWidth', 2)
title(strrep(file_name(1:end-4), '_', ' '));
xlabel('mse');
ylabel('dist');
hold off;


%()()
%('')HAANJU.YOO

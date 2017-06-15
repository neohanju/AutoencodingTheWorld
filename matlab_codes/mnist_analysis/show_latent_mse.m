clear;
% =========================================================================
file_name = 'latent_perturb_100_vae_0150.txt';
input_dir = '/home/mlpa/Workspace/github/VAE_regularization/data/mnist';
% =========================================================================
disp_name = strrep(file_name(1:end-4), '_', ' ');

input_path = fullfile(input_dir, file_name);
data = importdata(input_path);
z_data = data(:,1:end-2);
labels = data(:,end);
mse = data(:, end-1);
z_cov = cov(z_data);

% eigen value
[V, D] = eig(z_cov);
eig_val = diag(D);
eig_val = sort(eig_val, 'descend');

%--------------------------------------
% Covariance
%--------------------------------------
figure; clf;
subplot(1, 2, 1);
imagesc(abs(z_cov));
colormap('gray');
title(sprintf('Abs. Cov. matrix of %s', disp_name));
subplot(1, 2, 2);
plot(1:length(eig_val), eig_val, '-');
grid on;
title(sprintf('Eigen values of %s', disp_name));


%--------------------------------------
% MSE distribution & Entropy
%--------------------------------------
mse_sum = sum(mse);
P_mse = mse / mse_sum;
H_mse = sum(P_mse .* log(P_mse));


fprintf([file_name, '\n']);
fprintf('MSE: mean=%.3f, std=%.3f, max=%.3f, H=%.3f\n', ...
    mean(mse), std(mse), max(mse), H_mse);

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


%--------------------------------------
% 2D-Visualization
%--------------------------------------
[coef, score, root] = pca(z_data);
X = score(:,1);
Y = score(:,2);

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
title(disp_name);
hold off;


%()()
%('')HAANJU.YOO

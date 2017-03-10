-- Load dependencies
local mnist = require 'mnist';
local optim = require 'optim';
local gnuplot = require 'gnuplot';
local image = require 'image';
local hdf5 = require 'hdf5'
local cuda = pcall(require, 'cutorch'); -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn'); -- Use cuDNN if available
require 'dpnn';


--=============================================================================
-- Command-line options
--=============================================================================
local cmd = torch.CmdLine();
-- major parameters
cmd:option('-model', 'ConvAE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|CatVAE|WTA-AE');
cmd:option('-batchSize', 60, 'Batch size');
cmd:option('-epochs', 20, 'Training epochs');
-- data loading
cmd:option('-datasetPath', '', 'Path for dataset folder')
cmd:option('-nThreads', 2, '# of threads for data loading')
-- others
cmd:option('-denoising', false, 'Use denoising criterion');
cmd:option('-mcmc', 0, 'MCMC samples');
cmd:option('-sampleStd', 1, 'Standard deviation of Gaussian distribution to sample from');
-- cpu / gpu
cmd:option('-cpu', false, 'CPU only (useful if GPU memory is too low)');

local opt = cmd:parse(arg);
if opt.cpu then
	cuda = false;
end
if opt.model == 'DenoisingAE' then
	opt.denoising = false; -- Disable "extra" denoising
end
print(opt)


--=============================================================================
-- Set up Torch
--=============================================================================
print('Setting up');
torch.setdefaulttensortype('torch.FloatTensor');
torch.manualSeed(854);
if cuda then
	require 'cunn';
	cutorch.manualSeed(torch.random());
end


--=============================================================================
-- Test
--=============================================================================
print('Testing')
x = XTest:narrow(1, 1, 10)
local xHat
if opt.model == 'DenoisingAE' then
	-- Normally this should be switched to evaluation mode, but this lets us extract the noised version
	xHat = autoencoder:forward(x)
	-- Extract noised version from denoising AE
	x = Model.noiser.output
else
	autoencoder:evaluate()
	xHat = autoencoder:forward(x)
end


--=============================================================================
-- Plot reconstructions
--=============================================================================
image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))

if opt.model == 'AE' or opt.model == 'SparseAE' or opt.model == 'WTA-AE' then
	-- Plot filters
	image.save('Weights.png', image.toDisplayTensor(Model.decoder:findModules('nn.Linear')[1].weight:view(x:size(3), x:size(2), Model.features):transpose(1, 3), 1, math.floor(math.sqrt(Model.features))))
end

-- if opt.model == 'VAE' then
-- 	if opt.denoising then
-- 		autoencoder:training() -- Retain corruption process
-- 	end

-- 	-- Plot interpolations
-- 	local height, width = XTest:size(2), XTest:size(3)
-- 	local interpolations = torch.Tensor(15 * height, 15 * width):typeAs(XTest)
-- 	local step = 0.05 -- Use small steps in dense region of 2D Gaussian; TODO: Move to spherical interpolation?

-- 	-- Sample 15 x 15 points
-- 	for i = 1, 15  do
-- 		for j = 1, 15 do
-- 			local sample = torch.Tensor({2 * i * step - 16 * step, 2 * j * step - 16 * step}):typeAs(XTest):view(1, 2) -- Minibatch of 1 for batch normalisation
-- 			interpolations[{{(i-1) * height + 1, i * height}, {(j-1) * width + 1, j * width}}] = Model.decoder:forward(sample)
-- 		end
-- 	end
-- 	image.save('Interpolations.png', interpolations)

-- 	-- Plot samples
-- 	local output = Model.decoder:forward(torch.Tensor(15 * 15, 2):normal(0, opt.sampleStd):typeAs(XTest)):clone()

-- 	-- Perform MCMC sampling
-- 	for m = 0, opt.mcmc do
-- 		-- Save samples
-- 		if m == 0 then
-- 			image.save('Samples.png', image.toDisplayTensor(Model.decoder.output, 0, 15))
-- 		else
-- 			image.save('Samples (MCMC step ' .. m .. ').png', image.toDisplayTensor(Model.decoder.output, 0, 15))
-- 		end

-- 		-- Forward again
-- 		autoencoder:forward(output)
-- 	end
-- elseif opt.model == 'CatVAE' then
-- 	if opt.denoising then
-- 		autoencoder:training() -- Retain corruption process
-- 	end

-- 	-- Plot "interpolations"
-- 	local height, width = XTest:size(2), XTest:size(3)
-- 	local interpolations = torch.Tensor(Model.N * height, Model.k * width):typeAs(XTest)

-- 	for n = 1, Model.N do
-- 		for k = 1, Model.k do
-- 			local sample = torch.zeros(Model.N, Model.k):typeAs(XTest)
-- 			sample[{{}, {1}}] = 1 -- Start with first dimension "set"
-- 			sample[n] = 0 -- Zero out distribution
-- 			sample[n][k] = 1 -- "Set" cluster
-- 			interpolations[{{(n-1) * height + 1, n * height}, {(k-1) * width + 1, k * width}}] = Model.decoder:forward(sample:view(1, Model.N * Model.k)) -- Minibatch of 1 for batch normalisation
-- 		end
-- 	end
-- 	image.save('Interpolations.png', interpolations)

-- 	-- Plot samples
-- 	local samples = torch.Tensor(15 * 15 * Model.N, Model.k):bernoulli(1 / Model.k):typeAs(XTest):view(15 * 15, Model.N * Model.k)
-- 	local output = Model.decoder:forward(samples):clone()

-- 	-- Perform MCMC sampling
-- 	for m = 0, opt.mcmc do
-- 		-- Save samples
-- 		if m == 0 then
-- 			image.save('Samples.png', image.toDisplayTensor(Model.decoder.output, 0, 15))
-- 		else
-- 			image.save('Samples (MCMC step ' .. m .. ').png', image.toDisplayTensor(Model.decoder.output, 0, 15))
-- 		end

-- 		-- Forward again
-- 		autoencoder:forward(output)
-- 	end
-- end

-- ()()
-- ('') HAANJU.YOO
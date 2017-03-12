-- Load dependencies
local optim = require 'optim';
local gnuplot = require 'gnuplot';
local image = require 'image';
local display = require 'display';
local hdf5 = require 'hdf5'
local cuda = pcall(require, 'cutorch'); -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn'); -- Use cuDNN if available
util = paths.dofile('util/util.lua')
-- require 'dpnn';

debug_display = false;
local function print_debug(...)
	if debug_display then print(...) end
end


--=============================================================================
-- Command-line options
--=============================================================================
local cmd = torch.CmdLine();
-- major parameters
cmd:option('-model', 'ConvAE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|CatVAE|WTA-AE');
cmd:option('-batchSize', 64, 'Batch size');
cmd:option('-epochs', 20, 'Training epochs');
-- data loading
cmd:option('-datasetPath', '', 'Path for dataset folder')
cmd:option('-nThreads', 2, '# of threads for data loading')
-- optimizer
cmd:option('-optimiser', 'adagrad | adam', 'Optimiser');
cmd:option('-learningRate', 0.01, 'Learning rate');
cmd:option('-weightDecay', 0.0005, 'Weight decay coefficient for regularization');
-- others
cmd:option('-denoising', 0, 'Use denoising criterion');
cmd:option('-mcmc', 0, 'MCMC samples');
cmd:option('-sampleStd', 1, 'Standard deviation of Gaussian distribution to sample from');
-- cpu / gpu
cmd:option('-cpu', 0, 'CPU only (useful if GPU memory is too low)');
-- control
cmd:option('-continue_train', 0, "if continue training, load the latest model: true, false")
-- save
cmd:option('-save_epoch_freq', 1, "network saving frequency")
cmd:option('-save_point', './training_result', "path to trained network")
cmd:option('-save_name', 'autoencoder', 'name for saving')

local opt = cmd:parse(arg);
if 1 == opt.cpu then
	cuda = false;
else
	opt.gpu = 1;
end
if opt.model == 'DenoisingAE' then
	opt.denoising = false; -- Disable "extra" denoising
end
if 1 == opt.continue_train then
	opt.continue_train = true;
else
	opt.continue_train = false;
end
if hasCudnn then opt.cudnn = 1 end
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

local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

--=============================================================================
-- Load data
--=============================================================================
-- check file existance
local inputFileList = {}
for file in paths.files(opt.datasetPath, ".h5") do
	table.insert(inputFileList, file);
end
assert(nil ~= next(inputFileList), "There is no proper input file at " .. opt.datasetPath)

local XTrainFile = hdf5.open(paths.concat(opt.datasetPath, inputFileList[1]), 'r');
local dataDim = XTrainFile:read('/data'):dataspaceSize()
XTrainFile:close();

local sampleLength, sampleWidth, sampleHeight = dataDim[2], dataDim[3], dataDim[4]
local XTrain = torch.Tensor(opt.batchSize, sampleLength, sampleWidth, sampleHeight)

local function load_data_from_file(inputFileName)
	local readFile = hdf5.open(paths.concat(opt.datasetPath, inputFileName), 'r');
	local dim = readFile:read('/data'):dataspaceSize();
	local numSamples = dim[1];
	print(('Reading data from %s : %d samples'):format(
		inputFileName, numSamples))
	-- local data = readFile:read('/data'):all();
	local data = readFile:read('/data'):partial({1, 100}, {1, dim[2]}, {1, dim[3]}, {1, dim[4]});
	readFile:close();

	return data
end


--=============================================================================
-- Create model & loss function
--=============================================================================
local function weights_init(m)
	local name = torch.type(m)
	if name:find('Convolution') then
		m.weight:normal(0.0, 0.02)
		m.bias:fill(0)
	elseif name:find('BatchNormalization') then
		if m.weight then m.weight:normal(1.0, 0.02) end
		if m.bias then m.bias:fill(0) end
	end
end

if opt.continue_train then
	local netFileList = {}
	for file in paths.files(paths.concat(opt.save_point, opt.save_name), ".t7") do
		table.insert(netFileList, file);
	end
	assert(nil ~= next(netFileList), "There is no proper network saving data at" ..
		paths.concat(opt.save_point, opt.save_name))
	-- find the latest network
	table.sort(netFileList, 
		function (a, b) 
			return string.lower(a) > string.lower(b) 
		end)
	print(('load model from %s'):format(netFileList[1]))
	Model = util.load_model(paths.concat(opt.save_point, opt.save_name, netFileList[1]), opt)
else
	Model = require ('models/' .. opt.model);
	Model:createAutoencoder(XTrain);
end

-- if opt.denoising then
-- 	Model.autoencoder:insert(nn.WhiteNoise(0, 0.5), 1); -- Add noise during training
-- end

local autoencoder = Model.autoencoder;
local encoder = Model.encoder;

autoencoder:apply(weights_init);


local criterion = nn.MSECriterion()
local softmax = nn.SoftMax() -- Softmax for CatVAE KL divergence


--=============================================================================
-- Data buffer and GPU
--=============================================================================
local batchInput  = torch.Tensor(opt.batchSize, sampleLength, sampleHeight, sampleWidth)
local batchOutput = torch.Tensor(opt.batchSize, sampleLength, sampleHeight, sampleWidth)

if cuda then
	print('Transferring to gpu...')
	require 'cunn'
	cutorch.setDevice(opt.gpu)
	
	-- data buffer
	batchInput  = batchInput:cuda()
	batchOutput = batchOutput:cuda()

	-- network
	if hasCudnn then
		-- Use cuDNN if available
		cudnn.benchmark = true
		cudnn.fastest = true
		cudnn.verbose = false
		cudnn.convert(autoencoder, cudnn, function(module)
			return torch.type(module):find('SpatialMaxPooling') -- to associate with maxUnpooling
		end)
		-- autoencoder = util.cudnn(autoencoder)
		-- encoder = util.cudnn(encoder)
		print(autoencoder)
	end
	autoencoder:cuda();

	-- loss function
	criterion:cuda();
	
	print('done')
else
	print('Running model on CPU')
end

-- Get parameters
local params, gradParams = autoencoder:getParameters();

--=============================================================================
-- Create optimiser function evaluation
--=============================================================================
local feval = function(params)

	-- Zero gradients
	gradParams:zero()

	-- Reconstruction phase
	-- Forward propagation
	batchOutput = autoencoder:forward(batchInput); -- Reconstruction
	local loss = criterion:forward(batchOutput, batchInput); -- xHat = batchOutput
	-- Backpropagation
	local gradLoss = criterion:backward(batchOutput, batchInput);
	autoencoder:backward(batchInput, gradLoss);

	return loss, gradParams
end


--=============================================================================
-- Training
--=============================================================================
print('Training...')
autoencoder:training()

-- for network saving
paths.mkdir(opt.save_point)
paths.mkdir(opt.save_point .. '/' .. opt.save_name)

-- optimizer parameters
optimState = {
	learningRate = opt.learningRate,
	weightDecay  = opt.weightDecay,
}

local __, loss
local losses = {}
-- to make a consistant data size
local leftDataLength = 0


for epoch = 1, opt.epochs do
	epoch_tm:reset()

	-- shufflie the input file list
	local fileIndices = torch.randperm(#inputFileList);

	for k = 1, #inputFileList do
		local fIdx = fileIndices[k];

		data_tm:reset()
		local data = load_data_from_file(inputFileList[fIdx]);
		print(('Done: %3f secs'):format(data_tm:time().real))

		-- shufffle the data
		data = data:index(1, torch.randperm(data:size(1)):long())
		print_debug(('total samples: %d'):format(data:size(1)))
		print_debug(('left sample before loop: %d'):format(leftDataLength))

		local start = 1;
		while true do

		-- for start = leftDataLength+1, data:size(1), opt.batchSize do
			print_debug(('start: %d'):format(start))

			-- get minibatch
			local loadSize   = math.min(data:size(1) - start + 1, opt.batchSize - leftDataLength)
			local loadedData = data:sub(start, start + loadSize - 1);
			print_debug(('loaded samples: %d'):format(loadSize))

			-- total size (load size + left data size)
			local readySize = leftDataLength + loadSize;			

			if readySize < opt.batchSize then
				-- less than batch size

				if  0 == leftDataLength then
					-- save and skip
					batchInput:sub(1, loadSize):copy(loadedData);
				else
					batchInput:sub(leftDataLength + 1, readySize):copy(loadedData);					
				end
				leftDataLength = readySize;
				print_debug(('left sample: %d'):format(leftDataLength))
			else
				-- batch size

				if leftDataLength > 0 then
					print_debug(('left samples: %d, loaded sampled: %d'):format(leftDataLength, loadSize))
				else
					print_debug('full batch')
				end

				batchInput:sub(leftDataLength + 1, readySize):copy(loadedData);
				leftDataLength = 0;
				
				-- Optimize
				print('optimize start')
				__, loss = optim.adagrad(feval, params, optimState)
				print(('optimize end, current loss: %.7f'):format(loss[1]))

				losses[#losses + 1] = loss[1]
			end

			start = start + loadSize;
			if start > data:size(1) then
				print_debug(('loop end with start = %d'):format(start))
				break
			end
			-- print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
   --                  .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f'):format(
   --                   epoch, curItInBatch, totalItInBatch,
   --                   tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
   --                   errG, errD, errL1))
		end
	end
	
	-- Plot training curve(s)
	local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
	gnuplot.pngfigure('Training.png')
	gnuplot.plot(table.unpack(plots))
	gnuplot.ylabel('Loss')
	gnuplot.xlabel('Batch #')
	gnuplot.plotflush()

	print(('End of epoch %d / %d \t Time Taken: %.3f secs'):format(
		epoch, opt.epochs, epoch_tm:time().real))

	if epoch % opt.save_epoch_freq == 0 then
		-- autoencoder:clearState();
		-- torch.save(paths.concat(opt.save_point, opt.save_name, ('net_epoch_%05d.t7'):format(epoch)), Model)
	end
end


-- --=============================================================================
-- -- Plot reconstructions
-- --=============================================================================
-- image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))

-- if opt.model == 'AE' or opt.model == 'SparseAE' or opt.model == 'WTA-AE' then
-- 	-- Plot filters
-- 	image.save('Weights.png', image.toDisplayTensor(Model.decoder:findModules('nn.Linear')[1].weight:view(x:size(3), x:size(2), Model.features):transpose(1, 3), 1, math.floor(math.sqrt(Model.features))))
-- end

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

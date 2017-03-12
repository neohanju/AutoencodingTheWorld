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
cmd:option('-epochs', 10000, 'Training epochs');
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
cmd:option('-save_epoch_freq', 500, "network saving frequency")
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
if '' == opt.save_name then
	opt.save_name = opt.model
end
if hasCudnn then opt.cudnn = 1 end
print(opt)


--=============================================================================
-- Set up Torch
--=============================================================================
print('Setting up...');
torch.setdefaulttensortype('torch.FloatTensor');
torch.manualSeed(854);
if cuda then
	require 'cunn';
	cutorch.manualSeed(torch.random());
end

local tm = torch.Timer()
local epoch_tm = torch.Timer()
local data_tm = torch.Timer()
local file_tm = torch.Timer()

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
	print_debug(('Reading data from %s : %d samples'):format(
		inputFileName, numSamples))
	local data = readFile:read('/data'):all();
	-- local data = readFile:read('/data'):partial({1, 100}, {1, dim[2]}, {1, dim[3]}, {1, dim[4]});
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

if 'ConvVAE' == opt.model then
	require 'modules/Gaussian'
end

-- generate model
print('Prepare the model...')
if opt.continue_train then	
	-- find the latest network
	local netFileList = {}
	for file in paths.files(paths.concat(opt.save_point, opt.save_name), ".t7") do
		table.insert(netFileList, file);
	end
	assert(nil ~= next(netFileList), "There is no proper network saving data at" ..
		paths.concat(opt.save_point, opt.save_name))	
	table.sort(netFileList, 
		function (a, b) 
			return string.lower(a) > string.lower(b) 
		end)

	for token in string.gmatch(netFileList[1], "[%d]+") do
		epoch_start = tonumber(token) + 1;
		break;
	end
	print(('load model from %s with epoch at %d'):format(netFileList[1], epoch_start))

	Model = util.load_model(paths.concat(opt.save_point, opt.save_name, netFileList[1]), opt)
else
	Model = require ('models/' .. opt.model);
	Model:createAutoencoder(XTrain);
	epoch_start = 1;
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


for epoch = epoch_start, opt.epochs do
	epoch_tm:reset()

	-- shufflie the input file list
	local fileIndices = torch.randperm(#inputFileList);

	for k = 1, #inputFileList do
		local fIdx = fileIndices[k];

		file_tm:reset()
		data_tm:reset()
		local data = load_data_from_file(inputFileList[fIdx]);
		print_debug(('Done: %3f secs'):format(data_tm:time().real))

		-- shufffle the data
		data = data:index(1, torch.randperm(data:size(1)):long())
		print_debug(('total samples: %d'):format(data:size(1)))
		print_debug(('left sample before loop: %d'):format(leftDataLength))

		local start = 1;
		local count = 1;
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
				print_debug('optimize start')
				__, loss = optim.adagrad(feval, params, optimState)
				print_debug(('optimize end, current loss: %.7f'):format(loss[1]))

				losses[#losses + 1] = loss[1]
			end

			start = start + loadSize;
			
			if start > data:size(1) then
				print_debug(('loop end with start = %d'):format(start))
				break
			end
			count = count + 1;
		end

		print(('Epoch: [%d][%3d/%3d]   Batches: %3d, Time: %.2f, FileTime: %.2f, Loss: %.5f'):format(
			epoch, k, #inputFileList, count, tm:time().real, file_tm:time().real, loss[1]))
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
		torch.save(paths.concat(opt.save_point, opt.save_name, ('net_epoch_%05d.t7'):format(epoch)),
			autoencoder:clearState())
	end
end

-- ()()
-- ('') HAANJU.YOO

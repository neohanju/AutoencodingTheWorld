-- script example:
--$ th -ldisplay.start 8000 0.0.0.0
--$ nohup th train.lua -datasetPath ./dataset/train -model ConvVAE -coefL2 0.5 -display 1 -max_iter 150000 2>&1 | tee enter_train.log

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
-- model
cmd:option('-model', 'ConvAE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|CatVAE|WTA-AE');
cmd:option('-coefL1', 0, 'Param for L1 regularization on the weights')
cmd:option('-coefL2', 0, 'Param for L1 regularization on the weights')
-- training
cmd:option('-batchSize', 64, 'Batch size');
cmd:option('-epochs', 1000, 'Training epochs');
cmd:option('-max_iter', 200000, 'Maximum iteration number');
cmd:option('-partial_learning', 1, "Learing with partial data, but at least one sample from the file")
-- network loading
cmd:option('-continue_train', 0, "if continue training, load the latest model: true, false")
-- data loading
cmd:option('-datasetPath', '', 'Path for dataset folder')
cmd:option('-sequence', 'all', 'Target sequence to train')
-- optimizer
cmd:option('-optimiser', 'adam', 'Optimiser: adagrad | adam');
cmd:option('-learningRate', 0.01, 'Learning rate');
cmd:option('-weightDecay', 0.0005, 'Weight decay coefficient for regularization');
cmd:option('-beta1', 0.5, 'for Adam optimizer')
-- others
cmd:option('-denoising', 0, 'Use denoising criterion');
cmd:option('-mcmc', 0, 'MCMC samples');
cmd:option('-sampleStd', 1, 'Standard deviation of Gaussian distribution to sample from');
-- display
cmd:option('-display', 0, 'Whether use display or not')
cmd:option('-display_freq', 3, 'Display frequency')
-- cpu / gpu
cmd:option('-cpu', 0, 'CPU only (useful if GPU memory is too low)');
cmd:option('-gpu', 1, 'GPU index')
-- save
cmd:option('-save_epoch_freq', 1, "network saving frequency with epoch number")
cmd:option('-save_iter_freq', 500, "network saving frequency with iteration number")
cmd:option('-save_point', './training_result', "path to trained network")
cmd:option('-save_name', '', 'name for saving')

local opt = cmd:parse(arg);

-- because the below command cannot handle link
-- assert(paths.dirp(opt.datasetPath), "There is not directory named " .. opt.datasetPath)

if 1 == opt.cpu then
	cuda = false;
else
	cuda = true;	
end
if hasCudnn and cuda then 
	opt.cudnn = 1 
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
	if 'all' == opt.sequence or string.find(file, opt.sequence) then
		table.insert(inputFileList, file);		
	end
end
assert(nil ~= next(inputFileList), "There is no proper input file at " .. opt.datasetPath)
print_debug(inputFileList);

local XTrainFile = hdf5.open(paths.concat(opt.datasetPath, inputFileList[1]), 'r');
local dataDim = XTrainFile:read('/data'):dataspaceSize()
XTrainFile:close();

local sampleLength, sampleWidth, sampleHeight = dataDim[2], dataDim[3], dataDim[4]
local XTrain = torch.Tensor(opt.batchSize, sampleLength, sampleWidth, sampleHeight)

local function load_data_from_file(inputFileName)
	local readFile = hdf5.open(paths.concat(opt.datasetPath, inputFileName), 'r');
	local dim = readFile:read('data'):dataspaceSize();
	local numSamples = dim[1];
	print_debug(('Reading data from %s : %d samples'):format(
		inputFileName, numSamples))

	local data = {};
	if 1 == opt.partial_learning or numSamples < opt.batchSize then
		-- full read
		data = readFile:read('data'):all();
	else
		-- read at least one batch size
		local load_size = math.max(opt.batchSize, math.floor(numSamples * opt.partial_learning))
		-- to considering speed (because the data is read with chunk size),
		-- read the continuous data
		local start_pos = math.random(numSamples - load_size + 1)
		local end_pos = start_pos + load_size - 1;
		data = readFile:read('/data'):partial({start_pos, end_pos}, {1, dim[2]}, {1, dim[3]}, {1, dim[4]});
	end	
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

-- generate model
print('Prepare the model...')
if opt.continue_train then
	if string.find(opt.model, 'VAE') then
		require 'modules/Gaussian'
	end

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
		iter_start = tonumber(token);
		break;
	end
	print(('load model from %s with iteration %d'):format(netFileList[1], iter_start))

	Model = util.load_model(paths.concat(opt.save_point, opt.save_name, netFileList[1]), opt)
else
	Model = require ('models/' .. opt.model);
	Model:createAutoencoder(XTrain);
	iter_start = 1;
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
local inputs  = torch.Tensor(opt.batchSize, sampleLength, sampleHeight, sampleWidth)
local outputs = torch.Tensor(opt.batchSize, sampleLength, sampleHeight, sampleWidth)

if cuda then
	print('Transferring to gpu...')
	require 'cunn'
	cutorch.setDevice(opt.gpu)
	
	-- data buffer
	inputs  = inputs:cuda()
	outputs = outputs:cuda()

	-- network
	if hasCudnn then
		-- Use cuDNN if available
		cudnn.benchmark = true
		cudnn.fastest = true
		cudnn.verbose = false

		if opt.model == 'ConvAE' or opt.model == 'ConvVAE' then
			cudnn.convert(autoencoder, cudnn, function(module)
				return torch.type(module):find('SpatialMaxPooling') -- to associate with maxUnpooling
			end);
		else
			cudnn.convert(autoencoder, cudnn);
		end
		-- autoencoder = util.cudnn(autoencoder)		
	end
	autoencoder:cuda();

	-- loss function
	criterion:cuda();
	
	print('done')
else
	print('Running model on CPU')
end

print('Autoencoder summary')
print(autoencoder)
print('Encoder summary')
print(encoder)

-- Get parameters
local params, gradParams = autoencoder:getParameters();

--=============================================================================
-- Create optimiser function evaluation
--=============================================================================
local feval = function(x)

	-- just in case:
	collectgarbage()

	-- get new parameters
	if x ~= params then
		params:copy(x)
	end

	-- reset gradients
	gradParams:zero()

	-- evaluate function for complete mini batch
	outputs = autoencoder:forward(inputs); -- Reconstruction
	local loss = criterion:forward(outputs, inputs); -- target = inputs

	-- estimate df/dW
	local gradLoss = criterion:backward(outputs, inputs);
	autoencoder:backward(inputs, gradLoss);

	if opt.model == 'ConvVAE' or opt.model == 'VAE' then
	    -- Optimise Gaussian KL divergence between inference model and prior: DKL[q(z|x)||N(0, σI)] = log(σ2/σ1) + ((σ1^2 - σ2^2) + (μ1 - μ2)^2) / 2σ2^2
	    local nElements = outputs:nElement()
	    local mean, logVar = table.unpack(Model.encoder.output)
	    local var = torch.exp(logVar)
	    local KLLoss = 0.5 * torch.sum(torch.pow(mean, 2) + var - logVar - 1)
	    KLLoss = KLLoss / nElements -- Normalise loss (same normalisation as BCECriterion)
	    loss = loss + KLLoss
	    local gradKLLoss = {mean / nElements, 0.5*(var - 1) / nElements}  -- Normalise gradient of loss (same normalisation as BCECriterion)
	    Model.encoder:backward(inputs, gradKLLoss)
	end

	-- penalties (L1 and L2): reference -> https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua#L214
	if opt.coefL1 ~= 0 then
		
		-- Loss:
		loss = loss + opt.coefL1 * torch.norm(params, 1)
		
		-- Gradients:
		gradParams:add(torch.sign(params):mul(opt.coefL1))

	end
	if opt.coefL2 ~= 0 then
		
		-- Loss:
		loss = loss + opt.coefL2 * torch.norm(params, 2)^2 / 2
		
		-- Gradients:
		gradParams:add(params:clone():mul(opt.coefL2))

	end

	return loss, gradParams
end


--=============================================================================
-- Training
--=============================================================================
print('Training...')
autoencoder:training()

-- for network saving
paths.mkdir(opt.save_point)
paths.mkdir(paths.concat(opt.save_point, opt.save_name))

-- optimizer parameters
optimState = {
	adagrad = {
		learningRate = opt.learningRate,
		weightDecay  = opt.weightDecay,
	},
	adam = {
		learningRate = opt.learningRate,
		beta1 = opt.beta1,
	},
}


-- display setting
local loss_graph_config = {
	title = "Training losses",	
	xlabel = "Batch #",
	ylabel = "loss"
}

local __, loss
local losses = {}
-- to make a consistant data size
local leftDataLength = 0;

local iter_count = iter_start;
local continue_training = true;
for epoch = 1, opt.epochs do
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
		local file_count = 1;
		local prev_iter = iter_count;
		while continue_training do

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
					inputs:sub(1, loadSize):copy(loadedData);
				else
					inputs:sub(leftDataLength + 1, readySize):copy(loadedData);					
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

				inputs:sub(leftDataLength + 1, readySize):copy(loadedData);
				leftDataLength = 0;
				
				-- Optimize -----------------------------------------------------
				print_debug('optimize start')
				__, loss = optim[opt.optimiser](feval, params, optimState[opt.optimiser])
				print_debug(('optimize end, current loss: %.7f'):format(loss[1]))
				iter_count = iter_count + 1;
				-----------------------------------------------------------------	

				-- draw network result
				if opt.display > 0 and iter_count % opt.display_freq == 0 then					
					local batch_index = math.max(1, math.floor(opt.batchSize * 0.5));
					local frame_index = math.max(1, math.floor(sampleLength * 0.5));
					print_debug(('input_frame (%d x %d x %d x %d) at %d'):format(
						inputs:size(1), inputs:size(2), inputs:size(3), inputs:size(4), batch_index))
					local input_frame = inputs[batch_index][frame_index];					
					print_debug(('output_frame (%d x %d x %d x %d) at %d'):format(
						outputs:size(1), outputs:size(2), outputs:size(3), outputs:size(4), batch_index))
					local output_frame = outputs[batch_index][frame_index];
					print_debug('display')
					disp_in_win = display.image(input_frame, {win=disp_in_win, title='input frame at iter ' .. iter_count})
					disp_out_win = display.image(output_frame, {win=disp_out_win, title='output frame at iter ' .. iter_count})
				end

				-- save network
				if iter_count % opt.save_iter_freq == 0 then		
					torch.save(paths.concat(opt.save_point, opt.save_name, ('%s_iter_%05d.t7'):format(opt.model, iter_count)),
						autoencoder:clearState())
				end
				torch.save(paths.concat(opt.save_point, opt.save_name, ('%s_latest.t7'):format(opt.model)),
						autoencoder:clearState())

				if iter_count >= opt.max_iter then
					continue_training = false;
					print(('End of training with the maximum iteration number: %d'):format(iter_count));
				end

				table.insert(losses, loss[1]);
			end

			start = start + loadSize;
			
			if start > data:size(1) then
				print_debug(('loop end with start = %d'):format(start))
				break
			end
			file_count = file_count + 1;
						
		end

		-- print log to console
		if nil ~= loss then
			print(('Epoch: [%d][%3d/%3d] Iteration: %5d (%2d), Total time: %5.2f, Loss: %.5f'):format(
				epoch, k, #inputFileList, iter_count, iter_count - prev_iter, tm:time().real, loss[1]))
		else
			print(('Epoch: [%d][%3d/%3d] too small data for batch'):format(
				epoch, k, #inputFileList, file_count))
		end

		-- draw loss plot
		if opt.display > 0 and #losses > 0 then
			print_debug('Draw loss graph')
			loss_graph_config.win = display.plot(
				torch.cat(torch.linspace(1, #losses, #losses), torch.Tensor(losses), 2), 
				loss_graph_config)
		end
	end
	
	-- Plot training curve(s)
	-- local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
	-- gnuplot.pngfigure('Training.png')
	-- gnuplot.plot(table.unpack(plots))
	-- gnuplot.ylabel('Loss')
	-- gnuplot.xlabel('Batch #')
	-- gnuplot.plotflush()
	-- gnuplot.close()	

	print(('End of epoch %d / %d \t Time Taken: %.3f secs, Acc. Time: %.3f'):format(
		epoch, opt.epochs, epoch_tm:time().real, tm:time().real))

	-- if epoch % opt.save_epoch_freq == 0 then		
	-- 	torch.save(paths.concat(opt.save_point, opt.save_name, ('%s_epoch_%05d.t7'):format(opt.model, epoch)),
	-- 		autoencoder:clearState())
	-- end
end

-- ()()
-- ('') HAANJU.YOO

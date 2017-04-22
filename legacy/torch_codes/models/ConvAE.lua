local nn = require 'nn'
--local nn = require 'nngraph'

local Model = {}

function nn.tables.group(nin, nout, ngroup)
  local input_group_size  = math.floor(nin / ngroup + 0.5)
  local output_group_size = math.floor(nout / ngroup + 0.5)

  local in_start = 1; local out_start = 1;
  local input_group_boundary = {};
  local output_group_boundary = {};
  local num_table_element = 0;
  for k = 1, ngroup do
    in_end = math.min(in_start + input_group_size - 1, nin);
    out_end = math.min(out_start + output_group_size - 1, nout);
    input_group_boundary[k] = {in_start, in_end};
    output_group_boundary[k] = {out_start, out_end};
    num_table_element = num_table_element + (in_end - in_start + 1) * (out_end - out_start + 1);

    in_start = in_end + 1;
    out_start = out_end + 1;
  end

  local ft = torch.Tensor(num_table_element, 2);
  local p = 1
  for k = 1, ngroup do
    for i = input_group_boundary[k][1], input_group_boundary[k][2] do
      for o = output_group_boundary[k][1], output_group_boundary[k][2] do
        ft[p][1] = i;
        ft[p][2] = o;
        p = p + 1;
      end
    end
  end

  return ft
end

function Model:createAutoencoder(X)
  local L = X:size(2)

  -- to associate with unpooling layers
  local poolingLayer1 = nn.SpatialMaxPooling(2, 2, 2, 2);  -- differ from paper
  local poolingLayer2 = nn.SpatialMaxPooling(2, 2, 2, 2);
  -- differ from paper: torch only supports the same size of kernel and stride
  -- paper used 3 for kernel size and 2 for the sride

  -- Create encoder
  -- expected input: 10 x 227 x 227
  self.encoder = nn.Sequential();
  self.encoder:add(nn.SpatialConvolution(L, 256, 11, 11, 4, 4));
  self.encoder:add(nn.Tanh());
  self.encoder:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75));
  -- out: 256 x 55 x 55
  self.encoder:add(poolingLayer1);
  -- out: 256 x 27 x 27

  -- local groupTable = nn.tables.group(256, 128, 2)
  -- out: 128 x 27 x 27
  -- self.encoder:add(nn.SpatialZeroPadding(2, 2, 2, 2))
  -- -- out: 128 x 31 x 31
  -- self.encoder:add(nn.SpatialConvolutionMap(groupTable, 5, 5, 1, 1))  -- no way to add padding
  self.encoder:add(nn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2))
  self.encoder:add(nn.Tanh())
  self.encoder:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
  -- out: 128 x 27 x 27
  self.encoder:add(poolingLayer2)
  -- out: 128 x 13 x 13

  self.encoder:add(nn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
  self.encoder:add(nn.Tanh())
  -- out: 64 x 13 x 13

  -- Create decoder
  -- expected input: 64 x 13 x 13
  self.decoder = nn.Sequential()
  self.decoder:add(nn.SpatialFullConvolution(64, 128, 3, 3, 1, 1, 1, 1))
  self.decoder:add(nn.Tanh())
  -- out: 128 x 13 x 13
  self.decoder:add(nn.SpatialMaxUnpooling(poolingLayer2))
  -- out: 128 x 27 x 27
  self.decoder:add(nn.SpatialFullConvolution(128, 256, 5, 5, 1, 1, 2, 2))
  self.decoder:add(nn.Tanh())
  self.decoder:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
  -- out: 256 x 27 x 27
  self.decoder:add(nn.SpatialMaxUnpooling(poolingLayer1))
  -- out: 256 x 55 x 55
  self.decoder:add(nn.SpatialFullConvolution(256, 10, 11, 11, 4, 4))
  self.decoder:add(nn.Tanh())
  -- out: 10 x 227 x 227

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
end

return Model


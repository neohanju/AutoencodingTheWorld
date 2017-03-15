local nn = require 'nn'

local Model = {}

function Model:createAutoencoder(X)
  local L = X:size(2)

  -- Create encoder
  -- expected input: L x 227 x 227
  self.encoder = nn.Sequential();
  self.encoder:add(nn.SpatialConvolution(L, 64, 5, 5, 2, 2, 1, 1));
  -- self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Tanh());
  -- out: 64 x 113 x 113
  self.encoder:add(nn.SpatialConvolution(64, 64, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(64));
  -- self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Tanh());
  -- out: 64 x 56 x 56
  self.encoder:add(nn.SpatialConvolution(64, 128, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(128));
  -- self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Tanh());
  -- out: 128 x 27 x 27
  self.encoder:add(nn.SpatialConvolution(128, 256, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(256));
  -- self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Tanh());
  -- out: 256 x 13 x 13
  self.encoder:add(nn.SpatialConvolution(256, 512, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(512));
  -- self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Tanh());
  -- out: 512 x 6 x 6

  -- Create decoder
  self.decoder = nn.Sequential();
  self.decoder:add(nn.SpatialFullConvolution(512, 256, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(256));  
  -- self.decoder:add(nn.ReLU(true));
  self.decoder:add(nn.Tanh());
  self.decoder:add(nn.Dropout(0.5));
  -- out: 256 x 13 x 13
  self.decoder:add(nn.SpatialFullConvolution(256, 128, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(128));  
  -- self.decoder:add(nn.ReLU(true));
  self.decoder:add(nn.Tanh());
  self.decoder:add(nn.Dropout(0.5));
  -- out: 128 x 27 x 27
  self.decoder:add(nn.SpatialFullConvolution(128, 64, 4, 4, 2, 2, 0, 0));
  self.decoder:add(nn.SpatialBatchNormalization(64));  
  -- self.decoder:add(nn.ReLU(true));
  self.decoder:add(nn.Tanh());
  -- out: 64 x 56 x 56
  self.decoder:add(nn.SpatialFullConvolution(64, 64, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(64));  
  -- self.decoder:add(nn.ReLU(true));
  self.decoder:add(nn.Tanh());
  -- out: 64 x 113 x 113
  self.decoder:add(nn.SpatialFullConvolution(64, L, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(L));
  self.decoder:add(nn.Tanh());
  -- out: L x 227 x 227

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
end

return Model


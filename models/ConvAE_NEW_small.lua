local nn = require 'nn'

local Model = { zSize = 200 }

function Model:createAutoencoder(X)

  local L = X:size(2)

  --[[ ENCODER ]]--------------------------------------------------------------
  -- expected input: (L) x 113 x 113
  self.encoder = nn.Sequential();
  self.encoder:add(nn.SpatialConvolution(L, 64, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(64));
  self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Dropout(0.5));
  -- out: 64 x 56 x 56
  self.encoder:add(nn.SpatialConvolution(64, 128, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(128));
  self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Dropout(0.5));
  -- out: 128 x 27 x 27
  self.encoder:add(nn.SpatialConvolution(128, 128, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(128));
  self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Dropout(0.5));
  -- out: 128 x 13 x 13
  self.encoder:add(nn.SpatialConvolution(128, 64, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(64));
  self.encoder:add(nn.Tanh());
  -- out: 64 x 6 x 6


  --[[ DECODER ]]--------------------------------------------------------------
  -- expected input: (self.zSize) x 1 x 1
  self.decoder = nn.Sequential();
  self.decoder:add(nn.SpatialFullConvolution(self.zSize, 64, 6, 6, 2, 2, 0, 0));
  self.decoder:add(nn.SpatialBatchNormalization(64));
  self.decoder:add(nn.ReLU(true));  
  -- out: 64 x 6 x 6
  self.decoder:add(nn.SpatialFullConvolution(64, 128, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(128));
  self.decoder:add(nn.ReLU(true));  
  -- out: 128 x 13 x 13
  self.decoder:add(nn.SpatialFullConvolution(128, 128, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(128));
  self.decoder:add(nn.ReLU(true));  
  -- out: 128 x 27 x 27
  self.decoder:add(nn.SpatialFullConvolution(128, 64, 5, 5, 2, 2, 1, 1, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(64));
  self.decoder:add(nn.ReLU(true));
  -- out: 64 x 56 x 56
  self.decoder:add(nn.SpatialFullConvolution(64, L, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(L));
  self.decoder:add(nn.Tanh());
  -- out: (L) x 113 x 113

  
  --[[ AUTOENCODER ]]----------------------------------------------------------
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
end

return Model


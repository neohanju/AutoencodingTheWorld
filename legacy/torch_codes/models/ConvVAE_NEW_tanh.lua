local nn = require 'nn'
require '../modules/Gaussian'

local Model = {
  zSize = 200
}

function Model:createAutoencoder(X)
 
  local L = X:size(2)


  --[[ ENCODER ]]--------------------------------------------------------------
  -- expected input: (L) x 227 x 227
  self.encoder = nn.Sequential();
  self.encoder:add(nn.SpatialConvolution(L, 64, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(64));
  self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Dropout(0.5));
  -- out: 64 x 113 x 113
  self.encoder:add(nn.SpatialConvolution(64, 128, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(128));
  self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Dropout(0.5));
  -- out: 128 x 56 x 56
  self.encoder:add(nn.SpatialConvolution(128, 256, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(256));
  self.encoder:add(nn.LeakyReLU(0.2, true));
  self.encoder:add(nn.Dropout(0.5));
  -- out: 256 x 27 x 27
  self.encoder:add(nn.SpatialConvolution(256, 128, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(128));
  self.encoder:add(nn.LeakyReLU(0.2, true));
  -- out: 128 x 13 x 13
  self.encoder:add(nn.SpatialConvolution(128, 64, 5, 5, 2, 2, 1, 1));
  self.encoder:add(nn.SpatialBatchNormalization(64));
  self.encoder:add(nn.Tanh());
  -- out: 64 x 6 x 6

  -- Create latent Z parameter layer
  local zLayer = nn.ConcatTable();
  zLayer:add(nn.SpatialConvolution(64, self.zSize, 6, 6)); -- Mean μ of Z
  zLayer:add(nn.SpatialConvolution(64, self.zSize, 6, 6)); -- Log variance σ^2 of Z (diagonal covariance)
  self.encoder:add(zLayer); -- Add Z parameter layer
  -- out: parallel -> { (self.zSize) x 1 x 1 , (self.zSize) x 1 x 1 }


  --[[ SAMPLER ]]--------------------------------------------------------------
    -- Create σε module
  local noiseModule = nn.Sequential();
  local noiseModuleInternal = nn.ConcatTable();
  local stdModule = nn.Sequential();
  stdModule:add(nn.MulConstant(0.5)); -- Compute 1/2 log σ^2 = log σ
  stdModule:add(nn.Exp()); -- Compute σ
  noiseModuleInternal:add(stdModule); -- Standard deviation σ
  noiseModuleInternal:add(nn.Gaussian(0, 1)); -- Sample noise ε ~ N(0, 1)
  noiseModule:add(noiseModuleInternal);
  noiseModule:add(nn.CMulTable()); -- Compute σε

  -- Create sampler q(z) = N(z; μ, σI) = μ + σε (reparametrization trick)
  local sampler = nn.Sequential();
  local samplerInternal = nn.ParallelTable();
  samplerInternal:add(nn.Identity()); -- Pass through μ 
  samplerInternal:add(noiseModule); -- Create noise σ * ε
  sampler:add(samplerInternal);
  sampler:add(nn.CAddTable());


  --[[ DECODER ]]--------------------------------------------------------------
  -- expected input: (self.zSize) x 1 x 1
  self.decoder = nn.Sequential();
  self.decoder:add(nn.SpatialFullConvolution(self.zSize, 64, 6, 6, 2, 2, 0, 0));
  self.decoder:add(nn.SpatialBatchNormalization(64));
  self.decoder:add(nn.Tanh());  
  -- out: 64 x 6 x 6
  self.decoder:add(nn.SpatialFullConvolution(64, 128, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(128));
  self.decoder:add(nn.Tanh());  
  -- out: 128 x 13 x 13
  self.decoder:add(nn.SpatialFullConvolution(128, 256, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(256));
  self.decoder:add(nn.Tanh());  
  -- out: 256 x 27 x 27
  self.decoder:add(nn.SpatialFullConvolution(256, 128, 5, 5, 2, 2, 1, 1, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(128));
  self.decoder:add(nn.Tanh());
  -- out: 128 x 56 x 56
  self.decoder:add(nn.SpatialFullConvolution(128, 64, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(64));
  self.decoder:add(nn.Tanh());
  -- out: 64 x 113 x 113
  self.decoder:add(nn.SpatialFullConvolution(64, L, 5, 5, 2, 2, 1, 1));
  self.decoder:add(nn.SpatialBatchNormalization(L));
  self.decoder:add(nn.Tanh());
  -- out: (L) x 227 x 227
  

  --[[ AUTOENCODER ]]----------------------------------------------------------
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(sampler)
  self.autoencoder:add(self.decoder)
end

return Model


local nn = require 'nn'
require '../modules/Gaussian'

local Model = {
  zSize = 200
}

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

  self.encoder:add(nn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2))
  self.encoder:add(nn.Tanh())
  self.encoder:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
  -- out: 128 x 27 x 27
  self.encoder:add(poolingLayer2)
  -- out: 128 x 13 x 13

  self.encoder:add(nn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
  self.encoder:add(nn.Tanh())
  -- out: 64 x 13 x 13

  -- Create latent Z parameter layer
  local zLayer = nn.ConcatTable()
  zLayer:add(nn.SpatialConvolution(64, self.zSize, 13, 13)) -- Mean μ of Z
  zLayer:add(nn.SpatialConvolution(64, self.zSize, 13, 13)) -- Log variance σ^2 of Z (diagonal covariance)
  self.encoder:add(zLayer) -- Add Z parameter layer

  -- Create σε module
  local noiseModule = nn.Sequential()
  local noiseModuleInternal = nn.ConcatTable()
  local stdModule = nn.Sequential()
  stdModule:add(nn.MulConstant(0.5)) -- Compute 1/2 log σ^2 = log σ
  stdModule:add(nn.Exp()) -- Compute σ
  noiseModuleInternal:add(stdModule) -- Standard deviation σ
  noiseModuleInternal:add(nn.Gaussian(0, 1)) -- Sample noise ε ~ N(0, 1)
  noiseModule:add(noiseModuleInternal)
  noiseModule:add(nn.CMulTable()) -- Compute σε

  -- Create sampler q(z) = N(z; μ, σI) = μ + σε (reparametrization trick)
  local sampler = nn.Sequential()
  local samplerInternal = nn.ParallelTable()
  samplerInternal:add(nn.Identity()) -- Pass through μ 
  samplerInternal:add(noiseModule) -- Create noise σ * ε
  sampler:add(samplerInternal)
  sampler:add(nn.CAddTable())

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.SpatialFullConvolution(self.zSize, 64, 13, 13, 1, 1))
  self.decoder:add(nn.Tanh())
  -- out: 64 x 13 x 13
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
  self.autoencoder:add(sampler)
  self.autoencoder:add(self.decoder)
end

return Model


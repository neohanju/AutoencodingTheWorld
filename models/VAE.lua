-- require 'nngraph'
-- require '../modules/Gaussian'
require 'nn'
require 'nngraph'

local Model = {
  zSize = 2 -- Size of isotropic multivariate Gaussian Z
}

function Model:createAutoencoder(X)

  -- Create encoder (inference/recognition model q, variational approximation for posterior p(z|x))
  -- self.encoder = nn.Sequential()
  -- self.encoder:add(nn.View(-1, 1, X:size(2), X:size(3)))
  -- self.encoder:add(nn.SpatialConvolution(1, 16, 3, 3, 1, 1, 1, 1))
  -- self.encoder:add(nn.ReLU(true))
  -- self.encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
  -- self.encoder:add(nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1))
  -- self.encoder:add(nn.ReLU(true))
  -- self.encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
  -- self.encoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
  -- self.encoder:add(nn.ReLU(true))
  -- self.encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- ==========================================================================
  -- input is (1) x 28 x 28
  self.encoder = nil
  local e1 = - nn.SpatialConvolution(1, 16, 3, 3, 1, 1, 1, 1) - nn.ReLU(true) - nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1)
  -- input is (16) x 14 x 14
  local e2 = e1
            - nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1) - nn.ReLU(true) - nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1)
  -- input is (8) x 8 x 8
  local zMean = e2
            - nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1) - nn.ReLU(true) - nn.SpatialMaxPooling(2, 2, 2, 2)

  -- local zCov = e2
  --           - nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1) - nn.ReLU(true) - nn.SpatialMaxPooling(2, 2, 2, 2)

  -- input is (8) x 4 x 4
  self.encoder = nn.gModule({e1}, {zMean})
  -- -- ==========================================================================


  -- -- ==========================================================================
  -- -- Create σε module
  -- local stdModule   = - nn.MulConstant(0.5) -- Compute 1/2 log σ^2 = log σ
  --                     - nn.Exp()            -- Compute σ 
  -- local noiseModule = {stdModule, nn.Gaussian(0, 1)} -- Sample noise ε ~ N(0, 1)
  --                     - nn.CMulTable()                 -- Compute σε
  -- local meanModule = - nn.Identity()

  -- -- ==========================================================================

  -- -- Create sampler q(z) = N(z; μ, σI) = μ + σε (reparametrization trick)
  -- local samplerParallel = nn.ParalleTable()
  -- samplerParallel:add(meanModule)
  -- samplerParallel:add(noiseModule)
  
  -- local sampler = samplerParallel - nn.CAddTable()
  -- -- ==========================================================================

  -- -- ==========================================================================
  -- -- -- Create decoder (generative model q)

  -- -- input is (8) x 4 x 4
  -- local d1 = sampler - nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1) - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2)
  -- -- input is (8) x 8 x 8
  -- local d2 = d1 - nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1) - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2)
  -- -- input is (8) x 16 x 16
  -- local d3 = d2 - nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 0, 0) - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2)
  -- -- input is (16) x 28 x 28
  -- local o1 = d3 - nn.SpatialConvolution(16, 1, 3, 3, 1, 1, 1, 1) - nn.Sigmoid(true) - nn.View(X:size(2), X:size(3))
  -- -- ==========================================================================

  -- -- Create autoencoder
  -- self.autoencoder = nn.gModule({e1}, {o1})
  self.autoencoder = nil

end

return Model

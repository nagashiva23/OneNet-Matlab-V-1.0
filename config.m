function cfg = config()

cfg.imageSize = [28 28 1];
cfg.batchSize = 16;
cfg.numEpochs = 3;
cfg.learningRate = 1e-3;

cfg.lambda = 0.1;
cfg.maxIter = 10;

cfg.noiseSigma = 0.05;

end

Overview

Project: OneNet — simple MATLAB demo of a learned ADMM-style reconstruction using a small CNN projection network.
Purpose: Train a learned projection within an ADMM-like iterative solver to reconstruct images from blurred + downsampled measurements.
Requirements

MATLAB: R2019b or later (Deep Learning Toolbox, Image Processing Toolbox).
Optional GPU: Parallel Computing Toolbox for gpuArray acceleration.
Quick Start

Build network and load data:
cfg = config();
net = buildProjectionNet(cfg);
[trainData,testData] = loadDataset(cfg);
Train (demo):
net = trainModel(net, trainData, cfg);
Test / visualize:
testModel(net, testData, cfg);
Running (example)
cfg = config();
net = buildProjectionNet(cfg);
[trainData,testData] = loadDataset(cfg);
net = trainModel(net, trainData, cfg);
testModel(net, testData, cfg);
File summary

adjointOperator.m: Adjoint of blur+downsample forward model (upsample by zero-insertion + Gaussian blur).
forwardOperator.m: Forward model — Gaussian blur then 2x subsample.
buildProjectionNet.m: Defines the small CNN dlnetwork used as learned projection.
admmSolver.m: Runtime ADMM solver using predict for z-update (numeric inference).
admmSolverTest.m: Simplified test-time solver variant.
admmSolverTrain.m: Differentiable ADMM loop (uses forward to preserve dlarray for grads).
modelLoss.m: Computes per-sample loss and gradients (used by trainModel).
trainModel.m: Training loop (per-sample updates using adamupdate).
testModel.m: Visualization of reconstructions for demo/test samples.
loadDataset.m: Loads example digit dataset (MATLAB builtin).
Notes & Known Issues

Training forward mismatch: modelLoss.m currently adds noise to x_gt (identity forward) rather than using forwardOperator. This should be updated so training matches the test/problem forward model.
cfg.batchSize is unused — training is per-sample (inefficient). Consider minibatching.
No checkpointing or validation monitoring — add save/validation if training longer.
predict vs forward: predict returns numeric arrays (use for inference); forward preserves dlarray for differentiation (use in training).
Suggested next items (TODO)

Align modelLoss.m to use forwardOperator + noise.
Add minibatch support in trainModel.m.
Add optional GPU support (cfg.useGPU) and move dlarray/network to gpuArray when enabled.
Save model checkpoints and add a validation loop.
License

Add your preferred license (e.g., MIT) to LICENSE if you want this repo public.
Want me to apply the minimal fix to modelLoss.m to use forwardOperator and add a short README file in the repo?

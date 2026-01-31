# OneNet (MATLAB)

A minimal MATLAB implementation of a learned projection used inside an ADMM-style iterative solver for simple inverse problems (example: blur + downsample). This repo shows how to build a small convolutional projection network, train it end-to-end through a differentiable ADMM-like solver, and run reconstructions.

**Requirements**
- MATLAB (R2020a or newer recommended)
- Deep Learning Toolbox (for `dlnetwork`, `dlarray`, training utilities)
- Image Processing Toolbox (for `imfilter`, `fspecial`)

**Quick Start**
1. Open MATLAB and add this folder to the path.
2. Run the example workflow in the MATLAB command window:

```matlab
cfg = config();
[trainData, testData] = loadDataset(cfg);
net = buildProjectionNet(cfg);
net = trainModel(net, trainData, cfg);
testModel(net, testData, cfg);
```

The default configuration (`[config.m](config.m)`) is very small (28x28 images, 3 epochs) so you can run it quickly for testing.

**Files**
- [config.m](config.m): default parameters (image size, learning rate, ADMM hyperparameters).
- [loadDataset.m](loadDataset.m): loads example digits dataset (`digitTrain4DArrayData` / `digitTest4DArrayData`).
- [buildProjectionNet.m](buildProjectionNet.m): constructs a small CNN `dlnetwork` used as the learned projection.
- [modelLoss.m](modelLoss.m): computes the training loss by adding noise, running the differentiable solver, and returning MSE and gradients.
- [admmSolverTrain.m](admmSolverTrain.m): differentiable ADMM-like solver used during training (keeps `dlarray` flows).
- [admmSolverTest.m](admmSolverTest.m): ADMM solver variant used at test time (uses `predict` + `extractdata`).
- [admmSolver.m](admmSolver.m): alternate solver variant (simple example using `predict`).
- [forwardOperator.m](forwardOperator.m): forward sensing model — Gaussian blur then 2x downsample.
- [adjointOperator.m](adjointOperator.m): simple adjoint (upsample + blur) matched to the forward operator.
- [trainModel.m](trainModel.m): simple training loop that calls `modelLoss` and updates `net` with `adamupdate`.
- [testModel.m](testModel.m): visualization routine that runs reconstructions and shows GT / input / reconstruction.
- [main.mlx](main.mlx): example live script (if present) to run experiments.

**How it works (brief)**
- Forward model: `forwardOperator` blurs and downsamples images to simulate measurements.
- Adjoint: `adjointOperator` upsamples and applies a blur as a pseudo-inverse for initialization and updates.
- Learned projection: a small CNN (`buildProjectionNet`) acts as the projection step inside ADMM.
- Training: `modelLoss` injects Gaussian noise then runs `admmSolverTrain` (differentiable) to produce a reconstruction; gradients are computed with `dlgradient` and used to update the projection net.

**Usage notes & tips**
- The dataset uses MATLAB's built-in digit data; swap in your own data by modifying `loadDataset.m`.
- Increase `cfg.numEpochs` and `cfg.batchSize` in `[config.m](config.m)` for real training (defaults are kept tiny for quick iteration).
- The ADMM loop uses a simple quadratic `x`-update for demonstration. Replace with problem-specific solvers if needed.



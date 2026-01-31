#  OneNet (MATLAB)
### Learned Deep Projection Network for Generic Linear Inverse Problems

This repository provides a MATLAB implementation inspired by the paper:

> **One Network to Solve Them All — Solving Linear Inverse Problems using Deep Projection Models**  
> J. H. Rick Chang et al., arXiv:1703.09912

The project implements a **learned projection (proximal) network embedded inside an ADMM solver**, following the framework proposed in the paper. A single neural network is trained to act as a **universal signal prior** that can solve multiple inverse problems without retraining.

---

##  Overview

Many image restoration tasks can be modeled as:

\[
y = Ax + n
\]

Where:

- `x` → unknown clean image  
- `A` → linear degradation operator  
- `y` → observed measurement  
- `n` → noise  

Examples:

- Super-resolution → Downsampling operator  
- Inpainting → Masking operator  
- Compressive sensing → Random projections  
- Denoising → Identity operator  

These problems are often **ill-posed**, meaning multiple solutions exist. This project solves them using a combination of:

- Classical optimization (ADMM)
- Learned deep projection operator
- End-to-end training

---

##  Core Idea (From the Paper)

Traditional approaches:

| Method | Limitation |
|--------|------------|
| Hand-crafted priors | Too generic |
| End-to-end CNNs | Must retrain for every problem |

Proposed approach:

> Learn a **single projection network** that acts as a universal image prior and embed it inside ADMM.

This network approximates the **proximal operator** of an implicit learned prior.

### Key Insight

In ADMM, the prior appears only as a **proximal operator**:

\[
x^{k+1} = \text{prox}_{\phi}(v)
\]

Instead of designing φ, we learn this operator using a CNN.

---

##  Requirements

### Software

- MATLAB R2020a or newer (recommended)
- Deep Learning Toolbox
- Image Processing Toolbox

### Why These Are Needed

| Toolbox | Purpose |
|---------|----------|
| Deep Learning | dlnetwork, dlarray, gradients |
| Image Processing | Filtering, blur |
| MATLAB Core | Optimization, visualization |

---

##  Quick Start

### 1. Add Repository to Path

```matlab
addpath(genpath(pwd));
```

2. Run the Main Pipeline
cfg = config();

[trainData, testData] = loadDataset(cfg);

net = buildProjectionNet(cfg);

net = trainModel(net, trainData, cfg);

testModel(net, testData, cfg);


This performs:

1.) Data loading

2.) Network construction

3.) End-to-end training

4.) Testing and visualization



Repository Structure
Configuration
config.m

Contains all hyperparameters:

cfg.imgSize
cfg.numEpochs
cfg.batchSize
cfg.learningRate
cfg.rho
cfg.admmIters


Controls training and solver behavior.


Dataset
loadDataset.m

Loads MATLAB digit dataset:

digitTrain4DArrayData

digitTest4DArrayData

Output format:

H × W × C × N


To use your own data, modify this file.

Projection Network
buildProjectionNet.m

Builds a small CNN that approximates the proximal operator.

Structure:

Input → Conv → ReLU → Conv → ReLU → Conv → Output


Role:

Removes artifacts

Enforces image manifold constraint

Acts as learned regularizer

The network is wrapped as dlnetwork for automatic differentiation.

DMM Solvers
admmSolverTrain.m

Training-time solver.

Features:

Fully differentiable

Uses dlarray

Preserves gradient flow

Used inside backpropagation.

admmSolverTest.m

Testing-time solver.

Features:

Faster

Uses predict

Converts to numeric arrays

No gradient tracking

admmSolver.m

Reference implementation.

Used for debugging and demonstration.

Forward & Adjoint Models
forwardOperator.m

Simulates measurement:

Gaussian blur

Downsampling

𝑦
=
𝐷
(
𝐵
(
𝑥
)
)
y=D(B(x))
adjointOperator.m

Pseudo-inverse operator:

Upsampling

Blur

Used for initialization and updates.

Loss Computation
modelLoss.m

Implements the training objective:

Perturb input

Apply forward model

Run ADMM

Reconstruct image

Compute MSE

Compute gradients

Returns:

[loss, gradients]

Training Loop
trainModel.m

Custom training loop:

Mini-batch iteration

Adam optimizer

Gradient updates

Loss logging

Why custom loop?

Built-in trainNetwork cannot handle unrolled solvers.

Evaluation
testModel.m

Displays:

Ground truth

Degraded input

Reconstruction

Used for qualitative evaluation.

Live Script
main.mlx

Step-by-step demo (if present).

Useful for experimentation.

Training Pipeline

During training:

x → A → y → ADMM + CNN → x̂ → Loss → Backprop


Steps:

Start from clean image x

Apply forward operator A

Add noise

Initialize solution

Run K ADMM iterations

Apply CNN projection

Produce x_hat

Compute loss

Backpropagate

The entire solver is differentiable.



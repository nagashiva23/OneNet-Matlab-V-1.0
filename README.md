
# OneNet (MATLAB)
### Learned Deep Projection Network for Generic Linear Inverse Problems

This repository provides a MATLAB implementation inspired by the paper:

> **One Network to Solve Them All — Solving Linear Inverse Problems using Deep Projection Models**  
> J. H. Rick Chang et al., arXiv:1703.09912

The project implements a **learned projection (proximal) network embedded inside an ADMM solver**, following the framework proposed in the paper. A single neural network is trained to act as a **universal signal prior** that can solve multiple inverse problems without retraining.

---

## Overview

Many image restoration tasks can be modeled as:

$$y = Ax + n$$

Where:
- `x` → unknown clean image
- `A` → linear operator
- `y` → observed measurement
- `n` → noise

**Examples:**
- Super-resolution → Downsampling operator
- Inpainting → Masking operator
- Compressive sensing → Random projections
- Denoising → Identity operator

These problems are often **ill-posed**, meaning multiple solutions exist. This project solves them using a combination of:
- Classical optimization (ADMM)
- Learned deep projection operator
- End-to-end training

---

## Core Idea (From the Paper)

**Traditional approaches:**

| Method | Limitation |
|--------|------------|
| Hand-crafted priors | Too generic |
| End-to-end CNNs | Must retrain for every problem |

**Proposed approach:**

> Learn a **single projection network** that acts as a universal image prior and embed it inside ADMM.

This network approximates the **proximal operator** of an implicit learned prior.

### Key Insight

In ADMM, the prior appears only as a **proximal operator**:

$$x^{k+1} = \text{prox}_{\phi}(v)$$

Instead of designing φ, we learn this operator using a CNN.

---

## ADMM Formulation

The objective:

$$\min_x \frac{1}{2}\|Ax-y\|^2 + \lambda \phi(x)$$

Using variable splitting:

$$x = z$$

**ADMM updates:**

### x-update (Projection Step)

$$x^{k+1} = P(z^k - u^k)$$

Learned by CNN.

### z-update (Data Consistency)

$$z^{k+1} = \arg\min_z \|Az-y\|^2 + \rho\|x-z+u\|^2$$

Solved by least squares.

### u-update (Dual Variable)

$$u^{k+1} = u^k + x^{k+1} - z^{k+1}$$

---

## Relation to the Paper

This implementation is a simplified version of the framework in the paper.

| Paper Component | This Repo |
|----------------|-----------|
| Projection Net P | CNN |
| Classifier D | Not implemented |
| Adversarial Training | Simplified |
| Proximal Learning | Direct MSE training |
| Nonconvex ADMM | Unrolled ADMM |

**Differences:**
- No adversarial classifier
- Simpler training
- Educational focus
- Smaller networks

---

## Requirements

### Software
- MATLAB R2020a or newer (recommended)
- Deep Learning Toolbox
- Image Processing Toolbox

### Why These Are Needed

| Toolbox | Purpose |
|---------|---------|
| Deep Learning | dlnetwork, dlarray, gradients |
| Image Processing | Filtering, blur |
| MATLAB Core | Optimization, visualization |

---

## Quick Start

### 1. Add Repository to Path

```matlab
addpath(genpath(pwd));


### 2. Run the Main Pipeline

```matlab
cfg = config();
[trainData, testData] = loadDataset(cfg);
net = buildProjectionNet(cfg);
net = trainModel(net, trainData, cfg);
testModel(net, testData, cfg);
```

This performs:
1. Data loading
2. Network construction
3. End-to-end training
4. Testing and visualization

---

## Repository Structure

### Configuration

**`config.m`**  
Contains all hyperparameters:

```matlab
cfg.imgSize
cfg.numEpochs
cfg.batchSize
cfg.learningRate
cfg.rho
cfg.admmIters
```

Controls training and solver behavior.

---

### Dataset

**`loadDataset.m`**  
Loads MATLAB digit dataset:
- `digitTrain4DArrayData`
- `digitTest4DArrayData`

**Output format:** `H × W × C × N`

To use your own data, modify this file.

---

### Projection Network

**`buildProjectionNet.m`**  
Builds a small CNN that approximates the proximal operator.

**Structure:**  
Input → Conv → ReLU → Conv → ReLU → Conv → Output

**Role:**
- Removes artifacts
- Enforces image manifold constraint
- Acts as learned regularizer

The network is wrapped as `dlnetwork` for automatic differentiation.

---

### ADMM Solvers

**`admmSolverTrain.m`**  
Training-time solver.

**Features:**
- Fully differentiable
- Uses `dlarray`
- Preserves gradient flow
- Used inside backpropagation

**`admmSolverTest.m`**  
Testing-time solver.

**Features:**
- Faster
- Uses `predict`
- Converts to numeric arrays
- No gradient tracking

**`admmSolver.m`**  
Reference implementation.  
Used for debugging and demonstration.

---

### Forward & Adjoint Models

**`forwardOperator.m`**  
Simulates measurement:
- Gaussian blur
- Downsampling

$$y = D(B(x))$$

**`adjointOperator.m`**  
Pseudo-inverse operator:
- Upsampling
- Blur

Used for initialization and updates.

---

### Loss Computation

**`modelLoss.m`**  
Implements the training objective:

1. Perturb input
2. Apply forward model
3. Run ADMM
4. Reconstruct image
5. Compute MSE
6. Compute gradients

**Returns:** `[loss, gradients]`

---

### Training Loop

**`trainModel.m`**  
Custom training loop:
- Mini-batch iteration
- Adam optimizer
- Gradient updates
- Loss logging

**Why custom loop?**  
Built-in `trainNetwork` cannot handle unrolled solvers.

---

### Evaluation

**`testModel.m`**  
Displays:
- Ground truth
- Degraded input
- Reconstruction

Used for qualitative evaluation.

---

### Live Script

**`main.mlx`**  
Step-by-step demo (if present).  
Useful for experimentation.

---

## Training Pipeline

During training:

```
x → A → y → ADMM + CNN → x̂ → Loss → Backprop
```

**Steps:**
1. Start from clean image `x`
2. Apply forward operator `A`
3. Add noise
4. Initialize solution
5. Run `K` ADMM iterations
6. Apply CNN projection
7. Produce `x_hat`
8. Compute loss
9. Backpropagate

The entire solver is differentiable.

---

##  Design Choices

### Small Images (28×28)
- Faster iteration
- Lower memory
- Educational focus

### Few Epochs
- Quick testing
- Not optimized for performance

### Simple CNN
- Easier debugging
- Extendable

---

##  Usage Tips

### Improve Performance

Edit `config.m`:

```matlab
cfg.numEpochs = 50;
cfg.batchSize = 64;
cfg.admmIters = 10;
cfg.learningRate = 1e-4;
```

### Use Custom Dataset

Modify `loadDataset.m`:
1. Load images
2. Normalize to [0,1]
3. Convert to 4D tensor

**Format:**

```matlab
H × W × C × N
```





## Citation

If you use this code, please cite the original paper:

```bibtex
@article{chang2017one,
  title={One Network to Solve Them All--Solving Linear Inverse Problems using Deep Projection Models},
  author={Chang, JH Rick and Li, Chun-Liang and Poczos, Barnabas and Kumar, BVK Vijaya and Sankaranarayanan, Aswin C},
  journal={arXiv preprint arXiv:1703.09912},
  year={2017}
}
```

**Reference:**

```
J. H. R. Chang et al.,
"One Network to Solve Them All — Solving Linear Inverse Problems
using Deep Projection Models",
arXiv:1703.09912, 2017.
```

---

##  Summary

This repository demonstrates:

✔️ Deep unrolling with ADMM  
✔️ Learned proximal operators  
✔️ Universal image priors  
✔️ MATLAB-based implementation  
✔️ Optimization + deep learning integration

It follows the philosophy of:

> **One network for many inverse problems**

as proposed in the original paper.




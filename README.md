# OneNet — MATLAB Demo for Learned ADMM Reconstruction

## Overview

**Project:** OneNet — A simple MATLAB demo of a learned ADMM-style reconstruction using a small CNN projection network.

**Purpose:**  
Train a learned projection within an ADMM-like iterative solver to reconstruct images from blurred and downsampled measurements.

---

## Requirements

- **MATLAB:** R2019b or later  
  - Deep Learning Toolbox  
  - Image Processing Toolbox
- **Optional GPU:** Parallel Computing Toolbox (for `gpuArray` acceleration)

---

## Quick Start

### Build Network and Load Data

```matlab
cfg = config();
net = buildProjectionNet(cfg);
[trainData, testData] = loadDataset(cfg);



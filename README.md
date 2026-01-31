# 📘 OneNet (MATLAB)
### Learned Deep Projection Network for Generic Linear Inverse Problems

This repository provides a MATLAB implementation inspired by the paper:

> **One Network to Solve Them All — Solving Linear Inverse Problems using Deep Projection Models**  
> J. H. Rick Chang et al., arXiv:1703.09912

The project implements a **learned projection (proximal) network embedded inside an ADMM solver**, following the framework proposed in the paper. A single neural network is trained to act as a **universal signal prior** that can solve multiple inverse problems without retraining.

---

## 📌 Overview

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

## 🧠 Core Idea (From the Paper)

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

## ⚙️ Requirements

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

## 🚀 Quick Start

### 1. Add Repository to Path

```matlab
addpath(genpath(pwd));

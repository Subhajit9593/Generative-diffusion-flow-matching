# Generative-diffusion-flow-matching
# Generative Diffusion & Flow Matching Framework

A from-scratch PyTorch implementation of modern generative
models using DDPM, DDIM, and Probability Flow ODEs on AFHQ-v2.

## Overview

This project explores discrete-time diffusion models and their
continuous-time formulation through probability flow ODEs.

The framework implements conditional and unconditional image
generation, classifier guidance, classifier-free guidance,
deterministic inversion, and zero-shot semantic image translation.

## Key Implementations

- Conditional U-Net (~23M parameters)
- FiLM-based residual blocks and multi-head self-attention
- DDPM and DDIM sampling
- Linear and cosine noise schedules
- Epsilon, x0 and v prediction parameterizations
- Classifier Guidance and Classifier-Free Guidance
- Deterministic DDIM inversion with fixed-point refinement
- Cat-to-dog and dog-to-cat image translation
- Probability Flow ODE with second-order Heun solver

## Experimental Results

| Method | Reconstruction MSE |
|---|---:|
| DDIM inversion (50 steps) | 0.0236 |
| Refined DDIM inversion | 0.000022 |
| First-order DDIM (50 NFE) | 0.0228 |
| Probability Flow ODE (Heun, 50 NFE) | 0.0015 |

## Repository Structure

- `notebooks/`: Complete implementation and mathematical explanations
- `assets/`: Generated samples and experimental visualizations

## Dataset

AFHQ-v2 (Animal Faces-HQ), containing cat, dog and wildlife images.

## Framework

- Python
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Reproducibility

See the notebook for implementation details, model configuration,
training procedure and sampling experiments.

## Author

Subhajit Mukherjee

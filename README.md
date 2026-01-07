# VQ-VAE Reproduction (Course Project)

This repository contains a reproduction of the first stage of
**Neural Discrete Representation Learning** (van den Oord et al., NeurIPS 2017),
implemented in PyTorch.

## Overview
- Model: Vector Quantized Variational Autoencoder (VQ-VAE)
- Task: Image reconstruction with discrete latent representations
- Datasets: CIFAR-10, Fashion-MNIST

This project focuses on reproducing the discrete representation learning
stage (Stage 1) of VQ-VAE. The PixelCNN prior described in the original paper
is not included.

## Code Structure
- `vqvae.py`: VQ-VAE model definition
- `train_vqvae.py`: Training on CIFAR-10
- `train_vqvae_fashionmnist.py`: Training on Fashion-MNIST
- `results/`: Reconstruction results

## Acknowledgements
This implementation is based on the open-source repository:
https://github.com/airalcorn2/vqvae-pytorch

The original code is licensed under Apache License 2.0.

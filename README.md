# An-Adaptive-Sampling-Framework-for-Diffusion-based-Dataset-Distillation

Official Repository for the AAAI 2026 Accepted Paper
“An Adaptive Sampling Framework for Diffusion-based Dataset Distillation with High Fidelity and Diversity”

This repository will contain the official implementation of our adaptive sampling–based dataset distillation framework, which leverages diffusion models to generate compact yet high-quality synthetic datasets. Our method introduces:

Bayesian Optimization–driven adaptive sampling, enabling class-wise selection of optimal denoising strength and guidance scale without any model fine-tuning.

A principled repulsion regularizer that balances fidelity and diversity by mitigating redundancy among distilled samples.

A training-free, generalizable pipeline compatible with a wide range of diffusion models and downstream classifiers.

Our approach demonstrates state-of-the-art performance across multiple datasets and architectures, significantly improving fidelity–diversity trade-offs while maintaining training-free generation.

The full paper will be available soon on the AAAI 2026 proceedings.

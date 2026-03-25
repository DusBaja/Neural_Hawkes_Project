# Neural Hawkes Project

PyTorch implementation of a **physics-informed neural network (PINN)** estimator for **marked Hawkes process kernels**.

## Overview

Hawkes processes are used to model self-exciting event streams, where the occurrence of one event increases the probability of future events. They are especially useful in financial market microstructure, where events such as trades, order submissions, or cancellations arrive asynchronously and often cluster in time.

This project focuses on **high-dimensional marked Hawkes processes**, where each event may carry additional information such as volume or size. The main objective is to estimate the interaction kernels that describe how past events influence future intensities.

Our work is based on the paper:

**Timothée Fabre and Ioane Muni Toke**  
*Neural Hawkes: Non-Parametric Estimation in High Dimension and Causality Analysis in Cryptocurrency Markets*

## Project Goals

The goals of this repository are:

1. **Implement** the proposed PINN-based Hawkes kernel estimator in **PyTorch**.
2. **Reproduce** selected numerical experiments and empirical findings from the paper.
3. **Explore** practical issues such as stability, scaling, sampling, and training performance.

## Method Summary

Classical non-parametric Hawkes kernel estimation often relies on the **Wiener–Hopf approach**, which solves a discretized Fredholm equation by matrix inversion. In high-dimensional marked settings, this can become computationally expensive, noisy, and unstable.

The idea of the reference paper is to replace this direct numerical inversion with a **physics-informed neural network**:

- represent the unknown Hawkes kernel with a neural network,
- define a residual from the Fredholm characterization equation,
- train the network so that the residual is minimized on sampled collocation points.

Key ingredients of the method include:

- **Fredholm equation of the second kind** for kernel identification,
- **PINN loss** based on the characterization equation residual,
- **causal temporal weighting** to improve accuracy at short times,
- **scale-aware weighting** for kernel components with different magnitudes,
- **DGM-style architecture** with ReLU activations,
- **mixed sampling strategy** with more points at small times,
- **log-scaling of time** and **z-score normalization of marks**.

## Repository Structure

A possible structure for the repository is:

```text
Neural_Hawkes_Project/
│
├── README.md
├── requirements.txt
├── src/
│   └── neural_hawkes/
│       ├── __init__.py
│       ├── model.py
│       ├── loss.py
│       ├── data.py
│       ├── statistics.py
│       ├── train.py
│       └── utils.py
├── notebooks/
├── tests/
├── data/
└── docs/
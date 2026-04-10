# Neural Hawkes Project

PyTorch implementation of a **physics-informed neural network (PINN)** estimator for **marked Hawkes process kernels**.

This project is inspired by the paper:

**Timothée Fabre and Ioane Muni Toke**  
*Neural Hawkes: Non-Parametric Estimation in High Dimension and Causality Analysis in Cryptocurrency Markets*


## Overview

Hawkes processes are point processes used to model event arrivals that cluster over time. In the multivariate marked setting, each event has:

- a **time**
- a **type**
- an optional **mark** (for example volume, size, or another event attribute)

The main goal is to estimate the kernel matrix $\Phi = (\varphi^{ij})$ which describes how past events of type j influence the future intensity of events of type i.

Classical non-parametric estimation often relies on the **Wiener–Hopf** approach, which solves a discretized Fredholm equation by matrix inversion. In high dimension, this can become unstable and expensive.  
The approach implemented here replaces that inversion step with a **physics-informed neural network** trained to satisfy the Hawkes characterization equation.

---

## Project goals

This repository aims to:

1. implement the moment-based **Neural Hawkes** estimator in **PyTorch**
2. reproduce core experiments from the reference paper on synthetic data
3. compare the PINN estimator with a **Wiener–Hopf benchmark**
4. study practical issues such as:
   - time-grid design
   - mark discretization
   - causal temporal weighting
   - scaling and normalization
   - numerical stability

---

## Current repository structure

```text
Neural_Hawkes_Project/
├── README.md
├── requirements.txt
└── src/
    ├── data/
    └── neural_hawkes/
        ├── __init__.py
        ├── data.py
        ├── loss.py
        ├── models.py
        ├── StatisticEstimators.py
        ├── MHP.py
        ├── WH.py
        ├── Exp_Kernel.ipynb
        ├── Gaussien_Kernel.ipynb
        └── Kernel_with_inhibition.ipynb
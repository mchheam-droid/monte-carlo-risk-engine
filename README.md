# High-Performance Monte Carlo Portfolio Risk Engine

A hybrid C++ / Python quantitative risk analytics engine modeling correlated asset price paths via multi-threaded Monte Carlo simulation and computing regulatory market risk metrics (Value at Risk & Conditional VaR).

---

## Overview

In quantitative portfolio management, estimating tail risk across correlated assets requires simulating large volumes of future market scenarios. While Python is standard for data analysis, pure Python loops suffer from interpreter overhead and the Global Interpreter Lock (GIL).

This project implements a high-performance architecture:
- **C++ Core Engine**: High-throughput multi-threaded simulation using modern C++ (`std::thread`), linear algebra, and thread-safe random number generation.
- **Pybind11 Integration**: Seamless zero-overhead Python binding with explicit GIL release during intensive compute phases.
- **Python Risk Suite**: Clean financial abstraction for multi-asset portfolios, validation checks, and tail risk reporting.

---

## Mathematical Formulation

### 1. Correlated Geometric Brownian Motion (GBM)
The price path of each asset $i$ under standard Black-Scholes dynamics follows:

$$S_i(t + \Delta t) = S_i(t) \exp\left( \left(\mu_i - \frac{1}{2}\sigma_i^2\right)\Delta t + \sigma_i \sqrt{\Delta t} \, Z_{i, \text{corr}} \right)$$

where:
- $S_i(t)$ is the spot price of asset $i$
- $\mu_i$ is the annualized expected drift
- $\sigma_i$ is the annualized volatility
- $\Delta t = \frac{\text{horizon}}{252}$ is the trading time step (252 business days per year)

### 2. Cholesky Correlation Factorization
To generate correlated shocks from independent standard normal variates $Z \sim \mathcal{N}(0, \mathbf{I})$, we decompose the empirical correlation matrix $\mathbf{\Sigma}$:

$$\mathbf{\Sigma} = \mathbf{L} \mathbf{L}^T$$

where $\mathbf{L}$ is a lower-triangular matrix. The correlated shocks are obtained via:

$$Z_{\text{corr}} = \mathbf{L} \cdot Z \implies \text{Cov}(Z_{\text{corr}}) = \mathbf{L} \mathbf{I} \mathbf{L}^T = \mathbf{\Sigma}$$

A small numerical jitter ($\varepsilon = 10^{-10}$) is applied to the diagonal to guarantee strict positive-definiteness on empirical matrices.

### 3. Variance Reduction: Antithetic Variates
For every random shock $+Z$, the engine simultaneously generates its mirror counterpart $-Z$. Because $\mathbb{E}[Z + (-Z)] = 0$, this antithetic coupling substantially reduces Monte Carlo sampling variance without additional generator calls.

---

## Market Risk Metrics

From the terminal portfolio distribution, we compute the loss distribution $L = -(V_{\text{terminal}} - V_0)$:

1. **Value at Risk ($\text{VaR}_\alpha$)**:
   The maximum loss expected over the horizon at confidence level $\alpha$ (e.g., 95%, 99%, 99.9%):
   $$\text{VaR}_\alpha = \text{Percentile}(L, \alpha)$$

2. **Conditional Value at Risk ($\text{CVaR}_\alpha$ / Expected Shortfall)**:
   The average loss in the tail distribution beyond the VaR threshold (Basel III / FRTB standard):
   $$\text{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha]$$

Mathematically, $\text{CVaR}_\alpha \ge \text{VaR}_\alpha$ always holds.

---

## Project Structure

```text
├── cpp/
│   ├── include/
│   │   ├── cholesky.hpp          # Cholesky factorization & vector correlation
│   │   ├── rng.hpp               # Thread-safe 64-bit PRNG & antithetic pairs
│   │   └── simulation_engine.hpp # Multi-threaded Monte Carlo engine header
│   └── src/
│       ├── cholesky.cpp          # 1D flattened Cholesky implementation
│       ├── simulation_engine.cpp # Parallel worker threads & GBM loops
│       └── bindings.cpp         # Pybind11 module with GIL release
├── risk_engine/
│   ├── __init__.py               # Package init & Windows DLL directory resolution
│   ├── portfolio.py              # Portfolio dataclass with validation checks
│   └── risk_engine.py            # Quantitative VaR & CVaR evaluator
├── build_extension.py            # Automated compilation script (GCC / MinGW)
├── main.py                       # End-to-end demo simulating 1,000,000 paths
└── README.md
```
```

---

## Quickstart

### Prerequisites
```bash
pip install numpy pybind11
```

### 1. Build the C++ Extension
```bash
python build_extension.py
```

### 2. Run the Risk Pipeline
```bash
python main.py
```
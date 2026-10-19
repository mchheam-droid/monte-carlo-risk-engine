# Monte Carlo Portfolio Risk Engine

A Monte Carlo engine for portfolio market risk (VaR and Expected Shortfall). The simulation core is written in C++ and runs on several threads. It is exposed to Python through Pybind11, and Python handles the portfolio definition and the risk report.

I built it to learn how market risk is measured in practice, and to see how much a C++ core helps compared to a Python loop. Next step is derivatives pricing (see Roadmap).

## What it does

- Simulates correlated multi-asset prices over a short horizon (1 day in the demo) under geometric Brownian motion.
- Correlates the shocks with a Cholesky factorization of the correlation matrix.
- Optionally uses antithetic variates (each draw `Z` is paired with `-Z`).
- Splits the paths across `std::thread` workers, each with its own random number generator. The GIL is released during the computation.
- Computes VaR and Expected Shortfall (ES) from the simulated loss distribution.

## Model

For asset *i*, the price after a step of length Δt is

$$S_i(t+\Delta t) = S_i(t)\,\exp\!\Big(\big(\mu_i - \tfrac{1}{2}\sigma_i^2\big)\Delta t + \sigma_i\sqrt{\Delta t}\,Z_i\Big)$$

with Δt = horizon / 252 (horizon in business days), μ the annualized drift and σ the annualized volatility.

The shocks are correlated through the Cholesky factor L of the correlation matrix Σ:

$$\Sigma = L L^{\top}, \qquad Z = L\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

A jitter of 1e-10 is added to the diagonal so the factorization does not fail on nearly singular empirical matrices.

## Risk measures

With V₀ the initial portfolio value and V the simulated terminal value, the loss is L = V₀ − V.

VaR at level α is the α-quantile of the loss: the loss that is exceeded with probability 1 − α.

$$\mathrm{VaR}_{\alpha} = \inf\{\ell : P(L \le \ell) \ge \alpha\}$$

Expected Shortfall is the average loss in the cases where the loss reaches or exceeds the VaR:

$$\mathrm{ES}_{\alpha} = E\big[\,L \mid L \ge \mathrm{VaR}_{\alpha}\,\big]$$

By construction ES ≥ VaR. ES is the measure used in the FRTB framework.

## Example output

1,000,000 paths, 5 correlated stocks, 1-day horizon. The engine runs in 79.75 ms (about 12.5 million paths per second).

```
 Portfolio Value      : $244.25
 Expected PnL (Mean)  : $0.08
 Worst-Case Loss      : $14.03
 VaR 95.0%            : $5.01  (2.05%)
 VaR 99.0%            : $7.07  (2.89%)
 VaR 99.9%            : $9.39  (3.84%)
 ES  95.0%            : $6.27  (2.57%)
 ES  99.0%            : $8.10  (3.31%)
 ES  99.9%            : $10.23 (4.19%)
```

The VaR and ES ratios are close to the Gaussian ones (for example VaR 99% / VaR 95% ≈ 1.41), which is expected for a one-day GBM.

## Limitations

- Constant volatility and Gaussian returns: no volatility smile, no fat tails. Extreme losses (99.9%) are probably underestimated compared to real markets.
- Only the terminal value is simulated (one time step), which is exact for GBM but not enough for path-dependent products.
- `weight` is a number of shares, not a percentage of the portfolio value.
- The drift is a historical-style drift μ. For pricing, it would have to be replaced by the risk-free rate (risk-neutral measure).
- The effect of antithetic variates on the VaR/ES estimates has not been measured yet.

## Roadmap

- [ ] Input validation in the C++ engine (number of paths, matrix size)
- [ ] Check VaR against the parametric (variance-covariance) VaR
- [ ] Benchmark against a vectorized NumPy baseline, with scaling by number of threads
- [ ] Multi-step paths and European / barrier option pricing, validated against Black-Scholes
- [ ] Reproducible results independent of the number of threads

## Project structure

```
cpp/
  include/
    cholesky.hpp           Cholesky factorization and correlation of shocks
    rng.hpp                per-thread random number generator, antithetic pairs
    simulation_engine.hpp  engine interface
  src/
    cholesky.cpp
    simulation_engine.cpp  worker threads and GBM loop
    bindings.cpp           Pybind11 module (releases the GIL)
risk_engine/
  __init__.py
  portfolio.py             portfolio definition and input checks
  risk_engine.py           VaR / ES computation
build_extension.py         build script (g++ / MinGW)
main.py                    demo with 1,000,000 paths
```

## Usage

Requirements: Python 3, `numpy`, `pybind11`, and `g++` (MinGW on Windows).

```bash
pip install numpy pybind11
python build_extension.py
python main.py
```
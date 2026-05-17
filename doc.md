# Monte Carlo Simulation - Project Documentation

## Overview

Monte Carlo Stock Terminal — a professional GBM (Geometric Brownian Motion) engine for simulating equity paths and pricing options using Monte Carlo methods. It produces statistical summaries, risk metrics and publication-quality visualisations (Matplotlib PNGs and interactive Plotly HTML).

## Features

- Calibrate drift and volatility from historical data (yfinance fallback to synthetic data).
- Simulate thousands of paths with pseudorandom, antithetic and Sobol (QMC) sampling.
- Price European, Asian, barrier and American options (LSMC) with variance-reduction techniques.
- Compute pathwise Greeks, convergence analysis and multi-asset simulations via Cholesky.
- Generate Matplotlib (PNG) and Plotly (HTML) visualisations saved to `mc_outputs/`.

## Repository contents

- `sim.py` — main application and GBM engine (CLI entrypoint).
- `plotly.md` — supplemental notes about plotting.

## Requirements

- Python 3.8+ recommended
- Primary libraries:
  - numpy, pandas, scipy
  - matplotlib, plotly (optional)
  - yfinance (optional, for real historical data)
  - rich (optional, for pretty CLI)


```
pip install numpy pandas scipy matplotlib plotly yfinance rich
```

## Quick Start

```bash
python sim.py 
```

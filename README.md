# Monte-Carlo-Simulation-Advance

Monte Carlo simulation for stock-price scenario analysis using a Geometric Brownian Motion (GBM) engine.

It supports:
- Historical calibration from Yahoo Finance (with synthetic fallback)
- Monte Carlo path simulation (pseudo-random, antithetic variates, Sobol QMC)
- Advanced option pricing (European, Asian, Barrier, American LSMC)
- Pathwise Greeks (Delta, Vega, Rho)
- Variance-reduction diagnostics and convergence analysis
- Multi-asset correlated simulation (Cholesky)
- Matplotlib static charts and Plotly interactive dashboards

## Dependencies

```bash
pip install numpy pandas matplotlib scipy yfinance rich
```

### Optional dependency (interactive Plotly dashboards)

```bash
pip install plotly
```


## Quick Start
```bash
python sim.py
```

<img width="681" height="459" alt="Image" src="https://github.com/user-attachments/assets/23c995b9-9a49-4d89-a630-be43c7f64e4b" />

<img width="795" height="790" alt="Image" src="https://github.com/user-attachments/assets/576f3bb5-de9d-4b18-8e37-420e98118087" />

<img width="1049" height="483" alt="Image" src="https://github.com/user-attachments/assets/43796864-d34e-495a-9652-21406f82a172"/>

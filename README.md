# Monte Carlo Stock Terminal

A Monte Carlo simulation engine for stock price modeling and options pricing using Geometric Brownian Motion (GBM). This terminal provides analysis tools, advanced variance reduction techniques, and extensive diagnostic for quantitative finance applications.


## Features

### Core Simulation Engine
- Geometric Brownian Motion (GBM) path generation
- Real-time stock data fetching via Yahoo Finance
- Calibration from historical price data
- Multiple random number generation methods:
  - Pseudo-random (standard)
  - Quasi-Monte Carlo (Sobol sequences)
  - Antithetic variates for variance reduction

### Options Pricing
- **European Options**: Call and Put options with Black-Scholes benchmarking
- **Asian Options**: Path-dependent options based on average price
- **Barrier Options**: Up-and-out, down-and-out, up-and-in, down-and-in
- **American Options**: Early exercise via Longstaff-Schwartz Monte Carlo (LSMC)

### Variance Reduction Techniques
- Antithetic variates
- Control variates
- Quasi-Monte Carlo (QMC) with Sobol sequences
- Comparative analysis of variance reduction effectiveness

### Greeks Calculation
- Delta: Price sensitivity to underlying asset
- Vega: Price sensitivity to volatility
- Rho: Price sensitivity to interest rate
- Pathwise derivative method for unbiased estimates

### Advanced Analytics
- Multi-asset simulation with correlation modeling
- Convergence analysis and error estimation
- Monte Carlo Standard Error (MCSE) tracking
- Risk metrics: VaR, CVaR, Sharpe, Sortino, Calmar ratios
- Maximum drawdown analysis
- RSI divergence detection

### Diagnostic Tools
- **Convergence Heatmaps**: Price error and standard error surfaces across path/step grids
- **Bias Analysis**: Repeated-run convergence drift detection
- **Discretization Analysis**: Monitoring resolution impact on barrier and Asian options
- **Variance Decomposition**: Uncertainty attribution across multiple sources
- **Failure Region Maps**: Identification of fragile parameter regimes
- **Path Instability Analysis**: Sensitivity to seed and volatility perturbations
- **RNG Diagnostics**: Q-Q plots, autocorrelation, Ljung-Box, Shapiro-Wilk tests

### Visualization Suite
All visualizations are generated as interactive HTML files using Plotly:
- 3D price path surfaces with animation
- Probability density surfaces
- Convergence dashboards
- Sensitivity analysis charts
- Diagnostic failure maps
- Historical price analysis with technical indicators
- Multi-scenario comparison tools


```bash
pip install numpy plotly scipy pandas yfinance rich
```

### Package Details
- `numpy`: Numerical computing and array operations
- `plotly`: Interactive visualization and charting
- `scipy`: Statistical functions and optimization
- `pandas`: Data manipulation and time series
- `yfinance`: Real-time stock data fetching
- `rich`: Enhanced terminal output and progress bars

## Usage

Run the simulation with default parameters (interactive mode):

```bash
python sim.py
```

You will be prompted to enter:
- Ticker symbol (default: AAPL)
- Forecast days (default: 252 = 1 year)
- Number of simulations (default: 500)



## Output Files

All outputs are saved to the `mc_outputs/` directory:

### HTML Visualizations
- `mc_dashboard_TICKER.html` - Main dashboard with key metrics
- `mc_3d_TICKER.html` - 3D price path surface
- `mc_3d_animation_TICKER.html` - Animated 3D paths
- `mc_density_surface_TICKER.html` - Probability density visualization
- `mc_surface_TICKER.html` - Price surface analysis
- `mc_history_detail_TICKER.html` - Historical price analysis with RSI
- `mc_sensitivity_TICKER.html` - Parameter sensitivity analysis
- `mc_diagnostics_TICKER.html` - Comprehensive diagnostic dashboard
- `mc_diagnostics_failure_lab_TICKER.html` - Failure mode analysis
- `mc_rng_diagnostics_TICKER.html` - Random number generator quality tests
- `mc_advanced_extensions_TICKER.html` - Advanced pricing techniques
- `mc_setup_lab_TICKER.html` - Simulation setup and configuration
- `pathwise_intuition_TICKER.html` - Greeks calculation visualization

### Text Reports
- `mc_report_TICKER.txt` - Summary statistics and metrics

## Technical Details

### Geometric Brownian Motion Model

The simulation uses the standard GBM model for stock price evolution:

```
dS = μS dt + σS dW
```

Where:
- S: Stock price
- μ: Drift (expected return)
- σ: Volatility
- W: Wiener process (Brownian motion)

### Calibration

Parameters are calibrated from historical data:
- **Drift (μ)**: Annualized mean of log returns
- **Volatility (σ)**: Annualized standard deviation of log returns
- **Initial Price (S0)**: Most recent closing price

### Risk-Neutral Pricing

For options pricing, the simulation uses risk-neutral valuation where the drift is replaced by the risk-free rate:

```
dS = rS dt + σS dW
```

### Variance Reduction

The terminal implements multiple variance reduction techniques:

1. **Antithetic Variates**: Generates paired paths with negated random numbers
2. **Control Variates**: Uses correlated variables with known expectations
3. **Quasi-Monte Carlo**: Employs low-discrepancy Sobol sequences

Variance reduction ratios are computed and reported for comparison.

## Statistical Metrics

### Risk Metrics
- **Value at Risk (VaR)**: 95th percentile loss
- **Conditional VaR (CVaR)**: Expected loss beyond VaR
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return relative to maximum drawdown

### Convergence Metrics
- **Monte Carlo Standard Error (MCSE)**: Estimation uncertainty
- **Confidence Intervals**: 95% confidence bounds
- **Bias Analysis**: Systematic estimation error
- **Log-log Slope**: Convergence rate verification

## Diagnostic Features

### Failure Region Detection

The terminal identifies parameter regimes where Monte Carlo becomes unreliable:

1. **Deep OTM / Volatility Fragility**: High noise for out-of-the-money options
2. **Barrier Proximity**: Discretization bias near barrier levels
3. **Long Maturity / Low Sample Size**: Inefficiency for extended horizons

### RNG Quality Tests

Comprehensive random number generator diagnostics:
- Q-Q plots against theoretical normal distribution
- Autocorrelation function analysis
- Ljung-Box test for independence
- Shapiro-Wilk normality test
- Anderson-Darling test
- Kolmogorov-Smirnov test


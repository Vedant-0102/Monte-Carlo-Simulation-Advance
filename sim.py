"""
MONTE CARLO STOCK TERMINAL

Install:  pip install numpy plotly scipy pandas yfinance rich
Run:      python test.py
Flags:    --ticker AAPL --sims 500 --renderer plotly
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import norm, t as t_dist
from scipy.stats import qmc

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import box
    HAS_RICH = True
    con = Console()
except ImportError:
    HAS_RICH = False
    con = None

STDOUT_ENCODING = (getattr(sys.stdout, "encoding", None) or "").lower()
ASCII_TERMINAL = not STDOUT_ENCODING.startswith("utf")
if ASCII_TERMINAL:
    HAS_RICH = False

OUT = Path("mc_outputs")
OUT.mkdir(exist_ok=True)
# THEME
C = {
    "bg":      "#0d1117",
    "bg2":     "#161b22",
    "bg3":     "#1c2128",
    "border":  "#30363d",
    "muted":   "#8b949e",
    "text":    "#c9d1d9",
    "title":   "#f0f6fc",
    "blue":    "#58a6ff",
    "green":   "#3fb950",
    "red":     "#f85149",
    "amber":   "#e3b341",
    "purple":  "#a371f7",
    "cyan":    "#39c5cf",
    "pink":    "#f778ba",
}

def _compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _find_bearish_rsi_divergences(price: pd.Series, rsi: pd.Series, lookback: int = 20) -> list[pd.Timestamp]:
    hits: list[pd.Timestamp] = []
    if len(price) <= lookback:
        return hits

    rolling_price_max = price.rolling(lookback).max().shift(1)
    rolling_rsi_max = rsi.rolling(lookback).max().shift(1)
    for idx in range(lookback, len(price)):
        p_now = float(price.iloc[idx])
        p_prev_max = rolling_price_max.iloc[idx]
        r_now = float(rsi.iloc[idx]) if np.isfinite(rsi.iloc[idx]) else np.nan
        r_prev_max = rolling_rsi_max.iloc[idx]
        if not np.isfinite(p_prev_max) or not np.isfinite(r_now) or not np.isfinite(r_prev_max):
            continue
        if p_now > p_prev_max and r_now < r_prev_max - 2.5:
            hits.append(price.index[idx])
    return hits[-6:]

def _history_market_data(ticker: str, idx: pd.Index) -> pd.DataFrame:
    if HAS_YF:
        try:
            raw = yf.Ticker(ticker).history(period="2y")
            cols = [c for c in ["Open", "Close", "Volume"] if c in raw.columns]
            if cols:
                out = raw[cols].reindex(idx)
                if "Close" not in out:
                    out["Close"] = np.nan
                out["Close"] = out["Close"].fillna(method="ffill").fillna(method="bfill")
                if "Open" not in out:
                    out["Open"] = out["Close"].shift(1).fillna(out["Close"])
                else:
                    out["Open"] = out["Open"].fillna(out["Close"].shift(1)).fillna(out["Close"])
                if "Volume" not in out:
                    out["Volume"] = 0.0
                out["Volume"] = out["Volume"].fillna(0.0)
                return out
        except Exception:
            pass

    rng = np.random.default_rng(7)
    close = pd.Series(np.nan, index=idx, dtype=float)
    open_ = pd.Series(np.nan, index=idx, dtype=float)
    volume = pd.Series(np.abs(rng.standard_normal(len(idx))) * 5e7 + 3e7, index=idx, dtype=float)
    return pd.DataFrame({"Open": open_, "Close": close, "Volume": volume}, index=idx)

def _contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(mask) - 1))
    return ranges

def _ljung_box_pvalue(sample: np.ndarray, max_lag: int) -> float:
    n = len(sample)
    if n <= max_lag + 1 or max_lag <= 0:
        return float("nan")
    centered = sample - np.mean(sample)
    denom = float(np.sum(centered**2)) + 1e-12
    q_stat = 0.0
    for lag in range(1, max_lag + 1):
        acf = float(np.sum(centered[:-lag] * centered[lag:]) / denom)
        q_stat += (acf * acf) / max(n - lag, 1)
    q_stat *= n * (n + 2)
    return float(scipy_stats.chi2.sf(q_stat, df=max_lag))

def _rng_diagnostic_sample(n: int = 4096, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)

def _rng_uniform_pairs(n: int = 4096, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.random(n + 1)
    return u[:-1], u[1:]

def _compute_rng_diagnostics(sample: np.ndarray, max_lag: int = 40) -> dict:
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    n = len(sample)
    if n == 0:
        return {
            "sample": sample,
            "sorted_sample": np.array([]),
            "theoretical": np.array([]),
            "diag_min": -1.0,
            "diag_max": 1.0,
            "ks_band": 0.0,
            "acf_lags": np.array([]),
            "acf_vals": np.array([]),
            "acf_conf": 0.0,
            "acf_flags": np.array([], dtype=bool),
            "lag1": float("nan"),
            "ljung_box_p": float("nan"),
            "shapiro_p": float("nan"),
            "anderson_stat": float("nan"),
        }

    p = (np.arange(1, n + 1) - 0.5) / n
    theoretical = norm.ppf(p)
    sorted_sample = np.sort(sample)
    band = 1.36 / np.sqrt(n)
    diag_min = float(min(theoretical.min(), sorted_sample.min()))
    diag_max = float(max(theoretical.max(), sorted_sample.max()))

    lag_max = min(max_lag, n - 2)
    centered = sample - sample.mean()
    denom = float(np.sum(centered**2)) + 1e-12
    lags = np.arange(1, lag_max + 1)
    acf_vals = np.array([float(np.sum(centered[:-lag] * centered[lag:]) / denom) for lag in lags], dtype=float)
    acf_conf = 1.96 / np.sqrt(n)
    acf_flags = np.abs(acf_vals) > acf_conf

    shapiro_sample = sample[: min(5000, n)]
    try:
        shapiro_p = float(scipy_stats.shapiro(shapiro_sample).pvalue)
    except Exception:
        shapiro_p = float("nan")
    try:
        anderson_stat = float(scipy_stats.anderson(sample, dist="norm").statistic)
    except Exception:
        anderson_stat = float("nan")

    return {
        "sample": sample,
        "sorted_sample": sorted_sample,
        "theoretical": theoretical,
        "diag_min": diag_min,
        "diag_max": diag_max,
        "ks_band": band,
        "acf_lags": lags,
        "acf_vals": acf_vals,
        "acf_conf": acf_conf,
        "acf_flags": acf_flags,
        "lag1": float(acf_vals[0]) if len(acf_vals) else float("nan"),
        "ljung_box_p": _ljung_box_pvalue(sample, lag_max),
        "shapiro_p": shapiro_p,
        "anderson_stat": anderson_stat,
    }

def _surface_z_bounds(Z: np.ndarray, S0: float) -> tuple[float, float]:
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    anchor = max(abs(float(S0)), 1.0)
    span = max(z_max - z_min, anchor * 0.05)
    lo = min(z_min, float(S0)) - span * 0.12
    hi = max(z_max, float(S0)) + span * 0.08
    hi = max(hi, lo + 1.0)
    return lo, hi

# RICH HELPERS
def rlog(msg, style=""):
    import re

    if ASCII_TERMINAL:
        cleaned = re.sub(r"\[.*?\]", "", msg)
        cleaned = cleaned.replace("OK", "[OK]").replace("WARN", "[WARN]")
        print(cleaned.encode("ascii", "ignore").decode("ascii"))
        return

    if HAS_RICH:
        con.print(msg, style=style)
    else:
        print(re.sub(r"\[.*?\]", "", msg))

def print_banner():
    if ASCII_TERMINAL:
        print("=" * 72)
        print("  __  __  ___  _   _ _____ _____    ____    _    ____  _     ___")
        print(" |  \\/  |/ _ \\| \\ | |_   _| ____|  / ___|  / \\  |  _ \\| |   / _ \\")
        print(" | |\\/| | | | |  \\| | | | |  _|   | |     / _ \\ | |_) | |  | | | |")
        print(" | |  | | |_| | |\\  | | | | |___  | |___ / ___ \\|  _ <| |__| |_| |")
        print(" |_|  |_|\\___/|_| \\_| |_| |_____|  \\____/_/   \\_\\_| \\_\\_____\\___/")
        print("  Professional GBM Engine - v4.0")
        print("=" * 72)
        print()
        return

    b = r"""
  __  __  ___  _   _ _____ _____    ____    _    ____  _     ___
 |  \/  |/ _ \| \ | |_   _| ____|  / ___|  / \  |  _ \| |   / _ \
 | |\/| | | | |  \| | | | |  _|   | |     / _ \ | |_) | |  | | | |
 | |  | | |_| | |\  | | | | |___  | |___ / ___ \|  _ <| |__| |_| |
 |_|  |_|\___/|_| \_| |_| |_____|  \____/_/   \_\_| \_\_____\___/
    """
    if HAS_RICH:
        con.print(b, style="bold cyan")
        con.print("  [dim]Professional Monte Carlo Stock Terminal  -  GBM Engine  -  v4.0[/dim]\n")
    else:
        print(b)
        print("  Monte Carlo Stock Terminal  v4.0\n")

def plain_line(width: int = 55):
    print("-" * width)

############################################################################################################################################################################################


############################################################################################################################################################################################

############################################################################################################################################################################################


# GBM ENGINE
class GBMEngine:
    def __init__(self, ticker: str):
        self.ticker  = ticker.upper()
        self.mu      = None
        self.sigma   = None
        self.S0      = None
        self.history = None
        self.log_ret = None
        self.roll_vol = None

    def fetch(self, period: str = "2y", seed: int = 99) -> pd.Series:
        if HAS_YF:
            rlog(f"  Fetching [cyan]{period}[/cyan] of data for [bold]{self.ticker}[/bold] ...")
            try:
                df = yf.Ticker(self.ticker).history(period=period)
                if df.empty:
                    raise ValueError("Empty response")
                self.history = df["Close"].dropna()
                rlog(f"  [green]OK[/green] {len(self.history)} trading days loaded  "
                     f"(${self.history.iloc[0]:.2f} -> ${self.history.iloc[-1]:.2f})")
                return self.history
            except Exception as e:
                rlog(f"  [yellow]WARN {e} - falling back to synthetic data[/yellow]")
        rlog("  [dim]Generating synthetic price series ...[/dim]")
        return self._synthetic(seed=seed)

    def _synthetic(self, days: int = 504, S0: float = 150.0,
                   mu: float = 0.14, sigma: float = 0.25,
                   seed: int = 99) -> pd.Series:
        rng = np.random.default_rng(seed)
        dt  = 1.0 / 252
        inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(days)
        px  = S0 * np.exp(np.concatenate([[0.0], np.cumsum(inc)]))
        idx = pd.date_range(end=datetime.now(), periods=days + 1, freq="B")
        self.history = pd.Series(px, index=idx, name="Close")
        return self.history

    def calibrate(self):
        lr         = np.log(self.history / self.history.shift(1)).dropna()
        self.log_ret = lr
        self.mu    = float(lr.mean() * 252)
        self.sigma = float(lr.std()  * np.sqrt(252))
        self.S0    = float(self.history.iloc[-1])
        self.roll_vol = self.log_ret.rolling(30).std() * np.sqrt(252)
        return self.mu, self.sigma, self.S0

    def _generate_normals(
        self,
        num_sim: int,
        N: int,
        rng: np.random.Generator,
        random_method: str = "pseudo",
        antithetic: bool = False,
    ) -> np.ndarray:
        if antithetic and random_method == "sobol":
            raise ValueError("antithetic=True is incompatible with random_method='sobol'. Use one or the other.")

        if random_method == "sobol":
            m = int(np.ceil(np.log2(max(num_sim, 2))))
            sobol = qmc.Sobol(d=N, scramble=True, seed=int(rng.integers(1, 2**31 - 1)))
            u = sobol.random_base2(m=m)[:num_sim]
            u = np.clip(u, 1e-12, 1 - 1e-12)
            Z = norm.ppf(u)
        else:
            Z = rng.standard_normal((num_sim, N))

        if antithetic:
            half = (num_sim + 1) // 2
            base = Z[:half]
            Z = np.vstack([base, -base])[:num_sim]
        return Z

    def _paths_from_normals(self, Z: np.ndarray, drift: float, sigma: float, S0: float, T: float) -> np.ndarray:
        N = Z.shape[1]
        dt = T / N
        inc = (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        return S0 * np.exp(
            np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)
        )

    def simulate(self, N: int = 252, num_sim: int = 500,
                 seed: int = 42, random_method: str = "pseudo",
                 antithetic: bool = False) -> tuple:
        rng = np.random.default_rng(seed)
        T   = N / 252.0
        Z = self._generate_normals(
            num_sim=num_sim,
            N=N,
            rng=rng,
            random_method=random_method,
            antithetic=antithetic,
        )
        paths = self._paths_from_normals(Z=Z, drift=self.mu, sigma=self.sigma, S0=self.S0, T=T)
        t = np.linspace(0, T, N + 1)
        return paths, t

    def simulate_risk_neutral(
        self,
        r: float,
        N: int = 252,
        num_sim: int = 5000,
        seed: int = 42,
        random_method: str = "pseudo",
        antithetic: bool = False,
        return_Z: bool = False,
    ):
        rng = np.random.default_rng(seed)
        T = N / 252.0
        Z = self._generate_normals(
            num_sim=num_sim,
            N=N,
            rng=rng,
            random_method=random_method,
            antithetic=antithetic,
        )
        paths = self._paths_from_normals(Z=Z, drift=r, sigma=self.sigma, S0=self.S0, T=T)
        t = np.linspace(0, T, N + 1)
        if return_Z:
            return paths, t, Z
        return paths, t

    def black_scholes_price(self, K: float, T: float, r: float = 0.05, option_type: str = "call") -> float:
        if T <= 0:
            if option_type == "put":
                return max(K - self.S0, 0.0)
            return max(self.S0 - K, 0.0)
        if self.sigma <= 0:
            fwd = self.S0 * np.exp(r * T)
            disc = np.exp(-r * T)
            if option_type == "put":
                return disc * max(K - fwd, 0.0)
            return disc * max(fwd - K, 0.0)

        vol_sqrt = self.sigma * np.sqrt(T)
        d1 = (np.log(self.S0 / K) + (r + 0.5 * self.sigma**2) * T) / vol_sqrt
        d2 = d1 - vol_sqrt
        if option_type == "put":
            return float(K * np.exp(-r * T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1))
        return float(self.S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    @staticmethod
    def _summary_with_error(samples: np.ndarray) -> dict:
        n = len(samples)
        if n == 0:
            return {"estimate": 0.0, "stderr": 0.0, "ci95_halfwidth": 0.0, "sample_std": 0.0}
        est = float(np.mean(samples))
        if n == 1:
            return {"estimate": est, "stderr": 0.0, "ci95_halfwidth": 0.0, "sample_std": 0.0}
        sd = float(np.std(samples, ddof=1))
        se = float(sd / np.sqrt(n))
        return {
            "estimate": est,
            "stderr": se,
            "ci95_halfwidth": float(1.96 * se),
            "sample_std": sd,
        }

    def price_european_option_mc(
        self,
        K: float,
        r: float = 0.05,
        N: int = 252,
        num_sim: int = 5000,
        option_type: str = "call",
        seed: int = 42,
        random_method: str = "pseudo",
        antithetic: bool = True,
        control_variate: bool = False,
    ) -> dict:
        paths, _ = self.simulate_risk_neutral(
            r=r,
            N=N,
            num_sim=num_sim,
            seed=seed,
            random_method=random_method,
            antithetic=antithetic,
        )
        T = N / 252.0
        disc = np.exp(-r * T)
        ST = paths[:, -1]
        if option_type == "put":
            payoff = np.maximum(K - ST, 0.0)
        else:
            payoff = np.maximum(ST - K, 0.0)
        pv = disc * payoff

        beta = 0.0
        if control_variate:
            cv = disc * ST
            cv_target = self.S0
            cv_var = float(np.var(cv, ddof=1))
            if cv_var > 1e-14:
                beta = float(np.cov(pv, cv, ddof=1)[0, 1] / cv_var)
                pv = pv - beta * (cv - cv_target)

        summ = self._summary_with_error(pv)
        return {
            "price": summ["estimate"],
            "stderr": summ["stderr"],
            "ci95": (
                summ["estimate"] - summ["ci95_halfwidth"],
                summ["estimate"] + summ["ci95_halfwidth"],
            ),
            "sample_std": summ["sample_std"],
            "control_variate_beta": beta,
            "num_sim": num_sim,
            "random_method": random_method,
            "antithetic": antithetic,
            "control_variate": control_variate,
        }

    def price_asian_option_mc(
        self,
        K: float,
        r: float = 0.05,
        N: int = 252,
        num_sim: int = 5000,
        option_type: str = "call",
        seed: int = 42,
        random_method: str = "pseudo",
        antithetic: bool = True,
        control_variate: bool = True,
    ) -> dict:
        paths, _ = self.simulate_risk_neutral(
            r=r,
            N=N,
            num_sim=num_sim,
            seed=seed,
            random_method=random_method,
            antithetic=antithetic,
        )
        T = N / 252.0
        disc = np.exp(-r * T)
        avg_price = paths[:, 1:].mean(axis=1)
        ST = paths[:, -1]

        if option_type == "put":
            raw = np.maximum(K - avg_price, 0.0)
            euro = np.maximum(K - ST, 0.0)
        else:
            raw = np.maximum(avg_price - K, 0.0)
            euro = np.maximum(ST - K, 0.0)

        pv = disc * raw
        beta = 0.0
        if control_variate:
            cv = disc * euro
            cv_target = self.black_scholes_price(K=K, T=T, r=r, option_type=option_type)
            cv_var = float(np.var(cv, ddof=1))
            if cv_var > 1e-14:
                beta = float(np.cov(pv, cv, ddof=1)[0, 1] / cv_var)
                pv = pv - beta * (cv - cv_target)

        summ = self._summary_with_error(pv)
        return {
            "price": summ["estimate"],
            "stderr": summ["stderr"],
            "ci95": (
                summ["estimate"] - summ["ci95_halfwidth"],
                summ["estimate"] + summ["ci95_halfwidth"],
            ),
            "sample_std": summ["sample_std"],
            "control_variate_beta": beta,
        }

    def price_barrier_option_mc(
        self,
        K: float,
        barrier: float,
        r: float = 0.05,
        N: int = 252,
        num_sim: int = 5000,
        option_type: str = "call",
        barrier_type: str = "up-and-out",
        seed: int = 42,
        random_method: str = "pseudo",
        antithetic: bool = True,
    ) -> dict:
        paths, _ = self.simulate_risk_neutral(
            r=r,
            N=N,
            num_sim=num_sim,
            seed=seed,
            random_method=random_method,
            antithetic=antithetic,
        )
        T = N / 252.0
        disc = np.exp(-r * T)
        ST = paths[:, -1]
        crossed_up = paths.max(axis=1) >= barrier
        crossed_down = paths.min(axis=1) <= barrier

        if option_type == "put":
            vanilla = np.maximum(K - ST, 0.0)
        else:
            vanilla = np.maximum(ST - K, 0.0)

        if barrier_type == "up-and-out":
            active = ~crossed_up
        elif barrier_type == "down-and-out":
            active = ~crossed_down
        elif barrier_type == "up-and-in":
            active = crossed_up
        elif barrier_type == "down-and-in":
            active = crossed_down
        else:
            raise ValueError("barrier_type must be one of: up-and-out, down-and-out, up-and-in, down-and-in")

        pv = disc * vanilla * active.astype(float)
        summ = self._summary_with_error(pv)
        return {
            "price": summ["estimate"],
            "stderr": summ["stderr"],
            "ci95": (
                summ["estimate"] - summ["ci95_halfwidth"],
                summ["estimate"] + summ["ci95_halfwidth"],
            ),
            "sample_std": summ["sample_std"],
            "active_ratio": float(np.mean(active)),
        }

    def price_american_option_lsmc(
        self,
        K: float,
        r: float = 0.05,
        N: int = 252,
        num_sim: int = 8000,
        option_type: str = "put",
        seed: int = 42,
        random_method: str = "pseudo",
        antithetic: bool = True,
    ) -> dict:
        paths, _ = self.simulate_risk_neutral(
            r=r,
            N=N,
            num_sim=num_sim,
            seed=seed,
            random_method=random_method,
            antithetic=antithetic,
        )
        dt = (N / 252.0) / N
        if option_type == "put":
            intrinsic = np.maximum(K - paths, 0.0)
        else:
            intrinsic = np.maximum(paths - K, 0.0)

        exercise_time = np.full(num_sim, N, dtype=int)
        cashflow = intrinsic[:, -1].copy()

        for t in range(N - 1, 0, -1):
            itm_alive = (intrinsic[:, t] > 0) & (exercise_time > t)
            idx = np.where(itm_alive)[0]
            if len(idx) < 5:
                continue

            X = paths[idx, t]
            Y = cashflow[idx] * np.exp(-r * dt * (exercise_time[idx] - t))
            X_norm = (X - X.mean()) / (X.std() + 1e-9)
            A = np.column_stack([np.ones_like(X_norm), X_norm, X_norm**2])
            beta = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = A @ beta

            exercise_now = intrinsic[idx, t] > continuation
            ex_idx = idx[exercise_now]
            cashflow[ex_idx] = intrinsic[ex_idx, t]
            exercise_time[ex_idx] = t

        pv = cashflow * np.exp(-r * dt * exercise_time)
        summ = self._summary_with_error(pv)
        return {
            "price": summ["estimate"],
            "stderr": summ["stderr"],
            "ci95": (
                summ["estimate"] - summ["ci95_halfwidth"],
                summ["estimate"] + summ["ci95_halfwidth"],
            ),
            "sample_std": summ["sample_std"],
            "early_exercise_ratio": float(np.mean(exercise_time < N)),
        }

    def pathwise_greeks(
        self,
        K: float,
        r: float = 0.05,
        N: int = 252,
        num_sim: int = 8000,
        option_type: str = "call",
        seed: int = 42,
        random_method: str = "pseudo",
        antithetic: bool = True,
    ) -> dict:
        paths, _, Z = self.simulate_risk_neutral(
            r=r,
            N=N,
            num_sim=num_sim,
            seed=seed,
            random_method=random_method,
            antithetic=antithetic,
            return_Z=True,
        )
        T = N / 252.0
        disc = np.exp(-r * T)
        ST = paths[:, -1]
        # For antithetic sampling, paired paths have opposite Z and opposite z_terminal.
        # Averaging these pathwise terms preserves unbiased Greeks while reducing variance.
        z_terminal = Z.sum(axis=1) / np.sqrt(N)

        if option_type == "put":
            ind = (ST < K).astype(float)
            sign = -1.0
            payoff = np.maximum(K - ST, 0.0)
        else:
            ind = (ST > K).astype(float)
            sign = 1.0
            payoff = np.maximum(ST - K, 0.0)

        dST_dS0 = ST / self.S0
        dST_dsigma = ST * (np.sqrt(T) * z_terminal - self.sigma * T)
        delta_samples = disc * sign * ind * dST_dS0
        vega_samples = disc * sign * ind * dST_dsigma
        rho_samples = disc * T * (sign * ind * ST - payoff)

        delta = self._summary_with_error(delta_samples)
        vega = self._summary_with_error(vega_samples)
        rho = self._summary_with_error(rho_samples)
        return {
            "delta": delta["estimate"],
            "delta_stderr": delta["stderr"],
            "vega": vega["estimate"],
            "vega_stderr": vega["stderr"],
            "rho": rho["estimate"],
            "rho_stderr": rho["stderr"],
        }

    def convergence_analysis(
        self,
        K: float,
        r: float = 0.05,
        N: int = 252,
        option_type: str = "call",
        n_grid: list[int] | None = None,
        antithetic: bool = True,
        random_method: str = "pseudo",
        control_variate: bool = False,
        seed: int = 42,
    ) -> dict:
        if n_grid is None:
            n_grid = [250, 500, 1000, 2000, 4000, 8000]
        estimates = []
        stderrs = []
        for i, n in enumerate(n_grid):
            # Use distinct seeds per N to estimate independent MC error on the convergence curve.
            res = self.price_european_option_mc(
                K=K,
                r=r,
                N=N,
                num_sim=n,
                option_type=option_type,
                seed=seed + 31 * i,
                random_method=random_method,
                antithetic=antithetic,
                control_variate=control_variate,
            )
            estimates.append(float(res["price"]))
            stderrs.append(max(float(res["stderr"]), 1e-12))

        slope = float(np.polyfit(np.log(np.array(n_grid, dtype=float)), np.log(np.array(stderrs)), 1)[0])
        return {
            "n_grid": n_grid,
            "estimates": estimates,
            "stderrs": stderrs,
            "loglog_slope": slope,
            "expected_slope": -0.5,
        }

    def european_payoff_samples(
        self,
        K: float,
        r: float = 0.05,
        N: int = 252,
        num_sim: int = 8000,
        option_type: str = "call",
        seed: int = 42,
        random_method: str = "pseudo",
        antithetic: bool = False,
    ) -> np.ndarray:
        paths, _ = self.simulate_risk_neutral(
            r=r,
            N=N,
            num_sim=num_sim,
            seed=seed,
            random_method=random_method,
            antithetic=antithetic,
        )
        T = N / 252.0
        disc = np.exp(-r * T)
        ST = paths[:, -1]
        if option_type == "put":
            payoff = np.maximum(K - ST, 0.0)
        else:
            payoff = np.maximum(ST - K, 0.0)
        return disc * payoff

    def simulate_multi_asset_cholesky(
        self,
        S0_vec: np.ndarray,
        mu_vec: np.ndarray,
        sigma_vec: np.ndarray,
        corr: np.ndarray,
        N: int = 252,
        num_sim: int = 3000,
        seed: int = 42,
        risk_neutral_rate: float | None = None,
    ) -> np.ndarray:
        S0_vec = np.asarray(S0_vec, dtype=float)
        mu_vec = np.asarray(mu_vec, dtype=float)
        sigma_vec = np.asarray(sigma_vec, dtype=float)
        corr = np.asarray(corr, dtype=float)
        d = len(S0_vec)
        if corr.shape != (d, d):
            raise ValueError("corr matrix must have shape (n_assets, n_assets)")

        chol = np.linalg.cholesky(corr)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((num_sim, N, d))
        corr_Z = np.matmul(Z, chol.T)

        T = N / 252.0
        dt = T / N
        drift = np.full(d, risk_neutral_rate if risk_neutral_rate is not None else 0.0)
        if risk_neutral_rate is None:
            drift = mu_vec
        drift_term = (drift - 0.5 * sigma_vec**2) * dt
        diff_term = sigma_vec * np.sqrt(dt) * corr_Z
        inc = drift_term + diff_term
        log_paths = np.concatenate([np.zeros((num_sim, 1, d)), np.cumsum(inc, axis=1)], axis=1)
        return S0_vec * np.exp(log_paths)

    def compute_stats(self, paths: np.ndarray) -> dict:
        S0     = self.S0
        finals = paths[:, -1]
        lr     = self.log_ret.values
        n_sims = len(finals)

        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns   = (paths - running_max) / running_max
        max_dd_per  = drawdowns.min(axis=1)
        mean_max_dd = float(max_dd_per.mean())

        hist_cm     = np.maximum.accumulate(self.history.values)
        hist_dd     = (self.history.values - hist_cm) / hist_cm
        hist_max_dd = float(hist_dd.min())

        rf       = 0.05
        exc_ret  = self.mu - rf
        sharpe   = exc_ret / self.sigma if self.sigma > 0 else 0.0
        neg_lr   = lr[lr < 0]
        downside = float(neg_lr.std() * np.sqrt(252)) if len(neg_lr) > 1 else self.sigma
        sortino  = exc_ret / downside if downside > 0 else 0.0
        calmar   = exc_ret / abs(hist_max_dd) if hist_max_dd != 0 else 0.0

        var95  = S0 - float(np.percentile(finals, 5))
        var99  = S0 - float(np.percentile(finals, 1))
        tail95 = finals[finals < np.percentile(finals, 5)]
        tail99 = finals[finals < np.percentile(finals, 1)]
        cvar95 = S0 - float(tail95.mean()) if len(tail95) else var95
        cvar99 = S0 - float(tail99.mean()) if len(tail99) else var99

        wins   = finals[finals > S0]
        losses = finals[finals <= S0]

        pcts = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            pcts[str(p)] = np.percentile(paths, p, axis=0)

        roll_vol = self.roll_vol if self.roll_vol is not None else self.log_ret.rolling(30).std() * np.sqrt(252)
        jb_stat, jb_p = scipy_stats.jarque_bera(lr)

        return {
            "paths":     paths,
            "finals":    finals,
            "S0":        S0,
            "mean":      float(finals.mean()),
            "mean_stderr": float(finals.std(ddof=1) / np.sqrt(n_sims)) if n_sims > 1 else 0.0,
            "mean_ci95_halfwidth": float(1.96 * finals.std(ddof=1) / np.sqrt(n_sims)) if n_sims > 1 else 0.0,
            "median":    float(np.median(finals)),
            "std":       float(finals.std()),
            "min":       float(finals.min()),
            "max":       float(finals.max()),
            "prob_up":   float((finals > S0).mean()),
            "prob_up_stderr": float(np.sqrt(((finals > S0).mean()) * (1 - (finals > S0).mean()) / n_sims)) if n_sims > 0 else 0.0,
            "prob_2x":   float((finals > 2 * S0).mean()),
            "prob_half": float((finals < 0.5 * S0).mean()),
            "var95":     var95,
            "var99":     var99,
            "cvar95":    cvar95,
            "cvar99":    cvar99,
            "sharpe":    sharpe,
            "sortino":   sortino,
            "calmar":    calmar,
            "hist_max_dd": hist_max_dd,
            "sim_max_dd":  mean_max_dd,
            "win_rate":  float(len(wins) / len(finals)),
            "avg_win":   float(wins.mean() - S0) if len(wins) else 0.0,
            "avg_loss":  float(S0 - losses.mean()) if len(losses) else 0.0,
            "terminal_pct": {k: float(np.percentile(finals, int(k)))
                          for k in ["1","5","10","25","50","75","90","95","99"]},
            "bands":     pcts,
            "mean_path": paths.mean(axis=0),
            "roll_vol":  roll_vol,
            "jb_stat":   jb_stat,
            "jb_p":      jb_p,
            "lr_skew":   float(scipy_stats.skew(lr)),
            "lr_kurt":   float(scipy_stats.kurtosis(lr)),
        }
    


class AdvancedMonteCarloExtensions:
    """Educational Monte Carlo-only extensions layered on top of the core GBM engine."""

    def __init__(self, eng: GBMEngine):
        self.eng = eng

    @staticmethod
    def _nearest_psd_corr(mat: np.ndarray) -> np.ndarray:
        sym = 0.5 * (mat + mat.T)
        vals, vecs = np.linalg.eigh(sym)
        vals = np.clip(vals, 1e-6, None)
        psd = vecs @ np.diag(vals) @ vecs.T
        d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
        corr = psd / np.outer(d, d)
        return np.clip(corr, -0.999, 0.999)

    @staticmethod
    def _summary(samples: np.ndarray) -> dict:
        arr = np.asarray(samples, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    def _simulate_rate_paths(
        self,
        model: str,
        N: int,
        num_sim: int,
        seed: int,
        r0: float = 0.03,
        kappa: float = 1.4,
        theta: float = 0.035,
        eta: float = 0.09,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        dt = 1.0 / 252.0
        rates = np.empty((num_sim, N + 1), dtype=float)
        rates[:, 0] = r0
        for i in range(N):
            z = rng.standard_normal(num_sim)
            rt = rates[:, i]
            if model.lower() == "cir":
                drift = kappa * (theta - np.maximum(rt, 0.0)) * dt
                diff = eta * np.sqrt(np.maximum(rt, 1e-6)) * np.sqrt(dt) * z
                rates[:, i + 1] = np.maximum(rt + drift + diff, 1e-6)
            else:
                rates[:, i + 1] = rt + kappa * (theta - rt) * dt + eta * np.sqrt(dt) * z
        t = np.linspace(0.0, N / 252.0, N + 1)
        return rates, t

    def simulate_jump_diffusion(
        self,
        N: int,
        num_sim: int,
        seed: int,
        jump_lambda: float,
        jump_mean: float,
        jump_vol: float,
        model: str = "merton",
        drift: float | None = None,
        sigma: float | None = None,
        rate_paths: np.ndarray | None = None,
    ) -> dict:
        rng = np.random.default_rng(seed)
        sigma = float(self.eng.sigma if sigma is None else sigma)
        drift = float(self.eng.mu if drift is None else drift)
        dt = (N / 252.0) / N
        z_diff = rng.standard_normal((num_sim, N))
        z_jump = rng.standard_normal((num_sim, N))
        counts = rng.poisson(max(jump_lambda, 0.0) * dt, size=(num_sim, N))
        jump_log = counts * jump_mean + np.sqrt(np.maximum(counts, 0.0)) * jump_vol * z_jump
        if model.lower() == "jump_diffusion":
            jump_rel = np.clip(jump_log, -0.9, 3.0)
            gross_jump = np.maximum(1.0 + jump_rel, 0.05)
            drift_grid = np.full((num_sim, N), drift)
            if rate_paths is not None:
                drift_grid = rate_paths[:, :-1]
            inc = np.exp((drift_grid - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_diff) * gross_jump
            paths = np.concatenate(
                [np.full((num_sim, 1), self.eng.S0), self.eng.S0 * np.cumprod(inc, axis=1)],
                axis=1,
            )
        else:
            compensator = jump_lambda * (np.exp(jump_mean + 0.5 * jump_vol**2) - 1.0)
            drift_grid = np.full((num_sim, N), drift - compensator)
            if rate_paths is not None:
                drift_grid = rate_paths[:, :-1] - compensator
            inc = (drift_grid - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_diff + jump_log
            paths = self.eng.S0 * np.exp(
                np.concatenate([np.zeros((num_sim, 1)), np.cumsum(inc, axis=1)], axis=1)
            )
        jump_mask = counts > 0
        arrivals = np.cumsum(counts, axis=1)
        t = np.linspace(0.0, N / 252.0, N + 1)
        return {
            "paths": paths,
            "time": t,
            "jump_counts": counts,
            "jump_mask": jump_mask,
            "jump_arrivals": arrivals,
            "terminal": paths[:, -1],
            "jump_frequency": float(np.mean(counts.sum(axis=1) / max(N / 252.0, 1e-9))),
            "mean_jumps_per_path": float(np.mean(counts.sum(axis=1))),
        }

    @staticmethod
    def _brownian_bridge_path(z_increments: np.ndarray, bridge_noise: np.ndarray) -> np.ndarray:
        n = len(z_increments)
        dt = 1.0 / n
        w = np.full(n + 1, np.nan, dtype=float)
        w[0] = 0.0
        w[-1] = float(np.sum(z_increments) * np.sqrt(dt))
        noise_idx = 0

        def fill(left: int, right: int):
            nonlocal noise_idx
            if right - left <= 1:
                return
            mid = (left + right) // 2
            tl = left * dt
            tr = right * dt
            tm = mid * dt
            mean = ((tr - tm) * w[left] + (tm - tl) * w[right]) / max(tr - tl, 1e-12)
            var = max((tm - tl) * (tr - tm) / max(tr - tl, 1e-12), 0.0)
            w[mid] = mean + np.sqrt(var) * bridge_noise[noise_idx]
            noise_idx += 1
            fill(left, mid)
            fill(mid, right)

        fill(0, n)
        for idx in range(1, n):
            if not np.isfinite(w[idx]):
                w[idx] = np.interp(idx * dt, [0.0, 1.0], [w[0], w[-1]])
        return w

    def _barrier_bridge_survival(self, paths: np.ndarray, barrier: float, sigma: float, dt: float) -> np.ndarray:
        left = paths[:, :-1]
        right = paths[:, 1:]
        crossed = (left >= barrier) | (right >= barrier)
        safe_left = np.clip(barrier / np.maximum(left, 1e-9), 1.0, None)
        safe_right = np.clip(barrier / np.maximum(right, 1e-9), 1.0, None)
        exponent = -2.0 * np.log(safe_left) * np.log(safe_right) / max(sigma * sigma * dt, 1e-12)
        hit_prob = np.exp(np.clip(exponent, -50, 20))
        hit_prob = np.where(crossed, 1.0, np.clip(hit_prob, 0.0, 1.0))
        survival = np.prod(1.0 - hit_prob, axis=1)
        return np.clip(survival, 0.0, 1.0)

    def _dynamic_multi_asset_paths(
        self,
        N: int,
        num_sim: int,
        seed: int,
        base_corr: np.ndarray,
        stress_bump: np.ndarray,
    ) -> dict:
        rng = np.random.default_rng(seed)
        d = base_corr.shape[0]
        S0_vec = np.array([self.eng.S0, self.eng.S0 * 0.96, self.eng.S0 * 1.04], dtype=float)
        mu_vec = np.array([self.eng.mu, self.eng.mu * 0.92, self.eng.mu * 1.05], dtype=float)
        sigma_vec = np.array([self.eng.sigma, self.eng.sigma * 1.08, self.eng.sigma * 0.9], dtype=float)
        dt = (N / 252.0) / N
        t = np.linspace(0.0, N / 252.0, N + 1)
        stress_curve = 0.55 * np.exp(-((np.arange(N) - 0.68 * N) / max(0.16 * N, 1.0)) ** 2) - 0.12
        stress_curve += 0.16 * np.sin(np.linspace(0, 2 * np.pi, N))
        stress_curve = np.clip(stress_curve, -0.25, 0.65)
        corr_track = np.empty((N, d, d), dtype=float)
        dyn_paths = np.empty((num_sim, N + 1, d), dtype=float)
        dyn_paths[:, 0, :] = S0_vec
        static_paths = np.empty_like(dyn_paths)
        static_paths[:, 0, :] = S0_vec
        static_chol = np.linalg.cholesky(self._nearest_psd_corr(base_corr))
        for i in range(N):
            corr_t = self._nearest_psd_corr(base_corr + stress_curve[i] * stress_bump)
            corr_track[i] = corr_t
            chol_t = np.linalg.cholesky(corr_t)
            z_dyn = rng.standard_normal((num_sim, d)) @ chol_t.T
            z_static = rng.standard_normal((num_sim, d)) @ static_chol.T
            drift = (mu_vec - 0.5 * sigma_vec**2) * dt
            dyn_paths[:, i + 1, :] = dyn_paths[:, i, :] * np.exp(drift + sigma_vec * np.sqrt(dt) * z_dyn)
            static_paths[:, i + 1, :] = static_paths[:, i, :] * np.exp(drift + sigma_vec * np.sqrt(dt) * z_static)
        basket_dyn = dyn_paths[:, -1, :].mean(axis=1)
        basket_static = static_paths[:, -1, :].mean(axis=1)
        return {
            "time": t.tolist(),
            "stress_curve": stress_curve.tolist(),
            "corr_12": corr_track[:, 0, 1].tolist(),
            "corr_13": corr_track[:, 0, 2].tolist(),
            "corr_23": corr_track[:, 1, 2].tolist(),
            "basket_dynamic": self._summary(basket_dyn),
            "basket_static": self._summary(basket_static),
            "basket_dynamic_samples": basket_dyn[:800].tolist(),
            "basket_static_samples": basket_static[:800].tolist(),
        }

    def _adaptive_simulation(self, N: int, num_sim: int, seed: int, jump_lambda: float) -> dict:
        dt = (N / 252.0) / N
        barrier = 1.18 * self.eng.S0
        K = self.eng.S0
        r = 0.05

        def run(mode: str, rng_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            rng = np.random.default_rng(rng_seed)
            paths = np.empty((num_sim, N + 1), dtype=float)
            paths[:, 0] = self.eng.S0
            steps = np.ones((num_sim, N), dtype=int)
            counts_book = np.zeros((num_sim, N), dtype=int)
            for i in range(N):
                counts = rng.poisson(jump_lambda * dt, size=num_sim)
                stress = np.abs(rng.standard_normal(num_sim)) + 1.8 * counts
                if mode == "adaptive":
                    sub = np.clip(1 + (stress > 0.8).astype(int) + (stress > 1.5).astype(int) + 2 * counts, 1, 6)
                elif mode == "reference":
                    sub = np.full(num_sim, 6, dtype=int)
                else:
                    sub = np.ones(num_sim, dtype=int)
                steps[:, i] = sub
                counts_book[:, i] = counts
                next_vals = np.empty(num_sim, dtype=float)
                for j in range(num_sim):
                    price = paths[j, i]
                    m = int(sub[j])
                    sub_dt = dt / m
                    jump_count = counts[j]
                    for k in range(m):
                        z = rng.standard_normal()
                        jump_now = jump_count > 0 and k == m // 2
                        jump_term = (-0.04 + 0.18 * rng.standard_normal()) if jump_now else 0.0
                        price *= np.exp((self.eng.mu - 0.5 * self.eng.sigma**2) * sub_dt + self.eng.sigma * np.sqrt(sub_dt) * z + jump_term)
                    next_vals[j] = price
                paths[:, i + 1] = next_vals
            return paths, steps, counts_book

        start = perf_counter()
        fixed_paths, fixed_steps, fixed_counts = run("fixed", seed + 1)
        fixed_runtime = perf_counter() - start
        start = perf_counter()
        adaptive_paths, adaptive_steps, adaptive_counts = run("adaptive", seed + 2)
        adaptive_runtime = perf_counter() - start
        start = perf_counter()
        ref_paths, _, _ = run("reference", seed + 3)
        ref_runtime = perf_counter() - start

        def barrier_price(paths: np.ndarray) -> float:
            alive = paths.max(axis=1) < barrier
            payoff = np.maximum(paths[:, -1] - K, 0.0) * alive.astype(float)
            return float(np.exp(-r * N / 252.0) * payoff.mean())

        price_ref = barrier_price(ref_paths)
        price_fixed = barrier_price(fixed_paths)
        price_adaptive = barrier_price(adaptive_paths)
        sample_idx = int(np.argmax(adaptive_steps.sum(axis=1)))
        return {
            "fixed": {
                "runtime_ms": 1000.0 * fixed_runtime,
                "barrier_price": price_fixed,
                "abs_error_vs_ref": abs(price_fixed - price_ref),
            },
            "adaptive": {
                "runtime_ms": 1000.0 * adaptive_runtime,
                "barrier_price": price_adaptive,
                "abs_error_vs_ref": abs(price_adaptive - price_ref),
            },
            "reference": {
                "runtime_ms": 1000.0 * ref_runtime,
                "barrier_price": price_ref,
            },
            "sample_step_schedule": adaptive_steps[sample_idx].tolist(),
            "sample_jump_counts": adaptive_counts[sample_idx].tolist(),
            "sample_path": adaptive_paths[sample_idx].tolist(),
            "sample_fixed_path": fixed_paths[sample_idx].tolist(),
        }

    def _importance_sampling_experiment(self, N: int, num_sim: int, seed: int) -> dict:
        T = N / 252.0
        dt = T / N
        K = 1.35 * self.eng.S0
        theta = 0.38
        r = 0.05
        rng_std = np.random.default_rng(seed + 1)
        z_std = rng_std.standard_normal((num_sim, N))
        paths_std = self.eng.S0 * np.exp(
            np.concatenate(
                [np.zeros((num_sim, 1)), np.cumsum((r - 0.5 * self.eng.sigma**2) * dt + self.eng.sigma * np.sqrt(dt) * z_std, axis=1)],
                axis=1,
            )
        )
        pv_std = np.exp(-r * T) * np.maximum(paths_std[:, -1] - K, 0.0)

        rng_is = np.random.default_rng(seed + 2)
        y = rng_is.standard_normal((num_sim, N)) + theta
        weights = np.exp(-theta * np.sum(y, axis=1) + 0.5 * N * theta * theta)
        paths_is = self.eng.S0 * np.exp(
            np.concatenate(
                [np.zeros((num_sim, 1)), np.cumsum((r - 0.5 * self.eng.sigma**2) * dt + self.eng.sigma * np.sqrt(dt) * y, axis=1)],
                axis=1,
            )
        )
        pv_is = np.exp(-r * T) * np.maximum(paths_is[:, -1] - K, 0.0) * weights

        running_n = np.arange(1, num_sim + 1, dtype=float)
        run_std = np.cumsum(pv_std) / running_n
        run_is = np.cumsum(pv_is) / running_n
        hit_std = float(np.mean(paths_std[:, -1] > K))
        hit_is = float(np.mean(paths_is[:, -1] > K))
        ess = float((weights.sum() ** 2) / (np.sum(weights * weights) + 1e-12))
        return {
            "strike": float(K),
            "theta": theta,
            "standard_price": float(np.mean(pv_std)),
            "is_price": float(np.mean(pv_is)),
            "standard_stderr": float(np.std(pv_std, ddof=1) / np.sqrt(num_sim)),
            "is_stderr": float(np.std(pv_is, ddof=1) / np.sqrt(num_sim)),
            "variance_ratio": float((np.var(pv_is, ddof=1) + 1e-12) / (np.var(pv_std, ddof=1) + 1e-12)),
            "standard_hit_ratio": hit_std,
            "is_hit_ratio": hit_is,
            "effective_sample_size": ess,
            "running_standard": run_std[:1200].tolist(),
            "running_is": run_is[:1200].tolist(),
        }

    def build_report(self, N: int, num_sim: int, seed: int) -> dict:
        horizon = min(max(42, N), 126)
        sample_paths = min(max(120, num_sim // 2), 320)
        jump_grid_lambda = [0.0, 0.35, 0.8]
        jump_grid_mean = [-0.06, 0.0, 0.06]
        jump_grid_vol = [0.12, 0.28]
        jump_scenarios = []
        scenario_id = 0
        for model in ["jump_diffusion", "merton"]:
            for lam in jump_grid_lambda:
                for jmu in jump_grid_mean:
                    for jvol in jump_grid_vol:
                        sim = self.simulate_jump_diffusion(
                            N=horizon,
                            num_sim=sample_paths,
                            seed=seed + 300 + scenario_id,
                            jump_lambda=lam,
                            jump_mean=jmu,
                            jump_vol=jvol,
                            model=model,
                        )
                        paths = sim["paths"][:6]
                        jump_points = []
                        for idx in range(min(3, len(paths))):
                            hit_idx = np.where(sim["jump_mask"][idx])[0]
                            jump_points.append({
                                "x": sim["time"][hit_idx + 1].tolist(),
                                "y": paths[idx, hit_idx + 1].tolist(),
                            })
                        jump_scenarios.append({
                            "id": scenario_id,
                            "model": model,
                            "lambda": lam,
                            "jump_mean": jmu,
                            "jump_vol": jvol,
                            "time": sim["time"].tolist(),
                            "paths": [row.tolist() for row in paths],
                            "jump_points": jump_points,
                            "terminal_summary": self._summary(sim["terminal"]),
                            "jump_frequency": sim["jump_frequency"],
                            "mean_jumps_per_path": sim["mean_jumps_per_path"],
                        })
                        scenario_id += 1

        rate_payload = {}
        for idx, model in enumerate(["vasicek", "cir"]):
            rates, t = self._simulate_rate_paths(model=model, N=horizon, num_sim=360, seed=seed + 700 + idx)
            coupled = self.simulate_jump_diffusion(
                N=horizon,
                num_sim=360,
                seed=seed + 710 + idx,
                jump_lambda=0.25,
                jump_mean=-0.03,
                jump_vol=0.15,
                model="merton",
                rate_paths=rates,
            )
            rate_payload[model] = {
                "time": t.tolist(),
                "rate_p10": np.percentile(rates, 10, axis=0).tolist(),
                "rate_p50": np.percentile(rates, 50, axis=0).tolist(),
                "rate_p90": np.percentile(rates, 90, axis=0).tolist(),
                "stock_p10": np.percentile(coupled["paths"], 10, axis=0).tolist(),
                "stock_p50": np.percentile(coupled["paths"], 50, axis=0).tolist(),
                "stock_p90": np.percentile(coupled["paths"], 90, axis=0).tolist(),
            }

        base_corr = np.array([[1.0, 0.55, 0.35], [0.55, 1.0, 0.62], [0.35, 0.62, 1.0]], dtype=float)
        stress_bump = np.array([[0.0, 0.32, -0.10], [0.32, 0.0, 0.18], [-0.10, 0.18, 0.0]], dtype=float)
        dynamic_cov = self._dynamic_multi_asset_paths(
            N=horizon,
            num_sim=max(500, min(num_sim, 1200)),
            seed=seed + 820,
            base_corr=base_corr,
            stress_bump=stress_bump,
        )

        bridge_n = horizon
        rng = np.random.default_rng(seed + 900)
        z = rng.standard_normal(bridge_n)
        bridge_noise = rng.standard_normal(bridge_n - 1)
        t_unit = np.linspace(0.0, horizon / 252.0, bridge_n + 1)
        standard_w = np.concatenate([[0.0], np.cumsum(np.sqrt(1.0 / bridge_n) * z)])
        bridge_w = self._brownian_bridge_path(z, bridge_noise)
        standard_path = self.eng.S0 * np.exp((0.05 - 0.5 * self.eng.sigma**2) * t_unit + self.eng.sigma * np.sqrt(horizon / 252.0) * standard_w)
        bridge_path = self.eng.S0 * np.exp((0.05 - 0.5 * self.eng.sigma**2) * t_unit + self.eng.sigma * np.sqrt(horizon / 252.0) * bridge_w)
        barrier = 1.2 * self.eng.S0
        disc_paths, _ = self.eng.simulate_risk_neutral(r=0.05, N=horizon, num_sim=max(800, min(num_sim, 1800)), seed=seed + 910)
        payoff = np.exp(-0.05 * horizon / 252.0) * np.maximum(disc_paths[:, -1] - self.eng.S0, 0.0)
        discrete_alive = (disc_paths.max(axis=1) < barrier).astype(float)
        bridge_survival = self._barrier_bridge_survival(disc_paths, barrier=barrier, sigma=self.eng.sigma, dt=(horizon / 252.0) / horizon)
        bridge_payload = {
            "time": t_unit.tolist(),
            "standard_path": standard_path.tolist(),
            "bridge_path": bridge_path.tolist(),
            "barrier": float(barrier),
            "discrete_price": float(np.mean(payoff * discrete_alive)),
            "bridge_price": float(np.mean(payoff * bridge_survival)),
            "discrete_active_ratio": float(np.mean(discrete_alive)),
            "bridge_active_ratio": float(np.mean(bridge_survival)),
        }

        adaptive = self._adaptive_simulation(
            N=min(max(30, N // 2), 84),
            num_sim=min(max(180, num_sim // 2), 320),
            seed=seed + 1000,
            jump_lambda=0.85,
        )
        importance = self._importance_sampling_experiment(
            N=horizon,
            num_sim=max(1600, min(num_sim * 2, 4000)),
            seed=seed + 1100,
        )

        return {
            "horizon_days": int(horizon),
            "jump_scenarios": jump_scenarios,
            "rate_processes": rate_payload,
            "dynamic_covariance": dynamic_cov,
            "brownian_bridge": bridge_payload,
            "adaptive_timestep": adaptive,
            "importance_sampling": importance,
        }


class MonteCarloDiagnosticsLab:
    """Numerical experimentation layer for understanding Monte Carlo accuracy and failure modes."""

    def __init__(self, eng: GBMEngine):
        self.eng = eng

    def _paths_custom(self, Z: np.ndarray, r: float, sigma: float, S0: float, T: float) -> np.ndarray:
        return self.eng._paths_from_normals(Z=Z, drift=r, sigma=sigma, S0=S0, T=T)

    @staticmethod
    def _pv_call(paths: np.ndarray, K: float, r: float, T: float) -> np.ndarray:
        return np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0)

    @staticmethod
    def _pv_put(paths: np.ndarray, K: float, r: float, T: float) -> np.ndarray:
        return np.exp(-r * T) * np.maximum(K - paths[:, -1], 0.0)

    @staticmethod
    def _pv_barrier_up_out(paths: np.ndarray, K: float, barrier: float, r: float, T: float) -> np.ndarray:
        alive = paths.max(axis=1) < barrier
        return np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0) * alive.astype(float)

    @staticmethod
    def _stderr(samples: np.ndarray) -> float:
        if len(samples) <= 1:
            return 0.0
        return float(np.std(samples, ddof=1) / np.sqrt(len(samples)))

    def _call_price_estimate(
        self,
        K: float,
        r: float,
        N: int,
        num_sim: int,
        sigma: float,
        seed: int,
    ) -> dict:
        rng = np.random.default_rng(seed)
        T = N / 252.0
        Z = self.eng._generate_normals(num_sim=num_sim, N=N, rng=rng, random_method="pseudo", antithetic=False)
        paths = self._paths_custom(Z=Z, r=r, sigma=sigma, S0=self.eng.S0, T=T)
        pv = self._pv_call(paths=paths, K=K, r=r, T=T)
        return {"price": float(np.mean(pv)), "stderr": self._stderr(pv), "samples": pv, "paths": paths}

    def _convergence_heatmaps(self, seed: int) -> dict:
        path_grid = [200, 500, 1000, 2000]
        step_grid = [21, 63, 126, 252]
        vol_mults = [0.7, 1.0, 1.4]
        maturities = [21, 63, 126]
        K = self.eng.S0
        r = 0.05
        scenarios = []
        sid = 0
        for vol_mult in vol_mults:
            sigma = max(self.eng.sigma * vol_mult, 1e-6)
            for maturity in maturities:
                bs = self.eng.black_scholes_price(K=K, T=maturity / 252.0, r=r, option_type="call")
                zmat = []
                semat = []
                for n_paths in path_grid:
                    row_err = []
                    row_se = []
                    for steps in step_grid:
                        est = self._call_price_estimate(
                            K=K, r=r, N=steps, num_sim=n_paths, sigma=sigma, seed=seed + 50 * sid + n_paths + steps
                        )
                        row_err.append(float(est["price"] - bs))
                        row_se.append(float(est["stderr"]))
                    zmat.append(row_err)
                    semat.append(row_se)
                scenarios.append(
                    {
                        "id": sid,
                        "vol_mult": vol_mult,
                        "maturity": maturity,
                        "bs": float(bs),
                        "errors": zmat,
                        "stderrs": semat,
                    }
                )
                sid += 1
        return {"path_grid": path_grid, "step_grid": step_grid, "scenarios": scenarios}

    def _bias_analysis(self, seed: int) -> dict:
        K = self.eng.S0
        r = 0.05
        N = 126
        T = N / 252.0
        bs = self.eng.black_scholes_price(K=K, T=T, r=r, option_type="call")
        run_sizes = [300, 1200]
        profiles = []
        for idx, n_paths in enumerate(run_sizes):
            estimates = []
            for rep in range(36):
                est = self._call_price_estimate(
                    K=K, r=r, N=N, num_sim=n_paths, sigma=self.eng.sigma, seed=seed + 300 + idx * 100 + rep
                )
                estimates.append(est["price"])
            arr = np.asarray(estimates, dtype=float)
            running = np.cumsum(arr) / np.arange(1, len(arr) + 1, dtype=float)
            profiles.append(
                {
                    "num_sim": n_paths,
                    "estimates": arr.tolist(),
                    "running_mean": running.tolist(),
                    "bias": (arr - bs).tolist(),
                    "mean_bias": float(np.mean(arr - bs)),
                    "rmse": float(np.sqrt(np.mean((arr - bs) ** 2))),
                    "std_estimate": float(np.std(arr, ddof=1)),
                }
            )
        return {"bs": float(bs), "profiles": profiles}

    def _discretization_analysis(self, seed: int) -> dict:
        K = self.eng.S0
        barrier = 1.08 * self.eng.S0
        r = 0.05
        fine_steps = 504
        coarse_steps = [21, 63, 126, 252, 504]
        num_sim = 900
        T = 126 / 252.0
        rng = np.random.default_rng(seed + 600)
        Z_fine = rng.standard_normal((num_sim, fine_steps))
        reference = None
        rows = []
        ref_paths = None
        for steps in coarse_steps:
            block = fine_steps // steps
            Z = Z_fine.reshape(num_sim, steps, block).sum(axis=2) / np.sqrt(block)
            paths = self._paths_custom(Z=Z, r=r, sigma=self.eng.sigma, S0=self.eng.S0, T=T)
            pv_barrier = self._pv_barrier_up_out(paths, K=K, barrier=barrier, r=r, T=T)
            pv_asian = np.exp(-r * T) * np.maximum(paths[:, 1:].mean(axis=1) - K, 0.0)
            price_barrier = float(np.mean(pv_barrier))
            price_asian = float(np.mean(pv_asian))
            active_ratio = float(np.mean(paths.max(axis=1) < barrier))
            if steps == fine_steps:
                reference = {"barrier": price_barrier, "asian": price_asian}
                ref_paths = paths
            rows.append(
                {
                    "steps": steps,
                    "barrier_price": price_barrier,
                    "asian_price": price_asian,
                    "active_ratio": active_ratio,
                    "paths": paths,
                }
            )
        payload = []
        assert reference is not None and ref_paths is not None
        for row in rows:
            steps = row["steps"]
            comp = ref_paths[:, :: fine_steps // steps]
            rmse = float(np.sqrt(np.mean((row["paths"] - comp) ** 2)))
            max_dev = float(np.mean(np.max(np.abs(row["paths"] - comp), axis=1)))
            payload.append(
                {
                    "steps": steps,
                    "barrier_price": row["barrier_price"],
                    "asian_price": row["asian_price"],
                    "barrier_divergence": float(row["barrier_price"] - reference["barrier"]),
                    "asian_divergence": float(row["asian_price"] - reference["asian"]),
                    "active_ratio": row["active_ratio"],
                    "path_rmse": rmse,
                    "path_max_dev": max_dev,
                }
            )
        return {"barrier": float(barrier), "rows": payload}

    def _variance_decomposition(self, seed: int) -> dict:
        K = self.eng.S0
        r = 0.05
        N = 126
        T = N / 252.0
        num_sim = 1000
        base_rng = np.random.default_rng(seed + 900)
        base_Z = self.eng._generate_normals(num_sim=num_sim, N=N, rng=base_rng, random_method="pseudo", antithetic=False)

        vol_prices = []
        for mult in [0.75, 1.0, 1.35]:
            paths = self._paths_custom(base_Z, r=r, sigma=max(self.eng.sigma * mult, 1e-6), S0=self.eng.S0, T=T)
            vol_prices.append(float(np.mean(self._pv_call(paths, K=K, r=r, T=T))))

        step_prices = []
        for steps in [21, 63, 126, 252]:
            rng = np.random.default_rng(seed + 910 + steps)
            Z = self.eng._generate_normals(num_sim=num_sim, N=steps, rng=rng, random_method="pseudo", antithetic=False)
            paths = self._paths_custom(Z, r=r, sigma=self.eng.sigma, S0=self.eng.S0, T=T)
            step_prices.append(float(np.mean(self._pv_barrier_up_out(paths, K=K, barrier=1.08 * self.eng.S0, r=r, T=T))))

        sampling_prices = []
        for rep in range(20):
            est = self._call_price_estimate(K=K, r=r, N=N, num_sim=400, sigma=self.eng.sigma, seed=seed + 940 + rep)
            sampling_prices.append(est["price"])

        jumps = AdvancedMonteCarloExtensions(self.eng)
        jump_prices = []
        for lam in [0.0, 0.35, 0.8]:
            sim = jumps.simulate_jump_diffusion(
                N=N,
                num_sim=900,
                seed=seed + 980 + int(100 * lam),
                jump_lambda=lam,
                jump_mean=-0.04,
                jump_vol=0.18,
                model="merton",
                drift=r,
                sigma=self.eng.sigma,
            )
            jump_prices.append(float(np.exp(-r * T) * np.maximum(sim["paths"][:, -1] - K, 0.0).mean()))

        base_paths = self._paths_custom(base_Z, r=r, sigma=self.eng.sigma, S0=self.eng.S0, T=T)
        linear = np.exp(-r * T) * base_paths[:, -1]
        vanilla = self._pv_call(base_paths, K=K, r=r, T=T)
        barrier = self._pv_barrier_up_out(base_paths, K=K, barrier=1.08 * self.eng.S0, r=r, T=T)

        raw = {
            "volatility": float(np.var(vol_prices, ddof=1)),
            "timestep": float(np.var(step_prices, ddof=1)),
            "sampling_noise": float(np.var(sampling_prices, ddof=1)),
            "jump_events": float(np.var(jump_prices, ddof=1)),
            "payoff_nonlinearity": float(max(np.var(vanilla, ddof=1) - np.var(linear, ddof=1), 0.0) + np.var(barrier, ddof=1)),
        }
        total = sum(raw.values()) + 1e-12
        shares = {k: 100.0 * v / total for k, v in raw.items()}
        return {"raw": raw, "shares": shares}

    def _failure_region_maps(self, seed: int) -> dict:
        r = 0.05
        num_sim = 500
        maturity_grid = [21, 63, 126, 252]
        strike_mults = [0.9, 1.0, 1.1, 1.25, 1.4, 1.6]
        sigma_grid = [0.18, 0.28, 0.45, 0.70]
        path_grid = [150, 300, 700, 1500]
        barrier_offsets = [1.01, 1.03, 1.06, 1.10, 1.16]

        otm_map = []
        for sigma in sigma_grid:
            row = []
            for k_mult in strike_mults:
                K = self.eng.S0 * k_mult
                est = self._call_price_estimate(K=K, r=r, N=126, num_sim=num_sim, sigma=sigma, seed=seed + 1200 + int(100 * sigma) + int(100 * k_mult))
                price = max(est["price"], 1e-8)
                payoff_zero = float(np.mean(est["samples"] <= 1e-10))
                rel_noise = est["stderr"] / price
                score = min(100.0, 35 * rel_noise + 55 * payoff_zero + 20 * max(k_mult - 1.0, 0.0) + 12 * max(sigma - 0.28, 0.0))
                row.append(float(score))
            otm_map.append(row)

        barrier_map = []
        for offset in barrier_offsets:
            row = []
            barrier = self.eng.S0 * offset
            for n_paths in path_grid:
                res = self.eng.price_barrier_option_mc(
                    K=self.eng.S0,
                    barrier=barrier,
                    r=r,
                    N=126,
                    num_sim=n_paths,
                    option_type="call",
                    barrier_type="up-and-out",
                    seed=seed + 1300 + int(100 * offset) + n_paths,
                    random_method="pseudo",
                    antithetic=False,
                )
                price = max(float(res["price"]), 1e-8)
                rel_noise = float(res["stderr"]) / price
                fragility = abs(float(res["active_ratio"]) - 0.5)
                score = min(100.0, 45 * rel_noise + 30 * (1.0 - min(path_grid) / max(n_paths, 1)) + 35 * (1.0 - abs(fragility)))
                row.append(float(score))
            barrier_map.append(row)

        ineff = []
        for maturity in maturity_grid:
            row = []
            for n_paths in path_grid:
                est = self._call_price_estimate(K=1.35 * self.eng.S0, r=r, N=maturity, num_sim=n_paths, sigma=self.eng.sigma, seed=seed + 1400 + maturity + n_paths)
                rel_noise = est["stderr"] / max(est["price"], 1e-6)
                score = min(100.0, 28 * rel_noise + 18 * maturity / 252.0 + 22 * (path_grid[0] / n_paths))
                row.append(float(score))
            ineff.append(row)

        return {
            "strike_mults": strike_mults,
            "sigma_grid": sigma_grid,
            "otm_vol_score": otm_map,
            "barrier_offsets": barrier_offsets,
            "path_grid": path_grid,
            "barrier_score": barrier_map,
            "maturity_grid": maturity_grid,
            "inefficiency_score": ineff,
        }

    def _path_instability(self, seed: int) -> dict:
        r = 0.05
        N = 126
        T = N / 252.0
        K = self.eng.S0
        num_sim = 400
        rng = np.random.default_rng(seed + 1600)
        Z = self.eng._generate_normals(num_sim=num_sim, N=N, rng=rng, random_method="pseudo", antithetic=False)
        base_paths = self._paths_custom(Z, r=r, sigma=self.eng.sigma, S0=self.eng.S0, T=T)
        sigma_paths = self._paths_custom(Z, r=r, sigma=self.eng.sigma * 1.01, S0=self.eng.S0, T=T)
        rng2 = np.random.default_rng(seed + 1601)
        Z2 = self.eng._generate_normals(num_sim=num_sim, N=N, rng=rng2, random_method="pseudo", antithetic=False)
        seed_paths = self._paths_custom(Z2, r=r, sigma=self.eng.sigma, S0=self.eng.S0, T=T)
        base_payoff = self._pv_call(base_paths, K=K, r=r, T=T)
        sigma_payoff = self._pv_call(sigma_paths, K=K, r=r, T=T)
        seed_payoff = self._pv_call(seed_paths, K=K, r=r, T=T)
        idx = np.argsort(np.abs(sigma_payoff - base_payoff))[-6:]
        amplification = np.abs(sigma_payoff - base_payoff) / max(np.mean(base_payoff), 1e-9)
        return {
            "time": np.linspace(0.0, T, N + 1).tolist(),
            "sample_idx": idx.tolist(),
            "base_paths": [base_paths[i].tolist() for i in idx],
            "sigma_paths": [sigma_paths[i].tolist() for i in idx],
            "seed_paths": [seed_paths[i].tolist() for i in idx],
            "payoff_delta_sigma": (sigma_payoff - base_payoff)[:250].tolist(),
            "payoff_delta_seed": (seed_payoff - base_payoff)[:250].tolist(),
            "mean_abs_sigma_shift": float(np.mean(np.abs(sigma_payoff - base_payoff))),
            "mean_abs_seed_shift": float(np.mean(np.abs(seed_payoff - base_payoff))),
            "amplification_mean": float(np.mean(amplification)),
            "amplification_p95": float(np.percentile(amplification, 95)),
        }

    def build_report(self, seed: int) -> dict:
        return {
            "convergence_heatmaps": self._convergence_heatmaps(seed=seed),
            "bias_analysis": self._bias_analysis(seed=seed),
            "discretization": self._discretization_analysis(seed=seed),
            "variance_decomposition": self._variance_decomposition(seed=seed),
            "failure_regions": self._failure_region_maps(seed=seed),
            "path_instability": self._path_instability(seed=seed),
        }



############################################################################################################################################################################################


# PLOTLY DASHBOARD  - comprehensive, interactive, 6-panel
class PlotlyDashboard:
    def __init__(self, eng: GBMEngine):
        self.eng = eng

    def _plotly_theme(self):
        return dict(
            template="plotly_dark",
            paper_bgcolor=C["bg"],
            plot_bgcolor=C["bg"],
            font=dict(family="Consolas, Menlo, monospace", color=C["text"], size=11),
            hoverlabel=dict(
                bgcolor="#ffffff",
                bordercolor=C["border"],
                font=dict(color="#111111", size=11, family="Consolas, Menlo, monospace"),
                namelength=-1,
            ),
            legend=dict(
                bgcolor=C["bg2"], bordercolor=C["border"],
                borderwidth=1, font=dict(size=10)
            ),
        )

    def _setup_defaults(self) -> tuple[dict, dict]:
        lr = self.eng.log_ret.dropna()
        dist_defaults = {
            "distribution": "normal",
            "mu": float(lr.mean()) if len(lr) else 0.0,
            "sigma": max(float(lr.std()), 1e-4) if len(lr) else 0.2,
            "nu": 7.0,
        }
        corr_defaults = {
            "labels": [self.eng.ticker, f"{self.eng.ticker} Peer", "Market"],
            "matrix": [
                [1.0, 0.55, 0.35],
                [0.55, 1.0, 0.65],
                [0.35, 0.65, 1.0],
            ],
        }
        return dist_defaults, corr_defaults

    def render_setup_lab(self, show=False):
        dist_defaults, corr_defaults = self._setup_defaults()
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Monte Carlo Setup Lab - {self.eng.ticker}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: {C["bg"]};
      --bg2: {C["bg2"]};
      --bg3: {C["bg3"]};
      --border: {C["border"]};
      --muted: {C["muted"]};
      --text: {C["text"]};
      --title: {C["title"]};
      --blue: {C["blue"]};
      --green: {C["green"]};
      --red: {C["red"]};
      --amber: {C["amber"]};
      --purple: {C["purple"]};
      --cyan: {C["cyan"]};
      --pink: {C["pink"]};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Consolas, Menlo, monospace;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 0%, rgba(88,166,255,0.12), transparent 30%),
        linear-gradient(180deg, #0f141d 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px 22px 40px;
    }}
    .hero {{
      margin-bottom: 22px;
      padding: 22px 24px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(22,27,34,0.96), rgba(13,17,23,0.96));
      box-shadow: 0 24px 64px rgba(0,0,0,0.24);
    }}
    .eyebrow {{
      display: inline-block;
      margin-bottom: 10px;
      color: var(--cyan);
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      color: var(--title);
      font-size: 30px;
    }}
    .hero p {{
      margin: 0;
      max-width: 860px;
      line-height: 1.55;
      color: var(--muted);
      font-size: 13px;
    }}
    .hero-meta {{
      margin-top: 16px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .pill {{
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(28,33,40,0.92);
      color: var(--muted);
      font-size: 12px;
    }}
    .section {{
      margin-top: 18px;
      padding: 20px;
      border-radius: 20px;
      border: 1px solid var(--border);
      background: rgba(22,27,34,0.94);
      box-shadow: 0 20px 56px rgba(0,0,0,0.18);
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .section-head h2 {{
      margin: 0 0 6px;
      color: var(--title);
      font-size: 19px;
    }}
    .section-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      max-width: 760px;
    }}
    .distribution-layout {{
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 260px;
      gap: 18px;
      align-items: start;
    }}
    .side-card {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(13,17,23,0.76);
      padding: 16px;
    }}
    .side-card h3 {{
      margin: 0 0 12px;
      color: var(--title);
      font-size: 14px;
    }}
    .control {{
      margin-bottom: 16px;
    }}
    .control:last-child {{
      margin-bottom: 0;
    }}
    .control label {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--title);
    }}
    .control label span {{
      color: var(--cyan);
      font-weight: 700;
    }}
    select, input[type="range"], input[type="number"] {{
      width: 100%;
    }}
    select, input[type="number"] {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--bg3);
      color: var(--text);
      padding: 10px 12px;
      font-family: inherit;
    }}
    input[type="range"] {{
      accent-color: var(--blue);
    }}
    .plot-card {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(13,17,23,0.76);
      padding: 10px 10px 4px;
    }}
    #distributionPlot, #corrPlot {{
      min-height: 470px;
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 9px 0;
      border-bottom: 1px solid rgba(48,54,61,0.56);
      font-size: 12px;
    }}
    .stat-row:last-child {{
      border-bottom: 0;
    }}
    .stat-row .name {{
      color: var(--muted);
    }}
    .stat-row .val {{
      color: var(--cyan);
      text-align: right;
      font-weight: 700;
    }}
    .corr-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      align-items: start;
    }}
    .matrix-editor {{
      display: grid;
      gap: 10px;
    }}
    .editor-grid {{
      display: grid;
      gap: 8px;
      width: 100%;
    }}
    .editor-row {{
      display: grid;
      gap: 8px;
      align-items: center;
      width: 100%;
    }}
    .axis-label {{
      color: var(--muted);
      text-align: center;
      font-size: 12px;
      line-height: 1.15;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .editor-grid input {{
      text-align: center;
      font-weight: 700;
      padding: 10px 8px;
      min-width: 0;
    }}
    .editor-corner {{
      min-width: 0;
    }}
    .editor-grid .axis-label,
    .editor-grid input {{
      min-width: 0;
    }}
    .matrix-editor .axis-label {{
      font-size: 11px;
    }}
    .matrix-editor input {{
      padding: 8px 6px;
      font-size: 12px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 132px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--border);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 14px;
    }}
    .badge.valid {{
      background: rgba(63,185,80,0.16);
      border-color: rgba(63,185,80,0.32);
      color: #abf3b4;
    }}
    .badge.invalid {{
      background: rgba(248,81,73,0.16);
      border-color: rgba(248,81,73,0.32);
      color: #ffb2ac;
    }}
    .note {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }}
    @media (max-width: 1120px) {{
      .distribution-layout, .corr-layout {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Pre-Simulation Inputs</div>
      <h1>Layer 0 Setup Lab</h1>
      <p>Use this page to inspect the distribution assumptions and co-movement inputs before running Monte Carlo. The goal is quick sanity checking, not a crowded dashboard.</p>
      <div class="hero-meta">
        <div class="pill">Ticker: {self.eng.ticker}</div>
        <div class="pill">mu seed: {dist_defaults["mu"]:+.4f}</div>
        <div class="pill">sigma seed: {dist_defaults["sigma"]:.4f}</div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>01. Distribution Preview</h2>
          <p>Calibrated distribution from historical data showing PDF/CDF shape with moment statistics and standard deviation markers. This represents the actual distribution used in Monte Carlo simulation.</p>
        </div>
      </div>
      <div class="distribution-layout">
        <aside class="side-card">
          <h3>Calibrated Parameters</h3>
          <div class="control">
            <label>Distribution Type</label>
            <div style="padding: 10px 12px; background: var(--bg3); border-radius: 10px; color: var(--cyan); font-weight: bold;">Normal (Log Returns)</div>
          </div>
          <div class="control">
            <label>Drift (μ)</label>
            <div style="padding: 10px 12px; background: var(--bg3); border-radius: 10px; color: var(--text); font-weight: bold;">{dist_defaults["mu"]:+.4f}</div>
          </div>
          <div class="control">
            <label>Volatility (σ)</label>
            <div style="padding: 10px 12px; background: var(--bg3); border-radius: 10px; color: var(--text); font-weight: bold;">{dist_defaults["sigma"]:.4f}</div>
          </div>
          <div class="control">
            <label>Data Source</label>
            <div style="padding: 10px 12px; background: var(--bg3); border-radius: 10px; color: var(--muted); font-size: 11px;">Historical calibration from {self.eng.ticker}</div>
          </div>
        </aside>
        <div class="plot-card">
          <div id="distributionPlot"></div>
        </div>
        <aside class="side-card">
          <h3>Moments</h3>
          <div id="momentStats"></div>
        </aside>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>02. Correlation Matrix</h2>
          <p>Edit the off-diagonal entries, update the heatmap immediately, and keep an eye on the PSD validity badge before Cholesky-based simulation.</p>
        </div>
      </div>
      <div class="corr-layout">
        <div class="plot-card">
          <div id="corrPlot"></div>
        </div>
        <aside class="side-card matrix-editor">
          <div id="psdBadge" class="badge valid">Valid</div>
          <div id="psdMeta" class="note"></div>
          <h3>Matrix Editor</h3>
          <div id="corrGrid" class="editor-grid"></div>
          <div class="note">Blue means positive correlation, white is near zero, and red is negative correlation. Diagonal terms remain fixed at 1.00.</div>
        </aside>
      </div>
    </section>
  </div>

  <script>
    const theme = {json.dumps(C)};
    const distState = {json.dumps(dist_defaults)};
    const corrState = {json.dumps(corr_defaults)};

    const distType = document.getElementById("distType");
    const muSlider = document.getElementById("muSlider");
    const sigmaSlider = document.getElementById("sigmaSlider");
    const nuSlider = document.getElementById("nuSlider");
    const momentStats = document.getElementById("momentStats");
    const psdBadge = document.getElementById("psdBadge");
    const psdMeta = document.getElementById("psdMeta");
    const corrGrid = document.getElementById("corrGrid");

    function erfApprox(x) {{
      const sign = x >= 0 ? 1 : -1;
      const ax = Math.abs(x);
      const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
      const p = 0.3275911;
      const t = 1 / (1 + p * ax);
      const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
      return sign * y;
    }}

    function logGamma(z) {{
      const cof = [76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.001208650973866179,-0.000005395239384953];
      let x = z;
      let y = z;
      let tmp = x + 5.5;
      tmp -= (x + 0.5) * Math.log(tmp);
      let ser = 1.000000000190015;
      for (let j = 0; j < cof.length; j += 1) {{
        y += 1;
        ser += cof[j] / y;
      }}
      return Math.log(2.5066282746310005 * ser / x) - tmp;
    }}

    function normalPdf(x, mu, sigma) {{
      const z = (x - mu) / sigma;
      return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
    }}

    function normalCdf(x, mu, sigma) {{
      return 0.5 * (1 + erfApprox((x - mu) / (sigma * Math.sqrt(2))));
    }}

    function tPdf(x, mu, sigma, nu) {{
      const z = (x - mu) / sigma;
      const logCoeff = logGamma((nu + 1) / 2) - logGamma(nu / 2) - 0.5 * Math.log(nu * Math.PI) - Math.log(sigma);
      return Math.exp(logCoeff) * Math.pow(1 + (z * z) / nu, -(nu + 1) / 2);
    }}

    function simpsonIntegral(fn, a, b, n = 400) {{
      if (b <= a) return 0;
      const steps = n % 2 === 0 ? n : n + 1;
      const h = (b - a) / steps;
      let sum = fn(a) + fn(b);
      for (let i = 1; i < steps; i += 1) {{
        const x = a + i * h;
        sum += fn(x) * (i % 2 === 0 ? 2 : 4);
      }}
      return (h / 3) * sum;
    }}

    function tCdf(x, mu, sigma, nu) {{
      const z = (x - mu) / sigma;
      if (Math.abs(z) < 1e-12) return 0.5;
      const area = simpsonIntegral((u) => {{
        const coeff = Math.exp(logGamma((nu + 1) / 2) - logGamma(nu / 2) - 0.5 * Math.log(nu * Math.PI));
        return coeff * Math.pow(1 + (u * u) / nu, -(nu + 1) / 2);
      }}, 0, Math.abs(z), 500);
      return z > 0 ? Math.min(1, 0.5 + area) : Math.max(0, 0.5 - area);
    }}

    function lognormalPdf(x, mu, sigma) {{
      if (x <= 0) return 0;
      const z = (Math.log(x) - mu) / sigma;
      return Math.exp(-0.5 * z * z) / (x * sigma * Math.sqrt(2 * Math.PI));
    }}

    function lognormalCdf(x, mu, sigma) {{
      if (x <= 0) return 0;
      return normalCdf(Math.log(x), mu, sigma);
    }}

    function distMoments(kind, mu, sigma, nu) {{
      if (kind === "normal") {{
        return {{
          mean: mu,
          variance: sigma * sigma,
          std: sigma,
          skew: 0,
          kurtosis: 3,
          excessKurtosis: 0,
          support: "all real x",
        }};
      }}
      if (kind === "lognormal") {{
        const variance = (Math.exp(sigma * sigma) - 1) * Math.exp(2 * mu + sigma * sigma);
        const mean = Math.exp(mu + 0.5 * sigma * sigma);
        const std = Math.sqrt(variance);
        const skew = (Math.exp(sigma * sigma) + 2) * Math.sqrt(Math.exp(sigma * sigma) - 1);
        const excessKurtosis = Math.exp(4 * sigma * sigma) + 2 * Math.exp(3 * sigma * sigma) + 3 * Math.exp(2 * sigma * sigma) - 6;
        const kurtosis = excessKurtosis + 3;
        return {{ mean, variance, std, skew, kurtosis, excessKurtosis, support: "x > 0" }};
      }}
      const excessKurtosis = nu > 4 ? 6 / (nu - 4) : Infinity;
      return {{
        mean: nu > 1 ? mu : NaN,
        variance: nu > 2 ? (sigma * sigma * nu) / (nu - 2) : Infinity,
        std: nu > 2 ? Math.sqrt((sigma * sigma * nu) / (nu - 2)) : Infinity,
        skew: nu > 3 ? 0 : NaN,
        kurtosis: nu > 4 ? 3 + excessKurtosis : Infinity,
        excessKurtosis,
        support: "all real x",
      }};
    }}

    function cumulativeFromPdf(x, pdf) {{
      const cdf = new Array(x.length).fill(0);
      let total = 0;
      for (let i = 1; i < x.length; i += 1) {{
        total += 0.5 * (pdf[i] + pdf[i - 1]) * (x[i] - x[i - 1]);
        cdf[i] = total;
      }}
      if (total <= 0) return cdf;
      for (let i = 0; i < cdf.length; i += 1) cdf[i] = Math.max(0, Math.min(1, cdf[i] / total));
      cdf[cdf.length - 1] = 1;
      return cdf;
    }}

    function buildDistributionSeries(kind, mu, sigma, nu) {{
      let xMin = mu - 4 * sigma;
      let xMax = mu + 4 * sigma;
      if (kind === "lognormal") {{
        xMin = 1e-4;
        xMax = Math.exp(mu + 4 * sigma);
      }} else if (kind === "t") {{
        xMin = mu - 6 * sigma;
        xMax = mu + 6 * sigma;
      }}
      const x = [];
      const pdf = [];
      const cdf = [];
      const n = 320;
      for (let i = 0; i < n; i += 1) {{
        const xi = xMin + (i / (n - 1)) * (xMax - xMin);
        x.push(xi);
        if (kind === "normal") {{
          pdf.push(normalPdf(xi, mu, sigma));
          cdf.push(normalCdf(xi, mu, sigma));
        }} else if (kind === "lognormal") {{
          pdf.push(lognormalPdf(xi, mu, sigma));
          cdf.push(lognormalCdf(xi, mu, sigma));
        }} else {{
          pdf.push(tPdf(xi, mu, sigma, nu));
        }}
      }}
      if (kind === "t") {{
        return {{ x, pdf, cdf: cumulativeFromPdf(x, pdf) }};
      }}
      return {{ x, pdf, cdf }};
    }}

    function fmt(value) {{
      if (Number.isNaN(value)) return "undefined";
      if (!Number.isFinite(value)) return "infinite";
      const abs = Math.abs(value);
      if (abs >= 1000 || (abs > 0 && abs < 0.001)) return value.toExponential(3);
      return value.toFixed(4);
    }}

    function renderMoments(kind, mu, sigma, nu) {{
      const m = distMoments(kind, mu, sigma, nu);
      const rows = [
        ["E[X]", fmt(m.mean)],
        ["Var[X]", fmt(m.variance)],
        ["Std[X]", fmt(m.std)],
        ["Skew", fmt(m.skew)],
        ["Kurtosis", fmt(m.kurtosis)],
        ["Excess kurt.", fmt(m.excessKurtosis)],
        ["Support", m.support],
      ];
      if (kind === "t") rows.push(["Tail note", nu <= 4 ? "heavy / unstable" : "finite 4th moment"]);
      momentStats.innerHTML = rows.map(([name, val]) => `
        <div class="stat-row">
          <div class="name">${{name}}</div>
          <div class="val">${{val}}</div>
        </div>
      `).join("");
      return m;
    }}

    function renderDistributionPlot() {{
      // Use calibrated values from historical data
      const kind = "normal";
      const mu = distState.mu;
      const sigma = distState.sigma;

      const moments = renderMoments(kind, mu, sigma, 0);
      const series = buildDistributionSeries(kind, mu, sigma, 0);
      const shapes = [];
      const annotations = [];
      if (Number.isFinite(moments.mean) && Number.isFinite(moments.std)) {{
        [
          [moments.mean, theme.green, "E[X]"],
          [moments.mean - moments.std, theme.amber, "-1σ"],
          [moments.mean + moments.std, theme.amber, "+1σ"],
          [moments.mean - 2 * moments.std, theme.purple, "-2σ"],
          [moments.mean + 2 * moments.std, theme.purple, "+2σ"],
        ].forEach(([x, color, label]) => {{
          shapes.push({{
            type: "line",
            x0: x, x1: x, y0: 0, y1: 1,
            xref: "x", yref: "paper",
            line: {{ color, width: 1.2, dash: "dash" }}
          }});
          annotations.push({{
            x, y: 1.02, xref: "x", yref: "paper",
            text: label, showarrow: false,
            font: {{ size: 10, color }},
          }});
        }});
      }}

      Plotly.react("distributionPlot", [
        {{
          x: series.x,
          y: series.pdf,
          type: "scatter",
          mode: "lines",
          name: "PDF",
          line: {{ color: theme.blue, width: 3 }},
          hovertemplate: "x=%{{x:.4f}}<br>PDF=%{{y:.4f}}<extra></extra>",
        }},
        {{
          x: series.x,
          y: series.cdf,
          type: "scatter",
          mode: "lines",
          name: "CDF",
          yaxis: "y2",
          line: {{ color: theme.pink, width: 2.4 }},
          hovertemplate: "x=%{{x:.4f}}<br>CDF=%{{y:.4f}}<extra></extra>",
        }}
      ], {{
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        margin: {{ l: 58, r: 56, t: 46, b: 54 }},
        font: {{ family: "Consolas, Menlo, monospace", color: theme.text }},
        title: {{
          text: "Calibrated Distribution (Normal Log Returns)",
          font: {{ size: 15, color: theme.title }},
          x: 0.02,
          xanchor: "left",
        }},
        hoverlabel: {{
          bgcolor: "#ffffff",
          bordercolor: theme.border,
          font: {{ color: "#111111", family: "Consolas, Menlo, monospace" }},
        }},
        legend: {{
          orientation: "h",
          x: 0.5,
          xanchor: "center",
          y: 1.10,
          bgcolor: "rgba(22,27,34,0.86)",
          bordercolor: theme.border,
          borderwidth: 1,
        }},
        xaxis: {{
          title: "Log return",
          gridcolor: theme.border,
          zerolinecolor: theme.border,
        }},
        yaxis: {{
          title: "Density f(x)",
          gridcolor: theme.border,
          zerolinecolor: theme.border,
        }},
        yaxis2: {{
          title: "Probability F(x)",
          overlaying: "y",
          side: "right",
          range: [0, 1],
          showgrid: false,
          tickfont: {{ color: theme.pink }},
          titlefont: {{ color: theme.pink }},
        }},
        hovermode: "x unified",
        shapes,
        annotations,
      }}, {{
        responsive: true,
        displayModeBar: true,
        scrollZoom: true,
      }});
    }}

    function corrColor(v) {{
      const clamped = Math.max(-1, Math.min(1, v));
      if (clamped >= 0) {{
        const t = clamped;
        const r = Math.round(255 - 167 * t);
        const g = Math.round(255 - 73 * t);
        const b = 255;
        return `rgb(${{r}},${{g}},${{b}})`;
      }}
      const t = Math.abs(clamped);
      const r = 255;
      const g = Math.round(255 - 104 * t);
      const b = Math.round(255 - 106 * t);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}

    function choleskyStatus(matrix) {{
      const n = matrix.length;
      const L = Array.from({{ length: n }}, () => Array(n).fill(0));
      for (let i = 0; i < n; i += 1) {{
        for (let j = 0; j <= i; j += 1) {{
          let sum = matrix[i][j];
          for (let k = 0; k < j; k += 1) sum -= L[i][k] * L[j][k];
          if (i === j) {{
            if (sum < -1e-9) return false;
            L[i][j] = Math.sqrt(Math.max(sum, 0));
          }} else {{
            if (Math.abs(L[j][j]) < 1e-12) {{
              if (Math.abs(sum) > 1e-8) return false;
              L[i][j] = 0;
            }} else {{
              L[i][j] = sum / L[j][j];
            }}
          }}
        }}
      }}
      return true;
    }}

    function jacobiEigenvalues(matrix) {{
      const n = matrix.length;
      const a = matrix.map((row) => row.slice());
      const maxSweeps = Math.max(20, n * n * 8);
      for (let sweep = 0; sweep < maxSweeps; sweep += 1) {{
        let p = 0;
        let q = 1;
        let maxVal = 0;
        for (let i = 0; i < n; i += 1) {{
          for (let j = i + 1; j < n; j += 1) {{
            const val = Math.abs(a[i][j]);
            if (val > maxVal) {{
              maxVal = val;
              p = i;
              q = j;
            }}
          }}
        }}
        if (maxVal < 1e-10) break;
        const app = a[p][p];
        const aqq = a[q][q];
        const apq = a[p][q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        for (let k = 0; k < n; k += 1) {{
          if (k !== p && k !== q) {{
            const aik = a[k][p];
            const akq = a[k][q];
            a[k][p] = c * aik - s * akq;
            a[p][k] = a[k][p];
            a[k][q] = s * aik + c * akq;
            a[q][k] = a[k][q];
          }}
        }}
        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        a[p][q] = 0;
        a[q][p] = 0;
      }}
      return a.map((row, i) => row[i]).sort((lhs, rhs) => lhs - rhs);
    }}

    function updatePsdStatus(matrix) {{
      const eigenvalues = jacobiEigenvalues(matrix);
      const minEigen = eigenvalues.length ? eigenvalues[0] : NaN;
      const valid = choleskyStatus(matrix);
      psdBadge.textContent = valid ? "Valid" : "Cholesky failed";
      psdBadge.className = "badge " + (valid ? "valid" : "invalid");
      const worstPair = [];
      for (let i = 0; i < matrix.length; i += 1) {{
        for (let j = i + 1; j < matrix.length; j += 1) {{
          if (Math.abs(matrix[i][j]) > 0.95) worstPair.push(`[${{corrState.labels[i]}}-${{corrState.labels[j]}}]`);
        }}
      }}
      const stressNote = worstPair.length ? ` | high |rho|: ${{worstPair.join(", ")}}` : "";
      psdMeta.textContent = `min eigenvalue: ${{fmt(minEigen)}} | symmetric: yes${{stressNote}}`;
      return valid;
    }}

    function shortCorrLabel(label) {{
      const clean = String(label || "").trim();
      if (!clean) return "";
      if (/market/i.test(clean)) return "Mkt";
      if (/peer/i.test(clean)) return "Peer";
      if (clean.length <= 8) return clean;
      const acronym = clean.split(/\\s+/).map((part) => part[0]).join("").toUpperCase();
      return acronym || clean.slice(0, 8);
    }}

    function renderCorrGrid() {{
      const labels = corrState.labels;
      const matrix = corrState.matrix;
      const n = labels.length;
      corrGrid.innerHTML = "";
      const shell = document.createElement("div");
      shell.className = "matrix-grid";
      shell.style.gridTemplateColumns = `130px repeat(${{n}}, minmax(72px, 1fr))`;

      const headRow = document.createElement("div");
      headRow.className = "matrix-row";
      headRow.style.gridTemplateColumns = shell.style.gridTemplateColumns;
      headRow.innerHTML = `<div class="corner"></div>` + labels.map((label) => `<div class="axis-label">${{label}}</div>`).join("");
      shell.appendChild(headRow);

      for (let i = 0; i < n; i += 1) {{
        const row = document.createElement("div");
        row.className = "matrix-row";
        row.style.gridTemplateColumns = shell.style.gridTemplateColumns;
        row.appendChild(Object.assign(document.createElement("div"), {{
          className: "axis-label",
          textContent: labels[i],
        }}));

        for (let j = 0; j < n; j += 1) {{
          const cell = document.createElement("div");
          cell.className = "cell" + (i === j ? " diag" : "");
          cell.style.background = corrColor(matrix[i][j]);
          const input = document.createElement("input");
          input.type = "number";
          input.step = "0.01";
          input.min = "-1";
          input.max = "1";
          input.value = matrix[i][j].toFixed(2);
          input.disabled = i === j;
          input.addEventListener("input", (event) => {{
            let v = Number(event.target.value);
            if (!Number.isFinite(v)) return;
            v = Math.max(-1, Math.min(1, v));
            matrix[i][j] = v;
            matrix[j][i] = v;
            matrix[i][i] = 1;
            matrix[j][j] = 1;
            renderCorrGrid();
          }});
          cell.appendChild(input);
          row.appendChild(cell);
        }}
        shell.appendChild(row);
      }}

      corrGrid.appendChild(shell);
      updatePsdStatus(matrix);
    }}

    function renderCorrHeatmap() {{
      const labels = corrState.labels;
      const matrix = corrState.matrix;
      Plotly.react("corrPlot", [{{
        z: matrix,
        x: labels,
        y: labels,
        type: "heatmap",
        zmin: -1,
        zmax: 1,
        colorscale: [
          [0.0, theme.red],
          [0.5, "#f4f6f8"],
          [1.0, theme.blue]
        ],
        text: matrix.map((row) => row.map((v) => v.toFixed(2))),
        texttemplate: "%{{text}}",
        textfont: {{ size: 14 }},
        hovertemplate: "rho(%{{y}}, %{{x}}) = %{{z:.2f}}<extra></extra>",
        colorbar: {{
          title: "rho",
          tickfont: {{ color: theme.muted }},
          titlefont: {{ color: theme.muted }},
        }},
      }}], {{
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        margin: {{ l: 72, r: 28, t: 46, b: 54 }},
        title: {{
          text: "Editable correlation heatmap",
          font: {{ size: 15, color: theme.title }},
          x: 0.02,
          xanchor: "left",
        }},
        font: {{ family: "Consolas, Menlo, monospace", color: theme.text }},
        xaxis: {{ side: "top", tickfont: {{ size: 12 }} }},
        yaxis: {{ autorange: "reversed", tickfont: {{ size: 12 }} }},
      }}, {{
        responsive: true,
        displayModeBar: true,
        scrollZoom: true,
      }});
    }}

    function renderCorrEditor() {{
      const labels = corrState.labels;
      const matrix = corrState.matrix;
      const n = labels.length;
      corrGrid.innerHTML = "";
      corrGrid.style.gridTemplateColumns = "1fr";

      const head = document.createElement("div");
      head.className = "editor-row";
      head.style.gridTemplateColumns = `64px repeat(${{n}}, minmax(64px, 1fr))`;
      head.innerHTML = `<div class="editor-corner"></div>` + labels.map((label) => `<div class="axis-label" title="${{label}}">${{shortCorrLabel(label)}}</div>`).join("");
      corrGrid.appendChild(head);

      for (let i = 0; i < n; i += 1) {{
        const row = document.createElement("div");
        row.className = "editor-row";
        row.style.gridTemplateColumns = head.style.gridTemplateColumns;
        row.innerHTML = `<div class="axis-label" title="${{labels[i]}}">${{shortCorrLabel(labels[i])}}</div>`;
        for (let j = 0; j < n; j += 1) {{
          const input = document.createElement("input");
          input.type = "number";
          input.step = "0.01";
          input.min = "-1";
          input.max = "1";
          input.value = matrix[i][j].toFixed(2);
          input.disabled = i === j;
          input.addEventListener("input", (event) => {{
            let v = Number(event.target.value);
            if (!Number.isFinite(v)) return;
            v = Math.max(-1, Math.min(1, v));
            matrix[i][j] = v;
            matrix[j][i] = v;
            matrix[i][i] = 1;
            matrix[j][j] = 1;
            renderCorrEditor();
            renderCorrHeatmap();
            updatePsdStatus(matrix);
          }});
          row.appendChild(input);
        }}
        corrGrid.appendChild(row);
      }}
      updatePsdStatus(matrix);
    }}

    // Render calibrated distribution (no sliders needed)
    renderDistributionPlot();
    renderCorrEditor();
    renderCorrHeatmap();
  </script>
</body>
</html>
"""
        path_out = OUT / f"mc_setup_lab_{self.eng.ticker}.html"
        path_out.write_text(html_doc, encoding="utf-8")
        rlog(f"  [green]OK[/green] Setup lab -> [cyan]{path_out}[/cyan]")
        return path_out

    def render_history_detail(self, show=False):
        h = self.eng.history
        market = _history_market_data(self.eng.ticker, h.index)
        market["Close"] = market["Close"].fillna(h)
        market["Open"] = market["Open"].fillna(market["Close"].shift(1)).fillna(market["Close"])
        vol = market["Volume"].astype(float)
        vol_ma20 = vol.rolling(20).mean()
        up_day = market["Close"] >= market["Open"]
        spike = vol > 2 * vol_ma20.fillna(np.inf)
        ma20 = h.rolling(20).mean()
        ma50 = h.rolling(50).mean()
        std20 = h.rolling(20).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        band_width = (bb_upper - bb_lower) / ma20.replace(0, np.nan)
        squeeze_cutoff = float(band_width.dropna().quantile(0.10)) if band_width.notna().any() else np.nan
        squeeze_mask = (band_width <= squeeze_cutoff).fillna(False).to_numpy()
        rsi = _compute_rsi(h, 14)
        divergences = _find_bearish_rsi_divergences(h, rsi)
        w52 = min(252, len(h))
        hi52_idx = h.iloc[-w52:].idxmax()
        lo52_idx = h.iloc[-w52:].idxmin()
        hi52 = float(h.loc[hi52_idx])
        lo52 = float(h.loc[lo52_idx])
        cross_up = (ma20 > ma50) & (ma20.shift(1) <= ma50.shift(1))
        cross_dn = (ma20 < ma50) & (ma20.shift(1) >= ma50.shift(1))
        spike_idx = h.index[spike.fillna(False)]
        spike_vol = vol.loc[spike_idx]
        spike_colors = ["#8cff9b" if bool(up_day.loc[idx]) else "#ff8e88" for idx in spike_idx]

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.015,
            row_heights=[0.60, 0.20, 0.20],
            subplot_titles=(
                f"Historical Price  ({self.eng.ticker}, MC calibration context)",
                "Volume / Participation",
                "RSI-14 Momentum",
            ),
        )

        for start, end in _contiguous_true_ranges(squeeze_mask):
            fig.add_vrect(
                x0=h.index[start], x1=h.index[end],
                fillcolor="#9aa4b2", opacity=0.10, line_width=0,
                row=1, col=1,
            )

        fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines", name="Close",
                                 line=dict(color=C["cyan"], width=2.2),
                                 hovertemplate="Date: %{x|%d %b %Y}<br>Price: $%{y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=ma20, mode="lines", name="MA-20",
                                 line=dict(color=C["blue"], width=1.6),
                                 hovertemplate="Date: %{x|%d %b %Y}<br>MA20: $%{y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=ma50, mode="lines", name="MA-50",
                                 line=dict(color=C["amber"], width=1.6, dash="dash"),
                                 hovertemplate="Date: %{x|%d %b %Y}<br>MA50: $%{y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=bb_upper, mode="lines", name="BB upper",
                                 line=dict(color=C["purple"], width=0.9), opacity=0.6, showlegend=False,
                                 hovertemplate="Date: %{x|%d %b %Y}<br>BB Upper: $%{y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=bb_lower, mode="lines", name="BB band",
                                 line=dict(color=C["purple"], width=0.9), fill="tonexty",
                                 fillcolor="rgba(163,113,247,0.10)", opacity=0.6,
                                 hovertemplate="Date: %{x|%d %b %Y}<br>BB Lower: $%{y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[hi52_idx], y=[hi52], mode="markers+text", name="52w high",
                                 marker=dict(color=C["green"], size=12, symbol="triangle-up"),
                                 text=["52w high"], textposition="top center",
                                 hovertemplate="Date: %{x|%d %b %Y}<br>52w high: $%{y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[lo52_idx], y=[lo52], mode="markers+text", name="52w low",
                                 marker=dict(color=C["red"], size=12, symbol="triangle-down"),
                                 text=["52w low"], textposition="bottom center",
                                 hovertemplate="Date: %{x|%d %b %Y}<br>52w low: $%{y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index[cross_up.fillna(False)], y=h[cross_up.fillna(False)], mode="markers",
                                 name="Bull cross", marker=dict(color=C["green"], size=7, symbol="circle"),
                                 hovertemplate="Date: %{x|%d %b %Y}<br>Bull cross: $%{y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index[cross_dn.fillna(False)], y=h[cross_dn.fillna(False)], mode="markers",
                                 name="Bear cross", marker=dict(color=C["red"], size=7, symbol="circle"),
                                 hovertemplate="Date: %{x|%d %b %Y}<br>Bear cross: $%{y:,.2f}<extra></extra>"), row=1, col=1)

        bar_colors = np.where(spike, np.where(up_day, "#8cff9b", "#ff8e88"), np.where(up_day, C["green"], C["red"]))
        fig.add_trace(go.Bar(x=h.index, y=vol, name="Volume", marker_color=bar_colors,
                             hovertemplate="Date: %{x|%d %b %Y}<br>Volume: %{y:,.0f}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=vol_ma20, mode="lines", name="Vol MA-20",
                                 line=dict(color=C["muted"], width=1.4),
                                 hovertemplate="Date: %{x|%d %b %Y}<br>Vol MA20: %{y:,.0f}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=spike_idx, y=spike_vol, mode="markers", name="Volume spike",
            marker=dict(color=spike_colors, size=8, symbol="diamond"),
            hovertemplate="Date: %{x|%d %b %Y}<br>Volume spike: %{y:,.0f}<extra></extra>"
        ), row=2, col=1)

        fig.add_hrect(y0=70, y1=100, fillcolor=C["red"], opacity=0.10, line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor=C["green"], opacity=0.10, line_width=0, row=3, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=rsi.where((rsi >= 30) & (rsi <= 70)), mode="lines",
                                 name="RSI-14", line=dict(color=C["muted"], width=1.6),
                                 hovertemplate="Date: %{x|%d %b %Y}<br>RSI: %{y:.1f}<extra></extra>"), row=3, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=rsi.where(rsi > 70), mode="lines",
                                 name="RSI > 70", line=dict(color=C["red"], width=1.8), showlegend=False,
                                 hovertemplate="Date: %{x|%d %b %Y}<br>RSI: %{y:.1f}<extra></extra>"), row=3, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=rsi.where(rsi < 30), mode="lines",
                                 name="RSI < 30", line=dict(color=C["green"], width=1.8), showlegend=False,
                                 hovertemplate="Date: %{x|%d %b %Y}<br>RSI: %{y:.1f}<extra></extra>"), row=3, col=1)

        for ts in divergences:
            if not np.isfinite(rsi.loc[ts]):
                continue
            fig.add_annotation(
                x=ts, y=float(h.loc[ts]), row=1, col=1,
                text="RSI div", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
                ax=0, ay=-34, arrowcolor=C["amber"],
                font=dict(color=C["amber"], size=9),
            )
            fig.add_annotation(
                x=ts, y=float(rsi.loc[ts]), row=3, col=1,
                text="bear div", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
                ax=0, ay=-28, arrowcolor=C["amber"],
                font=dict(color=C["amber"], size=9),
            )

        fig.add_hline(y=70, line_color=C["red"], line_dash="dash", line_width=1.0, row=3, col=1)
        fig.add_hline(y=30, line_color=C["green"], line_dash="dash", line_width=1.0, row=3, col=1)

        layout = self._plotly_theme()
        layout.update(dict(
            title=dict(
                text=(f"<b>Monte Carlo Input Signal Stack</b> - {self.eng.ticker}<br>"
                      f"<span style='font-size:11px;color:{C['muted']}'>Historical context for drift, volatility regime, liquidity participation, and momentum stress signals</span>"),
                x=0.5, y=0.985, xanchor="center", yanchor="top",
                font=dict(size=15, color=C["title"])
            ),
            height=1080,
            margin=dict(l=70, r=70, t=130, b=90),
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5,
                        font=dict(size=9), bgcolor="rgba(13,17,23,0.82)", bordercolor=C["border"], borderwidth=1),
            bargap=0.08,
            hovermode="x unified",
            hoversubplots="axis",
            spikedistance=-1,
        ))
        fig.update_layout(**layout)
        for ann in fig.layout.annotations:
            if ann.text in {
                f"Historical Price  ({self.eng.ticker}, MC calibration context)",
                "Volume / Participation",
                "RSI-14 Momentum",
            }:
                ann.font = dict(size=12, color=C["title"])
        fig.update_yaxes(title_text="Price ($)", tickprefix="$", row=1, col=1)
        fig.update_yaxes(title_text="Volume", tickformat="~s", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        fig.update_xaxes(title_text="Calendar date", row=3, col=1)
        for r in [1, 2, 3]:
            fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=r, col=1,
                             showspikes=True, spikemode="across", spikesnap="cursor",
                             spikecolor=C["blue"], spikethickness=1)
            fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=r, col=1)

        path_out = OUT / f"mc_history_detail_{self.eng.ticker}.html"
        fig.write_html(str(path_out), include_plotlyjs="cdn",
                       config={"displayModeBar": True, "scrollZoom": True})
        rlog(f"  [green]OK[/green] History detail -> [cyan]{path_out}[/cyan]")
        if show:
            fig.show()
        return fig


    def render_rng_diagnostics(self, show=False):
        rng_diag = _compute_rng_diagnostics(_rng_diagnostic_sample())
        uni_x, uni_y = _rng_uniform_pairs()
        qx = rng_diag["theoretical"]
        qy = rng_diag["sorted_sample"]
        band = rng_diag["ks_band"]
        lags = rng_diag["acf_lags"]
        acf_vals = rng_diag["acf_vals"]
        conf = rng_diag["acf_conf"]
        colors = np.where(rng_diag["acf_flags"], C["red"], C["blue"])
        n_sample = len(rng_diag["sample"])

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Q-Q Validation of RNG Draws",
                "ACF Independence Check",
                "Random Uniformity Scatter",
            ),
            horizontal_spacing=0.09,
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        )

        fig.add_trace(go.Scatter(
            x=qx, y=qx + band, mode="lines", name="KS upper",
            line=dict(color="#9aa4b2", width=0.0), showlegend=False, hoverinfo="skip"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=qx, y=qx - band, mode="lines", name="KS band",
            fill="tonexty", fillcolor="rgba(154,164,178,0.18)",
            line=dict(color="#9aa4b2", width=0.0), hoverinfo="skip"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=qx, y=qy, mode="markers", name="Sim quantiles",
            marker=dict(color=C["blue"], size=5, opacity=0.52),
            hovertemplate="Theoretical: %{x:.3f}<br>Simulated: %{y:.3f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[rng_diag["diag_min"], rng_diag["diag_max"]],
            y=[rng_diag["diag_min"], rng_diag["diag_max"]],
            mode="lines", name="45deg ref",
            line=dict(color=C["red"], width=1.8, dash="dash"),
            hoverinfo="skip"
        ), row=1, col=1)
        fig.add_annotation(
            xref="x domain", yref="y domain",
            x=0.02, y=0.98,
            text="S-shape = fat tails | bow = skew",
            showarrow=False,
            align="left",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["muted"], size=9, family="Consolas, Menlo, monospace"),
            row=1, col=1,
        )

        fig.add_hrect(y0=-conf, y1=conf, fillcolor="rgba(154,164,178,0.10)", line_width=0, row=1, col=2)
        fig.add_trace(go.Bar(
            x=lags, y=acf_vals, name="ACF", marker_color=colors,
            hovertemplate="Lag %{x}<br>rho=%{y:.4f}<extra></extra>"
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=lags, y=acf_vals, mode="markers", name="Lag markers",
            marker=dict(color=colors, size=6, line=dict(color=C["bg"], width=0.5)),
            showlegend=False,
            hovertemplate="Lag %{x}<br>rho=%{y:.4f}<extra></extra>"
        ), row=1, col=2)
        fig.add_hline(y=0, line_color=C["muted"], line_width=1.0, row=1, col=2)
        fig.add_hline(y=conf, line_color=C["red"], line_dash="dash", line_width=1.0, row=1, col=2)
        fig.add_hline(y=-conf, line_color=C["red"], line_dash="dash", line_width=1.0, row=1, col=2)
        if len(lags):
            fig.add_annotation(
                x=int(lags[0]), y=float(acf_vals[0]), row=1, col=2,
                text=f"lag1={acf_vals[0]:+.3f}", showarrow=True, arrowhead=2,
                ax=28, ay=-22, arrowcolor=C["amber"],
                font=dict(color=C["amber"], size=9),
            )

        fig.add_trace(go.Scatter(
            x=uni_x, y=uni_y, mode="markers", name="U_t vs U_t+1",
            marker=dict(color=C["cyan"], size=3, opacity=0.30),
            hovertemplate="U_t=%{x:.3f}<br>U_t+1=%{y:.3f}<extra></extra>"
        ), row=1, col=3)
        fig.add_shape(
            type="line",
            x0=0, x1=1, y0=0.5, y1=0.5,
            xref="x3", yref="y3",
            line=dict(color=C["muted"], width=1, dash="dot")
        )
        fig.add_shape(
            type="line",
            x0=0.5, x1=0.5, y0=0, y1=1,
            xref="x3", yref="y3",
            line=dict(color=C["muted"], width=1, dash="dot")
        )
        fig.add_annotation(
            xref="x3 domain", yref="y3 domain",
            x=0.02, y=0.98,
            text="Cloud should fill the square without bands",
            showarrow=False,
            align="left",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["muted"], size=9, family="Consolas, Menlo, monospace"),
        )

        layout = self._plotly_theme()
        layout.update(dict(
            title=dict(
                text=(f"<b>RNG Diagnostics</b> - {self.eng.ticker}<br>"
                      f"<span style='font-size:11px;color:{C['muted']}'>N={n_sample} | SW p={rng_diag['shapiro_p']:.3g} | A-D={rng_diag['anderson_stat']:.3f} | Ljung-Box p={rng_diag['ljung_box_p']:.3g}</span>"),
                x=0.5, y=0.96, xanchor="center", yanchor="top",
                font=dict(size=15, color=C["title"]),
            ),
            height=700,
            margin=dict(l=70, r=70, t=140, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5,
                        font=dict(size=10), bgcolor="rgba(13,17,23,0.82)", bordercolor=C["border"], borderwidth=1),
        ))
        fig.update_layout(**layout)
        for ann in fig.layout.annotations:
            if ann.text in {"Q-Q Validation of RNG Draws", "ACF Independence Check", "Random Uniformity Scatter"}:
                ann.font = dict(size=12, color=C["title"])
        fig.update_xaxes(title_text="Theoretical quantiles N(0,1)", row=1, col=1)
        fig.update_yaxes(title_text="Simulated sample quantiles", row=1, col=1)
        fig.update_xaxes(title_text="Lag k (1-40)", row=1, col=2)
        fig.update_yaxes(title_text="Autocorrelation rho(k)", row=1, col=2)
        fig.update_xaxes(title_text="U_t", range=[0, 1], row=1, col=3)
        fig.update_yaxes(title_text="U_t+1", range=[0, 1], row=1, col=3, scaleanchor="x3", scaleratio=1)
        for c in [1, 2, 3]:
            fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=1, col=c)
            fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=1, col=c)

        path_out = OUT / f"mc_rng_diagnostics_{self.eng.ticker}.html"
        fig_html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            config={"displayModeBar": True, "scrollZoom": True},
        )
        page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RNG Diagnostics - {self.eng.ticker}</title>
  <style>
    html, body {{
      margin: 0;
      min-height: 100%;
      background: {C["bg"]};
      color: {C["text"]};
      font-family: Consolas, Menlo, monospace;
    }}
    body {{
      min-height: 100vh;
    }}
    .plot-shell {{
      background: {C["bg"]};
    }}
  </style>
</head>
<body>
  <div class="plot-shell">{fig_html}</div>
</body>
</html>
"""
        path_out.write_text(page_html, encoding="utf-8")
        rlog(f"  [green]OK[/green] RNG diagnostics -> [cyan]{path_out}[/cyan]")
        if show:
            fig.show()
        return fig

    def _paths_with_params(self, Z: np.ndarray, N: int, S0: float | None = None, mu: float | None = None, sigma: float | None = None) -> np.ndarray:
        S0_use = float(self.eng.S0 if S0 is None else S0)
        mu_use = float(self.eng.mu if mu is None else mu)
        sigma_use = float(self.eng.sigma if sigma is None else sigma)
        T = N / 252.0
        return self.eng._paths_from_normals(Z=Z[:, :N], drift=mu_use, sigma=sigma_use, S0=S0_use, T=T)

    def render_sensitivity_suite(self, N: int, show=False):
        num_sim = 1200
        rng = np.random.default_rng(123)
        max_horizon = max(N, int(round(N * 1.5)))
        Z = self.eng._generate_normals(num_sim=num_sim, N=max_horizon, rng=rng, random_method="pseudo", antithetic=False)
        base_paths = self._paths_with_params(Z, N=N)
        base_output = float(base_paths[:, -1].mean())

        lr = self.eng.log_ret.dropna()
        roll = self.eng.roll_vol.dropna() if self.eng.roll_vol is not None else pd.Series(dtype=float)
        mu_se = float(lr.std() * np.sqrt(252 / max(len(lr), 1))) if len(lr) > 2 else max(abs(self.eng.mu) * 0.2, 0.03)
        sigma_p10 = float(roll.quantile(0.10)) if len(roll) > 10 else max(self.eng.sigma * 0.75, 0.05)
        sigma_p90 = float(roll.quantile(0.90)) if len(roll) > 10 else self.eng.sigma * 1.25
        specs = [
            ("Spot S0", self.eng.S0 * 0.90, self.eng.S0 * 1.10),
            ("Drift mu", self.eng.mu - mu_se, self.eng.mu + mu_se),
            ("Vol sigma", max(0.03, sigma_p10), max(max(0.05, sigma_p10), sigma_p90)),
            ("Horizon N", max(21, int(round(N * 0.60))), max(42, int(round(N * 1.40)))),
        ]
        tornado_rows = []
        for name, low_v, high_v in specs:
            if name == "Spot S0":
                low_out = float(self._paths_with_params(Z, N=N, S0=low_v)[:, -1].mean())
                high_out = float(self._paths_with_params(Z, N=N, S0=high_v)[:, -1].mean())
            elif name == "Drift mu":
                low_out = float(self._paths_with_params(Z, N=N, mu=low_v)[:, -1].mean())
                high_out = float(self._paths_with_params(Z, N=N, mu=high_v)[:, -1].mean())
            elif name == "Vol sigma":
                low_out = float(self._paths_with_params(Z, N=N, sigma=low_v)[:, -1].mean())
                high_out = float(self._paths_with_params(Z, N=N, sigma=high_v)[:, -1].mean())
            else:
                low_out = float(self._paths_with_params(Z, N=int(low_v))[:, -1].mean())
                high_out = float(self._paths_with_params(Z, N=int(high_v))[:, -1].mean())
            tornado_rows.append({
                "name": name,
                "low_out": low_out,
                "high_out": high_out,
                "range": high_out - low_out,
            })
        tornado_rows.sort(key=lambda row: row["range"], reverse=True)
        labels = [row["name"] for row in tornado_rows]
        low_seg = [base_output - row["low_out"] for row in tornado_rows]
        high_seg = [row["high_out"] - base_output for row in tornado_rows]
        low_outs = [row["low_out"] for row in tornado_rows]
        high_outs = [row["high_out"] for row in tornado_rows]

        fig_tornado = go.Figure()
        fig_tornado.add_trace(go.Bar(
            y=labels, x=low_seg, base=low_outs, orientation="h", name="Downside swing",
            marker_color="rgba(248,81,73,0.72)",
            customdata=np.column_stack([low_outs, high_outs]),
            hovertemplate="%{y}<br>Low output=$%{customdata[0]:,.2f}<br>Base=$" + f"{base_output:,.2f}" + "<br>High output=$%{customdata[1]:,.2f}<extra></extra>",
        ))
        fig_tornado.add_trace(go.Bar(
            y=labels, x=high_seg, base=[base_output] * len(labels), orientation="h", name="Upside swing",
            marker_color="rgba(63,185,80,0.72)",
            customdata=np.column_stack([low_outs, high_outs]),
            hovertemplate="%{y}<br>Low output=$%{customdata[0]:,.2f}<br>Base=$" + f"{base_output:,.2f}" + "<br>High output=$%{customdata[1]:,.2f}<extra></extra>",
        ))
        for idx, row in enumerate(tornado_rows):
            fig_tornado.add_annotation(x=row["low_out"], y=row["name"], text=f"${row['low_out']:,.0f}", showarrow=False,
                                       xanchor="right", yanchor="middle", xshift=-6,
                                       font=dict(size=9, color=C["red"]))
            fig_tornado.add_annotation(x=row["high_out"], y=row["name"], text=f"${row['high_out']:,.0f}", showarrow=False,
                                       xanchor="left", yanchor="middle", xshift=6,
                                       font=dict(size=9, color=C["green"]))
        fig_tornado.add_vline(x=base_output, line_color=C["amber"], line_dash="dash", line_width=1.8)
        fig_tornado.update_layout(
            **self._plotly_theme(),
            title=dict(
                text=f"<b>Tornado Sensitivity</b><br><span style='font-size:11px;color:{C['muted']}'>Base output = mean terminal price E[S(T)] = ${base_output:,.2f}; widest bar = hedge first</span>",
                x=0.5, xanchor="center", font=dict(size=15, color=C["title"])
            ),
            barmode="overlay",
            height=520,
            margin=dict(l=90, r=70, t=95, b=60),
        )
        fig_tornado.update_xaxes(title_text="Output range when one input is varied", gridcolor=C["border"], zerolinecolor=C["border"], tickprefix="$")
        fig_tornado.update_yaxes(autorange="reversed", gridcolor=C["border"], zerolinecolor=C["border"])

        shock_avg = Z[:, :N].mean(axis=1)
        shock_disp = Z[:, :N].std(axis=1)
        terminal_shock = Z[:, N - 1]
        final_price = base_paths[:, -1]
        terminal_return = final_price / self.eng.S0 - 1.0
        running_max = np.maximum.accumulate(base_paths, axis=1)
        max_dd = ((base_paths - running_max) / running_max).min(axis=1)
        pair_df = pd.DataFrame({
            "Shock avg": shock_avg,
            "Shock disp": shock_disp,
            "Terminal shock": terminal_shock,
            "Final price": final_price,
            "Max drawdown": max_dd,
            "Return bucket": pd.qcut(terminal_return, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]),
        })
        fig_pair = px.scatter_matrix(
            pair_df,
            dimensions=["Shock avg", "Shock disp", "Terminal shock", "Final price", "Max drawdown"],
            color="Return bucket",
            color_discrete_map={"Q1 low": C["blue"], "Q2": C["cyan"], "Q3": C["amber"], "Q4 high": C["red"]},
            opacity=0.30,
        )
        fig_pair.update_traces(diagonal_visible=True, showupperhalf=False, marker=dict(size=4))
        fig_pair.update_layout(
            **self._plotly_theme(),
            title=dict(
                text=f"<b>Scatter Matrix / Pair Plot</b><br><span style='font-size:11px;color:{C['muted']}'>Path-level shock summaries vs outputs; colors show terminal return quartiles</span>",
                x=0.5, xanchor="center", font=dict(size=15, color=C["title"])
            ),
            height=880,
            margin=dict(l=50, r=30, t=95, b=40),
            dragmode="select",
        )

        sigma_grid = np.linspace(max(0.03, self.eng.sigma * 0.6), self.eng.sigma * 1.4, 19)
        pdp_vals = []
        ice_idx = np.linspace(0, num_sim - 1, 10, dtype=int)
        ice_curves = []
        for idx in ice_idx:
            ice_curves.append([])
        for s in sigma_grid:
            tmp_paths = self._paths_with_params(Z, N=N, sigma=float(s))
            terminal_vals = tmp_paths[:, -1]
            pdp_vals.append(float(terminal_vals.mean()))
            for k, idx in enumerate(ice_idx):
                ice_curves[k].append(float(terminal_vals[idx]))
        mid = len(sigma_grid) // 2
        pdp_slope = (pdp_vals[mid + 1] - pdp_vals[mid - 1]) / (sigma_grid[mid + 1] - sigma_grid[mid - 1]) if len(sigma_grid) > 2 else 0.0
        fig_pdp = go.Figure()
        for k, idx in enumerate(ice_idx):
            fig_pdp.add_trace(go.Scatter(
                x=sigma_grid, y=ice_curves[k], mode="lines", name=f"ICE {k+1}",
                line=dict(color=C["blue"], width=1.0), opacity=0.28, showlegend=False,
                hovertemplate="sigma=%{x:.2%}<br>Path terminal=$%{y:,.2f}<extra></extra>",
            ))
        fig_pdp.add_trace(go.Scatter(
            x=sigma_grid, y=pdp_vals, mode="lines+markers", name="PDP",
            line=dict(color=C["amber"], width=3), marker=dict(size=6),
            hovertemplate="sigma=%{x:.2%}<br>E[S(T)|sigma]=$%{y:,.2f}<extra></extra>",
        ))
        fig_pdp.add_vline(x=self.eng.sigma, line_color=C["green"], line_dash="dash", line_width=1.6)
        fig_pdp.add_annotation(
            xref="x domain", yref="y domain", x=0.98, y=0.98,
            text=f"Midpoint slope dE/dsigma ≈ {pdp_slope:,.1f}<br>Base sigma={self.eng.sigma:.2%}",
            showarrow=False, align="right",
            bgcolor="rgba(13,17,23,0.78)", bordercolor=C["border"], borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )
        fig_pdp.update_layout(
            **self._plotly_theme(),
            title=dict(
                text=f"<b>PDP + ICE</b><br><span style='font-size:11px;color:{C['muted']}'>Terminal price response to volatility sigma with shared shocks held fixed</span>",
                x=0.5, xanchor="center", font=dict(size=15, color=C["title"])
            ),
            height=520,
            margin=dict(l=70, r=60, t=95, b=60),
        )
        fig_pdp.update_xaxes(title_text="Input sigma", tickformat=".1%", gridcolor=C["border"], zerolinecolor=C["border"])
        fig_pdp.update_yaxes(title_text="Expected / path terminal price", tickprefix="$", gridcolor=C["border"], zerolinecolor=C["border"])

        tornado_html = fig_tornado.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": True, "scrollZoom": True})
        pair_html = fig_pair.to_html(include_plotlyjs=False, full_html=False, config={"displayModeBar": True, "scrollZoom": True})
        pdp_html = fig_pdp.to_html(include_plotlyjs=False, full_html=False, config={"displayModeBar": True, "scrollZoom": True})
        page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Monte Carlo Sensitivity - {self.eng.ticker}</title>
  <style>
    :root {{
      --bg: {C["bg"]};
      --bg2: {C["bg2"]};
      --border: {C["border"]};
      --muted: {C["muted"]};
      --text: {C["text"]};
      --title: {C["title"]};
    }}
    html, body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Consolas, Menlo, monospace;
    }}
    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px 20px 36px;
    }}
    .hero, .card {{
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(22,27,34,0.96);
      box-shadow: 0 18px 48px rgba(0,0,0,0.20);
    }}
    .hero {{
      padding: 20px 22px;
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      color: var(--title);
      font-size: 28px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 13px;
      max-width: 980px;
    }}
    .card {{
      padding: 10px 10px 2px;
      margin-bottom: 18px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Sensitivity & Interaction Lab</h1>
      <p>These views use shared Monte Carlo shocks so the visual differences come from the input being changed, not fresh randomness. Tornado ranks which parameter moves output most, the pair plot highlights path-level non-linearity, and PDP + ICE shows whether volatility response is linear or convex.</p>
    </section>
    <section class="card">{tornado_html}</section>
    <section class="card">{pair_html}</section>
    <section class="card">{pdp_html}</section>
  </div>
</body>
</html>
"""
        path_out = OUT / f"mc_sensitivity_{self.eng.ticker}.html"
        path_out.write_text(page_html, encoding="utf-8")
        rlog(f"  [green]OK[/green] Sensitivity lab -> [cyan]{path_out}[/cyan]")
        if show:
            fig_tornado.show()
            fig_pair.show()
            fig_pdp.show()
        return path_out

    def render_main(self, st: dict, N: int, show=True, adv: dict | None = None):
        adv = adv or {}
        h  = self.eng.history
        S0 = self.eng.S0
        ma20  = h.rolling(20).mean()
        ma50  = h.rolling(50).mean()
        std20 = h.rolling(20).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        band_width = (bb_upper - bb_lower) / ma20.replace(0, np.nan)
        squeeze_cutoff = float(band_width.dropna().quantile(0.10)) if band_width.notna().any() else np.nan
        squeeze_mask = (band_width <= squeeze_cutoff).fillna(False).to_numpy()
        w52 = min(252, len(h))
        hi52_idx = h.iloc[-w52:].idxmax()
        lo52_idx = h.iloc[-w52:].idxmin()
        hi52 = float(h.loc[hi52_idx])
        lo52 = float(h.loc[lo52_idx])
        cross_up = (ma20 > ma50) & (ma20.shift(1) <= ma50.shift(1))
        cross_dn = (ma20 < ma50) & (ma20.shift(1) >= ma50.shift(1))

        fig = make_subplots(
            rows=5, cols=3,
            subplot_titles=(
                f"Historical Price  ({self.eng.ticker})",
                "Terminal Distribution S(T)",
                "Log-Return Hist  Normal vs t",
                "Monte Carlo Sample Paths",
                "Price Path Fan",
                "Density Heatmap (column-normalised)",
                "Rolling 30d Volatility Regimes",
                "Return Q-Q vs Normal",
                "CDF / Exceedance Curve",
                "Return / Risk Profile",
                "Percentile Summary",
                "Colour-coded Risk Metrics",
                "Scenario Fan Chart",
            ),
            vertical_spacing=0.10,
            horizontal_spacing=0.09,
            specs=[
                [{"type": "xy"}, {"type": "xy", "secondary_y": True}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy", "secondary_y": True}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "table"}],
                [{"type": "xy", "colspan": 3}, None, None],
            ],
            row_heights=[0.19, 0.19, 0.19, 0.19, 0.24],
        )

        fig.add_trace(go.Scatter(
            x=h.index, y=h.values, mode="lines", name="Close",
            line=dict(color=C["blue"], width=2),
            hovertemplate="Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=h.index, y=ma20, mode="lines", name="MA-20",
            line=dict(color=C["purple"], width=1.5),
            hovertemplate="MA20: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=h.index, y=ma50, mode="lines", name="MA-50",
            line=dict(color=C["amber"], width=1.5, dash="dash"),
            hovertemplate="MA50: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)


        # Bollinger upper
        fig.add_trace(go.Scatter(
            x=h.index, y=bb_upper, mode="lines", name="BB+2sigma",
            line=dict(color=C["purple"], width=0.8, dash="dot"), opacity=0.7,
            hovertemplate="BB+2sigma: $%{y:,.2f}<extra></extra>",
            showlegend=False
        ), row=1, col=1)



        # Bollinger lower + fill
        fig.add_trace(go.Scatter(
            x=h.index, y=bb_lower, mode="lines", name="BB +/-2sigma",
            line=dict(color=C["purple"], width=0.8, dash="dot"), opacity=0.7,
            fill="tonexty", fillcolor="rgba(163,113,247,0.06)",
            hovertemplate="BB-2sigma: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[hi52_idx], y=[hi52], mode="markers", name="52w high",
            marker=dict(color=C["green"], size=11, symbol="triangle-up"),
            hovertemplate="52w high: %{x}<br>$%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[lo52_idx], y=[lo52], mode="markers", name="52w low",
            marker=dict(color=C["red"], size=11, symbol="triangle-down"),
            hovertemplate="52w low: %{x}<br>$%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=h.index[cross_up.fillna(False)], y=h[cross_up.fillna(False)], mode="markers",
            name="Bull cross", marker=dict(color=C["green"], size=6, symbol="circle")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=h.index[cross_dn.fillna(False)], y=h[cross_dn.fillna(False)], mode="markers",
            name="Bear cross", marker=dict(color=C["red"], size=6, symbol="circle")
        ), row=1, col=1)
        for start, end in _contiguous_true_ranges(squeeze_mask):
            fig.add_vrect(
                x0=h.index[start], x1=h.index[end],
                fillcolor="#9aa4b2", opacity=0.08, line_width=0,
                row=1, col=1,
            )

        finals = np.asarray(st["finals"], dtype=float)
        n_finals = max(len(finals), 1)
        fd_iqr = float(np.subtract(*np.percentile(finals, [75, 25]))) if len(finals) else 0.0
        fd_h = 2 * fd_iqr * (n_finals ** (-1 / 3)) if fd_iqr > 0 else 0.0
        fd_bins = int(np.ceil((finals.max() - finals.min()) / fd_h)) if fd_h > 0 else 0
        fd_bins = int(np.clip(fd_bins if fd_bins > 0 else np.sqrt(n_finals), 20, 120))
        var5_price = float(np.quantile(finals, 0.05))
        cvar5_price = float(finals[finals <= var5_price].mean()) if np.any(finals <= var5_price) else var5_price
        skew_terminal = float(scipy_stats.skew(finals))
        ex_kurt_terminal = float(scipy_stats.kurtosis(finals, fisher=True))
        fig.add_trace(go.Histogram(
            x=finals, nbinsx=fd_bins, name="Final prices",
            histnorm="probability density",
            marker=dict(
                color=finals,
                colorscale=[[0, C["red"]], [0.5, C["amber"]], [1.0, C["green"]]],
                colorbar=None,
                line=dict(width=0),
            ),
            opacity=0.82,
            hovertemplate="S(T): $%{x:,.2f}<br>Density: %{y:.4f}<extra></extra>"
        ), row=1, col=2, secondary_y=False)




        # KDE overlay
        kde_x = np.linspace(finals.min(), finals.max(), 250)
        kde   = scipy_stats.gaussian_kde(finals)
        kde_y = kde(kde_x)
        mode_x = float(kde_x[np.argmax(kde_y)])
        mode_y = float(np.max(kde_y))
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y, mode="lines", name="KDE",
            line=dict(color=C["cyan"], width=2.5),
            hovertemplate="$%{x:,.0f}: density=%{y:.4f}<extra></extra>"
        ), row=1, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=kde_x, y=np.clip(kde_y, 1e-9, None), mode="lines", name="Log density",
            line=dict(color=C["pink"], width=1.6, dash="dot"),
            hovertemplate="$%{x:,.0f}: log-density view=%{y:.4e}<extra></extra>",
            visible="legendonly",
        ), row=1, col=2, secondary_y=True)
        fig.add_trace(go.Scatter(
            x=[mode_x], y=[mode_y], mode="markers", name="Mode",
            marker=dict(color=C["green"], size=11, symbol="diamond"),
            hovertemplate="Mode: $%{x:,.2f}<br>Density=%{y:.4f}<extra></extra>"
        ), row=1, col=2, secondary_y=False)

        for val, col in [(S0, C["red"]), (st["mean"], C["green"]), (st["median"], C["amber"])]:
            fig.add_vline(
                x=val,
                line_color=col,
                line_dash="dash",
                line_width=1.8,
                row=1,
                col=2,
            )
        fig.add_vline(x=st["median"], line_color=C["amber"], line_dash="dot", line_width=1.8, row=1, col=2)
        fig.add_vrect(x0=finals.min(), x1=var5_price, fillcolor="rgba(248,81,73,0.12)", line_width=0, row=1, col=2)
        fig.add_annotation(
            xref="x2 domain",
            yref="y2 domain",
            x=0.02,
            y=0.98,
            text=(
                f"<b>S0</b> ${S0:,.0f}<br>"
                f"<b>Mean</b> ${st['mean']:,.0f}<br>"
                f"<b>Median</b> ${st['median']:,.0f}<br>"
                f"<b>Mode</b> ${mode_x:,.0f}<br>"
                f"<b>VaR 5%</b> ${var5_price:,.0f}"
            ),
            showarrow=False,
            align="left",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )
        fig.add_annotation(
            xref="x2 domain",
            yref="y2 domain",
            x=0.98,
            y=0.98,
            text=f"Skew={skew_terminal:+.3f}<br>Excess kurt={ex_kurt_terminal:+.3f}",
            showarrow=False,
            align="right",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        lr = self.eng.log_ret.values
        jb_stat, jb_p = scipy_stats.jarque_bera(lr)
        lr_ex_kurt = float(scipy_stats.kurtosis(lr, fisher=True))
        fig.add_trace(go.Histogram(
            x=lr, nbinsx=60, name="Log-returns",
            histnorm="probability density",
            marker=dict(color=C["cyan"], line=dict(width=0)), opacity=0.68,
            hovertemplate="Return: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>"
        ), row=1, col=3)

        x_lr = np.linspace(lr.min(), lr.max(), 300)
        mu_r, sd_r = lr.mean(), lr.std()
        fig.add_trace(go.Scatter(
            x=x_lr, y=norm.pdf(x_lr, mu_r, sd_r), mode="lines", name="Normal fit",
            line=dict(color=C["amber"], width=2.0),
            hovertemplate="Return: %{x:.2%}<br>Normal PDF: %{y:.4f}<extra></extra>"
        ), row=1, col=3)
        df_t, loc_t, scale_t = t_dist.fit(lr)
        fig.add_trace(go.Scatter(
            x=x_lr, y=t_dist.pdf(x_lr, df_t, loc_t, scale_t), mode="lines",
            name=f"t-fit (nu={df_t:.1f})",
            line=dict(color=C["purple"], width=1.8, dash="dash"),
            hovertemplate="Return: %{x:.2%}<br>t-PDF: %{y:.4f}<extra></extra>"
        ), row=1, col=3)
        fig.add_annotation(
            xref="x3 domain",
            yref="y3 domain",
            x=0.98,
            y=0.98,
            text=f"JB p={jb_p:.3g}<br>Excess kurt={lr_ex_kurt:+.3f}<br>t dof={df_t:.2f}",
            showarrow=False,
            align="right",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        days = np.arange(st["paths"].shape[1])
        sample_n = min(50, st["paths"].shape[0])
        sample_idx = np.random.default_rng(1).choice(st["paths"].shape[0], sample_n, replace=False)

        for i in sample_idx:
            f   = st["paths"][i, -1]
            col = C["green"] if f > S0*1.3 else C["red"] if f < S0*0.7 else C["muted"]
            fig.add_trace(go.Scatter(
                x=days, y=st["paths"][i], mode="lines",
                line=dict(color=col, width=0.6),
                opacity=0.20, showlegend=False, hoverinfo="skip"
            ), row=2, col=1)




        # 25-75 fill band
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["75"], mode="lines",
            line=dict(color="rgba(88,166,255,0)", width=0), showlegend=False, hoverinfo="skip"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["25"], mode="lines", name="25-75% band",
            fill="tonexty", fillcolor="rgba(88,166,255,0.18)",
            line=dict(color="rgba(88,166,255,0)", width=0), hoverinfo="skip"
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=days, y=st["mean_path"], mode="lines", name="Mean path",
            line=dict(color=C["amber"], width=3),
            hovertemplate="Day %{x}: Mean=$%{y:,.2f}<extra></extra>"
        ), row=2, col=1)
        fig.add_hline(y=S0, line_color=C["red"], line_dash="dash", line_width=1.5,
                      row=2, col=1,
                      annotation_text=f"S0 ${S0:,.0f}", annotation_font=dict(color=C["red"], size=8),
                      annotation_position="top left")



        # 5-95 fill
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["95"], mode="lines",
            line=dict(color=C["green"], width=1.5, dash="dash"), name="95th pct",
            hovertemplate="Day %{x}: 95th=$%{y:,.2f}<extra></extra>"
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["5"], mode="lines", name="5th pct",
            fill="tonexty", fillcolor="rgba(88,166,255,0.12)",
            line=dict(color=C["red"], width=1.5, dash="dash"),
            hovertemplate="Day %{x}: 5th=$%{y:,.2f}<extra></extra>",
            showlegend=False
        ), row=2, col=2)
        # 25-75 fill
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["75"], mode="lines",
            line=dict(color=C["green"], width=0.8, dash="dot"), showlegend=False, hoverinfo="skip"
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["25"], mode="lines",
            fill="tonexty", fillcolor="rgba(88,166,255,0.22)",
            line=dict(color=C["red"], width=0.8, dash="dot"),
            name="25-75%", hoverinfo="skip",
            showlegend=False
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["50"], mode="lines", name="Median",
            line=dict(color=C["amber"], width=2.5),
            hovertemplate="Day %{x}: Median=$%{y:,.2f}<extra></extra>",
            showlegend=False
        ), row=2, col=2)

        T    = st["paths"].shape[1] / 252.0
        t_arr = np.linspace(0, T, len(days))
        e_path = self.eng.S0 * np.exp(self.eng.mu * t_arr)
        fig.add_trace(go.Scatter(
            x=days, y=e_path, mode="lines", name="E[S_t]",
            line=dict(color=C["purple"], width=1.5, dash="dot"),
            hovertemplate="Day %{x}: E[S]=$%{y:,.2f}<extra></extra>",
            showlegend=False
        ), row=2, col=2)
        fig.add_annotation(
            xref="x5 domain",
            yref="y5 domain",
            x=0.02,
            y=0.98,
            text="Fan view of simulated price percentiles around drift reference",
            showarrow=False,
            align="left",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        paths_arr = st["paths"]
        n_bins = 60
        pmin = paths_arr.min() * 0.97
        pmax = paths_arr.max() * 1.03
        pbins = np.linspace(pmin, pmax, n_bins+1)
        stride = max(1, paths_arr.shape[1]//100)
        t_idx  = np.arange(0, paths_arr.shape[1], stride)
        hm = np.zeros((n_bins, len(t_idx)))
        for col_i, t in enumerate(t_idx):
            c, _ = np.histogram(paths_arr[:, t], bins=pbins)
            total = c.sum()
            hm[:, col_i] = c / total if total > 0 else c

        fig.add_trace(go.Heatmap(
            z=hm,
            x=t_idx,
            y=0.5*(pbins[:-1]+pbins[1:]),
            colorscale=[
                [0.00, C["bg"]],
                [0.15, "#0f2d4a"],
                [0.35, "#1f6feb"],
                [0.55, C["cyan"]],
                [0.75, C["green"]],
                [0.90, C["amber"]],
                [1.00, "#ffd700"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Density", font=dict(color=C["muted"], size=8)),
                tickfont=dict(color=C["muted"], size=7),
                thickness=12,
                len=0.34,
                x=1.11,
                xanchor="left",
                y=0.47,
                yanchor="middle",
                bgcolor="rgba(22,27,34,0.8)",
                bordercolor=C["border"],
                borderwidth=1
            ),
            hovertemplate="Day %{x}: $%{y:,.0f} | density=%{z:.4f}<extra></extra>",
            name="Density"
        ), row=2, col=3)
        fig.add_trace(go.Scatter(
            x=days, y=st["mean_path"], mode="lines", name="Mean path",
            line=dict(color="white", width=2.5), showlegend=False,
            hovertemplate="Day %{x}: Mean=$%{y:,.2f}<extra></extra>"
        ), row=2, col=3)
        fig.add_annotation(
            xref="x6 domain",
            yref="y6 domain",
            x=0.02,
            y=0.98,
            text="Each day column sums to 1.0",
            showarrow=False,
            align="left",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        rv = self.eng.log_ret.rolling(30).std() * np.sqrt(252)
        ewma_var = self.eng.log_ret.pow(2).ewm(alpha=1 - 0.94, adjust=False).mean()
        ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)
        rv_hi = float(rv.quantile(0.75)) if rv.notna().any() else float("nan")
        rv_lo = float(rv.quantile(0.25)) if rv.notna().any() else float("nan")
        high_mask = (rv > rv_hi).fillna(False).to_numpy()
        low_mask = (rv < rv_lo).fillna(False).to_numpy()
        for start, end in _contiguous_true_ranges(high_mask):
            fig.add_vrect(x0=rv.index[start], x1=rv.index[end], fillcolor="rgba(248,81,73,0.10)", line_width=0, row=3, col=1)
        for start, end in _contiguous_true_ranges(low_mask):
            fig.add_vrect(x0=rv.index[start], x1=rv.index[end], fillcolor="rgba(63,185,80,0.10)", line_width=0, row=3, col=1)
        fig.add_trace(go.Scatter(
            x=rv.index, y=rv.values, mode="lines", name="30d rolling vol",
            line=dict(color=C["amber"], width=1.8),
            fill="tozeroy", fillcolor=f"rgba(227,179,65,0.10)",
            hovertemplate="Date: %{x}<br>Vol: %{y:.1%}<extra></extra>"
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=ewma_vol.index, y=ewma_vol.values, mode="lines", name="EWMA vol",
            line=dict(color=C["cyan"], width=1.6, dash="dot"),
            hovertemplate="Date: %{x}<br>EWMA Vol: %{y:.1%}<extra></extra>"
        ), row=3, col=1)
        fig.add_hline(y=self.eng.sigma, line_color=C["purple"], line_dash="dash",
                      row=3, col=1,
                      annotation_text=f"Full-period sigma={self.eng.sigma:.1%}",
                      annotation_font=dict(color=C["purple"], size=8),
                      annotation_position="top left")
        fig.add_annotation(
            xref="x7 domain",
            yref="y7 domain",
            x=0.98,
            y=0.98,
            text=f"Current={rv.dropna().iloc[-1]:.1%}<br>Low<{rv_lo:.1%}<br>High>{rv_hi:.1%}",
            showarrow=False,
            align="right",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        terminal_r = np.log(finals / S0)
        terminal_r_std = (terminal_r - terminal_r.mean()) / (terminal_r.std() + 1e-12)
        p = (np.arange(1, len(terminal_r_std) + 1) - 0.5) / len(terminal_r_std)
        q_theory = norm.ppf(p)
        q_sample = np.sort(terminal_r_std)
        tail_mask = (p <= 0.05) | (p >= 0.95)
        fig.add_trace(go.Scatter(
            x=q_theory[~tail_mask], y=q_sample[~tail_mask], mode="markers", name="Q-Q core",
            marker=dict(color=C["blue"], size=3.5, opacity=0.65),
            hovertemplate="Theoretical: %{x:.3f}<br>Terminal return z-score: %{y:.3f}<extra></extra>"
        ), row=3, col=2)
        fig.add_trace(go.Scatter(
            x=q_theory[tail_mask], y=q_sample[tail_mask], mode="markers", name="Q-Q tails",
            marker=dict(color=C["red"], size=4.5, opacity=0.75),
            hovertemplate="Tail quantile: %{x:.3f}<br>Terminal return z-score: %{y:.3f}<extra></extra>"
        ), row=3, col=2)
        ql = np.array([q_theory[0], q_theory[-1]])
        fig.add_trace(go.Scatter(
            x=ql, y=ql, mode="lines", name="45deg ref",
            line=dict(color=C["amber"], width=2.0),
        ), row=3, col=2)
        fig.add_vrect(x0=ql[0], x1=norm.ppf(0.05), fillcolor="rgba(248,81,73,0.08)", line_width=0, row=3, col=2)
        fig.add_vrect(x0=norm.ppf(0.95), x1=ql[-1], fillcolor="rgba(248,81,73,0.08)", line_width=0, row=3, col=2)
        fig.add_annotation(
            xref="x8 domain",
            yref="y8 domain",
            x=0.98,
            y=0.98,
            text=f"Excess kurt={float(scipy_stats.kurtosis(terminal_r, fisher=True)):+.3f}",
            showarrow=False,
            align="right",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        ecdf_x = np.sort(finals)
        ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)
        surv_y = np.maximum(1 - np.arange(0, len(ecdf_x)) / len(ecdf_x), 1 / len(ecdf_x))
        var_cdf = float(np.mean(finals <= var5_price))
        var_surv = max(1 - var_cdf, 1 / len(ecdf_x))
        fig.add_trace(go.Scatter(
            x=ecdf_x, y=ecdf_y, mode="lines", name="ECDF",
            line=dict(color=C["blue"], width=2.0, shape="hv"),
            hovertemplate="At S(T)=$%{x:,.2f}, P(S(T) <= x)=%{y:.2%}<extra></extra>"
        ), row=3, col=3, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=ecdf_x, y=surv_y, mode="lines", name="Survival",
            line=dict(color=C["red"], width=1.8, dash="dot", shape="hv"),
            hovertemplate="At S(T)=$%{x:,.2f}, P(S(T) > x)=%{y:.2%}<extra></extra>"
        ), row=3, col=3, secondary_y=True)
        fig.add_vline(x=var5_price, line_color=C["amber"], line_dash="dash", line_width=1.6, row=3, col=3)
        fig.add_hline(y=var_cdf, line_color=C["blue"], line_dash="dot", line_width=1.0, row=3, col=3)
        fig.add_hline(y=var_surv, line_color=C["red"], line_dash="dot", line_width=1.0, row=3, col=3, secondary_y=True)
        fig.add_annotation(
            xref="x9 domain",
            yref="y9 domain",
            x=0.98,
            y=0.98,
            text=f"VaR95=${var5_price:,.0f}<br>CVaR95=${cvar5_price:,.0f}",
            showarrow=False,
            align="right",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        ret_terminal = finals / S0 - 1.0
        pnl_terminal = finals - S0
        alpha_levels = [0.90, 0.95, 0.99]
        var_map = {a: float(np.quantile(ret_terminal, 1 - a)) for a in alpha_levels}
        cvar_map = {
            a: float(ret_terminal[ret_terminal <= var_map[a]].mean()) if np.any(ret_terminal <= var_map[a]) else float(var_map[a])
            for a in alpha_levels
        }
        hist_sigma = float(self.eng.log_ret.std() * np.sqrt(252))
        sim_sigma = float(np.std(ret_terminal))
        kde_ret = scipy_stats.gaussian_kde(ret_terminal)
        ret_x = np.linspace(ret_terminal.min(), ret_terminal.max(), 260)
        ret_y = kde_ret(ret_x)
        fig.add_trace(go.Histogram(
            x=ret_terminal, nbinsx=60, histnorm="probability density", name="Terminal returns",
            marker=dict(color=C["blue"], line=dict(width=0)), opacity=0.70,
            hovertemplate="Return: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>"
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=ret_x, y=ret_y, mode="lines", name="Return KDE",
            line=dict(color=C["cyan"], width=2.2),
            hovertemplate="Return: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>"
        ), row=4, col=1)
        fig.add_vrect(x0=ret_terminal.min(), x1=var_map[0.95], fillcolor="rgba(248,81,73,0.10)", line_width=0, row=4, col=1)
        fig.add_vrect(x0=ret_terminal.min(), x1=var_map[0.99], fillcolor="rgba(248,81,73,0.18)", line_width=0, row=4, col=1)
        fig.add_vline(x=var_map[0.95], line_color=C["amber"], line_dash="dash", line_width=1.8, row=4, col=1)
        fig.add_vline(x=var_map[0.99], line_color=C["red"], line_dash="dot", line_width=1.8, row=4, col=1)
        fig.add_annotation(
            xref="x10 domain", yref="y10 domain", x=0.98, y=0.98,
            text=(
                f"alpha=95%<br>VaR={var_map[0.95]:.2%}<br>CVaR={cvar_map[0.95]:.2%}<br>"
                f"Sharpe={st['sharpe']:+.2f}<br>E[r]/sigma={ret_terminal.mean()/(ret_terminal.std()+1e-12):+.2f}<br>"
                f"hist sigma={hist_sigma:.1%}<br>sim sigma={sim_sigma:.1%}"
            ),
            showarrow=False, align="right",
            bgcolor="rgba(13,17,23,0.78)", bordercolor=C["border"], borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        pcts = {p: float(np.percentile(finals, p)) for p in [1, 5, 25, 50, 75, 95, 99]}
        fig.add_trace(go.Box(
            q1=[pcts[25]], median=[pcts[50]], q3=[pcts[75]],
            lowerfence=[pcts[5]], upperfence=[pcts[95]],
            x=[pcts[50]], name="Percentile box", orientation="h",
            marker_color=C["amber"], line=dict(color=C["amber"], width=2),
            fillcolor="rgba(227,179,65,0.22)", boxpoints=False, showlegend=False,
            hoverinfo="skip",
        ), row=4, col=2)
        pct_x = [pcts[1], pcts[5], pcts[25], pcts[50], pcts[75], pcts[95], pcts[99]]
        pct_labels = ["P1", "P5", "P25", "P50", "P75", "P95", "P99"]
        pct_colors = [C["red"], C["red"], C["amber"], C["green"], C["amber"], C["red"], C["red"]]
        fig.add_trace(go.Scatter(
            x=pct_x, y=[0]*len(pct_x), mode="markers+text", name="Percentiles",
            marker=dict(color=pct_colors, size=[8,9,10,12,10,9,8], symbol="diamond"),
            text=[f"{lab}<br>${val:,.0f}" for lab, val in zip(pct_labels, pct_x)],
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ), row=4, col=2)

        p10 = np.percentile(paths_arr, 10, axis=0)
        p90 = np.percentile(paths_arr, 90, axis=0)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["95"], mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False, hoverinfo="skip"
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["5"], mode="lines", name="P5-P95 band",
            fill="tonexty", fillcolor="rgba(88,166,255,0.12)",
            line=dict(color="rgba(0,0,0,0)", width=0), hoverinfo="skip"
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=days, y=p90, mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False, hoverinfo="skip"
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=days, y=p10, mode="lines", name="P10-P90 band",
            fill="tonexty", fillcolor="rgba(88,166,255,0.20)",
            line=dict(color="rgba(0,0,0,0)", width=0), hoverinfo="skip"
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["75"], mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False, hoverinfo="skip"
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["25"], mode="lines", name="P25-P75 band",
            fill="tonexty", fillcolor="rgba(88,166,255,0.30)",
            line=dict(color="rgba(0,0,0,0)", width=0), hoverinfo="skip"
        ), row=5, col=1)
        fig.add_trace(go.Scatter(x=days, y=p10, mode="lines", name="Bear P10",
                                 line=dict(color=C["red"], width=1.8, dash="dash"),
                                 hovertemplate="Day %{x}: Bear=$%{y:,.2f}<extra></extra>"), row=5, col=1)
        fig.add_trace(go.Scatter(x=days, y=st["bands"]["50"], mode="lines", name="Base P50",
                                 line=dict(color=C["amber"], width=2.4),
                                 hovertemplate="Day %{x}: Base=$%{y:,.2f}<extra></extra>"), row=5, col=1)
        fig.add_trace(go.Scatter(x=days, y=p90, mode="lines", name="Bull P90",
                                 line=dict(color=C["green"], width=1.8, dash="dash"),
                                 hovertemplate="Day %{x}: Bull=$%{y:,.2f}<extra></extra>"), row=5, col=1)

        rows_left = [
            ("Last Close",   f"${S0:,.2f}"),
            ("Drift  mu",     f"{self.eng.mu:+.2%}"),
            ("Volatility sigma", f"{self.eng.sigma:.2%}"),
            ("Sharpe",       f"{st['sharpe']:+.3f}"),
            ("Sortino",      f"{st['sortino']:+.3f}"),
            ("Calmar",       f"{st['calmar']:+.3f}"),
            ("Win Rate",     f"{st['win_rate']:.1%}"),
            ("Avg Win",      f"+${st['avg_win']:,.2f}"),
            ("Mean stderr",  f"${st['mean_stderr']:,.4f}"),
            ("P(Profit) SE", f"{st['prob_up_stderr']:.3%}"),
        ]
        rows_right = [
            ("VaR 95%",     f"-${st['var95']:,.2f}"),
            ("VaR 99%",     f"-${st['var99']:,.2f}"),
            ("CVaR 95%",    f"-${st['cvar95']:,.2f}"),
            ("P(Profit)",   f"{st['prob_up']:.1%}"),
            ("P(Double)",   f"{st['prob_2x']:.1%}"),
            ("P(Halve)",    f"{st['prob_half']:.1%}"),
            ("Max DrawDn",  f"{st['hist_max_dd']:.2%}"),
            ("Avg Loss",    f"-${st['avg_loss']:,.2f}"),
            ("Conv slope",  f"{adv['convergence']['loglog_slope']:+.3f}" if adv else "n/a"),
            ("Multi corr",  f"{adv['multi_asset_emp_corr']:+.3f}" if adv else "n/a"),
        ]

        if adv:
            rows_left.extend([
                ("Euro Pseudo", f"${adv['euro_plain']['price']:,.4f} ± {adv['euro_plain']['stderr']:.4f}"),
                ("Euro Anti",   f"${adv['euro_antithetic']['price']:,.4f} ± {adv['euro_antithetic']['stderr']:.4f}"),
                ("Euro QMC",    f"${adv['euro_qmc']['price']:,.4f} ± {adv['euro_qmc']['stderr']:.4f}"),
                ("Euro CV",     f"${adv['euro_control_variate']['price']:,.4f} ± {adv['euro_control_variate']['stderr']:.4f}"),
                ("Asian Call",  f"${adv['asian_call']['price']:,.4f} ± {adv['asian_call']['stderr']:.4f}"),
                ("Barrier O/O", f"${adv['barrier_up_out_call']['price']:,.4f} ± {adv['barrier_up_out_call']['stderr']:.4f}"),
                ("American Put", f"${adv['american_put_lsmc']['price']:,.4f} ± {adv['american_put_lsmc']['stderr']:.4f}"),
            ])
            rows_right.extend([
                ("Delta",        f"{adv['pathwise_greeks']['delta']:+.4f} ± {adv['pathwise_greeks']['delta_stderr']:.4f}"),
                ("Vega",         f"{adv['pathwise_greeks']['vega']:+.4f} ± {adv['pathwise_greeks']['vega_stderr']:.4f}"),
                ("Rho",          f"{adv['pathwise_greeks']['rho']:+.4f} ± {adv['pathwise_greeks']['rho_stderr']:.4f}"),
                ("Anti ratio",   f"{adv['variance_reduction_ratio_antithetic']:.3f}"),
                ("QMC ratio",    f"{adv['variance_reduction_ratio_qmc']:.3f}"),
                ("CV ratio",     f"{adv['variance_reduction_ratio_cv']:.3f}"),
                ("BS call",      f"${adv['black_scholes_call']:,.4f}"),
            ])



        def metric_fill(metric: str, value: str) -> str:
            good = "#EAF3DE"
            bad = "#FCEBEB"
            neutral = "#FAEEDA"
            num = None
            try:
                num = float(value.replace("$","").replace(",","").replace("%","").replace("+","").replace("Â±"," ").split()[0])
            except Exception:
                return neutral
            if metric == "Sharpe":
                return good if num > 1 else neutral if num >= 0.5 else bad
            if metric == "Sortino":
                return good if num > 1.5 else neutral if num >= 0.8 else bad
            if metric == "Calmar":
                return good if num > 0.5 else neutral if num >= 0.2 else bad
            if metric.startswith("VaR") or metric.startswith("CVaR") or metric == "Avg Loss" or metric == "Max DrawDn":
                return bad
            if metric in {"P(Profit)", "P(Double)", "Win Rate"}:
                return good if num > 55 else neutral if num >= 40 else bad
            return neutral

        left_fills = [metric_fill(k, v) for k, v in rows_left]
        right_fills = [metric_fill(k, v) for k, v in rows_right]
        fig.add_trace(go.Table(
            header=dict(
                values=["<b>RISK / PERF</b>", "<b>VALUE</b>", "<b>PROB / EXTRA</b>", "<b>VALUE</b>"],
                fill_color=C["bg3"], font=dict(color=C["title"], size=10),
                line_color=C["border"], align="left",
                height=28,
            ),
            cells=dict(
                values=[
                    [r[0] for r in rows_left],
                    [r[1] for r in rows_left],
                    [r[0] for r in rows_right],
                    [r[1] for r in rows_right],
                ],
                fill_color=[
                    left_fills,
                    left_fills,
                    right_fills,
                    right_fills,
                ],
                font=dict(color=[C["muted"], C["cyan"], C["muted"], C["amber"]], size=9.5),
                line_color=C["border"], align="left", height=25,
            )
        ), row=4, col=3)

        layout = self._plotly_theme()
        layout.update(dict(
            title=dict(
                text=(f"<b>Monte Carlo Dashboard</b> - {self.eng.ticker}<br>"
                      f"<span style='font-size:11px;color:{C['muted']}'>mu={self.eng.mu:+.2%} | sigma={self.eng.sigma:.2%} | "
                      f"S0=${S0:,.2f} | {st['paths'].shape[0]} sims x {N}d</span>"),
                x=0.5, y=0.985, xanchor="center", yanchor="top",
                font=dict(size=15, color=C["title"])
            ),
            height=3020,
            margin=dict(l=75, r=135, t=135, b=155),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.04,
                xanchor="center",
                x=0.5,
                font=dict(size=8.5),
                bgcolor="rgba(13, 17, 23, 0.85)",
                bordercolor=C["border"],
                borderwidth=1,
                tracegroupgap=4,
                itemwidth=72
            ),
        ))
        fig.update_layout(**layout)
        for ann in fig.layout.annotations:
            if ann.text in {
                f"Historical Price  ({self.eng.ticker})",
                "Terminal Distribution S(T)",
                "Log-Return Hist  Normal vs t",
                "Monte Carlo Sample Paths",
                "Price Path Fan",
                "Density Heatmap (column-normalised)",
                "Rolling 30d Volatility Regimes",
                "Return Q-Q vs Normal",
                "CDF / Exceedance Curve",
                "Return / Risk Profile",
                "Percentile Summary",
                "Colour-coded Risk Metrics",
                "Scenario Fan Chart",
            }:
                ann.font = dict(size=12, color=C["title"])

        # axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, tickprefix="$")
        fig.update_yaxes(title_text="Density", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Log density", type="log", row=1, col=2, secondary_y=True)
        fig.update_xaxes(title_text="Return (%)", row=1, col=3,
                         tickformat=".1%")
        fig.update_xaxes(title_text="Trading Day", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1, tickprefix="$")
        fig.update_xaxes(title_text="Trading Day", row=2, col=2)
        fig.update_yaxes(title_text="Price ($)", row=2, col=2, tickprefix="$")
        fig.update_xaxes(title_text="Trading Day", row=2, col=3)
        fig.update_yaxes(title_text="Price ($)", row=2, col=3, tickprefix="$")
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Annual Vol", row=3, col=1, tickformat=".0%")
        fig.update_xaxes(title_text="Theoretical quantiles N(0,1)", row=3, col=2)
        fig.update_yaxes(title_text="Simulated terminal return z-score", row=3, col=2)
        fig.update_xaxes(title_text="Outcome value S(T)", row=3, col=3)
        fig.update_yaxes(title_text="P(S(T) <= x)", row=3, col=3, secondary_y=False)
        fig.update_yaxes(title_text="P(S(T) > x)", type="log", row=3, col=3, secondary_y=True)
        fig.update_xaxes(title_text="Terminal return", tickformat=".1%", row=4, col=1)
        fig.update_yaxes(title_text="Density", row=4, col=1)
        fig.update_xaxes(title_text="Outcome S(T)", row=4, col=2, tickprefix="$")
        fig.update_yaxes(visible=False, row=4, col=2)
        fig.update_xaxes(title_text="Trading Day", row=5, col=1)
        fig.update_yaxes(title_text="Price ($)", row=5, col=1, tickprefix="$")



        # grid styling
        for r in range(1, 6):
            for c in range(1, 4):
                if not ((r == 4 and c == 3) or (r == 5 and c != 1)):  # skip table and empty cells
                    fig.update_xaxes(
                        gridcolor=C["border"], zerolinecolor=C["border"],
                        row=r, col=c
                    )
                    fig.update_yaxes(
                        gridcolor=C["border"], zerolinecolor=C["border"],
                        row=r, col=c
                    )

        path_out = OUT / f"mc_dashboard_{self.eng.ticker}.html"
        fig.write_html(str(path_out), include_plotlyjs="cdn",
                       config={"displayModeBar": True, "scrollZoom": True})
        rlog(f"  [green]OK[/green] Dashboard -> [cyan]{path_out}[/cyan]")
        if show: fig.show()
        return fig

    def render_dashboard6(self, st: dict, N: int, show=True):
        h = self.eng.history
        S0 = float(self.eng.S0)
        ma20 = h.rolling(20).mean()
        std20 = h.rolling(20).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        rv = self.eng.log_ret.rolling(30).std() * np.sqrt(252)
        ewma_var = self.eng.log_ret.pow(2).ewm(alpha=1 - 0.94, adjust=False).mean()
        ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)
        finals = np.asarray(st["finals"], dtype=float)
        lr = self.eng.log_ret.dropna()
        paths_arr = np.asarray(st["paths"], dtype=float)
        days = np.arange(paths_arr.shape[1], dtype=int)

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Price + Bollinger Bands",
                "Final Distribution + KDE",
                "Log-Return Distribution",
                "MC Paths + Bands",
                "Rolling Volatility",
                "Q-Q Plot",
            ),
            vertical_spacing=0.11,
            horizontal_spacing=0.10,
        )

        price_custom = np.column_stack([
            ma20.to_numpy(dtype=float),
            bb_upper.to_numpy(dtype=float),
            bb_lower.to_numpy(dtype=float),
        ])
        fig.add_trace(go.Scatter(
            x=h.index, y=h.values, mode="lines", name="Close",
            line=dict(color=C["cyan"], width=2.2),
            customdata=price_custom,
            hovertemplate="Date: %{x|%d %b %Y}<br>Price: $%{y:.2f}<br>MA20: $%{customdata[0]:.2f}<br>BB Upper: $%{customdata[1]:.2f}<br>BB Lower: $%{customdata[2]:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=h.index, y=ma20, mode="lines", name="MA20",
            line=dict(color=C["blue"], width=1.5),
            customdata=np.column_stack([h.to_numpy(dtype=float)]),
            hovertemplate="Date: %{x|%d %b %Y}<br>MA20: $%{y:.2f}<br>Close: $%{customdata[0]:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=h.index, y=bb_upper, mode="lines", name="BB Upper",
            line=dict(color=C["purple"], width=1.0, dash="dot"),
            hovertemplate="Date: %{x|%d %b %Y}<br>BB Upper: $%{y:.2f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=h.index, y=bb_lower, mode="lines", name="BB Band",
            line=dict(color=C["purple"], width=1.0, dash="dot"),
            fill="tonexty", fillcolor="rgba(163,113,247,0.10)",
            hovertemplate="Date: %{x|%d %b %Y}<br>BB Lower: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        fd_bins = int(np.clip(np.sqrt(max(len(finals), 1)), 20, 100))
        kde_x = np.linspace(finals.min(), finals.max(), 240)
        kde_y = scipy_stats.gaussian_kde(finals)(kde_x)
        fig.add_trace(go.Histogram(
            x=finals, nbinsx=fd_bins, histnorm="probability density", name="Final prices",
            marker=dict(color=C["blue"], line=dict(width=0)),
            opacity=0.72,
            hovertemplate="Terminal price: $%{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y, mode="lines", name="KDE",
            line=dict(color=C["amber"], width=2.2),
            hovertemplate="Terminal price: $%{x:.2f}<br>KDE: %{y:.4f}<extra></extra>",
        ), row=1, col=2)

        x_lr = np.linspace(lr.min(), lr.max(), 260)
        mu_r = float(lr.mean())
        sd_r = float(lr.std() + 1e-12)
        fig.add_trace(go.Histogram(
            x=lr, nbinsx=60, histnorm="probability density", name="Log-return hist",
            marker=dict(color=C["cyan"], line=dict(width=0)), opacity=0.72,
            hovertemplate="Log return: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=x_lr, y=norm.pdf(x_lr, mu_r, sd_r), mode="lines", name="Normal fit",
            line=dict(color=C["red"], width=2.0),
            hovertemplate="Log return: %{x:.2%}<br>Normal PDF: %{y:.4f}<extra></extra>",
        ), row=2, col=1)

        sample_n = min(24, paths_arr.shape[0])
        sample_idx = np.random.default_rng(11).choice(paths_arr.shape[0], sample_n, replace=False)
        for path_id, idx in enumerate(sample_idx, start=1):
            path = paths_arr[idx]
            fig.add_trace(go.Scatter(
                x=days, y=path, mode="lines", name=f"Path {path_id}",
                line=dict(color="rgba(88,166,255,0.22)", width=1),
                hovertemplate="Day %{x}<br>Price: $%{y:.2f}<extra></extra>",
                showlegend=False,
            ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["95"], mode="lines", name="P95",
            line=dict(color="rgba(0,0,0,0)", width=0),
            hovertemplate="Day %{x}<br>P95: $%{y:.2f}<extra></extra>",
            showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["5"], mode="lines", name="P5-P95 band",
            fill="tonexty", fillcolor="rgba(88,166,255,0.12)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            hovertemplate="Day %{x}<br>P5: $%{y:.2f}<extra></extra>",
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=days, y=st["bands"]["50"], mode="lines", name="Median",
            line=dict(color=C["amber"], width=2.4),
            hovertemplate="Day %{x}<br>Median: $%{y:.2f}<extra></extra>",
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=days, y=st["mean_path"], mode="lines", name="Mean path",
            line=dict(color=C["green"], width=2.6),
            hovertemplate="Day %{x}<br>Mean: $%{y:.2f}<extra></extra>",
        ), row=2, col=2)

        vol_custom = np.column_stack([ewma_vol.to_numpy(dtype=float)])
        fig.add_trace(go.Scatter(
            x=rv.index, y=rv.values, mode="lines", name="Rolling vol",
            line=dict(color=C["amber"], width=1.8),
            customdata=vol_custom,
            hovertemplate="Date: %{x|%d %b %Y}<br>Rolling vol: %{y:.1%}<br>EWMA vol: %{customdata[0]:.1%}<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=ewma_vol.index, y=ewma_vol.values, mode="lines", name="EWMA vol",
            line=dict(color=C["purple"], width=1.6, dash="dot"),
            customdata=np.column_stack([rv.reindex(ewma_vol.index).to_numpy(dtype=float)]),
            hovertemplate="Date: %{x|%d %b %Y}<br>EWMA vol: %{y:.1%}<br>Rolling vol: %{customdata[0]:.1%}<extra></extra>",
        ), row=3, col=1)

        terminal_r = np.log(finals / S0)
        terminal_r_std = (terminal_r - terminal_r.mean()) / (terminal_r.std() + 1e-12)
        p = (np.arange(1, len(terminal_r_std) + 1) - 0.5) / len(terminal_r_std)
        q_theory = norm.ppf(p)
        q_sample = np.sort(terminal_r_std)
        fig.add_trace(go.Scatter(
            x=q_theory, y=q_sample, mode="markers", name="Q-Q sample",
            marker=dict(color=C["blue"], size=4, opacity=0.68),
            hovertemplate="Theoretical z: %{x:.3f}<br>Sample z: %{y:.3f}<extra></extra>",
        ), row=3, col=2)
        ql = np.array([q_theory[0], q_theory[-1]])
        fig.add_trace(go.Scatter(
            x=ql, y=ql, mode="lines", name="45deg ref",
            line=dict(color=C["red"], width=1.8, dash="dash"),
            hovertemplate="Theoretical z: %{x:.3f}<br>Reference: %{y:.3f}<extra></extra>",
        ), row=3, col=2)

        layout = self._plotly_theme()
        layout.update(dict(
            title=dict(
                text=(f"<b>Quant 6-Panel Dashboard</b> - {self.eng.ticker}<br>"
                      f"<span style='font-size:11px;color:{C['muted']}'>Cross-hover on shared date panels | {paths_arr.shape[0]} sims x {N}d</span>"),
                x=0.5, y=0.985, xanchor="center", yanchor="top",
                font=dict(size=15, color=C["title"])
            ),
            height=1580,
            margin=dict(l=70, r=70, t=125, b=90),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5,
                        font=dict(size=9), bgcolor="rgba(13,17,23,0.82)", bordercolor=C["border"], borderwidth=1),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.01,
                    y=1.10,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[
                        dict(label="Linear Price", method="relayout", args=[{"yaxis.type": "linear"}]),
                        dict(label="Log Price", method="relayout", args=[{"yaxis.type": "log"}]),
                    ],
                )
            ],
        ))
        fig.update_layout(**layout)

        fig.update_xaxes(matches="x", row=3, col=1)
        for row, col in [(1, 1), (3, 1)]:
            fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=row, col=col,
                             showspikes=True, spikemode="across", spikesnap="cursor",
                             spikecolor=C["blue"], spikethickness=1)
            fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=row, col=col)
        for row, col in [(1, 2), (2, 1), (2, 2), (3, 2)]:
            fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=row, col=col)
            fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=row, col=col)

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Terminal price", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=2)
        fig.update_xaxes(title_text="Log return", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_xaxes(title_text="Day", row=2, col=2)
        fig.update_yaxes(title_text="Price ($)", row=2, col=2)
        fig.update_xaxes(title_text="Calendar date", row=3, col=1)
        fig.update_yaxes(title_text="Annualized vol", tickformat=".0%", row=3, col=1)
        fig.update_xaxes(title_text="Theoretical z", row=3, col=2)
        fig.update_yaxes(title_text="Sample z", row=3, col=2)

        path_out = OUT / f"mc_dashboard6_{self.eng.ticker}.html"
        fig.write_html(str(path_out), include_plotlyjs="cdn",
                       config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False})
        rlog(f"  [green]OK[/green] 6-panel dashboard -> [cyan]{path_out}[/cyan]")
        if show:
            fig.show(config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False})
        return fig

    def render_3d(self, paths: np.ndarray, N: int, show=True, no_anim=False):
        """Generate density, path-surface, explorer, and animated 3D Monte Carlo views."""
        S0 = float(self.eng.S0)
        Z = np.asarray(paths, dtype=float)
        n = int(Z.shape[0])
        total_days = int(Z.shape[1] - 1)
        x = np.arange(total_days + 1, dtype=float)
        y = np.arange(n, dtype=float)
        mean_path = Z.mean(axis=0)

        z_low, z_high = _surface_z_bounds(Z, S0)
        x_span = max(float(total_days), 1.0)
        y_span = max(float(n - 1), 1.0)
        base_span = max(x_span, y_span)
        z_ratio = max(float(z_high - z_low), 1e-9) / max(base_span, 1e-9)
        z_ratio = float(np.clip(z_ratio, 0.25, 0.95))

        n_bins = int(np.clip(np.sqrt(n), 28, 64))
        price_low = float(np.percentile(Z, 0.5))
        price_high = float(np.percentile(Z, 99.5))
        if not np.isfinite(price_low) or not np.isfinite(price_high) or price_high <= price_low:
            price_low, price_high = z_low, z_high
        price_edges = np.linspace(price_low, price_high, n_bins + 1)
        price_centers = 0.5 * (price_edges[:-1] + price_edges[1:])
        density = np.zeros((len(x), n_bins), dtype=float)
        ridge_idx = np.zeros(len(x), dtype=int)
        mean_density = np.zeros(len(x), dtype=float)
        for j in range(len(x)):
            hist, _ = np.histogram(Z[:, j], bins=price_edges, density=True)
            density[j] = hist
            ridge_idx[j] = int(np.argmax(hist))
            mean_density[j] = float(np.interp(mean_path[j], price_centers, hist, left=0.0, right=0.0))

        T_grid, P_grid = np.meshgrid(x, price_centers, indexing="ij")
        density_max = float(max(np.max(density), 1e-9))
        ridge_prices = price_centers[ridge_idx]

        if HAS_MPL:

            fig_density = plt.figure(figsize=(12, 7.2))
            ax = fig_density.add_subplot(111, projection="3d")
            surf = ax.plot_surface(T_grid, P_grid, density, cmap="plasma", linewidth=0, antialiased=True, alpha=0.96)
            ax.plot_wireframe(T_grid, P_grid, density, rstride=max(1, len(x) // 18), cstride=max(1, n_bins // 14),
                              color="white", linewidth=0.35, alpha=0.15)
            ax.contourf(T_grid, P_grid, density, zdir="z", offset=0.0, levels=18, cmap="plasma", alpha=0.9)
            ax.plot(x, mean_path, mean_density, color="#ff4d4f", linewidth=2.8, label="Mean path")
            ax.plot(x, ridge_prices, density[np.arange(len(x)), ridge_idx], color="white", linewidth=2.0,
                    linestyle="--", label="Peak density ridge")

            plane_x = np.array([[x[0], x[-1]], [x[0], x[-1]]], dtype=float)
            plane_y = np.array([[S0, S0], [S0, S0]], dtype=float)
            plane_z = np.array([[0.0, 0.0], [density_max, density_max]], dtype=float)
            ax.plot_surface(plane_x, plane_y, plane_z, color="#58a6ff", alpha=0.12, linewidth=0, shade=False)

            ax.set_title(f"{self.eng.ticker} Density Surface Over Time", fontsize=11, pad=14)
            ax.set_xlabel("Time step t", fontsize=9, labelpad=8)
            ax.set_ylabel("Price S(t)", fontsize=9, labelpad=8)
            ax.set_zlabel("Probability density f(S,t)", fontsize=9, labelpad=8)
            ax.tick_params(axis="both", which="major", labelsize=8, pad=1)
            ax.tick_params(axis="z", which="major", labelsize=8, pad=2)
            ax.view_init(elev=25, azim=-60)
            fig_density.colorbar(surf, ax=ax, shrink=0.68, pad=0.08, label="Density")
            ax.legend(loc="upper left", fontsize=8)
            fig_density.tight_layout()
            plt.close(fig_density)

        fig_density_html = go.Figure()
        fig_density_html.add_trace(
            go.Surface(
                x=T_grid,
                y=P_grid,
                z=density,
                colorscale="Plasma",
                opacity=0.96,
                colorbar=dict(title="Density", len=0.68),
                contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True), width=1)),
                hovertemplate="Day=%{x}<br>Price=$%{y:,.2f}<br>Density=%{z:.5f}<extra></extra>",
                name="Density surface",
            )
        )
        for price_idx in range(0, n_bins, max(1, n_bins // 12)):
            fig_density_html.add_trace(
                go.Scatter3d(
                    x=x,
                    y=np.full(len(x), price_centers[price_idx], dtype=float),
                    z=density[:, price_idx],
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.18)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig_density_html.add_trace(
            go.Scatter3d(
                x=x,
                y=mean_path,
                z=mean_density,
                mode="lines",
                line=dict(color=C["red"], width=5),
                name="Mean path",
                hovertemplate="Day=%{x}<br>Mean=$%{y:,.2f}<br>Density=%{z:.5f}<extra></extra>",
            )
        )
        fig_density_html.add_trace(
            go.Scatter3d(
                x=x,
                y=ridge_prices,
                z=density[np.arange(len(x)), ridge_idx],
                mode="lines",
                line=dict(color="white", width=4, dash="dash"),
                name="Peak density ridge",
                hovertemplate="Day=%{x}<br>Ridge=$%{y:,.2f}<br>Density=%{z:.5f}<extra></extra>",
            )
        )
        fig_density_html.add_trace(
            go.Surface(
                x=np.array([[x[0], x[-1]], [x[0], x[-1]]], dtype=float),
                y=np.array([[S0, S0], [S0, S0]], dtype=float),
                z=np.array([[0.0, 0.0], [density_max, density_max]], dtype=float),
                showscale=False,
                opacity=0.12,
                colorscale=[[0.0, C["blue"]], [1.0, C["blue"]]],
                hoverinfo="skip",
                name="S0 plane",
            )
        )
        fig_density_html.update_layout(
            **self._plotly_theme(),
            scene=dict(
                xaxis=dict(title="Time step t", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"]),
                yaxis=dict(title="Price S(t)", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"], range=[price_low, price_high]),
                zaxis=dict(title="Probability density f(S,t)", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"], range=[0, density_max * 1.03]),
                aspectmode="manual",
                aspectratio=dict(x=1.25, y=1.0, z=0.62),
                camera=dict(eye=dict(x=1.7, y=-1.75, z=1.0)),
                dragmode="orbit",
            ),
            margin=dict(l=20, r=20, b=20, t=96),
            title=dict(
                text=(
                    f"<b>Monte Carlo Density Surface</b> - {self.eng.ticker}<br>"
                    f"{n} paths x {total_days} days - widening lognormal cone"
                ),
                x=0.5, y=0.97, xanchor="center", yanchor="top",
                font=dict(size=17, color=C["title"]),
            ),
            scene_hovermode="closest",
        )
        density_html_path = OUT / f"mc_density_surface_{self.eng.ticker}.html"
        fig_density_html.write_html(
            str(density_html_path),
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
        )
        rlog(f"  [green]OK[/green] 3D density surface (html) -> [cyan]{density_html_path}[/cyan]")

        s0_plane_z = np.full((2, len(x)), S0, dtype=float)
        s0_plane_y = np.vstack([np.zeros(len(x)), np.full(len(x), max(y_span, 1.0))])
        s0_plane_x = np.vstack([x, x])

        surface_theme = dict(
            scene=dict(
                xaxis=dict(title="Day index", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"]),
                yaxis=dict(title="Simulation index", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"]),
                zaxis=dict(title="Price S(t)", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"], range=[z_low, z_high]),
                aspectmode="manual",
                aspectratio=dict(x=1.35, y=max(0.45, y_span / x_span), z=z_ratio),
                camera=dict(eye=dict(x=1.75, y=1.7, z=0.95)),
                dragmode="orbit",
            ),
            margin=dict(l=20, r=20, b=20, t=96),
            title=dict(
                text=(
                    f"<b>Monte Carlo Path Surface</b> - {self.eng.ticker}<br>"
                    f"{n} paths x {total_days} days - hover any (day, path) point"
                ),
                x=0.5, y=0.97, xanchor="center", yanchor="top",
                font=dict(size=17, color=C["title"]),
            ),
        )

        fig_surface = go.Figure()
        fig_surface.add_trace(
            go.Surface(
                x=x, y=y, z=Z, colorscale="Plasma", opacity=0.96,
                colorbar=dict(title="Price ($)", len=0.68),
                hovertemplate="Day=%{x}<br>Path=%{y}<br>Price=$%{z:,.2f}<extra></extra>",
                name="Path surface",
                showscale=True,
            )
        )
        fig_surface.add_trace(
            go.Surface(
                x=s0_plane_x,
                y=s0_plane_y,
                z=s0_plane_z,
                showscale=False,
                opacity=0.14,
                colorscale=[[0.0, C["blue"]], [1.0, C["blue"]]],
                hoverinfo="skip",
                name="S0 plane",
            )
        )
        fig_surface.add_trace(
            go.Scatter3d(
                x=x,
                y=np.full_like(x, y_span / 2.0, dtype=float),
                z=mean_path,
                mode="lines",
                line=dict(color=C["red"], width=4),
                name="Mean path",
                hovertemplate="Day=%{x}<br>Mean=$%{z:,.2f}<extra></extra>",
            )
        )
        fig_surface.update_layout(
            **self._plotly_theme(),
            **surface_theme,
            scene_hovermode="closest",
        )

        path_surf = OUT / f"mc_surface_{self.eng.ticker}.html"
        fig_surface.write_html(
            str(path_surf),
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
        )
        rlog(f"  [green]OK[/green] 3D interactive surface -> [cyan]{path_surf}[/cyan]")
        if show:
            fig_surface.show(config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False})

        n_paths = n
        sample_paths = Z

        fig_paths = go.Figure()
        for i in range(n_paths):
            final_val = sample_paths[i, -1]
            line_color = C["green"] if final_val > S0 * 1.35 else C["red"] if final_val < S0 * 0.75 else C["blue"]
            fig_paths.add_trace(
                go.Scatter3d(
                    x=x,
                    y=np.full_like(x, i, dtype=float),
                    z=sample_paths[i],
                    mode="lines",
                    line=dict(color=line_color, width=3),
                    opacity=0.72,
                    showlegend=False,
                    hovertemplate="Day=%{x}<br>Path=#" + str(i + 1) + "<br>Price=$%{z:,.2f}<extra></extra>",
                )
            )

        fig_paths.add_trace(
            go.Scatter3d(
                x=x,
                y=np.full_like(x, (n_paths - 1) / 2.0, dtype=float),
                z=sample_paths.mean(axis=0),
                mode="lines",
                line=dict(color=C["amber"], width=9),
                name="Sample mean",
                hovertemplate="Day=%{x}<br>Mean=$%{z:,.2f}<extra></extra>",
            )
        )
        fig_paths.add_trace(
            go.Scatter3d(
                x=x,
                y=np.full_like(x, (n_paths - 1) * 0.08, dtype=float),
                z=np.full_like(x, S0, dtype=float),
                mode="lines",
                line=dict(color=C["red"], width=5, dash="dash"),
                name="S0 reference",
                hovertemplate="Day=%{x}<br>S0=$%{z:,.2f}<extra></extra>",
            )
        )

        y_ratio_paths = max(float(n_paths - 1), 1.0) / x_span
        fig_paths.update_layout(
            **self._plotly_theme(),
            scene=dict(
                xaxis=dict(title="Days", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"]),
                yaxis=dict(title="Path #", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"]),
                zaxis=dict(title="Price ($)", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"], range=[z_low, z_high]),
                aspectmode="manual",
                aspectratio=dict(x=1.45, y=max(0.36, y_ratio_paths), z=max(0.28, z_ratio * 0.92)),
                camera=dict(eye=dict(x=1.65, y=-1.35, z=0.78)),
            ),
            margin=dict(l=20, r=20, b=20, t=96),
            title=dict(
                text=(
                    f"<b>3D Path Explorer</b> - {self.eng.ticker}<br>"
                    f"{n_paths} sampled paths - drag to rotate - zoom/pan enabled"
                ),
                x=0.5, y=0.97, xanchor="center", yanchor="top",
                font=dict(size=17, color=C["title"]),
            ),
            scene_hovermode="closest",
        )

        path_anim = OUT / f"mc_3d_{self.eng.ticker}.html"
        fig_paths.write_html(
            str(path_anim),
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
        )
        rlog(f"  [green]OK[/green] 3D movable path explorer -> [cyan]{path_anim}[/cyan]")
        if show:
            fig_paths.show(config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False})

        if no_anim:
            return

        frames = []
        for t in range(len(x)):
            frame_traces = []
            current_vals = sample_paths[:, t]
            above_count = int(np.sum(current_vals > S0))
            below_count = int(n_paths - above_count)
            for i in range(n_paths):
                final_val = sample_paths[i, -1]
                line_color = C["green"] if final_val > S0 else C["red"]
                frame_traces.append(
                    go.Scatter3d(
                        x=x[: t + 1],
                        y=np.full(t + 1, i, dtype=float),
                        z=sample_paths[i, : t + 1],
                        mode="lines",
                        line=dict(color=line_color, width=3),
                        opacity=0.78,
                        showlegend=False,
                        hovertemplate="Day=%{x}<br>Path=#" + str(i + 1) + "<br>Price=$%{z:,.2f}<extra></extra>",
                    )
                )
            frame_traces.append(
                go.Scatter3d(
                    x=x[: t + 1],
                    y=np.full(t + 1, (n_paths - 1) / 2.0, dtype=float),
                    z=sample_paths.mean(axis=0)[: t + 1],
                    mode="lines",
                    line=dict(color=C["amber"], width=7),
                    name="Mean path",
                    hovertemplate="Day=%{x}<br>Mean=$%{z:,.2f}<extra></extra>",
                )
            )
            frame_traces.append(
                go.Scatter3d(
                    x=x[: t + 1],
                    y=np.full(t + 1, (n_paths - 1) * 0.08, dtype=float),
                    z=np.full(t + 1, S0, dtype=float),
                    mode="lines",
                    line=dict(color=C["blue"], width=4, dash="dash"),
                    name="S0",
                    hovertemplate="Day=%{x}<br>S0=$%{z:,.2f}<extra></extra>",
                )
            )
            frames.append(
                go.Frame(
                    name=str(t),
                    data=frame_traces,
                    layout=go.Layout(
                        title=dict(
                            text=(
                                f"<b>3D Path Reveal</b> - {self.eng.ticker} | Day {t}/{total_days}<br>"
                                f"<span style='font-size:11px;color:{C['muted']}'>Above S0: {above_count} | Below S0: {below_count}</span>"
                            )
                        )
                    ),
                )
            )

        fig_anim = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames,
        )
        fig_anim.update_layout(
            **self._plotly_theme(),
            scene=dict(
                xaxis=dict(title="Day index", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"], range=[0, total_days]),
                yaxis=dict(title="Simulation index", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"], range=[0, max(n_paths - 1, 1)]),
                zaxis=dict(title="Price S(t)", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"], range=[z_low, z_high]),
                aspectmode="manual",
                aspectratio=dict(x=1.5, y=max(0.52, y_span / x_span), z=max(0.34, z_ratio * 0.88)),
                camera=dict(eye=dict(x=2.1, y=-1.95, z=1.15)),
                dragmode="orbit",
            ),
            margin=dict(l=20, r=20, b=20, t=104),
            title=dict(
                text=(
                    f"<b>3D Path Reveal</b> - {self.eng.ticker} | Day 0/{total_days}<br>"
                    f"<span style='font-size:11px;color:{C['muted']}'>Above S0: {int(np.sum(sample_paths[:, 0] > S0))} | Below S0: {int(np.sum(sample_paths[:, 0] <= S0))}</span>"
                ),
                x=0.5, y=0.97, xanchor="center", yanchor="top",
                font=dict(size=17, color=C["title"]),
            ),
            scene_hovermode="closest",
            uirevision="mc-3d-animation",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02,
                    y=1.02,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=35, redraw=True),
                                             transition=dict(duration=0),
                                             fromcurrent=True, mode="immediate")],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], dict(frame=dict(duration=0, redraw=False),
                                               transition=dict(duration=0),
                                               mode="immediate")],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.12,
                    y=0.02,
                    len=0.82,
                    currentvalue=dict(prefix="Day: ", font=dict(color=C["text"])),
                    steps=[
                        dict(
                            label=str(t),
                            method="animate",
                            args=[[str(t)], dict(frame=dict(duration=0, redraw=True),
                                                 transition=dict(duration=0), mode="immediate")],
                        )
                        for t in range(len(x))
                    ],
                )
            ],
        )

        anim_path = OUT / f"mc_3d_animation_{self.eng.ticker}.html"
        fig_anim.write_html(
            str(anim_path),
            include_plotlyjs="cdn",
            auto_play=False,
            config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
        )
        rlog(f"  [green]OK[/green] 3D animation -> [cyan]{anim_path}[/cyan]")
        if show:
            fig_anim.show(config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False})

    def render_mc_diagnostics(self, adv: dict, show=True):
        bs = float(adv.get("black_scholes_call", np.nan))
        horizon_days = int(adv.get("horizon_days", 252))
        profiles = adv.get("convergence_profiles", {})
        pseudo = profiles.get("pseudo", {})
        anti = profiles.get("antithetic", {})
        qmc_prof = profiles.get("qmc", {})
        n_grid = np.array(pseudo.get("n_grid", adv["convergence"]["n_grid"]), dtype=float)
        eps = float(pseudo.get("epsilon", max(abs(bs) * 0.01, 0.01)))
        min_n = pseudo.get("min_n_epsilon")
        final_n = int(pseudo.get("final_n", 0) or 0)
        final_mcse = float(pseudo.get("final_mcse", np.nan))
        final_std = float(pseudo.get("final_std", np.nan))

        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=(
                "Convergence Bands: Running Mean ± 2MCSE",
                "Convergence Slope: log|error| vs log(N)",
                "Trace Plot: Running Mean ± 2MCSE",
                "Trace Plot: Running Sigma",
                "ESS / N Applicability",
                "Advanced Option Prices",
                "Simulated vs Historical Overlay",
                "Variance Reduction Ratios",
            ),
            vertical_spacing=0.11,
            horizontal_spacing=0.10,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
        )
        colors = {"pseudo": C["cyan"], "antithetic": C["green"], "qmc": C["amber"]}
        labels = {"pseudo": "Standard MC", "antithetic": "Antithetic MC", "qmc": "Sobol QMC"}

        for key, prof in [("pseudo", pseudo), ("antithetic", anti), ("qmc", qmc_prof)]:
            grid = np.array(prof.get("n_grid", []), dtype=float)
            if len(grid) == 0:
                continue
            est = np.array(prof["estimates"], dtype=float)
            se = np.array(prof["stderrs"], dtype=float)
            err = np.array(prof["errors"], dtype=float)
            fig.add_trace(go.Scatter(
                x=grid, y=est, mode="lines+markers", name=labels[key],
                line=dict(color=colors[key], width=2), marker=dict(size=6),
                error_y=dict(type="data", array=2 * se, visible=True, color=colors[key], thickness=1),
                hovertemplate=f"{labels[key]}<br>N=%{{x}}<br>Mean=$%{{y:.4f}}<br>2MCSE=%{{error_y.array:.4f}}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=grid, y=np.clip(err, 1e-12, None), mode="lines+markers", name=f"{labels[key]} |error|",
                line=dict(color=colors[key], width=2), marker=dict(size=6), showlegend=False,
                hovertemplate=f"{labels[key]}<br>N=%{{x}}<br>|error|=%{{y:.5f}}<extra></extra>",
            ), row=1, col=2)
        if np.isfinite(bs):
            fig.add_hline(
                y=bs,
                line_color=C["amber"],
                line_dash="dash",
                row=1,
                col=1,
                annotation_text=f"BS ${bs:.4f}",
                annotation_font=dict(color=C["amber"], size=9),
            )
            fig.add_hrect(y0=bs - eps, y1=bs + eps, fillcolor="rgba(63,185,80,0.08)", line_width=0, row=1, col=1)
        if len(n_grid):
            ref = np.clip(np.array(pseudo.get("errors", [1.0]), dtype=float), 1e-12, None)
            ref_line = ref[0] * np.sqrt(n_grid[0] / n_grid)
            fig.add_trace(go.Scatter(
                x=n_grid,
                y=ref_line,
                mode="lines",
                name="Slope -1/2 reference",
                line=dict(color=C["purple"], width=2, dash="dash"),
                hovertemplate="N=%{x}<br>Ref=%{y:.5f}<extra></extra>",
            ), row=1, col=2)
        if min_n is not None:
            fig.add_vline(x=min_n, line_color=C["green"], line_dash="dot", line_width=1.4, row=1, col=1)
            fig.add_vline(x=min_n, line_color=C["green"], line_dash="dot", line_width=1.4, row=1, col=2)
            fig.add_annotation(
                x=min_n,
                y=0.98,
                xref="x2",
                yref="y2 domain",
                text=f"min N in ε-band: {min_n:,}",
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(13,17,23,0.82)",
                bordercolor=C["green"],
                borderwidth=1,
                font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
            )
        fig.add_annotation(
            xref="x2 domain", yref="y2 domain", x=0.98, y=0.06,
            text=(
                f"Std={pseudo.get('loglog_slope', float('nan')):+.3f}<br>"
                f"Anti={anti.get('loglog_slope', float('nan')):+.3f}<br>"
                f"Sobol={qmc_prof.get('loglog_slope', float('nan')):+.3f}"
            ),
            showarrow=False, align="right",
            bgcolor="rgba(13,17,23,0.78)", bordercolor=C["border"], borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        running_n = np.arange(1, len(pseudo.get("running_mean", [])) + 1, dtype=float)
        running_mean = np.array(pseudo.get("running_mean", []), dtype=float)
        running_std = np.array(pseudo.get("running_std", []), dtype=float)
        running_mcse = np.array(pseudo.get("running_mcse", []), dtype=float)
        if len(running_n):
            fig.add_trace(go.Scatter(
                x=running_n, y=running_mean + 2 * running_mcse, mode="lines",
                line=dict(color="#9aa4b2", width=0), showlegend=False, hoverinfo="skip"
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=running_n, y=running_mean - 2 * running_mcse, mode="lines",
                fill="tonexty", fillcolor="rgba(154,164,178,0.18)",
                line=dict(color="#9aa4b2", width=0), name="±2MCSE band", hoverinfo="skip"
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=running_n, y=running_mean, mode="lines", name="Running mean",
                line=dict(color=C["cyan"], width=2),
                hovertemplate="N=%{x}<br>Running mean=$%{y:.4f}<extra></extra>"
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=running_n, y=running_std, mode="lines", name="Running sigma",
                line=dict(color=C["green"], width=2),
                hovertemplate="N=%{x}<br>Running sigma=$%{y:.4f}<extra></extra>"
            ), row=2, col=2)
            if np.isfinite(bs):
                fig.add_hline(y=bs, line_color=C["amber"], line_dash="dash", row=2, col=1)
            if min_n is not None:
                fig.add_vline(x=min_n, line_color=C["green"], line_dash="dot", line_width=1.4, row=2, col=1)
                fig.add_vline(x=min_n, line_color=C["green"], line_dash="dot", line_width=1.4, row=2, col=2)
            if final_n > 1 and np.isfinite(final_mcse):
                fig.add_annotation(
                    xref="x3 domain", yref="y3 domain", x=0.98, y=0.98,
                    text=f"Final MCSE @ N={final_n:,}: ±{2 * final_mcse:.4f}<br>Final σ={final_std:.4f}",
                    showarrow=False, align="right",
                    bgcolor="rgba(13,17,23,0.78)", bordercolor=C["border"], borderwidth=1,
                    font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
                )

        ess_names = ["Pseudo MC", "Antithetic", "Sobol QMC"]
        fig.add_trace(go.Bar(
            x=[1.0, 1.0, 1.0], y=ess_names, orientation="h",
            marker_color=["rgba(250,238,218,0.16)"] * 3,
            marker_line_color=[C["border"]] * 3,
            marker_line_width=1,
            text=["Not applicable", "Not applicable", "Not applicable"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color=C["text"], size=10),
            hovertemplate="%{y}<br>ESS/N not applicable for iid or low-discrepancy paths<extra></extra>",
            name="ESS status",
            showlegend=False,
        ), row=3, col=1)
        fig.add_annotation(
            xref="x5 domain", yref="y5 domain", x=0.98, y=0.98,
            text="ESS applies to MCMC chains.<br>GBM pseudo, antithetic, and Sobol are independent samplers, not chain states.",
            showarrow=False, align="right",
            bgcolor="rgba(13,17,23,0.78)", bordercolor=C["border"], borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )
        fig.add_annotation(
            xref="x5 domain", yref="y5 domain", x=0.02, y=0.98,
            text="Use RNG ACF / Q-Q checks instead of ESS here",
            showarrow=False, align="left",
            bgcolor="rgba(13,17,23,0.78)", bordercolor=C["border"], borderwidth=1,
            font=dict(color=C["muted"], size=9, family="Consolas, Menlo, monospace"),
        )

        price_labels = ["Euro Pseudo", "Euro Anti", "Euro QMC", "Euro CV", "Asian", "Barrier", "American"]
        price_vals = [
            adv["euro_plain"]["price"],
            adv["euro_antithetic"]["price"],
            adv["euro_qmc"]["price"],
            adv["euro_control_variate"]["price"],
            adv["asian_call"]["price"],
            adv["barrier_up_out_call"]["price"],
            adv["american_put_lsmc"]["price"],
        ]
        price_se = [
            adv["euro_plain"]["stderr"],
            adv["euro_antithetic"]["stderr"],
            adv["euro_qmc"]["stderr"],
            adv["euro_control_variate"]["stderr"],
            adv["asian_call"]["stderr"],
            adv["barrier_up_out_call"]["stderr"],
            adv["american_put_lsmc"]["stderr"],
        ]
        fig.add_trace(go.Bar(
            x=price_labels,
            y=price_vals,
            marker=dict(color=[C["cyan"], C["blue"], C["green"], C["amber"], C["purple"], C["red"], C["muted"]]),
            error_y=dict(type="data", array=price_se, visible=True, thickness=1),
            name="Price +- SE",
            hovertemplate="%{x}<br>Price=$%{y:.4f}<extra></extra>",
        ), row=3, col=2)

        horizon = max(1, min(horizon_days, len(self.eng.history) - 1, 252))
        hist_ret = np.log(self.eng.history / self.eng.history.shift(horizon)).dropna().values
        sim_terminal_ret = np.log(
            np.asarray(self.eng.simulate(N=horizon, num_sim=4000, seed=909)[0][:, -1]) / self.eng.S0
        )
        if len(hist_ret) > 5 and len(sim_terminal_ret) > 10:
            sim_q5 = float(np.quantile(sim_terminal_ret, 0.05))
            sim_q95 = float(np.quantile(sim_terminal_ret, 0.95))
            sim_q1 = float(np.quantile(sim_terminal_ret, 0.01))
            sim_q99 = float(np.quantile(sim_terminal_ret, 0.99))
            hist_in_band = float(np.mean((hist_ret >= sim_q5) & (hist_ret <= sim_q95)))
            breach_vals = hist_ret[hist_ret < sim_q5]
            tail_flag = bool((hist_ret.min() < sim_q1) or (hist_ret.max() > sim_q99))
            x_hist = np.linspace(min(hist_ret.min(), sim_terminal_ret.min()), max(hist_ret.max(), sim_terminal_ret.max()), 260)
            kde_hist = scipy_stats.gaussian_kde(hist_ret)
            kde_sim = scipy_stats.gaussian_kde(sim_terminal_ret)
            fig.add_trace(go.Scatter(
                x=x_hist, y=kde_sim(x_hist), mode="lines", name="Simulated KDE",
                line=dict(color=C["blue"], width=2.4),
                hovertemplate="Return: %{x:.2%}<br>Sim density=%{y:.4f}<extra></extra>",
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=x_hist, y=kde_hist(x_hist), mode="lines", name="Historical KDE",
                line=dict(color=C["amber"], width=2.2),
                hovertemplate="Return: %{x:.2%}<br>Hist density=%{y:.4f}<extra></extra>",
            ), row=4, col=1)
            if len(breach_vals):
                fig.add_trace(go.Scatter(
                    x=breach_vals, y=np.zeros_like(breach_vals), mode="markers", name="Hist VaR breaches",
                    marker=dict(color=C["red"], size=8, symbol="circle"),
                    hovertemplate="Historical breach: %{x:.2%}<extra></extra>",
                ), row=4, col=1)
            fig.add_vline(x=sim_q5, line_color=C["red"], line_dash="dash", line_width=1.6, row=4, col=1)
            fig.add_vline(x=sim_q95, line_color=C["green"], line_dash="dash", line_width=1.6, row=4, col=1)
            fig.add_annotation(
                xref="x7 domain", yref="y7 domain", x=0.98, y=0.98,
                text=(
                    f"Hist in sim P5-P95: {hist_in_band:.1%}<br>"
                    f"VaR breaches: {len(breach_vals)}<br>"
                    f"P1/P99 tail miss: {'Yes' if tail_flag else 'No'}"
                ),
                showarrow=False, align="right",
                bgcolor="rgba(13,17,23,0.78)", bordercolor=C["border"], borderwidth=1,
                font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
            )
        else:
            fig.add_annotation(
                xref="x7 domain", yref="y7 domain", x=0.5, y=0.5,
                text="Not enough historical horizon returns to compare with simulation.",
                showarrow=False, align="center",
                bgcolor="rgba(13,17,23,0.78)", bordercolor=C["border"], borderwidth=1,
                font=dict(color=C["text"], size=10, family="Consolas, Menlo, monospace"),
            )

        vr_labels = ["Antithetic", "QMC Sobol", "Control Variate"]
        vr_vals = [
            float(adv["variance_reduction_ratio_antithetic"]),
            float(adv["variance_reduction_ratio_qmc"]),
            float(adv["variance_reduction_ratio_cv"]),
        ]
        fig.add_trace(go.Bar(
            x=vr_labels,
            y=vr_vals,
            marker=dict(color=[C["blue"], C["green"], C["amber"]]),
            name="Variance ratio",
            text=[f"{v:.3f}" for v in vr_vals],
            textposition="outside",
            hovertemplate="%{x}<br>Ratio=%{y:.4f}<extra></extra>",
        ), row=4, col=2)
        fig.add_hline(y=1.0, line_color=C["red"], line_dash="dot", row=4, col=2,
                      annotation_text="Baseline 1.0", annotation_font=dict(color=C["red"], size=9))

        layout = self._plotly_theme()
        layout.update(dict(
            title=dict(
                text=(
                    f"<b>Monte Carlo Diagnostics</b> - {self.eng.ticker}<br>"
                    f"<span style='font-size:11px;color:{C['muted']}'>Standard MC slope={pseudo.get('loglog_slope', float('nan')):+.3f} | Antithetic slope={anti.get('loglog_slope', float('nan')):+.3f} | Sobol QMC slope={qmc_prof.get('loglog_slope', float('nan')):+.3f} | epsilon band=±${eps:.4f}</span>"
                ),
                x=0.5, y=0.98, xanchor="center", yanchor="top",
                font=dict(size=15, color=C["title"]),
            ),
            height=1920,
            margin=dict(l=70, r=70, t=130, b=120),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5,
                        font=dict(size=9)),
        ))
        fig.update_layout(**layout)
        for ann in fig.layout.annotations:
            if ann.text in {
                "Convergence Bands: Running Mean ± 2MCSE",
                "Convergence Slope: log|error| vs log(N)",
                "Trace Plot: Running Mean ± 2MCSE",
                "Trace Plot: Running Sigma",
                "ESS / N Applicability",
                "Advanced Option Prices",
                "Simulated vs Historical Overlay",
                "Variance Reduction Ratios",
            }:
                ann.font = dict(size=12, color=C["title"])

        fig.update_xaxes(title_text="Paths N", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Paths N (log10)", type="log", row=1, col=2)
        fig.update_yaxes(title_text="|running mean - true| (log10)", type="log", row=1, col=2)
        fig.update_xaxes(title_text="Trials N", row=2, col=1)
        fig.update_yaxes(title_text="Running mean ($)", row=2, col=1)
        fig.update_xaxes(title_text="Trials N", row=2, col=2)
        fig.update_yaxes(title_text="Running sigma ($)", row=2, col=2)
        fig.update_xaxes(title_text="Applicability status", row=3, col=1, range=[0, 1.05], showticklabels=False)
        fig.update_yaxes(title_text="Sampler", row=3, col=1)
        fig.update_xaxes(title_text="Instrument", row=3, col=2)
        fig.update_yaxes(title_text="Price ($)", row=3, col=2)
        fig.update_xaxes(title_text="Horizon return", tickformat=".1%", row=4, col=1)
        fig.update_yaxes(title_text="Density", row=4, col=1)
        fig.update_xaxes(title_text="Method", row=4, col=2)
        fig.update_yaxes(title_text="Variance ratio", row=4, col=2)

        for r in [1, 2, 3, 4]:
            for c in [1, 2]:
                fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=r, col=c)
                fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=r, col=c)

        path_out = OUT / f"mc_diagnostics_{self.eng.ticker}.html"
        fig.write_html(str(path_out), include_plotlyjs="cdn", config={"displayModeBar": True, "scrollZoom": True})
        rlog(f"  [green]OK[/green] MC diagnostics -> [cyan]{path_out}[/cyan]")
        if show:
            fig.show()
        return fig


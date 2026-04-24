"""
MONTE CARLO STOCK TERMINAL

Install:  pip install numpy pandas matplotlib plotly scipy yfinance rich
Run:      python test.py
Flags:    --ticker AAPL --sims 500 --renderer plotly|matplotlib|both
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

def _select_matplotlib_backend():
    """Select a GUI backend; fail fast if none is available."""
    forced = os.environ.get("MPLBACKEND")
    if forced:
        return

    candidates = ["TkAgg", "QtAgg", "Qt5Agg"]
    for backend in candidates:
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue

    raise RuntimeError(
        "No interactive Matplotlib backend is available. "
        "Install Tk or Qt bindings (for example: pip install tk or pip install pyqt5), "
        "then re-run."
    )

_select_matplotlib_backend()
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
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

matplotlib.rcParams.update({
    "figure.facecolor":  C["bg"],
    "axes.facecolor":    C["bg"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["muted"],
    "axes.titlecolor":   C["title"],
    "axes.titlesize":    9,
    "axes.titlepad":     8,
    "xtick.color":       C["muted"],
    "ytick.color":       C["muted"],
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "text.color":        C["text"],
    "grid.color":        C["border"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.35,
    "figure.dpi":        100,
    "font.family":       "monospace",
    "legend.facecolor":  C["bg2"],
    "legend.edgecolor":  C["border"],
    "legend.labelcolor": C["text"],
    "legend.fontsize":   7,
    "legend.framealpha": 0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

_dollar = FuncFormatter(lambda x, _: f"${x:,.0f}")
_pct    = FuncFormatter(lambda x, _: f"{x:.1%}")

def _is_non_interactive_backend() -> bool:
    backend = str(matplotlib.get_backend()).lower()
    return "agg" in backend or "pdf" in backend or "svg" in backend or "ps" in backend

def _style(ax, grid=True):
    ax.set_facecolor(C["bg"])
    for spine in ax.spines.values():
        spine.set_color(C["border"])
        spine.set_linewidth(0.7)
    if grid:
        ax.grid(True, alpha=0.25, color=C["border"], linestyle="--", linewidth=0.5)

def _tag(ax, txt, x=0.02, y=0.96):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=6,
            color=C["muted"], va="top", ha="left", fontfamily="monospace",
            bbox=dict(fc=C["bg"], ec=C["border"], alpha=0.8, pad=2, linewidth=0.5))

def _vline(ax, x, color, label=None, lw=1.5, ls="--"):
    ax.axvline(x, color=color, lw=lw, ls=ls, label=label, zorder=6)

def _hline(ax, y, color, label=None, lw=1.5, ls="--", alpha=1.0):
    ax.axhline(y, color=color, lw=lw, ls=ls, label=label, zorder=6, alpha=alpha)

def _surface_z_bounds(Z: np.ndarray, S0: float) -> tuple[float, float]:
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    anchor = max(abs(float(S0)), 1.0)
    span = max(z_max - z_min, anchor * 0.05)
    lo = min(z_min, float(S0)) - span * 0.12
    hi = max(z_max, float(S0)) + span * 0.08
    hi = max(hi, lo + 1.0)
    return lo, hi

def _style_3d_axis(ax, N: int, n: int, z_low: float, z_high: float):
    ax.set_facecolor(C["bg"])
    for plane in [ax.xaxis, ax.yaxis, ax.zaxis]:
        plane.pane.fill = False
        plane.pane.set_edgecolor(C["border"])
        plane.pane.set_alpha(0.07)
    ax.set_xlim(0, N)
    ax.set_ylim(0, max(0, n - 1))
    ax.set_zlim(z_low, z_high)
    x_span = max(float(N), 1.0)
    y_span = max(float(n - 1), 1.0)
    base_span = max(x_span, y_span)
    # Keep geometric perspective stable regardless of dollar scale in z.
    ax.set_box_aspect((x_span, y_span, base_span * 0.58))
    ax.view_init(elev=24, azim=-132)
    ax.tick_params(colors=C["muted"], labelsize=8)
    ax.grid(True, alpha=0.10, color=C["border"])
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
    



############################################################################################################################################################################################


# MATPLOTLIB DASHBOARD  - fully fixed layout, no overlaps
class MatplotlibDashboard:
    """
    Layout (5 separate figures for maximum clarity):
      Fig 1: Historical Price + Volume + RSI  (3-row tall chart)
      Fig 2: Distribution + Log-Returns + QQ + Rolling Vol  (2x2)
      Fig 3: MC Paths + Percentile Bands + Heatmap  (1x3)
      Fig 4: Risk Metrics panel (full-width table)
      Fig 5: 3D Surface
    """

    def __init__(self, eng: GBMEngine):
        self.eng = eng

    def fig_history(self, show=True):
        h = self.eng.history

        fig = plt.figure(figsize=(16, 11), facecolor=C["bg"])
        fig.patch.set_facecolor(C["bg"])

        gs = gridspec.GridSpec(3, 1, figure=fig,
                               height_ratios=[4, 1.2, 1.2],
                               hspace=0.45,
                               left=0.08, right=0.96, top=0.88, bottom=0.08)

        ax_p = fig.add_subplot(gs[0])
        _style(ax_p)
        ax_p.plot(h.index, h.values, color=C["blue"], lw=1.6, label="Close", zorder=4)
        ax_p.fill_between(h.index, h.values, h.values.min(),
                          alpha=0.08, color=C["blue"])

        ma20  = h.rolling(20).mean()
        ma50  = h.rolling(50).mean()
        std20 = h.rolling(20).std()
        ax_p.plot(h.index, ma20, color=C["purple"], lw=1.0, alpha=0.9, label="MA-20")
        ax_p.plot(h.index, ma50, color=C["amber"],  lw=1.0, alpha=0.8, ls="--", label="MA-50")
        ax_p.fill_between(h.index, ma20 + 2*std20, ma20 - 2*std20,
                          alpha=0.08, color=C["purple"], label="BB +/-2sigma")

        w52  = min(252, len(h))
        hi52 = h.iloc[-w52:].max()
        lo52 = h.iloc[-w52:].min()
        _hline(ax_p, hi52, C["green"], f"52w H ${hi52:.0f}", lw=1.0, ls=":")
        _hline(ax_p, lo52, C["red"],   f"52w L ${lo52:.0f}", lw=1.0, ls=":")

        ax_p.yaxis.set_major_formatter(_dollar)
        ax_p.set_title(f"{self.eng.ticker} - Historical Price - Bollinger Bands - MA-20/50",
                       fontsize=9.5, color=C["title"], fontweight="bold", pad=8)
        ax_p.legend(loc="upper left", ncol=4, fontsize=6)
        ax_p.set_xticklabels([])
        _tag(ax_p, f"{len(h)} trading days")

        ax_v = fig.add_subplot(gs[1])
        _style(ax_v)
        vol = None
        if HAS_YF:
            try:
                raw = yf.Ticker(self.eng.ticker).history(period="2y")
                if "Volume" in raw.columns:
                    vol = raw["Volume"].reindex(h.index).fillna(0)
            except Exception:
                pass
        if vol is None:
            rng2 = np.random.default_rng(7)
            vol  = pd.Series(np.abs(rng2.standard_normal(len(h))) * 5e7 + 3e7, index=h.index)

        colors = [C["green"] if v > 0 else C["red"] for v in h.pct_change().fillna(0)]
        ax_v.bar(h.index, vol, color=colors, alpha=0.6, width=1.0)
        ax_v.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        ax_v.set_ylabel("Volume", fontsize=7.5, color=C["muted"])
        ax_v.set_xticklabels([])
        _tag(ax_v, "Volume  -  green=up day  red=down day")

        ax_r = fig.add_subplot(gs[2])
        _style(ax_r)
        delta = h.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)

        ax_r.plot(h.index, rsi, color=C["cyan"], lw=1.2, label="RSI-14")
        ax_r.axhline(70, color=C["red"],   lw=0.8, ls="--", alpha=0.7)
        ax_r.axhline(30, color=C["green"], lw=0.8, ls="--", alpha=0.7)
        ax_r.fill_between(h.index, rsi, 70, where=(rsi>=70), alpha=0.15, color=C["red"])
        ax_r.fill_between(h.index, rsi, 30, where=(rsi<=30), alpha=0.15, color=C["green"])
        ax_r.set_ylim(0, 100)
        ax_r.set_yticks([30, 50, 70])
        ax_r.set_ylabel("RSI", fontsize=7.5, color=C["muted"])
        _tag(ax_r, "RSI-14 - overbought >70  oversold <30")

        fig.suptitle(f"Historical Analysis - {self.eng.ticker}",
                     fontsize=11, color=C["title"], fontweight="bold", y=0.93)

        path_out = OUT / f"mc_history_{self.eng.ticker}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(str(path_out), dpi=130, bbox_inches="tight", facecolor=C["bg"])
        rlog(f"  [green]OK[/green] History chart -> [cyan]{path_out}[/cyan]")
        if show: plt.show()
        return fig

    def fig_analysis(self, st: dict, show=True):
        fig = plt.figure(figsize=(19, 12), facecolor=C["bg"])
        fig.patch.set_facecolor(C["bg"])

        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.55, wspace=0.35,
                               left=0.08, right=0.96, top=0.90, bottom=0.08)

        ax_d = fig.add_subplot(gs[0, 0])
        _style(ax_d)
        S0, finals = st["S0"], st["finals"]

        n, bins, patches = ax_d.hist(finals, bins=55, edgecolor="none", alpha=0.85, zorder=3)
        mids = 0.5 * (bins[:-1] + bins[1:])
        for patch, mid in zip(patches, mids):
            if mid < S0:
                t = np.clip((S0 - mid) / (S0 - finals.min() + 1e-9), 0, 1)
                patch.set_facecolor(plt.cm.Reds(0.30 + 0.55*t))
            else:
                t = np.clip((mid - S0) / (finals.max() - S0 + 1e-9), 0, 1)
                patch.set_facecolor(plt.cm.Greens(0.30 + 0.55*t))

        kde_x = np.linspace(finals.min(), finals.max(), 400)
        kde   = scipy_stats.gaussian_kde(finals)
        scale = n.max() / kde(kde_x).max()
        ax_d.plot(kde_x, kde(kde_x)*scale, color=C["cyan"], lw=2.0, zorder=5, label="KDE")

        _vline(ax_d, S0,           C["red"],   f"S0 ${S0:.2f}", lw=2.0)
        _vline(ax_d, st["mean"],   C["green"], f"Mean ${st['mean']:.2f}", lw=2.0)
        _vline(ax_d, st["median"], C["amber"], f"Median", lw=1.5, ls="-.")

        top = ax_d.get_ylim()[1]
        for lbl, val in [("5th", st["terminal_pct"]["5"]), ("95th", st["terminal_pct"]["95"])]:
            ax_d.axvline(val, color=C["border"], lw=1.0, ls=":", alpha=0.9)
            ax_d.text(val, top*0.92, f"{lbl}\n${val:.0f}",
                      color=C["muted"], fontsize=6, ha="center",
                      bbox=dict(fc=C["bg2"], ec=C["border"], alpha=0.9, pad=1.5))

        ax2 = ax_d.twiny()
        ax2.set_xlim(ax_d.get_xlim())
        ax2.set_facecolor(C["bg"])
        ticks = ax_d.get_xticks()
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{(x-S0)/S0:+.0%}" for x in ticks], fontsize=6, color=C["muted"])
        for sp in ax2.spines.values(): sp.set_color(C["border"])
        ax_d.xaxis.set_major_formatter(_dollar)
        ax_d.set_title("Final Price Distribution - 1-Year Forecast", fontsize=9, pad=4)
        ax_d.legend(loc="upper right", ncol=2, fontsize=6)
        ax_d.set_xlabel("Price ($)", fontsize=7.5, color=C["muted"])
        _tag(ax_d, f"P(profit) = {st['prob_up']:.1%} - {len(finals)} sims")

        ax_lr = fig.add_subplot(gs[0, 1])
        _style(ax_lr)
        lr = self.eng.log_ret.values
        ax_lr.hist(lr, bins=65, color=C["cyan"], alpha=0.65, edgecolor="none",
                   density=True, label="Log-returns")
        mu_r, sd_r = lr.mean(), lr.std()
        x = np.linspace(lr.min(), lr.max(), 300)
        ax_lr.plot(x, norm.pdf(x, mu_r, sd_r), color=C["amber"], lw=2.0, label="Normal fit")
        df_t, loc_t, scale_t = t_dist.fit(lr)
        ax_lr.plot(x, t_dist.pdf(x, df_t, loc_t, scale_t),
                   color=C["purple"], lw=1.6, ls="--", label=f"t-fit (nu={df_t:.1f})")
        ax_lr.axvline(0, color=C["muted"], lw=0.8, ls=":")
        ax_lr.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax_lr.set_title("Log-Return Distribution", fontsize=9, pad=4)
        ax_lr.legend(loc="upper right", fontsize=6)
        _tag(ax_lr, f"skew={st['lr_skew']:+.2f} kurt={st['lr_kurt']:+.2f} JB={st['jb_p']:.3f}")

        ax_acf = fig.add_subplot(gs[1, 0])
        _style(ax_acf)
        max_lag = min(30, len(lr) - 2)
        lags = np.arange(1, max_lag + 1)
        acf_vals = []
        lr_center = lr - lr.mean()
        denom = np.sum(lr_center**2) + 1e-12
        for lag in lags:
            num = np.sum(lr_center[:-lag] * lr_center[lag:])
            acf_vals.append(float(num / denom))
        conf = 1.96 / np.sqrt(len(lr))
        ax_acf.bar(lags, acf_vals, color=C["blue"], alpha=0.75, width=0.75)
        _hline(ax_acf, 0.0, C["muted"], lw=1.0, ls="-")
        _hline(ax_acf, conf, C["red"], lw=1.0, ls=":")
        _hline(ax_acf, -conf, C["red"], lw=1.0, ls=":")
        ax_acf.set_title("Return Autocorrelation (ACF)", fontsize=9, pad=4)
        ax_acf.set_xlabel("Lag", fontsize=7.5, color=C["muted"])
        ax_acf.set_ylabel("ACF", fontsize=7.5, color=C["muted"])
        _tag(ax_acf, "dotted bounds: ~95% white-noise interval")

        ax_vol = fig.add_subplot(gs[1, 1])
        _style(ax_vol)
        rv = self.eng.roll_vol if self.eng.roll_vol is not None else self.eng.log_ret.rolling(30).std() * np.sqrt(252)
        q25 = float(rv.quantile(0.25))
        q75 = float(rv.quantile(0.75))
        ax_vol.plot(rv.index, rv.values, color=C["amber"], lw=1.4, label="30d rolling vol")
        ax_vol.fill_between(rv.index, 0, rv.values, alpha=0.10, color=C["amber"])
        ax_vol.fill_between(rv.index, 0, rv.values, where=(rv.values >= q75),
                    alpha=0.13, color=C["red"], label="High-vol regime")
        ax_vol.fill_between(rv.index, 0, rv.values, where=(rv.values <= q25),
                    alpha=0.10, color=C["green"], label="Low-vol regime")
        ax_vol.axhline(self.eng.sigma, color=C["purple"], lw=1.4, ls="--",
                       label=f"Full-period sigma={self.eng.sigma:.1%}")
        ax_vol.yaxis.set_major_formatter(_pct)
        ax_vol.set_title("Rolling 30-Day Volatility", fontsize=9, pad=4)
        ax_vol.legend(loc="upper right", fontsize=6)
        _tag(ax_vol, "volatility clustering visible")

        fig.suptitle(f"Statistical Analysis - {self.eng.ticker}",
                     fontsize=11, color=C["title"], fontweight="bold", y=0.96)

        path_out = OUT / f"mc_analysis_{self.eng.ticker}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(str(path_out), dpi=130, bbox_inches="tight", facecolor=C["bg"])
        rlog(f"  [green]OK[/green] Analysis chart -> [cyan]{path_out}[/cyan]")
        if show: plt.show()
        return fig

    def fig_simulation(self, st: dict, show=True):
        fig = plt.figure(figsize=(24, 9), facecolor=C["bg"])
        fig.patch.set_facecolor(C["bg"])

        gs = gridspec.GridSpec(1, 3, figure=fig,
                               wspace=0.35,
                               left=0.06, right=0.96, top=0.92, bottom=0.10)

        ax_p = fig.add_subplot(gs[0])
        self._draw_paths(ax_p, st)

        ax_b = fig.add_subplot(gs[1])
        self._draw_bands(ax_b, st)

        ax_h = fig.add_subplot(gs[2])
        self._draw_heatmap(ax_h, st)

        fig.suptitle(f"Monte Carlo Simulation - {self.eng.ticker} - "
                     f"{st['paths'].shape[0]} sims x {st['paths'].shape[1]-1} days",
                     fontsize=10, color=C["title"], fontweight="bold", y=0.98)

        path_out = OUT / f"mc_simulation_{self.eng.ticker}.png"
        fig.savefig(str(path_out), dpi=130, bbox_inches="tight", facecolor=C["bg"])
        rlog(f"  [green]OK[/green] Simulation chart -> [cyan]{path_out}[/cyan]")
        if show: plt.show()
        return fig

    def _draw_paths(self, ax, st, max_show=80):
        _style(ax)
        S0, paths = st["S0"], st["paths"]
        N    = paths.shape[1] - 1
        days = np.arange(N + 1)
        rng  = np.random.default_rng(0)
        idx  = rng.choice(len(paths), min(max_show, len(paths)), replace=False)

        ax.fill_between(days, st["bands"]["5"], st["bands"]["95"],
                        alpha=0.07, color=C["blue"], label="5-95%")
        ax.fill_between(days, st["bands"]["25"], st["bands"]["75"],
                        alpha=0.13, color=C["blue"], label="25-75%")

        for i in idx:
            f = paths[i, -1]
            if   f > S0 * 1.5:  col, a = C["green"], 0.25
            elif f < S0 * 0.65: col, a = C["red"],   0.25
            else:               col, a = C["muted"],  0.07
            ax.plot(days, paths[i], color=col, alpha=a, lw=0.5)

        best_i  = int(paths[:, -1].argmax())
        worst_i = int(paths[:, -1].argmin())
        ax.plot(days, paths[best_i],  color=C["green"], lw=1.5, alpha=0.9,
                label=f"Best ${paths[best_i,-1]:.0f}")
        ax.plot(days, paths[worst_i], color=C["red"],   lw=1.5, alpha=0.9,
                label=f"Worst ${paths[worst_i,-1]:.0f}")
        ax.plot(days, st["mean_path"], color=C["amber"], lw=2.5,
                label=f"Mean ${st['mean_path'][-1]:.0f}", zorder=10)
        _hline(ax, S0, C["red"], f"S0 ${S0:.2f}", lw=1.5)

        mf = st["mean_path"][-1]
        ax.annotate(f" ${mf:.0f}", xy=(N, mf), color=C["amber"], fontsize=7.5,
                    fontweight="bold", va="center")

        ax.yaxis.set_major_formatter(_dollar)
        ax.set_xlabel("Trading Days", fontsize=7.5, color=C["muted"])
        ax.set_title("MC Paths & Confidence Bands", fontsize=9, pad=4)
        ax.legend(loc="lower left", ncol=2, fontsize=6)
        _tag(ax, f"{len(paths)} paths - {N}d")

    def _draw_bands(self, ax, st):
        _style(ax)
        S0   = st["S0"]
        N    = st["paths"].shape[1]
        days = np.arange(N)
        T    = N / 252.0
        t_arr = np.linspace(0, T, N)

        ax.fill_between(days, st["bands"]["5"],  st["bands"]["95"],
                        alpha=0.10, color=C["blue"], label="5-95%")
        ax.fill_between(days, st["bands"]["10"], st["bands"]["90"],
                        alpha=0.08, color=C["blue"])
        ax.fill_between(days, st["bands"]["25"], st["bands"]["75"],
                        alpha=0.18, color=C["blue"], label="25-75%")

        ax.plot(days, st["bands"]["95"], color=C["green"], lw=1.2, ls="--", alpha=0.9)
        ax.plot(days, st["bands"]["75"], color=C["green"], lw=0.8, ls="--", alpha=0.55)
        ax.plot(days, st["bands"]["50"], color=C["amber"], lw=2.2, label="Median")
        ax.plot(days, st["bands"]["25"], color=C["red"],   lw=0.8, ls="--", alpha=0.55)
        ax.plot(days, st["bands"]["5"],  color=C["red"],   lw=1.2, ls="--", alpha=0.9)

        e_path = S0 * np.exp(self.eng.mu * t_arr)
        ax.plot(days, e_path, color=C["purple"], lw=1.6, ls=":", label="E[S_t]", zorder=9)
        _hline(ax, S0, C["muted"], lw=1.2, ls=":")

        ax.fill_between(days, st["bands"]["5"], S0,
                        where=(st["bands"]["5"] < S0), alpha=0.07, color=C["red"])

        for key, col in [("95", C["green"]), ("50", C["amber"]), ("5", C["red"])]:
            v = st["bands"][key][-1]
            ax.text(N-1, v, f" ${v:.0f}", color=col, fontsize=7, va="center")

        ax.yaxis.set_major_formatter(_dollar)
        ax.set_xlabel("Trading Days", fontsize=7.5, color=C["muted"])
        ax.set_title("Percentile Bands 5/25/50/75/95", fontsize=9, pad=4)
        ax.legend(loc="lower left", ncol=2, fontsize=6)
        _tag(ax, "E[S_t] = S0-exp(mut)")

    def _draw_heatmap(self, ax, st):
        _style(ax, grid=False)
        paths = st["paths"]
        N     = paths.shape[1]
        pmin  = paths.min() * 0.96
        pmax  = paths.max() * 1.04
        pbins = np.linspace(pmin, pmax, 80)
        stride = max(1, N // 120)
        t_idx  = np.arange(0, N, stride)

        hm = np.zeros((len(pbins)-1, len(t_idx)))
        for col, t in enumerate(t_idx):
            c, _ = np.histogram(paths[:, t], bins=pbins)
            col_max = c.max()
            hm[:, col] = c / col_max if col_max > 0 else c

        cmap = LinearSegmentedColormap.from_list(
            "terminal",
            [C["bg"], "#0f2d4a", "#1f6feb", C["cyan"], C["green"], C["amber"], "#ffd700"]
        )
        im = ax.imshow(hm, aspect="auto", cmap=cmap, origin="lower",
                       extent=[0, N-1, pmin, pmax],
                       alpha=0.95, interpolation="bilinear")

        days = np.arange(N)
        ax.plot(days, st["mean_path"], color="white", lw=2.5, zorder=5, label="Mean")
        ax.plot(days, st["bands"]["5"],  color=C["red"],   lw=1.2, ls="--", alpha=0.85, label="5th")
        ax.plot(days, st["bands"]["95"], color=C["green"], lw=1.2, ls="--", alpha=0.85, label="95th")
        _hline(ax, st["S0"], "white", lw=1.0, ls=":", alpha=0.5)

        ax.yaxis.set_major_formatter(_dollar)
        ax.set_xlabel("Trading Days", fontsize=7.5, color=C["muted"])
        ax.set_title("Path Density Heatmap", fontsize=9, pad=4)
        ax.legend(loc="lower right", fontsize=6)
        cb = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
        cb.set_label("Density", color=C["muted"], fontsize=7)
        cb.ax.yaxis.set_tick_params(color=C["muted"])
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["muted"], fontsize=6)
        _tag(ax, "brighter = higher concentration")

    def fig_risk(self, st: dict, N: int, show=True):
        fig = plt.figure(figsize=(19, 8), facecolor=C["bg2"])
        fig.patch.set_facecolor(C["bg2"])

        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_facecolor(C["bg2"])

        metrics_left = [
            ("VaR 95%",        f"-${st['var95']:,.2f}",          C["red"]),
            ("VaR 99%",        f"-${st['var99']:,.2f}",          C["red"]),
            ("CVaR 95%",       f"-${st['cvar95']:,.2f}",         C["red"]),
            ("CVaR 99%",       f"-${st['cvar99']:,.2f}",         C["red"]),
            ("Hist Max DD",    f"{st['hist_max_dd']:.2%}",       C["red"]),
            ("Sim Avg Max DD", f"{st['sim_max_dd']:.2%}",        C["red"]),
        ]
        metrics_mid = [
            ("Sharpe Ratio",  f"{st['sharpe']:+.3f}",
             C["green"] if st["sharpe"] > 1 else C["amber"] if st["sharpe"] > 0 else C["red"]),
            ("Sortino Ratio", f"{st['sortino']:+.3f}",
             C["green"] if st["sortino"] > 1 else C["amber"] if st["sortino"] > 0 else C["red"]),
            ("Calmar Ratio",  f"{st['calmar']:+.3f}",
             C["green"] if st["calmar"] > 0.5 else C["amber"]),
            ("Win Rate",      f"{st['win_rate']:.1%}",
             C["green"] if st["win_rate"] > 0.5 else C["red"]),
            ("Avg Win",       f"+${st['avg_win']:,.2f}",         C["green"]),
            ("Avg Loss",      f"-${st['avg_loss']:,.2f}",        C["red"]),
        ]
        metrics_right = [
            ("P(Profit)",     f"{st['prob_up']:.1%}",
             C["green"] if st["prob_up"] > 0.5 else C["red"]),
            ("P(Double)",     f"{st['prob_2x']:.1%}",            C["purple"]),
            ("P(Halve)",      f"{st['prob_half']:.1%}",          C["red"]),
            ("Mean Target",   f"${st['mean']:,.2f}",
             C["green"] if st["mean"] > st["S0"] else C["red"]),
            ("Mean +- MCSE",  f"${st['mean']:,.2f} +- ${st['mean_stderr']:.2f}", C["cyan"]),
            ("5th Pctile",    f"${st['terminal_pct']['5']:,.2f}",         C["red"]),
            ("95th Pctile",   f"${st['terminal_pct']['95']:,.2f}",        C["green"]),
        ]

        # header row
        ax.text(0.5, 0.96, "RISK  &  PERFORMANCE  METRICS",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=14, fontweight="bold", color=C["title"], fontfamily="monospace")

        # subheaders
        info = (f"{self.eng.ticker} - S0=${self.eng.S0:,.2f} - "
                f"mu={self.eng.mu:+.2%} - sigma={self.eng.sigma:.2%} - "
                f"{st['paths'].shape[0]} sims - {N}d horizon")
        ax.text(0.5, 0.86, info, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=C["muted"], fontfamily="monospace")

        ax.plot([0.02, 0.98], [0.80, 0.80], color=C["border"], lw=0.8,
                transform=ax.transAxes)

        col_groups = [metrics_left, metrics_mid, metrics_right]
        col_titles = ["RISK METRICS", "PERFORMANCE RATIOS", "PROBABILITIES"]
        col_colors = [C["red"], C["amber"], C["purple"]]
        col_xs     = [0.18, 0.50, 0.83]

        for col_x, title, metrics, tcol in zip(col_xs, col_titles, col_groups, col_colors):
            ax.text(col_x, 0.77, title, transform=ax.transAxes, ha="center",
                    fontsize=8.5, color=tcol, fontweight="bold", fontfamily="monospace")

            for i, (label, value, col) in enumerate(metrics):
                y = 0.68 - i * 0.11
                ax.text(col_x, y, label, transform=ax.transAxes, ha="center",
                        fontsize=7.5, color=C["muted"], fontfamily="monospace")
                ax.text(col_x, y - 0.055, value, transform=ax.transAxes, ha="center",
                        fontsize=12, color=col, fontweight="bold", fontfamily="monospace")

        # bottom separator + parameter summary
        ax.plot([0.02, 0.98], [0.05, 0.05], color=C["border"], lw=0.8,
                transform=ax.transAxes)
        bottom = (f"log-return skew={st['lr_skew']:+.2f}  -  "
                  f"excess kurt={st['lr_kurt']:+.2f}  -  "
                  f"JB p={st['jb_p']:.4f}  -  "
                  f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        ax.text(0.5, 0.02, bottom, transform=ax.transAxes, ha="center",
                fontsize=7.5, color=C["muted"], fontfamily="monospace")

        path_out = OUT / f"mc_risk_{self.eng.ticker}.png"
        fig.savefig(str(path_out), dpi=130, bbox_inches="tight", facecolor=C["bg2"])
        rlog(f"  [green]OK[/green] Risk table -> [cyan]{path_out}[/cyan]")
        if show: plt.show()
        return fig

    def fig_mc_summary(self, adv: dict, show=True):
        fig = plt.figure(figsize=(18, 10), facecolor=C["bg"])
        fig.patch.set_facecolor(C["bg"])

        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_facecolor(C["bg"])

        left = [
            ("European Call (pseudo)", f"${adv['euro_plain']['price']:,.4f} ± {adv['euro_plain']['stderr']:.4f}"),
            ("European Call (anti)", f"${adv['euro_antithetic']['price']:,.4f} ± {adv['euro_antithetic']['stderr']:.4f}"),
            ("European Call (QMC)", f"${adv['euro_qmc']['price']:,.4f} ± {adv['euro_qmc']['stderr']:.4f}"),
            ("European Call (CV)", f"${adv['euro_control_variate']['price']:,.4f} ± {adv['euro_control_variate']['stderr']:.4f}"),
            ("Asian Call", f"${adv['asian_call']['price']:,.4f} ± {adv['asian_call']['stderr']:.4f}"),
            ("Barrier Up-Out", f"${adv['barrier_up_out_call']['price']:,.4f} ± {adv['barrier_up_out_call']['stderr']:.4f}"),
            ("American Put LSMC", f"${adv['american_put_lsmc']['price']:,.4f} ± {adv['american_put_lsmc']['stderr']:.4f}"),
        ]
        right = [
            ("Pathwise Delta", f"{adv['pathwise_greeks']['delta']:+.4f} ± {adv['pathwise_greeks']['delta_stderr']:.4f}"),
            ("Pathwise Vega", f"{adv['pathwise_greeks']['vega']:+.4f} ± {adv['pathwise_greeks']['vega_stderr']:.4f}"),
            ("Pathwise Rho", f"{adv['pathwise_greeks']['rho']:+.4f} ± {adv['pathwise_greeks']['rho_stderr']:.4f}"),
            ("Conv. slope", f"{adv['convergence']['loglog_slope']:+.3f} (target -0.500)"),
            ("Anti ratio", f"{adv['variance_reduction_ratio_antithetic']:.3f}"),
            ("QMC ratio", f"{adv['variance_reduction_ratio_qmc']:.3f}"),
            ("CV ratio", f"{adv['variance_reduction_ratio_cv']:.3f}"),
        ]

        ax.text(0.5, 0.96, "ADVANCED  MONTE  CARLO  SUMMARY",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=14, fontweight="bold", color=C["title"], fontfamily="monospace")
        ax.text(0.5, 0.90,
                f"{self.eng.ticker}  |  BS call=${adv['black_scholes_call']:,.4f}  |  multi-asset corr={adv['multi_asset_emp_corr']:+.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=C["muted"], fontfamily="monospace")

        ax.plot([0.03, 0.97], [0.86, 0.86], color=C["border"], lw=0.8, transform=ax.transAxes)

        col_xs = [0.24, 0.76]
        groups = [left, right]
        titles = ["PRICING METHODS", "GREEKS / EFFICIENCY"]
        colors = [C["cyan"], C["amber"]]

        for col_x, metrics, title, tcol in zip(col_xs, groups, titles, colors):
            ax.text(col_x, 0.80, title, transform=ax.transAxes, ha="center",
                    fontsize=8.8, color=tcol, fontweight="bold", fontfamily="monospace")
            for i, (label, value) in enumerate(metrics):
                y = 0.74 - i * 0.115
                ax.text(col_x, y, label, transform=ax.transAxes, ha="center",
                        fontsize=7.2, color=C["muted"], fontfamily="monospace")
                ax.text(col_x, y - 0.055, value, transform=ax.transAxes, ha="center",
                        fontsize=10.5, color=C["title"], fontweight="bold", fontfamily="monospace")

        ax.text(0.5, 0.02,
                f"Convergence slope should approach -0.5; lower variance ratios are better. Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.3, color=C["muted"], fontfamily="monospace")

        path_out = OUT / f"mc_advanced_{self.eng.ticker}.png"
        fig.savefig(str(path_out), dpi=130, bbox_inches="tight", facecolor=C["bg"])
        rlog(f"  [green]OK[/green] Advanced summary -> [cyan]{path_out}[/cyan]")
        if show:
            plt.show()
        return fig

    def fig_surface(self, paths: np.ndarray, show=True):
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        n  = min(60, len(paths))
        N  = paths.shape[1] - 1
        Z  = paths[:n]
        S0 = self.eng.S0
        X, Y = np.meshgrid(np.arange(N+1), np.arange(n))
        z_low, z_high = _surface_z_bounds(Z, S0)
        wire_r = max(1, n // 12)
        wire_c = max(1, (N + 1) // 20)

        fig = plt.figure(figsize=(16, 11), facecolor=C["bg"])
        ax  = fig.add_subplot(111, projection="3d")
        _style_3d_axis(ax, N, n, z_low, z_high)

        surf = ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.88,
                               rstride=2, cstride=4, linewidth=0,
                               antialiased=True, shade=True)
        ax.plot_wireframe(X[::wire_r, ::wire_c], Y[::wire_r, ::wire_c], Z[::wire_r, ::wire_c],
                          color="white", alpha=0.04, lw=0.3)

        ax.contourf(X, Y, Z, zdir="z", offset=z_low, cmap="plasma", alpha=0.25, levels=12)

        mean_z = Z.mean(axis=0)
        ax.plot(np.arange(N+1), np.full(N+1, n//2), mean_z,
                color="white", lw=3.0, alpha=0.95, zorder=10)

        xx, yy = np.meshgrid([0, N], [0, n-1])
        s0_plane = np.full(xx.shape, S0, dtype=float)
        ax.plot_surface(xx, yy, s0_plane, alpha=0.10, color=C["red"], linewidth=0)
        ax.text(N*0.02, n*0.48, S0*1.01, f"S0 ${S0:.2f}",
                color=C["red"], fontsize=10, fontweight="bold")

        ax.set_xlabel("Days",  fontsize=10, color=C["muted"], labelpad=12)
        ax.set_ylabel("Sim #", fontsize=10, color=C["muted"], labelpad=12)
        ax.set_zlabel("Price", fontsize=10, color=C["muted"], labelpad=12)
        fig.suptitle(
            f"Monte Carlo 3D Surface  -  {self.eng.ticker}\n"
            f"mu={self.eng.mu:+.2%}   sigma={self.eng.sigma:.2%}   {n} sims x {N}d",
            fontsize=11,
            color=C["title"],
            y=0.96,
        )

        cbar = fig.colorbar(surf, ax=ax, shrink=0.52, aspect=11, pad=0.08)
        cbar.set_label("Price ($)", color=C["muted"], fontsize=9)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C["muted"], fontsize=7)
        fig.subplots_adjust(left=0.02, right=0.92, bottom=0.03, top=0.87)

        path_out = OUT / f"mc_surface_{self.eng.ticker}.png"
        fig.savefig(str(path_out), dpi=130, bbox_inches="tight", facecolor=C["bg"])
        rlog(f"  [green]OK[/green] 3D surface -> [cyan]{path_out}[/cyan]")
        if show: plt.show()
        return fig

    def fig_mc_diagnostics(self, adv: dict, show=True):
        fig = plt.figure(figsize=(20, 6.8), facecolor=C["bg"])
        fig.patch.set_facecolor(C["bg"])
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.28, left=0.05, right=0.97, top=0.88, bottom=0.14)

        n_grid = np.array(adv["convergence"]["n_grid"], dtype=float)
        est = np.array(adv["convergence"]["estimates"], dtype=float)
        se = np.array(adv["convergence"]["stderrs"], dtype=float)
        bs = float(adv.get("black_scholes_call", np.nan))

        ax1 = fig.add_subplot(gs[0])
        _style(ax1)
        ax1.errorbar(n_grid, est, yerr=1.96 * se, fmt="o-", color=C["cyan"], ecolor=C["border"],
                     capsize=3, lw=1.6, ms=4.5, label="MC estimate +- 95% CI")
        if np.isfinite(bs):
            _hline(ax1, bs, C["amber"], f"Black-Scholes ${bs:.3f}", lw=1.6, ls="--")
        ax1.set_xlabel("Number of Paths", fontsize=8, color=C["muted"])
        ax1.set_ylabel("Price ($)", fontsize=8, color=C["muted"])
        ax1.set_title("Convergence (Estimate vs N)", fontsize=9.5)
        ax1.legend(loc="best", fontsize=6.5)
        _tag(ax1, f"slope={adv['convergence']['loglog_slope']:+.3f}")

        ax2 = fig.add_subplot(gs[1])
        _style(ax2)
        ax2.loglog(n_grid, se, "o-", color=C["green"], lw=1.8, ms=5, label="Observed stderr")
        ref = se[0] * np.sqrt(n_grid[0] / n_grid)
        ax2.loglog(n_grid, ref, "--", color=C["purple"], lw=1.5, label="Reference O(1/sqrt(N))")
        ax2.set_xlabel("Number of Paths (log)", fontsize=8, color=C["muted"])
        ax2.set_ylabel("Std Error (log)", fontsize=8, color=C["muted"])
        ax2.set_title("Error Decay", fontsize=9.5)
        ax2.legend(loc="best", fontsize=6.5)
        _tag(ax2, f"target slope=-0.5  observed={adv['convergence']['loglog_slope']:+.3f}")

        ax3 = fig.add_subplot(gs[2])
        _style(ax3)
        labels = ["Antithetic", "QMC Sobol", "Control Variate"]
        vals = [
            float(adv["variance_reduction_ratio_antithetic"]),
            float(adv["variance_reduction_ratio_qmc"]),
            float(adv["variance_reduction_ratio_cv"]),
        ]
        cols = [C["blue"], C["green"], C["amber"]]
        x = np.arange(len(labels))
        bars = ax3.bar(x, vals, color=cols, alpha=0.8, width=0.62)
        _hline(ax3, 1.0, C["red"], "Baseline variance ratio = 1.0", lw=1.2, ls=":")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=7)
        ax3.set_ylabel("Variance Ratio", fontsize=8, color=C["muted"])
        ax3.set_title("Variance Reduction Efficiency", fontsize=9.5)
        for i, b in enumerate(bars):
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{vals[i]:.3f}",
                     ha="center", va="bottom", fontsize=7, color=C["title"])
        _tag(ax3, "lower is better")

        fig.suptitle(f"Monte Carlo Diagnostics - {self.eng.ticker}", fontsize=11,
                     color=C["title"], fontweight="bold", y=0.96)

        path_out = OUT / f"mc_diagnostics_{self.eng.ticker}.png"
        fig.savefig(str(path_out), dpi=130, bbox_inches="tight", facecolor=C["bg"])
        rlog(f"  [green]OK[/green] MC diagnostics -> [cyan]{path_out}[/cyan]")
        if show:
            plt.show()
        return fig

    def render_all(self, st: dict, N: int, paths: np.ndarray, show=True, no3d=False, adv: dict | None = None):
        figs = []
        rlog("  [1/6] Historical price chart ...")
        figs.append(self.fig_history(show=False))
        rlog("  [2/6] Statistical analysis ...")
        figs.append(self.fig_analysis(st, show=False))
        rlog("  [3/6] Simulation panels ...")
        figs.append(self.fig_simulation(st, show=False))
        rlog("  [4/6] Risk metrics table ...")
        figs.append(self.fig_risk(st, N, show=False))
        if adv:
            rlog("  [5/6] Advanced MC summary ...")
            figs.append(self.fig_mc_summary(adv, show=False))
            rlog("  [6/6] MC diagnostics ...")
            figs.append(self.fig_mc_diagnostics(adv, show=False))
        if not no3d:
            rlog("  [+] 3D surface ...")
            figs.append(self.fig_surface(paths, show=False))
        if show:
            rlog("\n  [cyan]-----------------------------------------------------[/cyan]")
            rlog("  [yellow]ðŸ“Š All figures displayed![/yellow]")
            rlog("  [yellow]Close figure windows to continue, or press Enter.[/yellow]")
            rlog("  [cyan]-----------------------------------------------------[/cyan]\n")
            plt.show(block=True)
            # Raise figures in correct order (1, 2, 3, 4, 5) so first is on top
            for fig in figs:
                try:
                    fig.canvas.manager.window.lift()
                except Exception:
                    pass
            input("  Press Enter when done viewing figures...")
        return figs
    


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

    def render_main(self, st: dict, N: int, show=True, adv: dict | None = None):
        adv = adv or {}
        h  = self.eng.history
        S0 = self.eng.S0

        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                f"Historical Price  ({self.eng.ticker})",
                "Final Price Distribution",
                "Log-Return Distribution",
                "Monte Carlo Sample Paths",
                "Percentile Confidence Bands",
                "Path Density Heatmap",
                "Rolling Volatility",
                "Q-Q Plot",
                "Risk & Performance Summary",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.09,
            specs=[
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "table"}],
            ],
        )

        ma20  = h.rolling(20).mean()
        ma50  = h.rolling(50).mean()
        std20 = h.rolling(20).std()

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
            x=h.index, y=(ma20+2*std20), mode="lines", name="BB+2sigma",
            line=dict(color=C["purple"], width=0.8, dash="dot"), opacity=0.7,
            hovertemplate="BB+2sigma: $%{y:,.2f}<extra></extra>",
            showlegend=False
        ), row=1, col=1)



        # Bollinger lower + fill
        fig.add_trace(go.Scatter(
            x=h.index, y=(ma20-2*std20), mode="lines", name="BB +/-2sigma",
            line=dict(color=C["purple"], width=0.8, dash="dot"), opacity=0.7,
            fill="tonexty", fillcolor="rgba(163,113,247,0.06)",
            hovertemplate="BB-2sigma: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)

        finals = st["finals"]
        fig.add_trace(go.Histogram(
            x=finals, nbinsx=55, name="Final prices",
            marker=dict(
                color=finals,
                colorscale=[[0, C["red"]], [0.5, C["amber"]], [1.0, C["green"]]],
                colorbar=None,
                line=dict(width=0),
            ),
            opacity=0.82,
            hovertemplate="Price: $%{x:,.0f}<br>Count: %{y}<extra></extra>"
        ), row=1, col=2)




        # KDE overlay
        kde_x = np.linspace(finals.min(), finals.max(), 250)
        kde   = scipy_stats.gaussian_kde(finals)
        kde_y = kde(kde_x)
        scale = len(finals) * (finals.max()-finals.min()) / 55
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y * scale, mode="lines", name="KDE",
            line=dict(color=C["cyan"], width=2.5),
            hovertemplate="$%{x:,.0f}: density=%{y:.4f}<extra></extra>"
        ), row=1, col=2)

        for val, col in [(S0, C["red"]), (st["mean"], C["green"]), (st["median"], C["amber"])]:
            fig.add_vline(
                x=val,
                line_color=col,
                line_dash="dash",
                line_width=1.8,
                row=1,
                col=2,
            )
        fig.add_annotation(
            xref="x2 domain",
            yref="y2 domain",
            x=0.02,
            y=0.98,
            text=(
                f"<b>S0</b> ${S0:,.0f}<br>"
                f"<b>Mean</b> ${st['mean']:,.0f}<br>"
                f"<b>Median</b> ${st['median']:,.0f}"
            ),
            showarrow=False,
            align="left",
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["text"], size=9, family="Consolas, Menlo, monospace"),
        )

        lr = self.eng.log_ret.values
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

        rv = self.eng.log_ret.rolling(30).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rv.index, y=rv.values, mode="lines", name="30d rolling vol",
            line=dict(color=C["amber"], width=1.8),
            fill="tozeroy", fillcolor=f"rgba(227,179,65,0.10)",
            hovertemplate="Date: %{x}<br>Vol: %{y:.1%}<extra></extra>"
        ), row=3, col=1)
        fig.add_hline(y=self.eng.sigma, line_color=C["purple"], line_dash="dash",
                      row=3, col=1,
                      annotation_text=f"Full-period sigma={self.eng.sigma:.1%}",
                      annotation_font=dict(color=C["purple"], size=8),
                      annotation_position="top left")

        (osm, osr), (slope, intercept, _) = scipy_stats.probplot(lr)
        fig.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers", name="Q-Q quantiles",
            marker=dict(color=C["blue"], size=3, opacity=0.6),
            hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>"
        ), row=3, col=2)
        ql = np.array([osm[0], osm[-1]])
        fig.add_trace(go.Scatter(
            x=ql, y=slope*ql+intercept, mode="lines", name="Normal ref.",
            line=dict(color=C["amber"], width=2.0),
        ), row=3, col=2)

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

        fig.add_trace(go.Table(
            header=dict(
                values=["<b>METRIC</b>", "<b>VALUE</b>", "<b>METRIC</b>", "<b>VALUE</b>"],
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
                fill_color=[[C["bg2"] if i%2==0 else C["bg"] for i in range(len(rows_left))]]*4,
                font=dict(color=[C["muted"], C["cyan"], C["muted"], C["amber"]], size=9.5),
                line_color=C["border"], align="left", height=25,
            )
        ), row=3, col=3)

        layout = self._plotly_theme()
        layout.update(dict(
            title=dict(
                text=(f"<b>Monte Carlo Dashboard</b> - {self.eng.ticker}<br>"
                      f"mu={self.eng.mu:+.2%}  sigma={self.eng.sigma:.2%}  "
                      f"S0=${S0:,.2f}  {st['paths'].shape[0]} sims x {N}d"),
                x=0.5, y=0.985, xanchor="center", yanchor="top",
                font=dict(size=15, color=C["title"])
            ),
            height=1840,
            margin=dict(l=70, r=120, t=120, b=130),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.08,
                xanchor="center",
                x=0.5,
                font=dict(size=8),
                bgcolor="rgba(13, 17, 23, 0.85)",
                bordercolor=C["border"],
                borderwidth=1,
                tracegroupgap=4,
                itemwidth=60
            ),
        ))
        fig.update_layout(**layout)
        fig.update_annotations(font=dict(size=11, color=C["title"]))

        # axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, tickprefix="$")
        fig.update_yaxes(title_text="Count", row=1, col=2)
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
        fig.update_xaxes(title_text="Theoretical Quantiles", row=3, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=3, col=2)



        # grid styling
        for r in range(1, 4):
            for c in range(1, 4):
                if not (r == 3 and c == 3):  # skip table
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

    def render_3d(self, paths: np.ndarray, N: int, show=True, no_anim=False):
        """Generate fully interactive Plotly 3D surface and path explorer."""
        S0 = self.eng.S0
        n = len(paths)
        Z = paths[:n]
        x = np.arange(N + 1)
        y = np.arange(n)

        z_low, z_high = _surface_z_bounds(Z, S0)
        x_span = max(float(N), 1.0)
        y_span = max(float(n - 1), 1.0)
        base_span = max(x_span, y_span)
        z_ratio = max(float(z_high - z_low), 1e-9) / max(base_span, 1e-9)
        z_ratio = float(np.clip(z_ratio, 0.25, 0.95))

        surface_theme = dict(
            scene=dict(
                xaxis=dict(title="Days", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"]),
                yaxis=dict(title="Sim #", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"]),
                zaxis=dict(title="Price ($)", backgroundcolor=C["bg2"], gridcolor=C["border"],
                           zerolinecolor=C["border"], color=C["muted"], range=[z_low, z_high]),
                aspectmode="manual",
                aspectratio=dict(x=1.35, y=max(0.45, y_span / x_span), z=z_ratio),
                camera=dict(eye=dict(x=1.55, y=-1.45, z=0.72)),
            ),
            margin=dict(l=20, r=20, b=20, t=96),
            title=dict(
                text=(
                    f"<b>Monte Carlo 3D Surface</b> - {self.eng.ticker}<br>"
                    f"{n} sims x {N}d - mu={self.eng.mu:+.2%} - sigma={self.eng.sigma:.2%}"
                ),
                x=0.5, y=0.97, xanchor="center", yanchor="top",
                font=dict(size=17, color=C["title"]),
            ),
        )

        s0_mesh_x = np.array([[0, N], [0, N]], dtype=float)
        s0_mesh_y = np.array([[0, 0], [n - 1, n - 1]], dtype=float)
        s0_mesh_z = np.full((2, 2), S0, dtype=float)

        fig_surface = go.Figure()
        fig_surface.add_trace(
            go.Surface(
                x=x, y=y, z=Z, colorscale="Plasma", opacity=0.95,
                colorbar=dict(title="Price ($)", len=0.68),
                contours=dict(
                    z=dict(show=True, usecolormap=True, project_z=True,
                           start=z_low, end=z_high, size=max((z_high - z_low) / 12.0, 1e-6))
                ),
                hovertemplate="Day=%{x}<br>Sim=%{y}<br>Price=$%{z:,.2f}<extra></extra>",
            )
        )
        fig_surface.add_trace(
            go.Surface(
                x=s0_mesh_x, y=s0_mesh_y, z=s0_mesh_z,
                colorscale=[[0, C["red"]], [1, C["red"]]], opacity=0.18,
                showscale=False, hoverinfo="skip",
            )
        )
        fig_surface.add_trace(
            go.Scatter3d(
                x=x,
                y=np.full_like(x, (n - 1) / 2.0, dtype=float),
                z=Z.mean(axis=0),
                mode="lines",
                line=dict(color="white", width=6),
                name="Mean path",
                hovertemplate="Day=%{x}<br>Mean=$%{z:,.2f}<extra></extra>",
            )
        )
        fig_surface.update_layout(**self._plotly_theme(), **surface_theme)

        path_surf = OUT / f"mc_surface_{self.eng.ticker}.html"
        fig_surface.write_html(
            str(path_surf),
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
        )
        rlog(f"  [green]OK[/green] 3D interactive surface -> [cyan]{path_surf}[/cyan]")
        if show:
            fig_surface.show(config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False})

        if no_anim:
            return

        legacy_frames = OUT / f"frames_{self.eng.ticker}"
        if legacy_frames.exists():
            for png_file in legacy_frames.glob("frame_*.png"):
                try:
                    png_file.unlink()
                except OSError:
                    pass
            try:
                legacy_frames.rmdir()
            except OSError:
                pass

        n_paths = len(paths)
        rng = np.random.default_rng(17)
        idx = rng.choice(len(paths), n_paths, replace=False)
        sample_paths = paths[idx]

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
            go.Surface(
                x=np.array([[0, N], [0, N]], dtype=float),
                y=np.array([[0, 0], [n_paths - 1, n_paths - 1]], dtype=float),
                z=np.full((2, 2), S0, dtype=float),
                colorscale=[[0, C["red"]], [1, C["red"]]],
                opacity=0.12,
                showscale=False,
                hoverinfo="skip",
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
                           zerolinecolor=C["border"], color=C["muted"]),
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

    def render_mc_diagnostics(self, adv: dict, show=True):
        n_grid = np.array(adv["convergence"]["n_grid"], dtype=float)
        est = np.array(adv["convergence"]["estimates"], dtype=float)
        se = np.array(adv["convergence"]["stderrs"], dtype=float)
        bs = float(adv.get("black_scholes_call", np.nan))

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Convergence: Price Estimate",
                "Convergence: Std Error Decay",
                "Variance Reduction Ratios",
                "Advanced Option Prices",
            ),
            vertical_spacing=0.16,
            horizontal_spacing=0.10,
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        )

        fig.add_trace(go.Scatter(
            x=n_grid,
            y=est,
            mode="lines+markers",
            name="MC estimate",
            line=dict(color=C["cyan"], width=2),
            marker=dict(size=7),
            error_y=dict(type="data", array=1.96 * se, visible=True, color=C["muted"], thickness=1),
            hovertemplate="N=%{x}<br>Price=$%{y:.4f}<extra></extra>",
        ), row=1, col=1)
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

        ref = se[0] * np.sqrt(n_grid[0] / n_grid)
        fig.add_trace(go.Scatter(
            x=n_grid,
            y=se,
            mode="lines+markers",
            name="stderr",
            line=dict(color=C["green"], width=2),
            marker=dict(size=7),
            hovertemplate="N=%{x}<br>SE=%{y:.5f}<extra></extra>",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=n_grid,
            y=ref,
            mode="lines",
            name="O(1/sqrt(N))",
            line=dict(color=C["purple"], width=2, dash="dash"),
            hovertemplate="N=%{x}<br>Ref=%{y:.5f}<extra></extra>",
        ), row=1, col=2)

        labels = ["Antithetic", "QMC Sobol", "Control Variate"]
        vals = [
            float(adv["variance_reduction_ratio_antithetic"]),
            float(adv["variance_reduction_ratio_qmc"]),
            float(adv["variance_reduction_ratio_cv"]),
        ]
        fig.add_trace(go.Bar(
            x=labels,
            y=vals,
            marker=dict(color=[C["blue"], C["green"], C["amber"]]),
            name="Variance ratio",
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            hovertemplate="%{x}<br>Ratio=%{y:.4f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(
            y=1.0,
            line_color=C["red"],
            line_dash="dot",
            row=2,
            col=1,
            annotation_text="Baseline 1.0",
            annotation_font=dict(color=C["red"], size=9),
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
        ), row=2, col=2)

        layout = self._plotly_theme()
        layout.update(dict(
            title=dict(
                text=(
                    f"<b>Monte Carlo Diagnostics</b> - {self.eng.ticker}<br>"
                    f"Convergence slope={adv['convergence']['loglog_slope']:+.3f} (target -0.500)"
                ),
                x=0.5, y=0.98, xanchor="center", yanchor="top",
                font=dict(size=15, color=C["title"]),
            ),
            height=980,
            margin=dict(l=70, r=60, t=110, b=110),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="center", x=0.5),
        ))
        fig.update_layout(**layout)

        fig.update_xaxes(title_text="Paths (N)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Paths (N, log)", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Std Error (log)", type="log", row=1, col=2)
        fig.update_xaxes(title_text="Method", row=2, col=1)
        fig.update_yaxes(title_text="Variance Ratio", row=2, col=1)
        fig.update_xaxes(title_text="Instrument", row=2, col=2)
        fig.update_yaxes(title_text="Price ($)", row=2, col=2)

        for r in [1, 2]:
            for c in [1, 2]:
                fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=r, col=c)
                fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=r, col=c)

        path_out = OUT / f"mc_diagnostics_{self.eng.ticker}.html"
        fig.write_html(str(path_out), include_plotlyjs="cdn", config={"displayModeBar": True, "scrollZoom": True})
        rlog(f"  [green]OK[/green] MC diagnostics -> [cyan]{path_out}[/cyan]")
        if show:
            fig.show()
        return fig

    def render_all(self, st: dict, N: int, paths: np.ndarray, show=True, no3d=False, adv: dict | None = None):
        rlog("  [1/3] Rendering Plotly main dashboard ...")
        self.render_main(st, N, show=show, adv=adv)
        if adv:
            rlog("  [2/3] Rendering Plotly MC diagnostics ...")
            self.render_mc_diagnostics(adv, show=show)
        if not no3d:
            rlog("  [3/3] Rendering Plotly 3D views ...")
            self.render_3d(paths, N, show=show)




# RICH STATS TABLE
def print_stats_table(eng: GBMEngine, st: dict, N: int, adv: dict | None = None):
    adv = adv or {}
    if not HAS_RICH:
        print(f"\n{'=' * 55}")
        print(f"  {eng.ticker}  Monte Carlo Results  ({N}d horizon)")
        print(f"{'=' * 55}")
        for k, v in [
            ("S0 (last close)", f"${st['S0']:,.2f}"),
            ("Drift mu", f"{eng.mu:+.2%}"),
            ("Volatility sigma", f"{eng.sigma:.2%}"),
            ("Horizon", f"{N} trading days"),
            ("Simulations", str(st['paths'].shape[0])),
            ("Mean final", f"${st['mean']:,.2f}"),
            ("Mean stderr", f"${st['mean_stderr']:,.4f}"),
            ("Median final", f"${st['median']:,.2f}"),
            ("P(profit)", f"{st['prob_up']:.1%}"),
            ("P(profit) stderr", f"{st['prob_up_stderr']:.3%}"),
            ("VaR 95%", f"-${st['var95']:,.2f}"),
            ("CVaR 95%", f"-${st['cvar95']:,.2f}"),
            ("Sharpe", f"{st['sharpe']:.3f}"),
        ]:
            print(f"  {k:<24} {v}")
        if adv:
            print("\n  Advanced MC Pricing")
            print(f"  {'European Call (anti)':<24} ${adv['euro_antithetic']['price']:,.4f} +- {adv['euro_antithetic']['stderr']:.4f}")
            print(f"  {'Asian Call':<24} ${adv['asian_call']['price']:,.4f} +- {adv['asian_call']['stderr']:.4f}")
            print(f"  {'Barrier Up-Out':<24} ${adv['barrier_up_out_call']['price']:,.4f} +- {adv['barrier_up_out_call']['stderr']:.4f}")
            print(f"  {'American Put (LSMC)':<24} ${adv['american_put_lsmc']['price']:,.4f} +- {adv['american_put_lsmc']['stderr']:.4f}")
            print(f"  {'Convergence slope':<24} {adv['convergence']['loglog_slope']:.3f} (target -0.5)")
        return

    tbl = Table(
        title=f"[bold cyan]{eng.ticker}[/bold cyan]  -  Monte Carlo Results  "
              f"[dim]({N}d / {N//21}mo horizon)[/dim]",
        box=box.SIMPLE_HEAVY, border_style="dim",
        show_header=True, header_style="bold dim"
    )
    tbl.add_column("Metric",  style="dim",  width=24)
    tbl.add_column("Value",   style="bold", justify="right", width=18)
    tbl.add_column("Context", style="dim",  width=32)

    def row(lbl, val, ctx=""):
        tbl.add_row(lbl, val, ctx)
    def gr(v): return "green" if v else "red"

    row("Last close",      f"[cyan]${st['S0']:,.2f}[/cyan]", "calibration anchor")
    row("Drift  mu",        f"[{gr(eng.mu>=0)}]{eng.mu:+.2%}[/{gr(eng.mu>=0)}]",
                           "annualised log-return drift")
    row("Volatility  sigma",   f"[yellow]{eng.sigma:.2%}[/yellow]", "annualised std")
    row("Horizon",         f"{N}d  /  {N//21}mo", f"T = {N/252:.2f} years")
    row("Simulations",     f"{st['paths'].shape[0]}", "GBM paths")
    tbl.add_section()
    row("Mean (T)",        f"[{gr(st['mean']>st['S0'])}]${st['mean']:,.2f}[/{gr(st['mean']>st['S0'])}]")
    row("Mean stderr",      f"${st['mean_stderr']:,.4f}", "MC standard error")
    row("95% CI (mean)",    f"${st['mean']-st['mean_ci95_halfwidth']:,.2f} .. ${st['mean']+st['mean_ci95_halfwidth']:,.2f}")
    row("Median (T)",      f"[yellow]${st['median']:,.2f}[/yellow]")
    row("5th / 95th",      f"${st['terminal_pct']['5']:,.2f}  /  ${st['terminal_pct']['95']:,.2f}")
    tbl.add_section()
    row("P(profit)",       f"[{gr(st['prob_up']>0.5)}]{st['prob_up']:.1%}[/{gr(st['prob_up']>0.5)}]",
                           "paths ending above S0")
    row("P(profit) stderr", f"{st['prob_up_stderr']:.3%}", "binomial MC error bar")
    row("P(double)",       f"[purple]{st['prob_2x']:.1%}[/purple]", "above 2xS0")
    row("P(halve)",        f"[red]{st['prob_half']:.1%}[/red]", "below 1/2S0")
    tbl.add_section()
    row("VaR  95% / 99%",  f"[red]-${st['var95']:,.2f}  /  -${st['var99']:,.2f}[/red]")
    row("CVaR 95% / 99%",  f"[red]-${st['cvar95']:,.2f}  /  -${st['cvar99']:,.2f}[/red]")
    tbl.add_section()
    row("Sharpe ratio",    f"[{gr(st['sharpe']>1)}]{st['sharpe']:+.3f}[/{gr(st['sharpe']>1)}]",
                           "vs 5% risk-free rate")
    row("Sortino ratio",   f"[{gr(st['sortino']>1)}]{st['sortino']:+.3f}[/{gr(st['sortino']>1)}]")
    row("Calmar ratio",    f"[{gr(st['calmar']>0.5)}]{st['calmar']:+.3f}[/{gr(st['calmar']>0.5)}]")
    row("Hist max DD",     f"[red]{st['hist_max_dd']:.2%}[/red]")
    row("Win rate",        f"[{gr(st['win_rate']>0.5)}]{st['win_rate']:.1%}[/{gr(st['win_rate']>0.5)}]")
    row("Avg win / loss",  f"[green]+${st['avg_win']:,.2f}[/green]  /  [red]-${st['avg_loss']:,.2f}[/red]")
    tbl.add_section()
    row("Log-return skew", f"{st['lr_skew']:+.3f}", ">0 = right tail")
    row("Excess kurtosis", f"{st['lr_kurt']:+.3f}", ">0 = fatter tails")
    row("Jarque-Bera p",   f"{st['jb_p']:.4f}", "< 0.05 -> reject normality")

    if adv:
        tbl.add_section()
        row("Euro call (pseudo)", f"${adv['euro_plain']['price']:,.4f}", f"SE={adv['euro_plain']['stderr']:.4f}")
        row("Euro call (anti)", f"${adv['euro_antithetic']['price']:,.4f}", f"SE={adv['euro_antithetic']['stderr']:.4f}")
        row("Euro call (QMC)", f"${adv['euro_qmc']['price']:,.4f}", f"SE={adv['euro_qmc']['stderr']:.4f}")
        row("Euro call (CV)", f"${adv['euro_control_variate']['price']:,.4f}", f"SE={adv['euro_control_variate']['stderr']:.4f}")
        row("Asian call", f"${adv['asian_call']['price']:,.4f}", f"SE={adv['asian_call']['stderr']:.4f}")
        row("Barrier up-out", f"${adv['barrier_up_out_call']['price']:,.4f}", f"SE={adv['barrier_up_out_call']['stderr']:.4f}")
        row("American put LSMC", f"${adv['american_put_lsmc']['price']:,.4f}", f"SE={adv['american_put_lsmc']['stderr']:.4f}")
        row("Pathwise Delta", f"{adv['pathwise_greeks']['delta']:+.4f}", f"SE={adv['pathwise_greeks']['delta_stderr']:.4f}")
        row("Pathwise Vega", f"{adv['pathwise_greeks']['vega']:+.4f}", f"SE={adv['pathwise_greeks']['vega_stderr']:.4f}")
        row("Conv slope", f"{adv['convergence']['loglog_slope']:+.3f}", "expected -0.500")
        row("Anti variance ratio", f"{adv['variance_reduction_ratio_antithetic']:.3f}", "SE(anti)^2 / SE(pseudo)^2")
        row("QMC variance ratio", f"{adv['variance_reduction_ratio_qmc']:.3f}", "SE(qmc)^2 / SE(pseudo)^2")
        row("CV variance ratio", f"{adv['variance_reduction_ratio_cv']:.3f}", "SE(cv)^2 / SE(pseudo)^2")
        row("Multi-asset corr", f"{adv['multi_asset_emp_corr']:+.3f}", "empirical corr (2-asset sample)")

    con.print(tbl)

    p = OUT / f"mc_report_{eng.ticker}.txt"
    with open(p, "w") as f:
        f.write(f"Monte Carlo Report - {eng.ticker}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for k, v in [
            ("Ticker", eng.ticker), ("S0", f"${st['S0']:,.2f}"),
            ("Mu", f"{eng.mu:+.2%}"), ("Sigma", f"{eng.sigma:.2%}"),
            ("N_days", str(N)), ("Sims", str(st['paths'].shape[0])),
            ("Mean", f"${st['mean']:,.2f}"), ("Median", f"${st['median']:,.2f}"),
            ("Mean_stderr", f"${st['mean_stderr']:,.4f}"),
            ("P_profit", f"{st['prob_up']:.1%}"),
            ("P_profit_stderr", f"{st['prob_up_stderr']:.3%}"),
            ("VaR95", f"-${st['var95']:,.2f}"), ("CVaR95", f"-${st['cvar95']:,.2f}"),
            ("Sharpe", f"{st['sharpe']:.3f}"), ("Sortino", f"{st['sortino']:.3f}"),
            ("Calmar", f"{st['calmar']:.3f}"), ("MaxDD", f"{st['hist_max_dd']:.2%}"),
        ]:
            f.write(f"{k:<20} {v}\n")
        if adv:
            f.write("\nAdvanced_Monte_Carlo\n")
            f.write(f"{'Euro_call_pseudo':<20} ${adv['euro_plain']['price']:,.4f} | SE={adv['euro_plain']['stderr']:.4f}\n")
            f.write(f"{'Euro_call_anti':<20} ${adv['euro_antithetic']['price']:,.4f} | SE={adv['euro_antithetic']['stderr']:.4f}\n")
            f.write(f"{'Euro_call_qmc':<20} ${adv['euro_qmc']['price']:,.4f} | SE={adv['euro_qmc']['stderr']:.4f}\n")
            f.write(f"{'Euro_call_cv':<20} ${adv['euro_control_variate']['price']:,.4f} | SE={adv['euro_control_variate']['stderr']:.4f}\n")
            f.write(f"{'Asian_call':<20} ${adv['asian_call']['price']:,.4f} | SE={adv['asian_call']['stderr']:.4f}\n")
            f.write(f"{'Barrier_up_out':<20} ${adv['barrier_up_out_call']['price']:,.4f} | SE={adv['barrier_up_out_call']['stderr']:.4f}\n")
            f.write(f"{'American_put_lsmc':<20} ${adv['american_put_lsmc']['price']:,.4f} | SE={adv['american_put_lsmc']['stderr']:.4f}\n")
            f.write(f"{'Pathwise_delta':<20} {adv['pathwise_greeks']['delta']:+.4f} | SE={adv['pathwise_greeks']['delta_stderr']:.4f}\n")
            f.write(f"{'Pathwise_vega':<20} {adv['pathwise_greeks']['vega']:+.4f} | SE={adv['pathwise_greeks']['vega_stderr']:.4f}\n")
            f.write(f"{'Convergence_slope':<20} {adv['convergence']['loglog_slope']:+.4f}\n")
            f.write(f"{'Var_ratio_anti':<20} {adv['variance_reduction_ratio_antithetic']:.4f}\n")
            f.write(f"{'Var_ratio_qmc':<20} {adv['variance_reduction_ratio_qmc']:.4f}\n")
            f.write(f"{'Var_ratio_cv':<20} {adv['variance_reduction_ratio_cv']:.4f}\n")
            f.write(f"{'Multi_asset_corr':<20} {adv['multi_asset_emp_corr']:+.4f}\n")
    rlog(f"  [green]OK[/green] Report saved -> [cyan]{p}[/cyan]")




############################################################################################################################################################################################



# MAIN
def get_args():
    p = argparse.ArgumentParser(
        description="Monte Carlo Stock Terminal  v4.0",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--ticker",   default="",    help="Stock ticker, e.g. AAPL")
    p.add_argument("--sims",     type=int, default=0,   help="Number of simulations")
    p.add_argument("--days",     type=int, default=0,   help="Forecast horizon (trading days)")
    p.add_argument("--seed",     type=int, default=42,  help="Random seed for reproducible simulation")
    p.add_argument("--no3d",     action="store_true",   help="Skip 3D visualisations")
    p.add_argument("--period",   default="2y",          help="Historical data period (yfinance)")
    p.add_argument(
        "--renderer",
        choices=["matplotlib", "plotly", "both"],
        default="",
        help=(
            "Rendering engine:\n"
            "  matplotlib  -> static PNG files (publication-quality)\n"
            "  plotly      -> interactive HTML files (hover/zoom/pan)\n"
            "  both        -> generate both sets of outputs\n"
            "  (default: prompt user if not set)"
        )
    )
    return p.parse_args()

def _prompt(msg, default, cast=str):
    try:
        raw = input(msg).strip()
        return cast(raw) if raw else default
    except (ValueError, EOFError):
        return default

def _choose_renderer(arg_renderer: str) -> str:
    if arg_renderer in ("matplotlib", "plotly", "both"):
        return arg_renderer

    if HAS_RICH:
        con.rule("[dim]Renderer[/dim]")
        con.print("  Choose output renderer:")
        con.print("  [cyan]1[/cyan] -> [bold]Plotly[/bold]      interactive HTML (hover, zoom, pan)")
        con.print("  [cyan]2[/cyan] -> [bold]Matplotlib[/bold]  static PNG  (publication quality)")
        con.print("  [cyan]3[/cyan] -> [bold]Both[/bold]        generate all outputs")
        choice = _prompt("  Your choice [1/2/3, default 1]: ", "1")
    else:
        print("\nRenderer: 1=Plotly  2=Matplotlib  3=Both")
        choice = _prompt("Choice [1]: ", "1")

    return {"1": "plotly", "2": "matplotlib", "3": "both"}.get(choice.strip(), "plotly")


def _run_advanced_pricing(eng: GBMEngine, N: int, nsims: int, seed: int) -> dict:
    k_atm = eng.S0
    pricing_sims = max(2000, nsims)
    base = int(seed)

    euro_plain = eng.price_european_option_mc(
        K=k_atm, r=0.05, N=N, num_sim=pricing_sims,
        seed=base + 101,
        random_method="pseudo", antithetic=False, control_variate=False,
    )
    euro_anti = eng.price_european_option_mc(
        K=k_atm, r=0.05, N=N, num_sim=pricing_sims,
        seed=base + 102,
        random_method="pseudo", antithetic=True, control_variate=False,
    )
    euro_qmc = eng.price_european_option_mc(
        K=k_atm, r=0.05, N=N, num_sim=pricing_sims,
        seed=base + 103,
        random_method="sobol", antithetic=False, control_variate=False,
    )
    euro_cv = eng.price_european_option_mc(
        K=k_atm, r=0.05, N=N, num_sim=pricing_sims,
        seed=base + 104,
        random_method="pseudo", antithetic=True, control_variate=True,
    )
    asian_call = eng.price_asian_option_mc(
        K=k_atm, r=0.05, N=N, num_sim=pricing_sims,
        seed=base + 105,
        option_type="call", random_method="pseudo", antithetic=True, control_variate=True,
    )
    barrier_call = eng.price_barrier_option_mc(
        K=k_atm, barrier=1.2 * eng.S0, r=0.05, N=N, num_sim=pricing_sims,
        seed=base + 106,
        option_type="call", barrier_type="up-and-out", random_method="pseudo", antithetic=True,
    )
    american_put = eng.price_american_option_lsmc(
        K=k_atm, r=0.05, N=N, num_sim=max(3000, pricing_sims),
        seed=base + 107,
        option_type="put", random_method="pseudo", antithetic=True,
    )
    greeks = eng.pathwise_greeks(
        K=k_atm, r=0.05, N=N, num_sim=pricing_sims,
        seed=base + 108,
        option_type="call", random_method="pseudo", antithetic=True,
    )
    conv = eng.convergence_analysis(
        K=k_atm, r=0.05, N=N, option_type="call",
        n_grid=[250, 500, 1000, 2000, 4000, 8000],
        antithetic=True, random_method="pseudo", control_variate=False,
        seed=base + 109,
    )
    multi_paths = eng.simulate_multi_asset_cholesky(
        S0_vec=np.array([eng.S0, eng.S0 * 0.95]),
        mu_vec=np.array([eng.mu, eng.mu * 0.9]),
        sigma_vec=np.array([eng.sigma, eng.sigma * 1.1]),
        corr=np.array([[1.0, 0.55], [0.55, 1.0]]),
        N=N,
        num_sim=max(2000, nsims),
        seed=base + 110,
        risk_neutral_rate=None,
    )
    ret_a = np.log(multi_paths[:, -1, 0] / multi_paths[:, 0, 0])
    ret_b = np.log(multi_paths[:, -1, 1] / multi_paths[:, 0, 1])
    emp_corr = float(np.corrcoef(ret_a, ret_b)[0, 1])

    return {
        "euro_plain": euro_plain,
        "euro_antithetic": euro_anti,
        "euro_qmc": euro_qmc,
        "euro_control_variate": euro_cv,
        "asian_call": asian_call,
        "barrier_up_out_call": barrier_call,
        "american_put_lsmc": american_put,
        "pathwise_greeks": greeks,
        "convergence": conv,
        "black_scholes_call": eng.black_scholes_price(K=k_atm, T=N / 252.0, r=0.05, option_type="call"),
        "variance_reduction_ratio_antithetic": (euro_anti["stderr"] ** 2) / (euro_plain["stderr"] ** 2 + 1e-16),
        "variance_reduction_ratio_qmc": (euro_qmc["stderr"] ** 2) / (euro_plain["stderr"] ** 2 + 1e-16),
        "variance_reduction_ratio_cv": (euro_cv["stderr"] ** 2) / (euro_plain["stderr"] ** 2 + 1e-16),
        "multi_asset_emp_corr": emp_corr,
    }

def main():
    args = get_args()
    print_banner()

    if HAS_RICH:
        con.rule("[dim]Configuration[/dim]")
    else:
        plain_line()

    ticker   = (args.ticker or
                _prompt("  Ticker symbol [default: AAPL]: ", "AAPL")).upper().strip()
    N        = (args.days or
                _prompt("  Forecast days  [default: 252 = 1 year]: ", 252, int))
    nsims    = (args.sims or
                _prompt("  Simulations    [default: 500]: ", 500, int))
    renderer = _choose_renderer(args.renderer)

    N     = max(1, N)
    nsims = max(1, nsims)

    if HAS_RICH:
        con.rule()
    else:
        plain_line()

    eng = GBMEngine(ticker)
    adv = {}

    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=35),
            TimeElapsedColumn(),
            console=con, transient=True
        ) as prog:
            t1 = prog.add_task("Fetching data ...",            total=4)
            eng.fetch(args.period, seed=args.seed);            prog.advance(t1)
            t2 = prog.add_task("Calibrating parameters ...",   total=4)
            eng.calibrate();                                  prog.advance(t2)
            t3 = prog.add_task(f"Simulating {nsims} paths ...", total=4)
            paths, tpts = eng.simulate(N=N, num_sim=nsims, seed=args.seed)
            prog.advance(t3)
            t4 = prog.add_task("Computing statistics ...",     total=4)
            st = eng.compute_stats(paths);                    prog.advance(t4)
    else:
        print("  Fetching data ...")
        eng.fetch(args.period, seed=args.seed)
        print("  Calibrating ...")
        eng.calibrate()
        print(f"  Simulating {nsims} paths x {N} days ...")
        paths, tpts = eng.simulate(N=N, num_sim=nsims, seed=args.seed)
        print("  Computing statistics ...")
        st = eng.compute_stats(paths)

    adv = _run_advanced_pricing(eng=eng, N=N, nsims=nsims, seed=args.seed)

    rlog(f"\n  [green]OK[/green] Calibrated:  "
         f"S0=${eng.S0:,.2f}  mu={eng.mu:+.2%}  sigma={eng.sigma:.2%}")

    if HAS_RICH:
        con.rule("[dim]Results[/dim]")
    print_stats_table(eng, st, N, adv=adv)

    if HAS_RICH:
        con.rule(f"[dim]Visualisations  -  renderer=[cyan]{renderer}[/cyan][/dim]")
    else:
        print(f"\n  Renderer: {renderer}")

    use_mpl     = renderer in ("matplotlib", "both")
    use_plotly  = renderer in ("plotly", "both")

    if use_plotly and not HAS_PLOTLY:
        rlog("  [yellow]WARN Plotly not installed - switching to Matplotlib[/yellow]")
        use_plotly = False
        use_mpl    = True

    figs = []

    if use_mpl:
        rlog("  [bold]Matplotlib[/bold] charts ...")
        mpl_dash = MatplotlibDashboard(eng)
        figs = mpl_dash.render_all(st, N, paths, show=True, no3d=args.no3d, adv=adv)

    if use_plotly:
        rlog("  [bold]Plotly[/bold] interactive ...")
        ply_dash = PlotlyDashboard(eng)
        ply_dash.render_all(st, N, paths, show=False, no3d=args.no3d, adv=adv)

    if HAS_RICH:
        con.rule()
        con.print(f"  [bold green]Done.[/bold green]  "
                  f"All outputs saved to [cyan]{OUT}/[/cyan]")
    else:
        print(f"\n  Done. Outputs saved to {OUT}/")

if __name__ == "__main__":
    main()


"""
stylized_facts.py — Statistical tests and plots for financial stylized facts.

Purpose:
    Provides functions to test whether a simulated price series exhibits the two
    key stylized facts of real financial markets:
    1. Fat tails: Return distributions have excess kurtosis (heavier tails than normal).
    2. Volatility clustering: Large price moves tend to follow large moves (of either sign).

    These are the empirical criteria the hero experiment is evaluated against.

Key design decisions:
    - All statistics are computed from log returns (not simple returns). Log returns are
      approximately additive over time and have nicer statistical properties.
    - We use scipy.stats for the Jarque-Bera test (fat tails) and manual ACF computation
      for volatility clustering. Both are straightforward and well-understood.
    - Plotting functions accept an optional matplotlib Axes object so they can be
      embedded in subplots (as in the notebook) or called standalone.
    - The run_hero_analysis() and generate_hero_plots() functions are convenience
      wrappers that load the hero experiment JSON and compute everything at once.
      They are called by the notebook and the Streamlit app.

References:
    - Cont, R. (2001). Empirical properties of asset returns: stylized facts and
      statistical issues. Quantitative Finance, 1(2), 223-236.
    - Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules
      and the stochastic properties of stock returns. Journal of Finance, 47(5), 1731-1764.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------------
# Core statistical computations
# ---------------------------------------------------------------------------


def compute_log_returns(price_series: list[float]) -> np.ndarray:
    """
    Compute the series of log returns from a price series.

    Log return at time t: r_t = log(P_t / P_{t-1})

    Args:
        price_series: A list or array of prices (length N).

    Returns:
        A numpy array of log returns (length N-1).

    Why this matters:
        Log returns are the standard unit of analysis in financial econometrics.
        Unlike simple returns, log returns are symmetric (a 10% gain followed by
        a 10% loss returns to the starting price only with simple returns).
    """
    prices = np.array(price_series, dtype=float)
    if len(prices) < 2:
        return np.array([])

    # Filter out zero or negative prices to avoid log(0) errors.
    # SIMPLIFICATION: A production version would investigate why prices became
    # non-positive rather than silently skipping them.
    valid_mask = prices > 0
    if not np.all(valid_mask):
        prices = prices[valid_mask]

    returns = np.log(prices[1:] / prices[:-1])
    return returns


def test_fat_tails(returns: np.ndarray) -> dict:
    """
    Test whether a return series has fat tails relative to a normal distribution.

    Uses kurtosis and the Jarque-Bera test.

    Kurtosis of a normal distribution = 3. Excess kurtosis = kurtosis - 3.
    A positive excess kurtosis indicates heavier tails than normal (fat tails).

    The Jarque-Bera test checks whether skewness and excess kurtosis are jointly
    consistent with a normal distribution. A small p-value rejects normality.

    Args:
        returns: A numpy array of log returns.

    Returns:
        A dict with keys:
            - kurtosis: Fisher kurtosis (= excess kurtosis, normalized by N)
            - excess_kurtosis: kurtosis - 3 (for direct comparison to normal)
            - skewness: Sample skewness
            - jb_statistic: Jarque-Bera test statistic
            - jb_pvalue: Jarque-Bera p-value
            - is_fat_tailed: True if excess_kurtosis > 0.5 and jb_pvalue < 0.05
            - n_returns: Number of returns in the sample

    Why this matters:
        Fat tails are the most robust stylized fact in financial data. Any model
        that cannot reproduce them is fundamentally misspecified.
    """
    if len(returns) < 4:
        return {
            "kurtosis": float("nan"),
            "excess_kurtosis": float("nan"),
            "skewness": float("nan"),
            "jb_statistic": float("nan"),
            "jb_pvalue": float("nan"),
            "is_fat_tailed": False,
            "n_returns": len(returns),
        }

    # scipy.stats.kurtosis returns Fisher's kurtosis (excess kurtosis, i.e., kurtosis - 3).
    excess_kurtosis = float(stats.kurtosis(returns, fisher=True))
    kurtosis = excess_kurtosis + 3.0
    skewness = float(stats.skew(returns))

    # Jarque-Bera test for normality.
    jb_stat, jb_pvalue = stats.jarque_bera(returns)

    # We declare fat tails if excess kurtosis is meaningfully positive and JB rejects normality.
    is_fat_tailed = (excess_kurtosis > 0.5) and (jb_pvalue < 0.05)

    return {
        "kurtosis": round(kurtosis, 4),
        "excess_kurtosis": round(excess_kurtosis, 4),
        "skewness": round(skewness, 4),
        "jb_statistic": round(float(jb_stat), 4),
        "jb_pvalue": round(float(jb_pvalue), 6),
        "is_fat_tailed": is_fat_tailed,
        "n_returns": len(returns),
    }


def test_volatility_clustering(returns: np.ndarray, max_lag: int = 20) -> dict:
    """
    Test for volatility clustering via the autocorrelation of squared returns.

    Volatility clustering means that large moves (in either direction) tend to be
    followed by large moves. Formally, the squared return r_t^2 is a proxy for
    volatility; positive autocorrelation in r_t^2 indicates clustering.

    We compute the ACF of squared returns up to max_lag and test whether the
    Ljung-Box statistic is significant (indicating non-zero autocorrelation).

    Args:
        returns: A numpy array of log returns.
        max_lag: Maximum lag for the ACF computation (default: 20).

    Returns:
        A dict with keys:
            - acf_squared_returns: list of ACF values at lags 1..max_lag
            - lags: list of lag indices (1..max_lag)
            - mean_acf_lags_1_5: Mean ACF of squared returns at lags 1-5
            - lb_statistic: Ljung-Box statistic (at lag max_lag)
            - lb_pvalue: Ljung-Box p-value
            - has_clustering: True if mean_acf > 0.02 and lb_pvalue < 0.05
            - n_returns: Number of returns in the sample

    Why this matters:
        Volatility clustering is the empirical basis for GARCH models. Showing that
        heterogeneous agent populations produce it (while homogeneous ones do not)
        is a strong result for the behavioral hypothesis.
    """
    if len(returns) < max_lag + 2:
        return {
            "acf_squared_returns": [],
            "lags": [],
            "mean_acf_lags_1_5": float("nan"),
            "lb_statistic": float("nan"),
            "lb_pvalue": float("nan"),
            "has_clustering": False,
            "n_returns": len(returns),
        }

    squared_returns = returns ** 2
    n = len(squared_returns)
    mean_sq = np.mean(squared_returns)
    variance_sq = np.var(squared_returns)

    if variance_sq < 1e-12:
        # Degenerate case: all returns are identical (e.g., all zeros).
        return {
            "acf_squared_returns": [0.0] * max_lag,
            "lags": list(range(1, max_lag + 1)),
            "mean_acf_lags_1_5": 0.0,
            "lb_statistic": 0.0,
            "lb_pvalue": 1.0,
            "has_clustering": False,
            "n_returns": n,
        }

    # Compute ACF of squared returns at each lag.
    # ACF(k) = Cov(r^2_t, r^2_{t-k}) / Var(r^2_t)
    acf_values = []
    for lag in range(1, max_lag + 1):
        cov = np.mean(
            (squared_returns[lag:] - mean_sq) * (squared_returns[:-lag] - mean_sq)
        )
        acf_values.append(round(float(cov / variance_sq), 4))

    # Ljung-Box statistic (tests whether all ACF values up to max_lag are jointly zero).
    n_obs = n
    lb_stat = n_obs * (n_obs + 2) * sum(
        acf_values[k] ** 2 / (n_obs - (k + 1)) for k in range(max_lag)
    )

    # Ljung-Box p-value: chi-squared with max_lag degrees of freedom.
    lb_pvalue = 1.0 - stats.chi2.cdf(lb_stat, df=max_lag)

    # Mean ACF at short lags (1-5) as a summary measure.
    mean_acf_short = float(np.mean(acf_values[:5]))

    # Clustering declared if short-lag ACF is meaningfully positive and LB rejects.
    has_clustering = (mean_acf_short > 0.02) and (float(lb_pvalue) < 0.05)

    return {
        "acf_squared_returns": acf_values,
        "lags": list(range(1, max_lag + 1)),
        "mean_acf_lags_1_5": round(mean_acf_short, 4),
        "lb_statistic": round(float(lb_stat), 4),
        "lb_pvalue": round(float(lb_pvalue), 6),
        "has_clustering": has_clustering,
        "n_returns": n,
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_return_distribution(
    returns: np.ndarray,
    label: str,
    ax: Optional[plt.Axes] = None,
    color: str = "#3A86FF",
) -> plt.Axes:
    """
    Plot a histogram of log returns with a fitted normal distribution overlay.

    Args:
        returns: A numpy array of log returns.
        label: A label string used in the title and legend.
        ax: Optional matplotlib Axes to plot into. If None, creates a new figure.
        color: Bar color for the histogram.

    Returns:
        The matplotlib Axes object containing the plot.

    Why this matters:
        The visual contrast between the histogram and the normal fit is the
        most intuitive way to demonstrate fat tails to a non-technical audience.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram of returns.
    n_bins = min(50, len(returns) // 5)
    n_bins = max(n_bins, 10)

    ax.hist(
        returns,
        bins=n_bins,
        density=True,
        alpha=0.6,
        color=color,
        label=f"{label} returns",
        edgecolor="none",
    )

    # Fitted normal overlay.
    mu, sigma = float(np.mean(returns)), float(np.std(returns))
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    ax.plot(x_range, normal_pdf, "w--", linewidth=1.5, label="Fitted normal", alpha=0.85)

    # Compute and annotate excess kurtosis.
    excess_kurtosis = float(stats.kurtosis(returns, fisher=True))
    annotation = f"Excess kurtosis: {excess_kurtosis:.2f}"
    ax.text(
        0.97, 0.95, annotation,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1F2E", edgecolor="#3A86FF", alpha=0.8),
    )

    ax.set_title(f"Return Distribution — {label}", fontsize=11)
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    return ax


def plot_volatility_clustering(
    returns: np.ndarray,
    label: str,
    max_lag: int = 20,
    ax: Optional[plt.Axes] = None,
    color: str = "#3A86FF",
) -> plt.Axes:
    """
    Plot the autocorrelation function (ACF) of squared returns.

    Bars above zero indicate that large moves tend to follow large moves
    (volatility clustering). Bars near zero indicate no clustering.

    Args:
        returns: A numpy array of log returns.
        label: A label string used in the title.
        max_lag: Maximum lag to display (default: 20).
        ax: Optional matplotlib Axes to plot into. If None, creates a new figure.
        color: Bar color.

    Returns:
        The matplotlib Axes object containing the plot.

    Why this matters:
        The ACF of squared returns is the canonical diagnostic for volatility clustering.
        Persistent positive values at short lags are a hallmark of GARCH-type dynamics.
    """
    clustering_stats = test_volatility_clustering(returns, max_lag=max_lag)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    lags = clustering_stats["lags"]
    acf_values = clustering_stats["acf_squared_returns"]

    if not lags:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        return ax

    # 95% confidence band for zero autocorrelation: ±1.96 / sqrt(N).
    n = clustering_stats["n_returns"]
    conf_bound = 1.96 / math.sqrt(max(n, 1))

    ax.bar(lags, acf_values, color=color, alpha=0.7, width=0.7)
    ax.axhline(y=conf_bound, color="white", linestyle="--", linewidth=1, alpha=0.6, label="95% CI")
    ax.axhline(y=-conf_bound, color="white", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.4)

    # Annotate whether clustering is present.
    clustering_present = clustering_stats["has_clustering"]
    annotation = "Clustering DETECTED" if clustering_present else "No clustering"
    annotation_color = "#50FA7B" if clustering_present else "#FF6B6B"

    ax.text(
        0.97, 0.95, annotation,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        color=annotation_color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1F2E", edgecolor=annotation_color, alpha=0.8),
    )

    ax.set_title(f"ACF of Squared Returns — {label}", fontsize=11)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend(fontsize=8)

    return ax


def plot_price_series(
    price_series: list[float],
    label: str,
    ax: Optional[plt.Axes] = None,
    color: str = "#3A86FF",
) -> plt.Axes:
    """
    Plot the price series over time.

    Args:
        price_series: A list of prices (one per tick).
        label: A label string used in the title.
        ax: Optional matplotlib Axes to plot into. If None, creates a new figure.
        color: Line color.

    Returns:
        The matplotlib Axes object containing the plot.

    Why this matters:
        The price series is the most direct visual representation of the market's
        behavior. Comparing price series across conditions shows the qualitative
        differences even before computing formal statistics.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    ticks = list(range(len(price_series)))
    ax.plot(ticks, price_series, color=color, linewidth=1.0, alpha=0.9)
    ax.axhline(y=price_series[0], color="white", linewidth=0.5, linestyle=":", alpha=0.4)

    ax.set_title(f"Price Series — {label}", fontsize=11)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Price")

    return ax


# ---------------------------------------------------------------------------
# Hero experiment analysis
# ---------------------------------------------------------------------------


def run_hero_analysis(hero_results_path: str) -> dict:
    """
    Load hero experiment results and compute all stylized facts statistics.

    Args:
        hero_results_path: Path to the hero_experiment.json file.

    Returns:
        A dict mapping condition label -> statistics dict. Each statistics dict
        contains: fat_tails (from test_fat_tails), volatility_clustering (from
        test_volatility_clustering), and n_returns.

    Why this matters:
        This is the main analysis entry point. It encapsulates the full pipeline
        from raw price series to interpretable statistics in one function.
    """
    with open(hero_results_path, "r") as f:
        hero_data = json.load(f)

    analysis = {}

    for label, condition_data in hero_data.items():
        price_series = condition_data["price_series"]
        returns = compute_log_returns(price_series)

        fat_tails_stats = test_fat_tails(returns)
        clustering_stats = test_volatility_clustering(returns, max_lag=20)

        analysis[label] = {
            "name": condition_data["name"],
            "fat_tails": fat_tails_stats,
            "volatility_clustering": clustering_stats,
            "n_returns": len(returns),
        }

    return analysis


def generate_hero_plots(hero_results_path: str, output_dir: str) -> None:
    """
    Generate and save matplotlib figures for the hero experiment analysis.

    Saves the following figures to output_dir:
    - price_series.png: Price series for all three conditions (3 subplots)
    - return_distributions.png: Return histograms with normal overlay (3 subplots)
    - volatility_clustering.png: ACF of squared returns (3 subplots)
    - kurtosis_comparison.png: Bar chart of excess kurtosis across conditions

    Args:
        hero_results_path: Path to the hero_experiment.json file.
        output_dir: Directory to save the figures.

    Returns:
        None. Saves figures to disk.

    Why this matters:
        Saved figures can be embedded in the talk slides and README without
        needing to re-run the analysis.
    """
    with open(hero_results_path, "r") as f:
        hero_data = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Color palette consistent with the app.
    condition_colors = {
        "momentum_only": "#E8A838",
        "value_only": "#3A86FF",
        "mixed": "#FF6B6B",
    }

    conditions = list(hero_data.keys())
    n_conditions = len(conditions)

    # --- Price series ---
    fig, axes = plt.subplots(n_conditions, 1, figsize=(12, 3 * n_conditions))
    fig.patch.set_facecolor("#0E1117")
    for i, label in enumerate(conditions):
        ax = axes[i] if n_conditions > 1 else axes
        ax.set_facecolor("#1A1F2E")
        color = condition_colors.get(label, "#FFFFFF")
        price_series = hero_data[label]["price_series"]
        plot_price_series(price_series, hero_data[label]["name"], ax=ax, color=color)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    plt.tight_layout()
    plt.savefig(output_path / "price_series.png", dpi=150, bbox_inches="tight", facecolor="#0E1117")
    plt.close()

    # --- Return distributions ---
    fig, axes = plt.subplots(1, n_conditions, figsize=(5 * n_conditions, 4))
    fig.patch.set_facecolor("#0E1117")
    for i, label in enumerate(conditions):
        ax = axes[i] if n_conditions > 1 else axes
        ax.set_facecolor("#1A1F2E")
        color = condition_colors.get(label, "#FFFFFF")
        price_series = hero_data[label]["price_series"]
        returns = compute_log_returns(price_series)
        plot_return_distribution(returns, hero_data[label]["name"], ax=ax, color=color)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    plt.tight_layout()
    plt.savefig(output_path / "return_distributions.png", dpi=150, bbox_inches="tight", facecolor="#0E1117")
    plt.close()

    # --- Volatility clustering ---
    fig, axes = plt.subplots(1, n_conditions, figsize=(5 * n_conditions, 4))
    fig.patch.set_facecolor("#0E1117")
    for i, label in enumerate(conditions):
        ax = axes[i] if n_conditions > 1 else axes
        ax.set_facecolor("#1A1F2E")
        color = condition_colors.get(label, "#FFFFFF")
        price_series = hero_data[label]["price_series"]
        returns = compute_log_returns(price_series)
        plot_volatility_clustering(returns, hero_data[label]["name"], ax=ax, color=color)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    plt.tight_layout()
    plt.savefig(output_path / "volatility_clustering.png", dpi=150, bbox_inches="tight", facecolor="#0E1117")
    plt.close()

    # --- Kurtosis comparison ---
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1A1F2E")

    kurtosis_values = []
    condition_labels = []
    bar_colors = []

    for label in conditions:
        price_series = hero_data[label]["price_series"]
        returns = compute_log_returns(price_series)
        fat_tails = test_fat_tails(returns)
        kurtosis_values.append(fat_tails["excess_kurtosis"])
        condition_labels.append(hero_data[label]["name"])
        bar_colors.append(condition_colors.get(label, "#FFFFFF"))

    x_positions = range(len(conditions))
    bars = ax.bar(x_positions, kurtosis_values, color=bar_colors, alpha=0.8, width=0.5)
    ax.axhline(y=0, color="white", linewidth=1, linestyle="--", alpha=0.5, label="Normal distribution")

    # Annotate each bar with the value.
    for bar, val in zip(bars, kurtosis_values):
        y_pos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=10, color="white", fontweight="bold",
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(condition_labels, fontsize=9, color="white")
    ax.set_ylabel("Excess Kurtosis", color="white")
    ax.set_title("Excess Kurtosis by Condition", fontsize=11, color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=8, labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    plt.savefig(output_path / "kurtosis_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0E1117")
    plt.close()

    print(f"Figures saved to: {output_path}")

"""
analysis.py
===========
Statistics and plots covering all four required measurement dimensions:

  1. Waiting time  – time from joining queue to boarding the elevator.
  2. Total trip time (system time) – arrival at floor to reaching destination.
  3. Time-of-day effects – morning rush vs. midday vs. afternoon vs. evening.
  4. Floor fairness – do upper floors get served while lower floors starve?
     Includes pass_count analysis (how often a full elevator passed someone).

All plotting functions save to disk when out_path is provided and also
call plt.show() — comment out plt.show() if running headless.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict

# ─────────────────────────────────────────────
# Time-of-day buckets (seconds since midnight)
# Must be ordered; gaps are labelled "Other".
# ─────────────────────────────────────────────
TOD_BUCKETS = {
    "Morning\n06-09":    ( 6 * 3600,  9 * 3600),
    "Mid-morning\n09-12":(  9 * 3600, 12 * 3600),
    "Lunchtime\n12-14":  (12 * 3600, 14 * 3600),
    "Afternoon\n14-17":  (14 * 3600, 17 * 3600),
    "Evening\n17-20":    (17 * 3600, 20 * 3600),
}

TOD_ORDER = list(TOD_BUCKETS.keys())   # for plot ordering


def _tod_label(t: float) -> str:
    for label, (lo, hi) in TOD_BUCKETS.items():
        if lo <= t < hi:
            return label
    return "Other"


# ─────────────────────────────────────────────
# Core data preparation
# ─────────────────────────────────────────────

def records_to_df(records: list) -> pd.DataFrame:
    """
    Convert a list of completed passenger dicts into a tidy DataFrame.

    Columns added:
        waiting_time  – board_time − arrival_time  (seconds in queue)
        system_time   – arrival_dest_time − arrival_time  (door-to-door)
        ride_time     – arrival_dest_time − board_time  (time inside car)
        time_of_day   – categorical bucket label
        pass_count    – how many times a full elevator passed this passenger
    """
    df = pd.DataFrame(records)
    df = df.dropna(subset=["board_time", "arrival_dest_time"])
    df["waiting_time"] = df["board_time"]         - df["arrival_time"]
    df["system_time"]  = df["arrival_dest_time"]  - df["arrival_time"]
    df["ride_time"]    = df["arrival_dest_time"]  - df["board_time"]
    df["time_of_day"]  = df["arrival_time"].apply(_tod_label)
    df["pass_count"]   = df["pass_count"].fillna(0).astype(int)
    return df


# ─────────────────────────────────────────────
# Summary statistics helpers
# ─────────────────────────────────────────────

def _stats(series: pd.Series) -> pd.Series:
    """Mean / std / percentiles / max for a numeric series."""
    return pd.Series({
        "n":    len(series),
        "mean": series.mean(),
        "std":  series.std(),
        "p50":  series.quantile(0.50),
        "p90":  series.quantile(0.90),
        "p95":  series.quantile(0.95),
        "max":  series.max(),
    })


def waiting_time_stats(df: pd.DataFrame) -> pd.Series:
    """Overall waiting-time statistics."""
    return _stats(df["waiting_time"])


def system_time_stats(df: pd.DataFrame) -> pd.Series:
    """Overall total-trip-time statistics."""
    return _stats(df["system_time"])


def stats_by_floor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-floor summary for BOTH waiting_time and system_time.
    Also includes mean pass_count to measure starvation severity.
    """
    rows = []
    for floor, g in df.groupby("origin_floor"):
        wt = _stats(g["waiting_time"]).add_prefix("wait_")
        st = _stats(g["system_time"]).add_prefix("trip_")
        pc = pd.Series({"mean_pass_count": g["pass_count"].mean(),
                        "pct_passed_ge1":  (g["pass_count"] >= 1).mean() * 100})
        row = pd.concat([wt, st, pc])
        row["floor"] = floor
        rows.append(row)
    return pd.DataFrame(rows).set_index("floor").sort_index()


def stats_by_tod(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-time-of-day summary for waiting_time and system_time.
    """
    rows = []
    for tod, g in df.groupby("time_of_day"):
        wt = _stats(g["waiting_time"]).add_prefix("wait_")
        st = _stats(g["system_time"]).add_prefix("trip_")
        row = pd.concat([wt, st])
        row["time_of_day"] = tod
        rows.append(row)
    result = pd.DataFrame(rows).set_index("time_of_day")
    # Sort by the defined bucket order, put "Other" last
    order = [t for t in TOD_ORDER if t in result.index] + \
            [t for t in result.index if t not in TOD_ORDER]
    return result.reindex(order)


def save_records(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Records saved → '{path}'")


# ─────────────────────────────────────────────
# Plot 1: Waiting-time histogram
# ─────────────────────────────────────────────

def plot_waiting_histogram(
    df: pd.DataFrame,
    title: str = "Waiting-time distribution",
    out_path: Optional[str] = None,
) -> None:
    """Histogram with mean and P95 reference lines."""
    fig, ax = plt.subplots(figsize=(8, 4))
    wt = df["waiting_time"]
    ax.hist(wt, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(wt.mean(), color="tomato", lw=2,
               label=f"Mean = {wt.mean():.0f} s")
    ax.axvline(wt.quantile(0.95), color="orange", lw=2, linestyle="--",
               label=f"P95 = {wt.quantile(0.95):.0f} s")
    ax.set_xlabel("Waiting time (s)")
    ax.set_ylabel("Passengers")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150); print(f"Saved → '{out_path}'")
    plt.show(); plt.close(fig)


# ─────────────────────────────────────────────
# Plot 2: Waiting time vs. number of elevators
# ─────────────────────────────────────────────

def plot_boxplot_by_elevators(
    scenario_dfs: Dict[int, pd.DataFrame],   # {num_elevators: df}
    metric: str = "waiting_time",
    title: Optional[str] = None,
    out_path: Optional[str] = None,
) -> None:
    """
    Side-by-side boxplots showing how waiting OR system time changes
    with more elevators.  Pass metric='system_time' for total trip time.
    """
    frames = []
    for label, df in scenario_dfs.items():
        tmp = df[[metric]].copy()
        tmp["Elevators"] = str(label)
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)

    ylabel = "Waiting time (s)" if metric == "waiting_time" else "Total trip time (s)"
    if title is None:
        title = f"{ylabel} by number of elevators (outliers hidden)"

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=combined, x="Elevators", y=metric,
                palette="Blues", ax=ax, showfliers=False)
    ax.set_xlabel("Number of elevators")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150); print(f"Saved → '{out_path}'")
    plt.show(); plt.close(fig)


# ─────────────────────────────────────────────
# Plot 3: Floor fairness — waiting time by floor
# ─────────────────────────────────────────────

def plot_floor_fairness(
    df: pd.DataFrame,
    title: str = "Waiting time by origin floor",
    out_path: Optional[str] = None,
) -> None:
    """
    Two-panel figure:
      Left  – boxplot of waiting time by floor.
      Right – mean pass_count by floor (how often a full elevator
              passed passengers on each floor).
    Together these reveal whether lower floors are starved during
    downward rush-hour sweeps.
    """
    floors = sorted(df["origin_floor"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: waiting time boxplot
    sns.boxplot(data=df, x="origin_floor", y="waiting_time",
                order=floors, palette="RdYlGn_r", ax=ax1, showfliers=False)
    ax1.set_xlabel("Origin floor")
    ax1.set_ylabel("Waiting time (s)")
    ax1.set_title("Waiting time distribution")

    # Right: mean pass_count per floor (starvation indicator)
    pass_means = df.groupby("origin_floor")["pass_count"].mean().reindex(floors)
    colors = ["#e74c3c" if v > pass_means.mean() else "#2ecc71"
              for v in pass_means]
    ax2.bar(floors, pass_means, color=colors, edgecolor="white")
    ax2.axhline(pass_means.mean(), color="gray", linestyle="--",
                label=f"Mean = {pass_means.mean():.2f}")
    ax2.set_xlabel("Origin floor")
    ax2.set_ylabel("Mean times passed by a full elevator")
    ax2.set_title("'Full elevator' pass-by count\n(higher = more starvation)")
    ax2.legend()

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150); print(f"Saved → '{out_path}'")
    plt.show(); plt.close(fig)


# ─────────────────────────────────────────────
# Plot 4: Time-of-day effects
# ─────────────────────────────────────────────

def plot_tod_comparison(
    df: pd.DataFrame,
    title: str = "Waiting & trip time by time of day",
    out_path: Optional[str] = None,
) -> None:
    """
    Two-panel figure comparing waiting_time and system_time across
    time-of-day buckets.  Shows both the median bar and the spread,
    making morning vs. afternoon rush clearly visible.
    """
    order = [t for t in TOD_ORDER if t in df["time_of_day"].unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    palette = sns.color_palette("muted", len(order))

    sns.boxplot(data=df, x="time_of_day", y="waiting_time", order=order,
                palette=palette, ax=ax1, showfliers=False)
    ax1.set_xlabel("Time of day")
    ax1.set_ylabel("Waiting time (s)")
    ax1.set_title("Waiting time")
    ax1.tick_params(axis='x', labelsize=8)

    sns.boxplot(data=df, x="time_of_day", y="system_time", order=order,
                palette=palette, ax=ax2, showfliers=False)
    ax2.set_xlabel("Time of day")
    ax2.set_ylabel("Total trip time (s)")
    ax2.set_title("Total trip time (queue + ride)")
    ax2.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150); print(f"Saved → '{out_path}'")
    plt.show(); plt.close(fig)


# ─────────────────────────────────────────────
# Plot 5: Waiting time over the day (timeline)
# ─────────────────────────────────────────────

def plot_waiting_over_time(
    df: pd.DataFrame,
    bin_minutes: int = 15,
    title: str = "Mean waiting time over the day",
    out_path: Optional[str] = None,
) -> None:
    """
    Line chart of mean waiting_time in rolling 15-min bins, with a
    shaded band showing ±1 std.  Rush-hour spikes are immediately visible.
    """
    bin_sec = bin_minutes * 60
    d = df.copy()
    d["hour"] = (d["arrival_time"] // bin_sec) * bin_sec / 3600
    agg = d.groupby("hour")["waiting_time"].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(agg["hour"],
                    agg["mean"] - agg["std"],
                    agg["mean"] + agg["std"],
                    alpha=0.2, color="#3498db", label="±1 std")
    ax.plot(agg["hour"], agg["mean"], color="#2980b9", lw=2, marker="o",
            markersize=3, label="Mean wait")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Waiting time (s)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):02d}:00"))
    ax.legend()
    # Shade rush-hour periods
    for (lo, hi), label in [((8, 9.5), "AM rush"), ((12, 14), "Lunch"),
                              ((17, 18.5), "PM rush")]:
        ax.axvspan(lo, hi, alpha=0.08, color="red")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150); print(f"Saved → '{out_path}'")
    plt.show(); plt.close(fig)


# ─────────────────────────────────────────────
# Plot 6: Floor starvation during rush hour
# ─────────────────────────────────────────────

def plot_rush_floor_heatmap(
    df: pd.DataFrame,
    title: str = "Mean waiting time: floor × time-of-day",
    out_path: Optional[str] = None,
) -> None:
    """
    Heatmap: rows = origin floors, cols = time-of-day buckets.
    Reveals whether specific floors are disproportionately penalised
    at specific times (e.g. floors 2-3 during PM downward rush).
    """
    order = [t for t in TOD_ORDER if t in df["time_of_day"].unique()]
    pivot = (df.groupby(["origin_floor", "time_of_day"])["waiting_time"]
               .mean()
               .unstack("time_of_day")
               .reindex(columns=order))

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Mean waiting time (s)"})
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Origin floor")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150); print(f"Saved → '{out_path}'")
    plt.show(); plt.close(fig)


# ─────────────────────────────────────────────
# Plot 7: Elevator count comparison — cost vs. service
# ─────────────────────────────────────────────

def plot_cost_vs_service(
    summary_df: pd.DataFrame,
    elevator_cost: int = 100_000,
    out_path: Optional[str] = None,
) -> None:
    """
    Scatter / line chart: x = total capital cost, y = mean waiting time,
    one series per traffic_scale.  Makes the cost-vs-quality trade-off
    explicit for the decision-maker.

    summary_df must have columns: num_elevators, traffic_scale, mean_wait_s.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    scales  = sorted(summary_df["traffic_scale"].unique())
    colors  = sns.color_palette("Set2", len(scales))

    for scale, color in zip(scales, colors):
        sub = summary_df[summary_df["traffic_scale"] == scale].sort_values("num_elevators")
        costs = sub["num_elevators"] * elevator_cost
        ax.plot(costs, sub["mean_wait_s"], marker="o", color=color,
                lw=2, label=f"Traffic scale = {scale:.1f}")
        for _, row in sub.iterrows():
            ax.annotate(f"{int(row['num_elevators'])} elev.",
                        (row["num_elevators"] * elevator_cost, row["mean_wait_s"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7)

    ax.set_xlabel("Total elevator capital cost ($)")
    ax.set_ylabel("Mean waiting time (s)")
    ax.set_title("Cost vs. service quality trade-off")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.legend()
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150); print(f"Saved → '{out_path}'")
    plt.show(); plt.close(fig)

"""
Runs the full factorial experiment and produces all required measurements:
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from data_input_full      import load_passengers
from elevator_model_full  import run_simulation_full
from analysis_full        import (
    records_to_df, waiting_time_stats, system_time_stats,
    stats_by_floor, stats_by_tod,
    save_records,
    plot_waiting_histogram,
    plot_boxplot_by_elevators,
    plot_floor_fairness,
    plot_tod_comparison,
    plot_waiting_over_time,
    plot_rush_floor_heatmap,
    plot_cost_vs_service,
)

ON_COUNTS_PATH  = "/Users/lionaelsmac/Documents/code/BDA 450/project/OnCounts.xlsx"
OFF_COUNTS_PATH = "/Users/lionaelsmac/Documents/code/BDA 450/project/OffCounts.xlsx"
OUTPUT_DIR      = "results"

N_REPS          = 20          # replications per scenario
ELEVATOR_COUNTS = [1, 2, 3]
TRAFFIC_SCALES  = [0.8, 1.0, 1.2]
BASE_SEED       = 1000
ELEVATOR_COST   = 100_000    # dollars per elevator


# ─────────────────────────────────────────────
# Scenario runner
# ─────────────────────────────────────────────

def run_scenario(num_elevators: int, traffic_scale: float,
                 n_reps: int, base_seed: int) -> pd.DataFrame:
    """
    Run n_reps replications of one scenario.
    Returns a DataFrame of all completed passenger records.
    """
    all_records    = []
    total_unserved = 0

    scenario_offset = num_elevators * 1000 + int(round(traffic_scale * 100))

    for rep in range(n_reps):
        seed = base_seed + scenario_offset + rep
        rng  = np.random.default_rng(seed)
        passengers = load_passengers(ON_COUNTS_PATH, OFF_COUNTS_PATH,
                                     traffic_scale=traffic_scale, rng=rng)

        sim = run_simulation_full(passengers, num_elevators=num_elevators, seed=seed)
        df  = records_to_df(sim["completed"])
        df["replication"]   = rep
        df["num_elevators"] = num_elevators
        df["traffic_scale"] = traffic_scale
        all_records.append(df)
        total_unserved += sim["n_unserved"]

        print(
            f"  elev={num_elevators}, scale={traffic_scale:.1f}, rep={rep+1:2d}: "
            f"n={len(df):4d}, unserved={sim['n_unserved']:3d}, "
            f"mean_wait={df['waiting_time'].mean():6.1f}s, "
            f"mean_trip={df['system_time'].mean():6.1f}s"
        )

    result = pd.concat(all_records, ignore_index=True)
    print(f"  → avg unserved per rep: {total_unserved / n_reps:.1f}")
    return result


def build_summary_row(num_elevators: int, traffic_scale: float,
                      df: pd.DataFrame) -> dict:
    """
    Collapse one scenario's records into a single summary dict.
    """
    wt = waiting_time_stats(df)
    st = system_time_stats(df)
    return {
        "num_elevators":   num_elevators,
        "traffic_scale":   traffic_scale,
        "elevator_cost_$": num_elevators * ELEVATOR_COST,
        "n_passengers":    int(wt["n"]),
        # Waiting time
        "mean_wait_s":     round(wt["mean"], 1),   # name expected by plot_cost_vs_service
        "std_wait_s":      round(wt["std"],  1),
        "p50_wait_s":      round(wt["p50"],  1),
        "p90_wait_s":      round(wt["p90"],  1),
        "p95_wait_s":      round(wt["p95"],  1),
        "max_wait_s":      round(wt["max"],  1),
        # Total trip time
        "mean_trip_s":     round(st["mean"], 1),
        "p90_trip_s":      round(st["p90"],  1),
        "p95_trip_s":      round(st["p95"],  1),
        # Fairness
        "mean_pass_count": round(df["pass_count"].mean(), 3),
        "pct_passed_ge1":  round((df["pass_count"] >= 1).mean() * 100, 1),
    }



def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for path in [ON_COUNTS_PATH, OFF_COUNTS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input file not found: '{path}'")

    summary_rows = []
    all_dfs      = []

    for elev in ELEVATOR_COUNTS:
        for scale in TRAFFIC_SCALES:
            print(f"\n{'='*65}")
            print(f"SCENARIO: {elev} elevator(s), traffic scale = {scale:.1f}")
            print('='*65)
            df = run_scenario(elev, scale, N_REPS, BASE_SEED)
            all_dfs.append(df)
            summary_rows.append(build_summary_row(elev, scale, df))

    # ── Save raw records ─────────────────────────────────────────────
    all_records = pd.concat(all_dfs, ignore_index=True)
    save_records(all_records, f"{OUTPUT_DIR}/all_records.csv")

    # ── Summary table ────────────────────────────────────────────────
    summary      = pd.DataFrame(summary_rows)
    summary_path = f"{OUTPUT_DIR}/summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{'='*65}")
    print("SUMMARY TABLE")
    print('='*65)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))
    print(f"\nSaved → '{summary_path}'")

    # ── Per-floor fairness tables (baseline traffic) ──────────────────
    print(f"\n{'='*65}")
    print("FLOOR FAIRNESS — baseline (scale=1.0)")
    print('='*65)
    for elev in ELEVATOR_COUNTS:
        sub = all_records[(all_records["num_elevators"] == elev) &
                          (all_records["traffic_scale"] == 1.0)]
        print(f"\n  {elev} elevator(s):")
        print(stats_by_floor(sub).round(1).to_string())

    # ── Time-of-day table (2 elevators, baseline) ─────────────────────
    print(f"\n{'='*65}")
    print("TIME-OF-DAY EFFECTS — 2 elevators, scale=1.0")
    print('='*65)
    baseline_2 = all_records[(all_records["num_elevators"] == 2) &
                              (all_records["traffic_scale"] == 1.0)]
    print(stats_by_tod(baseline_2).round(1).to_string())

    # ── Plots ─────────────────────────────────────────────────────────

    # 1. Waiting-time histogram — each elevator count at baseline
    for elev in ELEVATOR_COUNTS:
        sub = all_records[(all_records["num_elevators"] == elev) &
                          (all_records["traffic_scale"] == 1.0)]
        plot_waiting_histogram(
            sub,
            title=f"Waiting time — {elev} elevator(s), scale=1.0",
            out_path=f"{OUTPUT_DIR}/histogram_wait_{elev}elev.png",
        )

    # 2a. Waiting time vs. #elevators (baseline)
    baseline_dfs = {
        e: all_records[(all_records["num_elevators"] == e) &
                       (all_records["traffic_scale"] == 1.0)]
        for e in ELEVATOR_COUNTS
    }
    plot_boxplot_by_elevators(
        baseline_dfs, metric="waiting_time",
        title="Waiting time vs. number of elevators (scale=1.0, outliers hidden)",
        out_path=f"{OUTPUT_DIR}/boxplot_wait_elevators.png",
    )

    # 2b. Total trip time vs. #elevators (baseline)
    plot_boxplot_by_elevators(
        baseline_dfs, metric="system_time",
        title="Total trip time vs. number of elevators (scale=1.0, outliers hidden)",
        out_path=f"{OUTPUT_DIR}/boxplot_trip_elevators.png",
    )

    # 3. Floor fairness — each elevator count at baseline
    for elev in ELEVATOR_COUNTS:
        sub = all_records[(all_records["num_elevators"] == elev) &
                          (all_records["traffic_scale"] == 1.0)]
        plot_floor_fairness(
            sub,
            title=f"Floor fairness — {elev} elevator(s), scale=1.0",
            out_path=f"{OUTPUT_DIR}/floor_fairness_{elev}elev.png",
        )

    # 4. Time-of-day comparison — each elevator count at baseline
    for elev in ELEVATOR_COUNTS:
        sub = all_records[(all_records["num_elevators"] == elev) &
                          (all_records["traffic_scale"] == 1.0)]
        plot_tod_comparison(
            sub,
            title=f"Time-of-day effects — {elev} elevator(s), scale=1.0",
            out_path=f"{OUTPUT_DIR}/tod_{elev}elev.png",
        )

    # 5. Waiting timeline over the day — each elevator count at baseline
    for elev in ELEVATOR_COUNTS:
        sub = all_records[(all_records["num_elevators"] == elev) &
                          (all_records["traffic_scale"] == 1.0)]
        plot_waiting_over_time(
            sub,
            title=f"Mean waiting time over the day — {elev} elevator(s), scale=1.0",
            out_path=f"{OUTPUT_DIR}/waiting_timeline_{elev}elev.png",
        )

    # 6. Floor × time-of-day heatmap — each elevator count at baseline
    for elev in ELEVATOR_COUNTS:
        sub = all_records[(all_records["num_elevators"] == elev) &
                          (all_records["traffic_scale"] == 1.0)]
        plot_rush_floor_heatmap(
            sub,
            title=f"Mean waiting time: floor × time-of-day  ({elev} elevators)",
            out_path=f"{OUTPUT_DIR}/rush_heatmap_{elev}elev.png",
        )

    # 7. Cost vs. service quality — all traffic scales
    plot_cost_vs_service(
        summary,
        elevator_cost=ELEVATOR_COST,
        out_path=f"{OUTPUT_DIR}/cost_vs_service.png",
    )

    print(f"\nAll outputs written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()

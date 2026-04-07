def run_scenario(num_elevators: int, traffic_scale: float,
                 n_reps: int, base_seed: int) -> pd.DataFrame:
    """
    Run n_reps replications of one scenario.
    Returns a DataFrame of all completed passenger records.
    Prints replication-level summaries to stdout.
    """
    all_records = []
    total_unserved = 0

    # Build a scenario-unique offset so seeds never collide across scenarios.
    # e.g. (elev=1, scale=0.8) → offset 1080
    #      (elev=2, scale=1.0) → offset 2100
    #      (elev=3, scale=1.2) → offset 3120
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
    avg_unserved = total_unserved / n_reps
    print(f"  → avg unserved per rep: {avg_unserved:.1f}")
    return result

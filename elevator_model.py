"""
elevator_model.py
=================
SimPy discrete-event simulation of a multi-elevator building.

Key design decisions
--------------------
* Time unit : seconds.
* Floors    : 0 (parking) through 4 (top business floor).

* Each floor has two FIFO queues (plain Python lists):
    - up_queues[floor]   : passengers waiting to go UP
    - down_queues[floor] : passengers waiting to go DOWN
  Plain lists let us inspect lengths and increment pass_count on
  stranded passengers without SimPy Store overhead.

* SCAN algorithm with realistic capacity enforcement:
    - Sweep in one direction; stop at every floor where someone exits
      OR same-direction waiters can board (capacity permitting).
    - If the car arrives FULL at a floor with waiting passengers, those
      passengers have pass_count incremented — they explicitly experience
      the "full elevator passes me by" effect that drives fairness issues.
    - Direction reverses when nothing remains ahead in the current
      direction (no on-board need + no waiting requests ahead).
    - Idle elevators sleep on a shared wakeup Event.

* pass_count per passenger is recorded in output for fairness analysis.

Timing constants — all at the top, easy to change.
"""

import simpy
import numpy as np
import copy
from typing import List, Optional

# ─────────────────────────────────────────────
# Timing / capacity constants
# ─────────────────────────────────────────────
FLOOR_TRAVEL_TIME: float = 3.0
DOOR_OPEN_TIME:    float = 2.0
DOOR_CLOSE_TIME:   float = 2.0
ELEVATOR_CAPACITY: int   = 8
MIN_FLOOR:         int   = 0
MAX_FLOOR:         int   = 4

UP   =  1
DOWN = -1


class Building:
    """Shared floor queues and idle-wakeup signalling."""

    def __init__(self, env: simpy.Environment):
        self.env = env
        n = MAX_FLOOR - MIN_FLOOR + 1
        self.up_queues:   List[List[dict]] = [[] for _ in range(n)]
        self.down_queues: List[List[dict]] = [[] for _ in range(n)]
        self.wakeup: simpy.Event = env.event()

    def queue_for(self, floor: int, direction: int) -> List[dict]:
        idx = floor - MIN_FLOOR
        return self.up_queues[idx] if direction == UP else self.down_queues[idx]

    def any_requests(self) -> bool:
        return any(q for q in self.up_queues + self.down_queues)

    def requests_ahead(self, floor: int, direction: int) -> bool:
        """Any unserved passenger on a floor strictly ahead in direction?"""
        floors = range(floor + 1, MAX_FLOOR + 1) if direction == UP else range(MIN_FLOOR, floor)
        for f in floors:
            idx = f - MIN_FLOOR
            if self.up_queues[idx] or self.down_queues[idx]:
                return True
        return False

    def signal_arrival(self):
        if not self.wakeup.triggered:
            self.wakeup.succeed()
        self.wakeup = self.env.event()


class Elevator:
    """
    One elevator car running the SCAN algorithm with capacity enforcement.

    Fairness modelling
    ------------------
    When the car arrives at a floor full and cannot board waiting passengers,
    each stranded passenger gets pass_count += 1.  This lets analysis.py
    identify floors where passengers are repeatedly bypassed — the classic
    rush-hour starvation effect.
    """

    def __init__(self, env, eid, building, results, start_floor=MIN_FLOOR):
        self.env       = env
        self.eid       = eid
        self.building  = building
        self.results   = results
        self.floor     = start_floor
        self.direction = UP
        self.onboard:  List[dict] = []
        self.process   = env.process(self._run())

    def _full(self): return len(self.onboard) >= ELEVATOR_CAPACITY

    def _need_dir(self, direction):
        for p in self.onboard:
            if direction == UP   and p["destination_floor"] > self.floor: return True
            if direction == DOWN and p["destination_floor"] < self.floor: return True
        return False

    def _should_stop(self):
        # Stop if someone exits here
        if any(p["destination_floor"] == self.floor for p in self.onboard):
            return True
        # Stop if same-direction waiters can board
        if self.building.queue_for(self.floor, self.direction) and not self._full():
            return True
        return False

    def _unload(self):
        remaining = []
        for p in self.onboard:
            if p["destination_floor"] == self.floor:
                p["arrival_dest_time"] = self.env.now
                self.results.append(p)
            else:
                remaining.append(p)
        self.onboard = remaining

    def _load(self):
        q = self.building.queue_for(self.floor, self.direction)
        while q and not self._full():
            p = q.pop(0)
            p["board_time"] = self.env.now
            self.onboard.append(p)
        # If still full and passengers remain, they are passed by
        if self._full():
            for p in q:
                p["pass_count"] += 1

    def _run(self):
        while True:
            # Idle wait
            while not self.building.any_requests() and not self.onboard:
                yield self.building.wakeup

            # Stop at current floor?
            if self._should_stop():
                yield self.env.timeout(DOOR_OPEN_TIME)
                self._unload()
                self._load()
                yield self.env.timeout(DOOR_CLOSE_TIME)
            else:
                # Not stopping — but mark any same-dir waiters as passed if full
                if self._full():
                    for p in self.building.queue_for(self.floor, self.direction):
                        p["pass_count"] += 1

            # Decide direction
            ahead_onboard = self._need_dir(self.direction)
            ahead_waiting = self.building.requests_ahead(self.floor, self.direction)

            if not ahead_onboard and not ahead_waiting:
                opposite = -self.direction
                opp_onboard = self._need_dir(opposite)
                opp_waiting = self.building.requests_ahead(self.floor, opposite)
                opp_here    = bool(self.building.queue_for(self.floor, opposite))
                if opp_onboard or opp_waiting or opp_here:
                    self.direction = opposite
                elif not self.onboard:
                    yield self.building.wakeup
                    continue

            # Enforce boundaries
            if self.floor == MAX_FLOOR: self.direction = DOWN
            if self.floor == MIN_FLOOR: self.direction = UP

            # Move one floor
            nxt = max(MIN_FLOOR, min(MAX_FLOOR, self.floor + self.direction))
            if nxt != self.floor:
                yield self.env.timeout(FLOOR_TRAVEL_TIME)
                self.floor = nxt
            else:
                self.direction = -self.direction
                yield self.env.timeout(0.5)


# ─────────────────────────────────────────────
# Arrival process
# ─────────────────────────────────────────────

def _passenger_arrivals(env, building, passengers):
    for p in passengers:
        delay = p["arrival_time"] - env.now
        if delay > 0:
            yield env.timeout(delay)
        direction = UP if p["destination_floor"] > p["origin_floor"] else DOWN
        building.queue_for(p["origin_floor"], direction).append(p)
        building.signal_arrival()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def run_simulation(
    passengers: List[dict],
    num_elevators: int = 2,
    sim_duration: Optional[float] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Run one replication; return completed passenger records.
    Passengers still waiting at sim_duration are dropped silently.
    Use run_simulation_full() to also capture unserved passengers.
    """
    if not passengers:
        return []
    passengers = [copy.deepcopy(p) for p in passengers]
    for p in passengers:
        p.setdefault("pass_count", 0)
    if sim_duration is None:
        sim_duration = max(p["arrival_time"] for p in passengers) + 7200

    env = simpy.Environment()
    results  = []
    building = Building(env)
    starts = np.linspace(MIN_FLOOR, MAX_FLOOR, num_elevators, endpoint=True)
    for i in range(num_elevators):
        Elevator(env, i, building, results, start_floor=int(round(starts[i])))
    env.process(_passenger_arrivals(env, building, passengers))
    env.run(until=sim_duration)
    return results


def run_simulation_full(
    passengers: List[dict],
    num_elevators: int = 2,
    sim_duration: Optional[float] = None,
    seed: int = 42,
) -> dict:
    """
    Like run_simulation but also returns unserved passengers.

    Returns dict with keys:
        completed   : list of finished-trip passenger dicts
        unserved    : list of passengers still waiting at end
        n_total     : total injected
        n_completed : trips finished
        n_unserved  : still waiting (unserved demand metric)
    """
    if not passengers:
        return {"completed": [], "unserved": [], "n_total": 0,
                "n_completed": 0, "n_unserved": 0}

    passengers = [copy.deepcopy(p) for p in passengers]
    for p in passengers:
        p.setdefault("pass_count", 0)
    if sim_duration is None:
        sim_duration = max(p["arrival_time"] for p in passengers) + 7200

    env = simpy.Environment()
    results  = []
    building = Building(env)
    starts = np.linspace(MIN_FLOOR, MAX_FLOOR, num_elevators, endpoint=True)
    for i in range(num_elevators):
        Elevator(env, i, building, results, start_floor=int(round(starts[i])))
    env.process(_passenger_arrivals(env, building, passengers))
    env.run(until=sim_duration)

    completed_ids = {p["id"] for p in results}
    unserved = [p for p in passengers if p["id"] not in completed_ids]

    return {
        "completed":   results,
        "unserved":    unserved,
        "n_total":     len(passengers),
        "n_completed": len(results),
        "n_unserved":  len(unserved),
    }

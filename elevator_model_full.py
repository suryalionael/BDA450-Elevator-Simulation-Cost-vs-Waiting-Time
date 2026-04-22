"""

SimPy discrete-event simulation of a multi-elevator building.
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


# ─────────────────────────────────────────────
# Building — shared queues and wakeup signal
# ─────────────────────────────────────────────

class Building:
    """
    Holds the shared floor queues and the idle-wakeup event.
    """

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
        """True if any floor has at least one waiting passenger."""
        return any(q for q in self.up_queues + self.down_queues)

    def requests_ahead(self, floor: int, direction: int) -> bool:
        """
        True if any floor strictly ahead in direction has waiting passengers
        (either direction). 
        """
        floors = (range(floor + 1, MAX_FLOOR + 1) if direction == UP
                  else range(MIN_FLOOR, floor))
        for f in floors:
            idx = f - MIN_FLOOR
            if self.up_queues[idx] or self.down_queues[idx]:
                return True
        return False

    def signal_arrival(self) -> None:
        """Wake all idle elevators when a new passenger joins a queue."""
        if not self.wakeup.triggered:
            self.wakeup.succeed()
        self.wakeup = self.env.event()


# ─────────────────────────────────────────────
# Elevator
# ─────────────────────────────────────────────

class Elevator:

    def __init__(self, env: simpy.Environment, eid: int,
                 building: Building, results: list,
                 start_floor: int = MIN_FLOOR):
        self.env      = env
        self.eid      = eid
        self.building = building
        self.results  = results
        self.floor    = start_floor
        self.direction = UP
        self.onboard: List[dict] = []
        self.process  = env.process(self._run())

    # ── Helpers ───────────────────────────────────────────────────────

    def _full(self) -> bool:
        return len(self.onboard) >= ELEVATOR_CAPACITY

    def _need_dir(self, direction: int) -> bool:
        """Any on-board passenger still needs to travel in this direction?"""
        for p in self.onboard:
            if direction == UP   and p["destination_floor"] > self.floor:
                return True
            if direction == DOWN and p["destination_floor"] < self.floor:
                return True
        return False

    def _should_stop(self) -> bool:
        """Stop here? Someone exits, or same-direction waiters can board."""
        if any(p["destination_floor"] == self.floor for p in self.onboard):
            return True
        if self.building.queue_for(self.floor, self.direction) and not self._full():
            return True
        return False

    def _unload(self) -> None:
        remaining = []
        for p in self.onboard:
            if p["destination_floor"] == self.floor:
                p["arrival_dest_time"] = self.env.now
                self.results.append(p)
            else:
                remaining.append(p)
        self.onboard = remaining

    def _load(self) -> None:
        """
        Board as many same-direction waiters as capacity allows.
        """
        q = self.building.queue_for(self.floor, self.direction)
        while q and not self._full():
            p = q.pop(0)
            p["board_time"] = self.env.now
            self.onboard.append(p)
        if self._full():
            for p in q:
                p["pass_count"] += 1

    def _serve_floor(self):
        """Open doors, unload, load, close doors — two SimPy timeouts."""
        yield self.env.timeout(DOOR_OPEN_TIME)
        self._unload()
        self._load()
        yield self.env.timeout(DOOR_CLOSE_TIME)

    # ── Main loop ──────────────────────────────────────────────────────

    def _run(self):
        while True:

            # ── Idle: sleep until a passenger arrives ────────────────────
            while not self.building.any_requests() and not self.onboard:
                yield self.building.wakeup

            # ── Serve current floor if stopping criteria are met ─────────
            if self._should_stop():
                yield from self._serve_floor()
            else:
                # Not stopping — if full, mark same-dir waiters as passed
                if self._full():
                    for p in self.building.queue_for(self.floor, self.direction):
                        p["pass_count"] += 1

            # ── Decide direction ─────────────────────────────────────────
            ahead_onboard = self._need_dir(self.direction)
            ahead_waiting = self.building.requests_ahead(self.floor, self.direction)

            if not ahead_onboard and not ahead_waiting:
                # Nothing remaining in current direction — consider reversing
                opposite    = -self.direction
                opp_onboard = self._need_dir(opposite)
                opp_waiting = self.building.requests_ahead(self.floor, opposite)
                opp_here    = bool(self.building.queue_for(self.floor, opposite))

                if opp_onboard or opp_waiting or opp_here:
                    self.direction = opposite

                    # ── Reversal re-check ────────────────────────────────
                    # Passengers at this floor in the NEW direction were
                    # invisible to _should_stop() above (it checked the old
                    # direction). Without this re-check they would be left
                    # behind even though the car is right in front of them.
                    if self._should_stop():
                        yield from self._serve_floor()

                elif not self.onboard:
                    # Truly idle — wait for the next arrival signal
                    yield self.building.wakeup
                    continue

            # ── Hard floor boundaries ────────────────────────────────────
            if self.floor == MAX_FLOOR:
                self.direction = DOWN
            if self.floor == MIN_FLOOR:
                self.direction = UP

            # ── Move one floor ───────────────────────────────────────────
            nxt = max(MIN_FLOOR, min(MAX_FLOOR, self.floor + self.direction))
            if nxt != self.floor:
                yield self.env.timeout(FLOOR_TRAVEL_TIME)
                self.floor = nxt
            else:
                # Stuck at a boundary with no work — flip and yield briefly
                self.direction = -self.direction
                yield self.env.timeout(0.5)


# ─────────────────────────────────────────────
# Arrival injector
# ─────────────────────────────────────────────

def _passenger_arrivals(env: simpy.Environment,
                        building: Building,
                        passengers: List[dict]):
    for p in passengers:
        delay = p["arrival_time"] - env.now
        if delay > 0:
            yield env.timeout(delay)
        direction = UP if p["destination_floor"] > p["origin_floor"] else DOWN
        building.queue_for(p["origin_floor"], direction).append(p)
        building.signal_arrival()



def _build_env(passengers: List[dict],
               num_elevators: int,
               sim_duration: float):
    """Create and wire up the SimPy environment, building, and elevators."""
    env      = simpy.Environment()
    results  = []
    building = Building(env)
    starts   = np.linspace(MIN_FLOOR, MAX_FLOOR, num_elevators, endpoint=True)
    for i in range(num_elevators):
        Elevator(env, i, building, results, start_floor=int(round(starts[i])))
    env.process(_passenger_arrivals(env, building, passengers))
    return env, results


def run_simulation(
    passengers: List[dict],
    num_elevators: int = 2,
    sim_duration: Optional[float] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Run one replication; return completed passenger records.
    """
    if not passengers:
        return []
    passengers = [copy.deepcopy(p) for p in passengers]
    for p in passengers:
        p.setdefault("pass_count", 0)
    if sim_duration is None:
        sim_duration = max(p["arrival_time"] for p in passengers) + 7200

    env, results = _build_env(passengers, num_elevators, sim_duration)
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
    """
    if not passengers:
        return {"completed": [], "unserved": [], "n_total": 0,
                "n_completed": 0, "n_unserved": 0}

    passengers = [copy.deepcopy(p) for p in passengers]
    for p in passengers:
        p.setdefault("pass_count", 0)
    if sim_duration is None:
        sim_duration = max(p["arrival_time"] for p in passengers) + 7200

    env, results = _build_env(passengers, num_elevators, sim_duration)
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
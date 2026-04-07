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

* Nearest-car dispatcher (Building.assign_call):
    When a passenger registers a call, it is assigned to the elevator
    with the lowest cost score:
        score = distance + large penalty if elevator is moving away.
    Each elevator only responds to its own assigned calls, so multiple
    cars never race to the same floor.  Calls are released as soon as
    the assigned elevator boards the waiting passengers.

* SCAN algorithm with realistic capacity enforcement:
    - Sweep in current direction; stop at every floor where someone
      exits OR assigned same-direction waiters can board (if not full).
    - "Ahead" means floors where THIS elevator has an assigned call,
      so the turnaround point is the last assigned floor — not the
      physical boundary — eliminating needless overshooting.
    - Direction reverses when nothing assigned remains ahead and no
      on-board passenger needs to continue.
    - After reversing, the current floor is immediately re-checked so
      passengers whose direction matches the new heading are boarded
      at once (fixes the "abandoned at reversal floor" bug).
    - When the car arrives full at an assigned floor it cannot board,
      each stranded passenger gets pass_count += 1 (starvation metric).

* pass_count per passenger is recorded for floor-fairness analysis.
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

_SPAN = MAX_FLOOR - MIN_FLOOR          # used for dispatcher penalty scale


class Building:
    """
    Shared floor queues, nearest-car dispatch table, and idle-wakeup
    signalling for all elevators.
    """

    def __init__(self, env: simpy.Environment):
        self.env = env
        n = MAX_FLOOR - MIN_FLOOR + 1
        self.up_queues:   List[List[dict]] = [[] for _ in range(n)]
        self.down_queues: List[List[dict]] = [[] for _ in range(n)]
        self.wakeup: simpy.Event = env.event()
        self.elevators: List = []
        # _calls[eid] = set of (floor, direction) assigned to that elevator
        self._calls: dict = {}

    # ── Elevator registration ──────────────────────────────────────────

    def register(self, elev) -> None:
        """Register an elevator with the dispatcher. Call after creation."""
        self.elevators.append(elev)
        self._calls[elev.eid] = set()

    # ── Nearest-car dispatcher ─────────────────────────────────────────

    def assign_call(self, floor: int, direction: int) -> None:
        """
        Assign (floor, direction) to the best available elevator.
        Idempotent: already-assigned calls are silently ignored.

        Scoring per elevator
        --------------------
        score = distance
              + _SPAN * 3   if moving away from the call floor
        Lowest score wins.
        """
        # Already assigned to someone?
        for calls in self._calls.values():
            if (floor, direction) in calls:
                return
        if not self.elevators:
            return
        best, best_score = None, float("inf")
        for elev in self.elevators:
            dist = abs(elev.floor - floor)
            moving_away = (
                (elev.direction == UP   and floor < elev.floor) or
                (elev.direction == DOWN and floor > elev.floor)
            )
            score = dist + (_SPAN * 3 if moving_away else 0)
            if score < best_score:
                best_score, best = score, elev
        self._calls[best.eid].add((floor, direction))

    def release_call(self, eid: int, floor: int, direction: int) -> None:
        self._calls[eid].discard((floor, direction))

    def has_call(self, eid: int, floor: int, direction: int) -> bool:
        return (floor, direction) in self._calls.get(eid, set())

    # ── Queue helpers ──────────────────────────────────────────────────

    def queue_for(self, floor: int, direction: int) -> List[dict]:
        idx = floor - MIN_FLOOR
        return self.up_queues[idx] if direction == UP else self.down_queues[idx]

    def any_requests(self) -> bool:
        return any(q for q in self.up_queues + self.down_queues)

    def requests_ahead(self, eid: int, floor: int, direction: int) -> bool:
        """
        Any call assigned to this elevator strictly ahead in direction?

        Both up- and down-queue calls at each forward floor are included:
        the elevator must travel there regardless of call direction, because
        it will serve opposite-direction calls after reversing at that floor.
        Filtering by eid means each elevator only sees its own workload, so
        the turnaround happens at the last *assigned* floor rather than the
        physical boundary.
        """
        floors = (range(floor + 1, MAX_FLOOR + 1) if direction == UP
                  else range(MIN_FLOOR, floor))
        for f in floors:
            idx = f - MIN_FLOOR
            if (self.up_queues[idx]   and self.has_call(eid, f, UP)) or \
               (self.down_queues[idx] and self.has_call(eid, f, DOWN)):
                return True
        return False

    def signal_arrival(self) -> None:
        if not self.wakeup.triggered:
            self.wakeup.succeed()
        self.wakeup = self.env.event()


class Elevator:
    """
    One elevator car — SCAN dispatch, capacity-enforced, nearest-car aware.
    Only boards / travels toward calls assigned by Building.assign_call().
    """

    def __init__(self, env, eid: int, building: Building,
                 results: list, start_floor: int = MIN_FLOOR):
        self.env       = env
        self.eid       = eid
        self.building  = building
        self.results   = results
        self.floor     = start_floor
        self.direction = UP
        self.onboard:  List[dict] = []
        self.process   = env.process(self._run())

    # ── Internal helpers ───────────────────────────────────────────────

    def _full(self) -> bool:
        return len(self.onboard) >= ELEVATOR_CAPACITY

    def _need_dir(self, direction: int) -> bool:
        """Any on-board passenger needs to travel in this direction?"""
        for p in self.onboard:
            if direction == UP   and p["destination_floor"] > self.floor: return True
            if direction == DOWN and p["destination_floor"] < self.floor: return True
        return False

    def _should_stop(self) -> bool:
        """
        Stop at current floor?
          - Someone on board exits here, OR
          - Same-direction waiters (assigned to this car) can board.
        """
        if any(p["destination_floor"] == self.floor for p in self.onboard):
            return True
        q = self.building.queue_for(self.floor, self.direction)
        if q and not self._full() and \
                self.building.has_call(self.eid, self.floor, self.direction):
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
        Board same-direction assigned passengers, then release the call.
        If the car fills up before the queue is drained, remaining
        passengers get pass_count += 1 (starvation indicator).
        """
        if not self.building.has_call(self.eid, self.floor, self.direction):
            return
        q = self.building.queue_for(self.floor, self.direction)
        if not q:
            self.building.release_call(self.eid, self.floor, self.direction)
            return
        self.building.release_call(self.eid, self.floor, self.direction)
        while q and not self._full():
            p = q.pop(0)
            p["board_time"] = self.env.now
            self.onboard.append(p)
        if self._full():
            for p in q:
                p["pass_count"] += 1
        # If passengers remain because car filled, re-assign so they are
        # picked up by the next available elevator.
        if q:
            self.building.assign_call(self.floor, self.direction)

    def _serve_floor(self):
        """Open doors, unload, load, close doors — yields two timeouts."""
        yield self.env.timeout(DOOR_OPEN_TIME)
        self._unload()
        self._load()
        yield self.env.timeout(DOOR_CLOSE_TIME)

    # ── Main loop ──────────────────────────────────────────────────────

    def _run(self):
        while True:
            # ── Idle wait ────────────────────────────────────────────────
            while not self.building.any_requests() and not self.onboard:
                yield self.building.wakeup

            # ── Serve current floor if needed ────────────────────────────
            if self._should_stop():
                yield from self._serve_floor()
            else:
                # Not stopping — if full and this is our call, mark as passed
                if self._full() and \
                        self.building.has_call(self.eid, self.floor, self.direction):
                    for p in self.building.queue_for(self.floor, self.direction):
                        p["pass_count"] += 1

            # ── Decide direction ─────────────────────────────────────────
            ahead_onboard = self._need_dir(self.direction)
            ahead_waiting = self.building.requests_ahead(
                self.eid, self.floor, self.direction)

            if not ahead_onboard and not ahead_waiting:
                opposite    = -self.direction
                opp_onboard = self._need_dir(opposite)
                opp_waiting = self.building.requests_ahead(
                    self.eid, self.floor, opposite)
                opp_here    = (
                    bool(self.building.queue_for(self.floor, opposite)) and
                    self.building.has_call(self.eid, self.floor, opposite)
                )

                if opp_onboard or opp_waiting or opp_here:
                    self.direction = opposite
                    # Re-check current floor after direction flip:
                    # passengers here in the new direction were invisible to
                    # _should_stop() above (it checked the old direction).
                    if self._should_stop():
                        yield from self._serve_floor()

                elif not self.onboard:
                    yield self.building.wakeup
                    continue

            # ── Enforce floor boundaries ─────────────────────────────────
            if self.floor == MAX_FLOOR: self.direction = DOWN
            if self.floor == MIN_FLOOR: self.direction = UP

            # ── Move one floor ───────────────────────────────────────────
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

def _passenger_arrivals(env, building: Building, passengers: List[dict]):
    for p in passengers:
        delay = p["arrival_time"] - env.now
        if delay > 0:
            yield env.timeout(delay)
        direction = UP if p["destination_floor"] > p["origin_floor"] else DOWN
        building.queue_for(p["origin_floor"], direction).append(p)
        building.assign_call(p["origin_floor"], direction)
        building.signal_arrival()


# ─────────────────────────────────────────────
# Helpers shared by both public functions
# ─────────────────────────────────────────────

def _build_env(passengers: List[dict], num_elevators: int,
               sim_duration: float) -> tuple:
    """Create env, building, elevators; return (env, building, results)."""
    env      = simpy.Environment()
    results  = []
    building = Building(env)
    starts   = np.linspace(MIN_FLOOR, MAX_FLOOR, num_elevators, endpoint=True)
    for i in range(num_elevators):
        e = Elevator(env, i, building, results,
                     start_floor=int(round(starts[i])))
        building.register(e)          # register AFTER creation, before run
    env.process(_passenger_arrivals(env, building, passengers))
    return env, building, results


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
    """
    if not passengers:
        return []
    passengers = [copy.deepcopy(p) for p in passengers]
    for p in passengers:
        p.setdefault("pass_count", 0)
    if sim_duration is None:
        sim_duration = max(p["arrival_time"] for p in passengers) + 7200
    env, _, results = _build_env(passengers, num_elevators, sim_duration)
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
        n_unserved  : still waiting at sim end
    """
    if not passengers:
        return {"completed": [], "unserved": [], "n_total": 0,
                "n_completed": 0, "n_unserved": 0}
    passengers = [copy.deepcopy(p) for p in passengers]
    for p in passengers:
        p.setdefault("pass_count", 0)
    if sim_duration is None:
        sim_duration = max(p["arrival_time"] for p in passengers) + 7200
    env, _, results = _build_env(passengers, num_elevators, sim_duration)
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

def _run(self):
        while True:
            # ── Idle wait ────────────────────────────────────────────────
            while not self.building.any_requests() and not self.onboard:
                yield self.building.wakeup

            # ── Stop at current floor? ───────────────────────────────────
            if self._should_stop():
                yield self.env.timeout(DOOR_OPEN_TIME)
                self._unload()
                self._load()
                yield self.env.timeout(DOOR_CLOSE_TIME)
            else:
                # Not stopping — mark same-dir waiters as passed if full
                if self._full():
                    for p in self.building.queue_for(self.floor, self.direction):
                        p["pass_count"] += 1

            # ── Decide direction ─────────────────────────────────────────
            ahead_onboard = self._need_dir(self.direction)
            ahead_waiting = self.building.requests_ahead(self.floor, self.direction)

            if not ahead_onboard and not ahead_waiting:
                opposite    = -self.direction
                opp_onboard = self._need_dir(opposite)
                opp_waiting = self.building.requests_ahead(self.floor, opposite)
                opp_here    = bool(self.building.queue_for(self.floor, opposite))

                if opp_onboard or opp_waiting or opp_here:
                    self.direction = opposite

                    # ── FIX: re-check current floor after direction flip ──
                    # Passengers waiting here in the new direction were
                    # invisible to _should_stop() above (it only checked the
                    # old direction). Without this block they would be left
                    # behind even though the car is right in front of them.
                    if self._should_stop():
                        yield self.env.timeout(DOOR_OPEN_TIME)
                        self._unload()
                        self._load()
                        yield self.env.timeout(DOOR_CLOSE_TIME)

                elif not self.onboard:
                    yield self.building.wakeup
                    continue

            # ── Enforce floor boundaries ─────────────────────────────────
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
                self.direction = -self.direction
                yield self.env.timeout(0.5)

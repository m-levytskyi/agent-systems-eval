"""Simple request rate limiter for Gemini API usage.

Enforces per-minute and per-day request caps using an in-memory counter.
"""

import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque


class RequestRateLimiter:
    """Throttle outbound requests to avoid exceeding provider quotas."""

    def __init__(self, max_per_minute: int = 4, max_per_day: int = 15):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        self._recent_calls: Deque[float] = deque()
        self._current_day = self._utc_day()
        self._day_count = 0

    def _utc_day(self) -> int:
        """Return integer day ordinal in UTC for rollover checks."""
        return datetime.now(timezone.utc).toordinal()

    def _reset_day_if_needed(self) -> None:
        today = self._utc_day()
        if today != self._current_day:
            self._current_day = today
            self._day_count = 0
            self._recent_calls.clear()

    def acquire(self) -> None:
        """Block until a request slot is available or raise if daily limit hit."""
        while True:
            self._reset_day_if_needed()
            now = time.time()

            # Drop calls older than 60s
            while self._recent_calls and now - self._recent_calls[0] >= 60:
                self._recent_calls.popleft()

            if self._day_count >= self.max_per_day:
                raise RuntimeError("Daily request limit reached; stop issuing Gemini calls until tomorrow.")

            if len(self._recent_calls) < self.max_per_minute:
                self._recent_calls.append(now)
                self._day_count += 1
                return

            # Wait for the earliest call to exit the 60s window
            wait_time = max(0.0, 60 - (now - self._recent_calls[0]))
            time.sleep(wait_time)

    def remaining_daily(self) -> int:
        """Helper for diagnostics: how many calls left today."""
        self._reset_day_if_needed()
        return max(0, self.max_per_day - self._day_count)

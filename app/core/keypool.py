"""Round-robin pool of API keys with per-key cooldowns. Keys are never logged; use label()."""
import threading
import time


class KeyPool:
    def __init__(self, keys: list[str], clock=time.monotonic):
        cleaned: list[str] = []
        for key in keys:
            key = (key or "").strip()
            if key and key not in cleaned:
                cleaned.append(key)
        self._keys = cleaned
        self._until = {k: 0.0 for k in cleaned}   # cooldown deadline per key
        self._dead: set[str] = set()              # keys rejected as invalid
        self._next = 0
        self._clock = clock
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    def label(self, key: str) -> str:
        return f"key-{self._keys.index(key) + 1}/{len(self._keys)}"

    def pick(self) -> tuple[str, float]:
        """Next live key in round-robin order that is not cooling down: (key, 0.0).
        If every live key is cooling down, returns the one that frees up first plus the
        seconds to wait. Raises RuntimeError when no key is usable."""
        with self._lock:
            live = [k for k in self._keys if k not in self._dead]
            if not live:
                raise RuntimeError("no usable API keys")
            now, n = self._clock(), len(self._keys)
            for step in range(n):
                idx = (self._next + step) % n
                key = self._keys[idx]
                if key not in self._dead and self._until[key] <= now:
                    self._next = (idx + 1) % n
                    return key, 0.0
            key = min(live, key=lambda k: self._until[k])
            self._next = (self._keys.index(key) + 1) % n
            return key, max(0.0, self._until[key] - now)

    def cooldown(self, key: str, seconds: float) -> None:
        with self._lock:
            self._until[key] = max(self._until[key], self._clock() + seconds)

    def disable(self, key: str) -> None:
        with self._lock:
            self._dead.add(key)
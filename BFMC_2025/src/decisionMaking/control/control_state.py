import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Tuple

# Add speed, angle flag, and override functionality (hard control)
# This helps vehicle not blind when time.sleep()
@dataclass
class ControlState:
    speed: float = 0.0
    angle: float = 0.0
    timestamp: float = field(default_factory=time.time)

    override_until: float = 0.0
    override_speed: float = 0.0
    override_angle: float = 0.0

    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def set_override(self, speed: float, angle: float, duration: float):
        with self._lock:
            self.override_speed = speed
            self.override_angle = angle
            self.override_until = time.time() + duration

    def cancel_override(self):
        with self._lock:
            self.override_until = 0.0

    def update(self, speed: float, angle: float):
        with self._lock:
            if time.time() > self.override_until:
                self.speed = speed
                self.angle = angle
                self.timestamp = time.time()

    def snapshot(self):
        with self._lock:
            if time.time() < self.override_until:
                return self.override_speed, self.override_angle, self.override_until
            else:
                return self.speed, self.angle, self.timestamp

@dataclass
class LightState:
    is_on: bool = False
    timestamp: float = field(default_factory=time.time)

    override_until: float = 0.0
    override_value: bool = False

    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def set_override(self, value: bool, duration: float):
        with self._lock:
            self.override_value = value
            self.override_until = time.time() + duration

    def update(self, value: bool):
        with self._lock:
            if time.time() > self.override_until:
                self.is_on = value
                self.timestamp = time.time()

    def snapshot(self) -> Tuple[bool, float]:
        with self._lock:
            if time.time() < self.override_until:
                return self.override_value, self.override_until
            else:
                return self.is_on, self.timestamp

    def cancel_override(self):
        with self._lock:
            self.override_until = 0.0

# speed = 0.35
# angle = -10.5
# CONTROL_STATE.update(speed, angle)
# CONTROL_STATE.set_override(speed=0.3, angle=0, duration=1.5)
CONTROL_STATE = ControlState()
LIGHT_STATE = LightState()
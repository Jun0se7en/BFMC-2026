import time
from enum import IntFlag, auto
from dataclasses import dataclass, field
from threading import RLock
from collections import defaultdict
from typing import Dict

# ----------  A. Generic debouncer  ----------
class Debouncer:
    def __init__(self, tau_on=0.10, tau_off=0.20):
        self.tau_on = tau_on
        self.tau_off = tau_off
        self._state = defaultdict(lambda: (False, 0.0))  # (latched, t_last_change)

    def update(self, raw_active: bool, key: str) -> bool:
        latched, t_change = self._state[key]
        now = time.monotonic()
        if raw_active != latched:
            deadline = self.tau_on if raw_active else self.tau_off
            if now - t_change >= deadline:
                latched = raw_active
                self._state[key] = (latched, now)
        return latched

class ObjectDebouncer:
    def __init__(self, area_thresholds: Dict[str, float], tau_on=0.1, tau_off=0.2):
        self.debouncer = Debouncer(tau_on=tau_on, tau_off=tau_off)
        self.area_thresholds = area_thresholds

    def update(self, object_key: str, detected_area: float) -> bool:
        min_area = self.area_thresholds.get(object_key.upper(), 0)
        is_active = detected_area >= min_area
        return self.debouncer.update(is_active, object_key)


# ----------  B. Flag enums  ----------
# Choose drive mode: angle out by img processing or GPS
class DrivingMode(IntFlag):
    NONE = 0
    AUTO_LANE = auto()
    GPS_EKF = auto()
    OVERRIDE = auto()

# Choose speed mode: normal, slow, fast, brake
class SpeedMode(IntFlag):
    NONE = 0
    NORMAL = auto()
    LIGHT_SLOW = auto()
    SLOW = auto()
    LIGHT_FAST = auto()
    FAST = auto()
    BRAKE = auto()

# Show specific areas
class AreaFlag(IntFlag):
    NONE = 0
    PARKING = auto()
    OVERTAKE = auto()
    MISSING_ROAD = auto()
    TUNNEL = auto()
    INTERSECTION = auto()
    FOG = auto()
    # HIGHWAY = auto()

# Show all objects detectable by image processing
class ObjectFlag(IntFlag):
    NONE = 0
    RED_LIGHT = auto()
    YELLOW_LIGHT = auto()
    GREEN_LIGHT = auto()
    STOP_SIGN = auto()
    HIGHWAY_ENTRY_SIGN = auto()
    HIGHWAY_END_SIGN = auto()
    CROSSWALK_SIGN = auto()

    CAR_OBJECT = auto()
    PEDESTRIAN = auto()

    # Soft signs, affect path planning
    PARKING_SIGN = auto()
    ROUNDABOUT_SIGN = auto()
    ONE_WAY_SIGN = auto() 
    NO_ENTRY_SIGN = auto()
    PRIORITY_SIGN = auto() # Non-stop 

class LightFlag(IntFlag):
    NONE = 0
    LIGHT_ON = auto()

class SensorFlag(IntFlag):
    NONE = 0
    FRONT =  auto()
    BACK = auto()
    LEFT = auto()
    RIGHT = auto()

class EnableFlag(IntFlag):
    NONE = 0
    ENABLE_PARKING = auto()
    ENABLE_OVERTAKE = auto()
    ENABLE_INTERSECTION = auto()
    ENABLE_ROUNDABOUT = auto()
    ENABLE_STOP_SIGN = auto()

# ----------  C. Thread-safe shared state  ----------
@dataclass
class VehicleState:
    drive_state: DrivingMode = DrivingMode.NONE
    speed_state: SpeedMode = SpeedMode.NORMAL
    area_state: AreaFlag = AreaFlag.NONE
    sign_state: ObjectFlag = ObjectFlag.NONE
    light_state: LightFlag = LightFlag.NONE

    last_update: float = field(default_factory=time.time)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def set(self, category: str, flag: IntFlag) -> None:
        with self._lock:
            current = getattr(self, category)
            if not current & flag:
                print(f"[{category.upper()}] Set {flag.name}")
            setattr(self, category, current | flag)
            self.last_update = time.time()

    def clear(self, category: str, flag: IntFlag) -> None:
        with self._lock:
            current = getattr(self, category)
            if current & flag:
                print(f"[{category.upper()}] Cleared {flag.name}")
            setattr(self, category, current & ~flag)
            self.last_update = time.time()

    def snapshot(self):
        with self._lock:
            return (
                self.drive_state,
                self.speed_state,
                self.area_state,
                self.sign_state,
                self.light_state,
                self.last_update,
            )

    def any(self, category: str) -> bool:
        with self._lock:
            return bool(getattr(self, category))

@dataclass
class ActionState:
    """Thread‑safe container analogous to *VehicleState*.

    Use the *set*/*clear* helpers with category strings "sensor_flag" or
    "enable_flag" to update the masks.
    """

    sensor_flag: SensorFlag = SensorFlag.NONE
    enable_flag: EnableFlag = EnableFlag.NONE

    last_update: float = field(default_factory=time.time)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def set(self, category: str, flag: IntFlag) -> None:
        with self._lock:
            current = getattr(self, category)
            if not current & flag:
                print(f"[{category.upper()}] Set {flag.name}")
            setattr(self, category, current | flag)
            self.last_update = time.time()

    def clear(self, category: str, flag: IntFlag) -> None:
        with self._lock:
            current = getattr(self, category)
            if current & flag:
                print(f"[{category.upper()}] Cleared {flag.name}")
            setattr(self, category, current & ~flag)
            self.last_update = time.time()

    def snapshot(self):
        with self._lock:
            return (
                self.sensor_flag,
                self.enable_flag,
                self.last_update,
            )

    def any(self, category: str) -> bool:
        with self._lock:
            return bool(getattr(self, category))

STATE = VehicleState()
ACTION_STATE = ActionState()
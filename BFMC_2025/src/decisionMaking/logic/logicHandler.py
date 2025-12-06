from src.decisionMaking.control.control_state import CONTROL_STATE, LIGHT_STATE
from src.decisionMaking.control.vehicle_state import (
    STATE, ACTION_STATE, SpeedMode, DrivingMode, ObjectFlag, AreaFlag, LightFlag, EnableFlag, Debouncer
)
from src.decisionMaking.logic.overrideHandler import OverrideHandler
import json
from typing import Tuple

class LogicHandler:
    def __init__(self):
        self.debouncer = Debouncer(tau_on=0.10, tau_off=0.20)
        self.speed_map = {
            SpeedMode.FAST: 40,
            SpeedMode.LIGHT_FAST: 35,
            SpeedMode.NORMAL: 30,
            SpeedMode.LIGHT_SLOW: 20,
            SpeedMode.SLOW: 5,
            SpeedMode.BRAKE: 0,
        }
        self.light_map = {
            LightFlag.NONE: False,
            LightFlag.LIGHT_ON: True,
        }
        self.angle = 0
        self.direction = 'right'
        self.override = OverrideHandler()
        self._pending_override_flags = EnableFlag.NONE

    def load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
            sensor_cfg = config.get("SENSOR_RANGE", {})
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
        
        return sensor_cfg if sensor_cfg else None


    def get_angle(self, driving_mode: DrivingMode) -> Tuple[float, str]:
        if driving_mode == DrivingMode.AUTO_LANE:
            # angle = self.update_auto_lane_angle()
            return self.angle, "Auto Lane"
        elif driving_mode == DrivingMode.OVERRIDE:
            # angle = self.gps_ekf_angle()
            return self.angle, "GPS EKF"
        else:
            return 0.0, "No Driving Mode"
    def set_direction(self, direction):
        self.direction = direction
    
    def update_auto_lane_angle(self, angle) -> int:
        self.angle = angle
   
    def update_ekf_angle(self, angle) -> int:
        self.angle = angle

    def traffic_sign_logic(self, sign: ObjectFlag) -> Tuple[SpeedMode, str]:
        if sign & ObjectFlag.RED_LIGHT:
            return SpeedMode.BRAKE, "🔴 Red light"
        elif sign & ObjectFlag.YELLOW_LIGHT:
            return SpeedMode.SLOW, "🟡 Yellow light"
        elif sign & ObjectFlag.GREEN_LIGHT:
            return SpeedMode.NORMAL, "🟢 Green light"
        elif sign & ObjectFlag.STOP_SIGN:
            ACTION_STATE.set("enable_flag", EnableFlag.ENABLE_STOP_SIGN)
            return SpeedMode.SLOW, "🛑 Stop sign"
        elif sign & ObjectFlag.HIGHWAY_ENTRY_SIGN:
            return SpeedMode.FAST, "🚀 Highway entry"
        elif sign & ObjectFlag.HIGHWAY_END_SIGN:
            return SpeedMode.NORMAL, "🏁 Highway end"
        elif sign & ObjectFlag.CROSSWALK_SIGN:
            return SpeedMode.SLOW, "🚸 Crosswalk"
        elif sign & ObjectFlag.CAR_OBJECT:
            return SpeedMode.SLOW, "🚗 Car object"
        elif sign & ObjectFlag.PEDESTRIAN:
            return SpeedMode.BRAKE, "👣 Pedestrian"
        elif sign & ObjectFlag.PARKING_SIGN:
            ACTION_STATE.set("enable_flag", EnableFlag.ENABLE_PARKING)
            return SpeedMode.LIGHT_SLOW, "🅿️  Parking sign"
        elif sign & ObjectFlag.ROUNDABOUT_SIGN:
            ACTION_STATE.set("enable_flag", EnableFlag.ENABLE_ROUNDABOUT)
            return SpeedMode.LIGHT_SLOW, "🌀  Roundabout sign"
        else:
            return SpeedMode.NORMAL, "Nothing detected"

    def area_logic(self, area: AreaFlag) -> Tuple[SpeedMode, DrivingMode, LightFlag, str]:
        light_mode = LightFlag.NONE
        drive_mode = DrivingMode.AUTO_LANE

        if area & AreaFlag.PARKING:
            return SpeedMode.LIGHT_SLOW, drive_mode, light_mode, "🚗 Parking area"
        elif area & AreaFlag.TUNNEL:
            return SpeedMode.NORMAL, drive_mode, LightFlag.LIGHT_ON, "🌉 Tunnel"
        elif area & AreaFlag.INTERSECTION:
            return SpeedMode.BRAKE, DrivingMode.OVERRIDE, light_mode, "🚦 Intersection"
        elif area & AreaFlag.OVERTAKE:
            return SpeedMode.LIGHT_FAST, drive_mode, light_mode, "🚗 Overtake"
        elif area & AreaFlag.FOG:
            return SpeedMode.LIGHT_SLOW, drive_mode, LightFlag.LIGHT_ON, "🌫️  Foggy area"
        else:
            return SpeedMode.NORMAL, drive_mode, light_mode, "Normal area"


    def update_control(self):
        drive_mode, _, area, sign, _, _ = STATE.snapshot()

        # Sign & area logic
        sign_speed_mode, sign_note = self.traffic_sign_logic(sign)
        area_speed_mode, area_drive_mode, light_flag, area_note = self.area_logic(area)

        # Step the override every tick
        self.override_active = self.override.update(
            emergency_active=(sign_speed_mode == SpeedMode.BRAKE)
        )

        _, enable_flags, _ = ACTION_STATE.snapshot()

        if not self.override_active:
            # if enable_flags & EnableFlag.ENABLE_PARKING:
            #     self.override.parking()
            if enable_flags & EnableFlag.ENABLE_ROUNDABOUT and enable_flags & EnableFlag.ENABLE_INTERSECTION:
                if self.direction == 'right':
                    self.override.override_roundabout_right()
                elif self.direction == 'forward':
                    self.override.override_roundabout_right()

            elif enable_flags & EnableFlag.ENABLE_OVERTAKE:
                self.override.left_overtake()

            elif enable_flags & EnableFlag.ENABLE_INTERSECTION:
                # print('direction: ', self.direction)
                self.override.handle_intersection(True, direction=self.direction)
            
            elif enable_flags & EnableFlag.ENABLE_STOP_SIGN:
                # print('🛑 stopppp')
                self.override.override_stop()
                # STATE.clear("sign_state", ObjectFlag.STOP_SIGN)
                # ACTION_STATE.clear("enable_flag", EnableFlag.ENABLE_STOP_SIGN)
                # self.override_active = True

        # If no override, apply control logic
        if not self.override_active:
            chosen_drive_mode = (
                area_drive_mode if area_drive_mode != DrivingMode.AUTO_LANE else drive_mode
            )
            angle_value, drive_mode_note = self.get_angle(chosen_drive_mode)

            # sign_speed = self.speed_map.get(sign_speed_mode, 0)
            # area_speed = self.speed_map.get(area_speed_mode, 0)
            # Always obey cautionary or braking signs
            if sign_speed_mode in (SpeedMode.BRAKE, SpeedMode.SLOW, SpeedMode.LIGHT_SLOW):
                final_speed_mode = sign_speed_mode
            else:
                # Otherwise allow speeding up in highways, tunnels, etc.
                final_speed_mode = (
                    sign_speed_mode if self.speed_map[sign_speed_mode] >= self.speed_map[area_speed_mode]
                    else area_speed_mode
                )

            speed_value = self.speed_map.get(final_speed_mode, 0)
            light_value = self.light_map.get(light_flag, False)

            CONTROL_STATE.update(speed=speed_value, angle=angle_value)
            LIGHT_STATE.update(light_value)
        else:
            drive_mode_note = "🚨 Override Active"

        # Return current state snapshot
        control_snapshot = CONTROL_STATE.snapshot()
        light_snapshot = LIGHT_STATE.snapshot()
        return {
            "speed": control_snapshot[0],
            "angle": control_snapshot[1],
            "control_timestamp": control_snapshot[2],
            "light_on": light_snapshot[0],
            "light_timestamp": light_snapshot[1],
            "sign_note": sign_note,
            "area_note": area_note,
            "drive_mode_note": drive_mode_note,
        }






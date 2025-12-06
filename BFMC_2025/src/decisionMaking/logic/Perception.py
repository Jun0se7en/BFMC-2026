import logging
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import time
import json
from src.decisionMaking.control.vehicle_state import (
    STATE,
    ACTION_STATE,
    Debouncer,
    # ObjectDebouncer,
    DrivingMode,
    ObjectFlag,
    AreaFlag,
    EnableFlag
)
from src.decisionMaking.control.control_state import CONTROL_STATE
from src.decisionMaking.logic.logicHandler import LogicHandler
from src.decisionMaking.logic.nodeHandler import NodeHandler
from src.decisionMaking.planning.Navigator import Navigator
from src.decisionMaking.logic.randomStartHandler import RandomStartHandler
from src.decisionMaking.logic.overrideHandler import OverrideHandler

class Perception:  
    _SIGN_MAPPINGS: Dict[str, Tuple[str, ObjectFlag]] = {
        "stop": ("sign_state", ObjectFlag.STOP_SIGN),
        "pedestrian": ("sign_state", ObjectFlag.PEDESTRIAN),
        "redlight": ("sign_state", ObjectFlag.RED_LIGHT),
        "yellowlight": ("sign_state", ObjectFlag.YELLOW_LIGHT),
        "greenlight": ("sign_state", ObjectFlag.GREEN_LIGHT),
        "crosswalk": ("sign_state", ObjectFlag.CROSSWALK_SIGN),
        "highwayentry": ("sign_state", ObjectFlag.HIGHWAY_ENTRY_SIGN),
        "highwayend": ("sign_state", ObjectFlag.HIGHWAY_END_SIGN),
        "car": ("sign_state", ObjectFlag.CAR_OBJECT),
        "parking": ("sign_state", ObjectFlag.PARKING_SIGN),
        "roundabout": ("sign_state", ObjectFlag.ROUNDABOUT_SIGN), 
        "noentry": ("sign_state", ObjectFlag.NO_ENTRY_SIGN),
        "oneway": ("sign_state", ObjectFlag.ONE_WAY_SIGN),
        "priority": ("sign_state", ObjectFlag.PRIORITY_SIGN)
    }

    _AREA_MAPPINGS: Dict[str, Tuple[str, AreaFlag, Optional[DrivingMode]]] = {
        "intersection": ("area_state", AreaFlag.INTERSECTION, DrivingMode.OVERRIDE),
        "tunnel": ("area_state", AreaFlag.TUNNEL, None),
        "fog": ("area_state", AreaFlag.FOG, None),
    }

    _ENABLE_MAPPINGS: Dict[str, Tuple[str, EnableFlag]] = {
        "parking": ("enable_flag", EnableFlag.ENABLE_PARKING),
        "overtake": ("enable_flag", EnableFlag.ENABLE_OVERTAKE),
        "intersection": ("enable_flag", EnableFlag.ENABLE_INTERSECTION),
        "stop": ("enable_flag", EnableFlag.ENABLE_STOP_SIGN)
    }

    def __init__(self, *, logger: Optional[logging.Logger] = None, area_path) -> None:
        self.debouncer = Debouncer(tau_on=0.15, tau_off=0.2)
        with open(area_path, 'r') as f:
            area_cfg = json.load(f)
        # area_thresholds = {k: v["area"] for k, v in area_cfg.items()}
        # self.object_debouncer = ObjectDebouncer(area_thresholds=area_thresholds, tau_on=0.1, tau_off=0.2)
        self.handler = LogicHandler()

        # Runtime state used for log‑spam reduction
        self._last_context: Optional[Dict[str, Any]] = None
        self._last_sign_note: str = ""
        self._last_area_note: str = ""
        self._last_drive_mode_note: str = ""
        self.override = OverrideHandler()
        self.last_speed = 0
        self.last_angle = 0
        self.lane_angle = 0
        self.final_angle = 0
        # Logger setup – inherit root handlers so that the user can configure them globally
        self.log = logger or logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)

    def update(self, **raw_inputs: bool) -> Dict[str, Any]:
        self._update_vehicle_state(raw_inputs)
        
        context = self.handler.update_control()
        self.last_speed = int(context["speed"])
        self.last_angle = int(context["angle"])
        # if self.last_angle != self.lane_angle:
            
        self._update_log(context)
        return context
    

    def update_angle(self, lane_angle: int = 0):
        try:
            if not self.handler.override_active:    
                self.lane_angle = lane_angle
                self.handler.update_auto_lane_angle(lane_angle)
        except:
            pass

    
    # def get_current_angle(self) -> int:
    #     if self.override.is_active():
    #         return self.last_angle
    #     return self.lane_angle

        # if time.time() > CONTROL_STATE.override_until :
        #     print(self.last_angle)
            # print('angle: ', CONTROL_STATE.snapshot())
        # else:
        #     self.last_angle = lane_angle
        # self.handler.update_auto_lane_angle(lane_angle)
        # self.handler.update_ekf_angle(ekf_angle)

    # def get_current_angle(self) -> int:
        # if time.time() <= CONTROL_STATE.override_until:
            # return self.last_angle
        # return self.lane_angle
        # return self.last_angle
        
    def get_current_angle(self) -> int:
        return int(CONTROL_STATE.snapshot()[1])

    
    def get_current_speed(self) -> int:
        return self.last_speed

    def _update_vehicle_state(self, raw: Dict[str, bool]) -> None:
        """Debounce raw signals and reflect them in :data:`STATE`."""
        # --- sign flags --------------------------------------------------
        for key, (category, flag) in self._SIGN_MAPPINGS.items():
            active = bool(raw.get(key, False))
            if self.debouncer.update(active, key):
                STATE.set(category, flag)
            else:
                STATE.clear(category, flag)

        # --- area flags --------------------------------------------------
        for key, (category, flag, mode) in self._AREA_MAPPINGS.items():
            active = bool(raw.get(key, False))
            if self.debouncer.update(active, key):
                STATE.set(category, flag)
                if mode is not None:
                    STATE.set("drive_state", mode)
            else:
                STATE.clear(category, flag)
                if mode is not None:
                    STATE.clear("drive_state", mode)
                    STATE.set("drive_state", DrivingMode.AUTO_LANE)
                    
        for key, (category, flag) in self._ENABLE_MAPPINGS.items():
            active = bool(raw.get(key, False))
            if self.debouncer.update(active, f"enable_{key}"):
                ACTION_STATE.set(category, flag)
            else:
                ACTION_STATE.clear(category, flag)
        

    def _update_log(self, context: Dict[str, Any]) -> None:
        if context == self._last_context:
            return  # nothing new

        sign_note = context["sign_note"]
        area_note = context["area_note"]
        drive_mode_note = context["drive_mode_note"]

        if (
            area_note != self._last_area_note
            or drive_mode_note != self._last_drive_mode_note
            or sign_note != self._last_sign_note
        ):
            self._last_area_note = area_note
            self._last_drive_mode_note = drive_mode_note
            self._last_sign_note = sign_note

            speed = context["speed"]
            angle = context["angle"]
            light = context["light_on"]

            self.log.info("[UPDATE] Area: %s, Driving mode: %s", area_note, drive_mode_note)
            self.log.info("[UPDATE] Sign: %s", sign_note)
            self.log.info(
                # "[UPDATE] Speed: %.2f, Angle: %.2f, Light: %s",
                "[UPDATE] Speed: %.2f, Angle: %.2f",
                speed,
                angle,
                # "ON" if light else "OFF",
            )

        self._last_context = context.copy() 

if __name__ == "__main__":
    import time
    import random
    from src.decisionMaking.planning.PathFinder import PathFinder
    graph_path = "src/decisionMaking/cfg/output.graphml"
    cfg_path = "src/decisionMaking/cfg/cfg.json"
    sign_path = "src/decisionMaking/cfg/sign_area.json"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    pth_finder = PathFinder(graph_path)
    handler = NodeHandler(cfg_path, graph_path)
    random_start_handler = RandomStartHandler(graph_path)
    nav = Navigator(graph_path, cfg_path)
    perception = Perception(area_path=sign_path)

    gps_data = [
        (2.88, 2.18, 80),
        (2.87, 2.23, 82),
        (2.87, 2.26, 84),
        (2.86, 2.27, 85),
        (2.87, 2.28, 85)
    ]
    prev_node, current_node = random_start_handler.find_best_matching_edge(gps_data)
    # x, y = 2.8, 1.6  # Get current coordinates here
    # node_id, coords, distance = handler.get_current_node((x, y))
    nav.capture_all(current_node)
    # nav.plot_segments(save_path='map_plot.png')
    print("Direction: ", nav.get_direction_map())
    direction_map = {}
    for segment in nav.full_path:
        for curr_node in segment:
            time.sleep(0.5)
            print(f"Node: {curr_node}")
            if curr_node in list(nav.intersections): # Random change intersection direction
                current_direction = nav.get_direction_map()[curr_node]
                if random.random() < 0.5:
                    next_lst = pth_finder._find_next_node(curr_node)
                    if len(next_lst) > 1:
                        next_node = random.choice(next_lst)
                        print(f'Changed direction to: {next_node}')
                        nav.capture_all(next_node)
                        # print(f"New path: {nav.full_path[0]}")
                # else:
                print(f"Intersection, turn {current_direction} at {curr_node}")
            area_next = handler.check_area_next(curr_node)
            if area_next is not None:
                area_next = [i.lower() for i in area_next]
                # print(f"Area next: {area_next}, Node: {curr_node}")
                
            def simulate_area(threshold, margin=500):
                return random.uniform(threshold - margin, threshold + margin)
            
            obj_sample = {
                "stop": (random.random() < 0.1, simulate_area(1000)),
                "pedestrian": (random.random() < 0.1, simulate_area(3500)),
                "redlight": (random.random() < 0.001, simulate_area(2000)),
                "yellowlight": (random.random() < 0.1, simulate_area(2000)),
                "greenlight": (random.random() < 0.1, simulate_area(2000)),
                "crosswalk": (random.random() < 0.1, simulate_area(1300)),
            }
            
            area_sample = {
                "intersection": "intersection" in area_next,
                # "tunnel": random.random() < 0.01,
                # "fog": random.random() < 0.01,
            }
            raw_sample = {}
            for key, (detected, area) in obj_sample.items():
                raw_sample[key] = detected
                raw_sample[f"{key}_area"] = area
            # print(raw_sample)
            raw_sample.update(area_sample)
            perception.update(**raw_sample)
            
            
                

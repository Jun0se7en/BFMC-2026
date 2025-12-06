import math
import json
from collections import deque

class DirectionPicking:
    def __init__(self, config_file):
        self.load_data(config_file)
    
    def load_data(self, config_file):
        with open(config_file, 'r') as f:
            data = json.load(f)
        self.dict_nodes_coordination = data["dict_nodes_coordination"]
        self.neighbor_nodes = data["neighbor_nodes"]
        self.from_lst = data["from_lst"]
        self.to_lst = data["to_lst"]
    
    def compute_heading(self, frm, to):
        x1, y1 = self.dict_nodes_coordination[frm]
        x2, y2 = self.dict_nodes_coordination[to]
        return math.degrees(math.atan2(y2 - y1, x2 - x1))
    
    def relative_turn(self, reference_heading, leg_heading, threshold=15):
        diff = (leg_heading - reference_heading + 180) % 360 - 180
        if -threshold <= diff <= threshold:
            return "straight"
        elif diff > threshold:
            return "left"
        else:
            return "right"
    
    def find_alternating_path(self, start, destination):
        if start not in self.from_lst:
            raise ValueError(f"Start node {start} must be in from_lst.")
        
        queue = deque([(start, [start])])
        visited = set([(start, 0)])
        
        while queue:
            current, path = queue.popleft()
            idx = len(path) - 1
            
            if current == destination:
                return path
            
            expected_next = self.to_lst if (idx % 2 == 0) else self.from_lst
            
            for neighbor in self.neighbor_nodes.get(current, []):
                if neighbor != destination and neighbor not in expected_next:
                    continue
                new_state = (neighbor, (idx + 1) % 2)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((neighbor, path + [neighbor]))
        
        raise ValueError(f"No alternating path found from {start} to {destination}.")
    
    def find_path_directions(self, traveled_nodes, threshold=15):
        directions = []
        start = traveled_nodes[0]
        if start not in self.neighbor_nodes or not self.neighbor_nodes[start]:
            raise ValueError(f"Node {start} has no neighbor defined to set an initial heading.")
        
        ref_heading = self.compute_heading(start, self.neighbor_nodes[start][0])
        current_ref = ref_heading
        
        for i in range(len(traveled_nodes) - 1):
            frm = traveled_nodes[i]
            to = traveled_nodes[i+1]
            x1, y1 = self.dict_nodes_coordination[frm]
            x2, y2 = self.dict_nodes_coordination[to]
            leg_heading = self.compute_heading(frm, to)
            
            if x1 == x2 or y1 == y2:
                turn = "straight"
            else:
                turn = self.relative_turn(current_ref, leg_heading, threshold)
            
            directions.append(turn)
            current_ref = leg_heading
        
        return directions
    
    def process_route(self, start, destination, threshold=15):
        if destination in self.neighbor_nodes.get(start, []):
            path = [start, destination]
        else:
            path = self.find_alternating_path(start, destination)
        
        directions = self.find_path_directions(path, threshold)
        return path, directions

class CarInstruction:
    def __init__(self):
        """
        Initializes the traffic sign handler.
        Stores recognized traffic signs and their corresponding actions.
        """
        self.sign_actions = {
            "priority": self.priority,
            "crosswalk": self.crosswalk,
            "stop": self.stop,
            "no_entry": self.no_entry,
            "straight": self.straight,
            "highway_entry": self.highway_entry,
            "highway_exit": self.highway_exit,
            "parking": self.parking,
            "roundabout": self.roundabout,
            "pedestrian": self.pedestrian,
            "greenlight": self.green_light,
            "yellowlight": self.yellow_light,
            "redlight": self.red_light,
        }
        
        self.intersection_actions = {
            "go_forward": self.go_forward,
            "go_right": self.go_right,
            "go_left": self.go_left,
            "stop": self.stop_long,
        }
    
    def priority(self):
        # Literally do nothing
        return [[[30, 0, 0]], True]
    
    def crosswalk(self):
        # Slow down until pass the crossroad
       return [[[20, 0, 0]], True]
        
    
    def stop(self):
        # Stop for 3 second
        return [[[30, 0, 0.7], [0, 0, 3]], False]
                
    def no_entry(self):
        # If seeing this, find another path to go
        return None
            
    def straight(self): # Oneway
        # Must go straight
        return None
    
    def highway_entry(self):
        # Speed up until end of highway
        return [[[46, 0, 0]], True]
    
    def highway_exit(self):
        # Slow down
        return [[[40, 0, 0]], True]
    
    def parking(self):
        # Park the car
        return [[[0, 0, 0.5],[40, 0, 2.2], [0, 0, 0.5], [-30, 16, 2.7], [-30, -20, 2.7], [0, 0, 0.5], [20, 6, 1], [0, 0, 6]], False]
    
    def roundabout(self):
        # No idea
        return [[[30, 0, 0]], True]
    
    def pedestrian(self):
        # Slow down
        return [[[0, 0, 0]], True]
    
    def green_light(self):
        return [[[40, 0, 0]], True]
    
    def yellow_light(self):
        # Slow down
        return [[[20, 0, 3]], True]
    
    def red_light(self):
        # Stop for 3 second
        return [[[0, 0, 3]], True]
    
    def choose_sign(self, sign_msg):
        if sign_msg in self.sign_actions:
            # The result will include: [[speed, angle, time], is_strict_instruction]
            return self.sign_actions[sign_msg]()
            
    def go_forward(self):
        return [[[0, 0, 0.2], [40, 0, 3.5]], False]
    
    def go_right(self):
        return [[[40, 0, 0.3], [40, 24, 2]], False]
    
    def go_left(self):
        return [[[30, 0, 1], [40, -20, 2.5]], False]
    
    def stop_long(self):
        return [[[0, 0, 5]], False]
    
    def choose_direction(self, direction):
        if direction in self.instruction:
            return self.instruction[direction]()
import numpy as np
import math

# Hàm normalize để điều chỉnh góc về khoảng [-pi, pi]
def normalize(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Lớp mô hình xe đạp kinematik
class KinematicBicycleModel:
    def __init__(self, wheelbase: float, length_rear: float, delta_time: float, max_steer: float):
        self.delta_time = delta_time
        self.wheelbase = wheelbase
        self.length_rear = length_rear
        self.max_steer = max_steer

    def discrete_kbm(self, velocity: float, steering: float, x: float, y: float, heading: float, acceleration: float):
        # slip_angle = np.arctan(self.length_rear * np.tan(steering) / self.wheelbase)
        # # new_velocity = velocity + acceleration * self.delta_time
        # # new_steer = steering + steering_rate * self.delta_time
        # angular_velocity = (velocity * np.tan(steering) * np.cos(slip_angle)) / self.wheelbase
        # new_x = x + velocity * np.cos(slip_angle + heading) * self.delta_time
        # new_y = y + velocity * np.sin(slip_angle + heading) * self.delta_time
        # new_heading = normalize(heading + angular_velocity * self.delta_time)
        # return new_x, new_y, new_heading
        new_velocity = velocity + self.delta_time * acceleration
        new_steer = self.max_steer if steering > self.max_steer else -self.max_steer if steering < -self.max_steer else steering
        angular_velocity = new_velocity * math.tan(new_steer) / self.wheelbase
        new_heading = heading + angular_velocity * self.delta_time

        state_x = x + velocity * math.cos(new_heading) * self.delta_time
        state_y = y + velocity * math.sin(new_heading) * self.delta_time
        state_heading = math.atan2(math.sin(new_heading), math.cos(new_heading))
        state_steer = new_steer
        state_velocity = new_velocity
        state_angular_velocity = angular_velocity

        return state_x, state_y, state_heading, state_steer, state_velocity, state_angular_velocity

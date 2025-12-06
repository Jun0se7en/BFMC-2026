import numpy as np
from lib.kinematic_bycicle_model import KinematicBicycleModel

# Lớp Bộ lọc Kalman Mở rộng (EKF)
class ExtendedKalmanFilter:
    def __init__(self, length_rear: float, delta_time: float):
        self.kbm = KinematicBicycleModel(length_rear, delta_time)
        self.length_rear = length_rear 
        self.delta_time = delta_time
    
    def P_Jacobian(self, velocity, slip_angle, heading,dt):
        
        cos_term = np.cos(np.radians(slip_angle + heading))
        sin_term = np.sin(np.radians(slip_angle + heading))

        jacobian = np.array([
        [1, 0, -dt * velocity * sin_term],
        [0, 1,  dt * velocity * cos_term],
        [0, 0, 1]
        ])
        return jacobian
    
    def Q_Jacobian(self,slip_angle, heading,dt):
        
        cos_term = np.cos(np.radians(slip_angle + heading))
        sin_term = np.sin(np.radians(slip_angle + heading))
    
        jacobian = np.array([
            [dt * cos_term, 0,0],
            [dt * sin_term, 0,0],
            [0, 0,dt]
        ])
        return jacobian
    
    def prediction_state(self,P_cov,Q_cov, x, y, heading, velocity, angular_velocity, dt):

        # 1. Predicted State Estimate
        new_x, new_y, new_heading, slip_angle = self.kbm.discrete_kbm(x, y, heading, velocity, angular_velocity, dt)
        predicted_state = np.array([new_x, new_y, new_heading])
        # 2. Predicted Error Covariance
        F_jacobian = self.P_Jacobian(velocity, slip_angle, heading,dt)

        G_jacobian = self.Q_Jacobian(slip_angle, heading,dt)

        predicted_err = F_jacobian @ P_cov @ F_jacobian.T + G_jacobian @ Q_cov @ G_jacobian.T

        return predicted_state , predicted_err
    
    def update_state(self,x_measure,y_measure,h_measure,sigma_x,sigma_y,sigma_h,predicted_state, predicted_err):

        H = np.array([
            [1,0,0],
            [0,1, 0],
            [0,0,1]
        ])

        M = np.array([
             [1,0,0],
             [0,1,0],
             [0,0,1]
        ])

        R = np.array([
            [sigma_x**2, 0,0],
            [0, sigma_y**2,0],
            [0, 0,sigma_h**2]
        ])    

        y_array = np.array([x_measure,y_measure,h_measure]) ## Measurements

        y_diff = y_array -  H @ predicted_state 

        S = H @ predicted_err @ H.T + M @ R @ M.T
        
        K = predicted_err @ H.T @ np.linalg.inv(S)

        update_state  = predicted_state + K @ y_diff

        I = np.eye(predicted_err.shape[0])

        update_err = (I - K @ H) @ predicted_err

        return update_state, update_err
    
    def update_state_yaw(self,heading_measure,sigma_h,predicted_state, predicted_err):

        H = np.array([[0,0,1]])

        M = np.array([[1]])

        R = np.array([[sigma_h**2]])    

        y_array = np.array([heading_measure]) ## Measurements

        y_diff = y_array - H @ predicted_state

        S = H @ predicted_err @ H.T + M @ R @ M.T
        
        K = predicted_err @ H.T @ np.linalg.inv(S)

        update_state  = predicted_state + K @ y_diff

        I = np.eye(predicted_err.shape[0])

        update_err = (I - K @ H) @ predicted_err

        return update_state, update_err
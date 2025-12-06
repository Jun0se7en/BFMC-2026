import numpy as np
from scipy.optimize import minimize
from kinematic_bycicle_model import KinematicBicycleModel
from extended_kalman_filter import ExtendedKalmanFilter
import time
class ModelPredictiveControl:
    def __init__(self, wheelbase, length_rear, dt, horizon, Q, R, ekf: ExtendedKalmanFilter,learning_rate, max_iter, beta1, beta2, epsilon, clip_value):
        self.kbm = KinematicBicycleModel(wheelbase, length_rear, dt)
        self.dt = dt
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.ekf = ekf
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.m = np.zeros(2 * horizon) # vector lưu trữ giá trị trung bình di chuyển (momentum) của gradient
        self.v = np.zeros(2 * horizon) # vector lưu trữ giá trị trung bình di chuyển của bình phương gradient.
        self.t = 0 #biến đếm số lần cập nhật.
        self.loss_values = []
    def control(self, state, reference, P, Q):
        start_time = time.time()  # Bắt đầu đo thời gian
        vars = np.array([state[0], state[1]] * self.horizon)
        bounds = [(0, 20), (-np.pi/4, np.pi/4)] * self.horizon  # Adjust bounds to increase velocity limit

        for _ in range(self.max_iter):
            grad = self.compute_gradient(vars, state, reference)
            grad = self.clip_gradient(grad, self.clip_value)
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            vars -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Apply bounds
            for i in range(len(vars)):
                vars[i] = max(bounds[i][0], min(vars[i], bounds[i][1]))
            cost = self.objective(vars, state, reference)
            self.loss_values.append(cost)
        velocity = vars[0]
        steering_angle = vars[1]

        # Cập nhật EKF - Bước dự đoán
        acceleration = (velocity - state[0]) / self.dt 
        steering_rate = (steering_angle - state[1]) / self.dt 

        predicted_state, predicted_P = self.ekf.prediction_state(state, [acceleration, steering_rate], P, Q)
        end_time = time.time()  # Kết thúc đo thời gian
        elapsed_time = end_time - start_time
        print(f"Time taken for control step: {elapsed_time:.6f} seconds")
        return velocity, steering_angle, predicted_state, predicted_P

    def compute_gradient(self, vars, state, reference):
        grad = np.zeros_like(vars)
        next_state = state.copy()

        for t in range(self.horizon):
            velocity = vars[2 * t]
            steering_angle = vars[2 * t + 1]

            next_state, _ = self.ekf.prediction_state(next_state, [velocity, steering_angle], self.ekf.P, self.ekf.Q)
            reference_state = reference[t]

            # Calculate gradients
            grad[2 * t] = 2 * self.Q[0] * (next_state[2] - reference_state[2]) + 2 * self.R[0] * velocity
            grad[2 * t + 1] = 2 * self.Q[1] * (next_state[3] - reference_state[3]) + 2 * self.R[1] * steering_angle

        return grad
    
    def clip_gradient(self, grad, clip_value):
        return np.clip(grad, -clip_value, clip_value)
    
    def objective(self, vars, state, reference):
        cost = 0
        next_state = state.copy()

        for t in range(self.horizon):
            velocity = vars[2 * t]
            steering_angle = vars[2 * t + 1]

            next_state, _ = self.ekf.prediction_state(next_state, [velocity, steering_angle], self.ekf.P, self.ekf.Q)
            reference_state = reference[t]

            cost += self.Q[0] * (next_state[2] - reference_state[2]) ** 2  # x
            cost += self.Q[1] * (next_state[3] - reference_state[3]) ** 2  # y
            cost += self.Q[2] * (next_state[4] - reference_state[4]) ** 2  # heading
            cost += self.R[0] * velocity ** 2  # control effort for velocity
            cost += self.R[1] * steering_angle ** 2  # control effort for steering angle

        return cost
    
    def update_with_measurement(self, state, P, measurement, R):
        updated_state, updated_P = self.ekf.update_state(state, P, measurement, R)
        return updated_state, updated_P

    def predict_trajectory(self, state, control_inputs):
        trajectory = [state]
        current_state = state

        for t in range(self.horizon):
            velocity, steering_angle = control_inputs[t*2], control_inputs[t*2 + 1]
            next_state = self.kbm.discrete_kbm(velocity, steering_angle, current_state[2], current_state[3], current_state[4], 0, 0)
            trajectory.append(next_state)
            current_state = next_state

        return trajectory

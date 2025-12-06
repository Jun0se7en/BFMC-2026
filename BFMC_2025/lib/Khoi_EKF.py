"""
EKF vị trí x, y dựa trên IMU + heading (quay tốc độ) và GPS
===========================================================

**Sửa lỗi đường thẳng** – phiên bản trước không xoay vector vận tốc khi
heading đổi (vì θ không nằm trong state).  Giờ ta lưu `prev_heading` nội bộ
và xoay `(vx, vy)` mỗi bước bằng ∆θ.

Công thức:
```
∆θ  = wrap(θ_now − θ_prev)
[vx′]   [ cos∆θ −sin∆θ ] [vx]
[vy′] = [ sin∆θ  cos∆θ ] [vy]
```
Sau đó áp dụng gia tốc thân → world rồi tích phân.

State: `[x, y, vx, vy]`  (4 biến).  Heading **không** là state → đơn giản.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

DEG2RAD = math.pi / 180.0

RAD2DEG = 180.0 / math.pi

# ---------------- util ---------------------------------------------------

def _wrap(rad: float) -> float:
    """Wrap angle to (−π, π]."""
    return (rad + math.pi) % (2*math.pi) - math.pi

# ---------------- EKF ----------------------------------------------------

@dataclass
class PositionEKF:
    """EKF 4‑state x, y, vx, vy – heading (deg) là đầu vào."""

    imu_rate_hz: float = 42.0
    sigma_acc:  float = 0.3     # m/s² 1σ
    sigma_gps:  float = 0.20    # m 1σ

    def __post_init__(self):
        self.dt  = 1.0 / self.imu_rate_hz
        self.state   = np.zeros(4)               # [x, y, vx, vy]
        self.P   = np.diag([10.0, 10.0, 1.0, 1.0])
        self.H   = np.array([[1,0,0,0], [0,1,0,0]])
        self.R   = (self.sigma_gps**2) * np.eye(2)
        self.prev_heading_rad: float | None = None

    # ------------------ predict -----------------------------------------
    def predict(self, ax_b: float, ay_b: float, heading_deg: float, dt: float | None = None):
        if dt is None:
            dt = self.dt

        # --- rotate existing velocity by Δθ -----------------------------
        theta = heading_deg * math.radians(heading_deg)
        if self.prev_heading_rad is None:
            dtheta = 0.0
        else:
            dtheta = _wrap(theta - self.prev_heading_rad)
        cos_d, sin_d = math.cos(dtheta), math.sin(dtheta)
        vx, vy = self.state[2], self.state[3]
        vx, vy = vx*cos_d - vy*sin_d, vx*sin_d + vy*cos_d
        # self.state[2:] = [vx, vy]
        self.state[2] = vx; self.state[3] = vy
        self.prev_heading_rad = theta

        # --- body accel → world frame -----------------------------------
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        ax_n = ax_b*cos_t - ay_b*sin_t
        ay_n = ax_b*sin_t + ay_b*cos_t

        # --- integrate ---------------------------------------------------
        self.state[0] += vx*dt + 0.5*ax_n*dt*dt
        self.state[1] += vy*dt + 0.5*ay_n*dt*dt
        self.state[2] += ax_n*dt
        self.state[3] += ay_n*dt

        # Jacobian F
        F = np.eye(4)
        F[0,2] = dt
        F[1,3] = dt

        # Q
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
        q = self.sigma_acc**2
        Q = q * np.array([[0.25*dt4,0,0.5*dt3,0],
                          [0,0.25*dt4,0,0.5*dt3],
                          [0.5*dt3,0,dt2,0],
                          [0,0.5*dt3,0,dt2]])
        self.P = F @ self.P @ F.T + Q

    # ------------------ update ------------------------------------------
    def update_xy(self, x_m: float, y_m: float):
        z = np.array([[x_m],[y_m]])
        y_res = z - self.H @ np.asarray(self.state, float).reshape(-1,1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = (np.asarray(self.state, float).reshape(-1,1) + K @ y_res).ravel()
        self.P = (np.eye(4) - K @ self.H) @ self.P

# ---------------- demo ---------------------------------------------------
if __name__ == "__main__":
    ekf = PositionEKF()
    # simulate robot chạy vòng tròn R=1 m, tốc độ 0.5 m/s (ω = v/R)
    R, v = 1.0, 0.5
    omega = v / R           # rad/s
    total_t = 10.0
    steps = int(total_t * ekf.imu_rate_hz)
    for i in range(steps):
        t = i / ekf.imu_rate_hz
        heading_deg = (omega*t)*RAD2DEG  # tăng tuyến tính → vòng tròn
        # gia tốc hướng tâm ~ v²/R hướng −y_body, nên ax=0, ay= -v²/R
        ax_b, ay_b = 0.0, -(v**2)/R
        ekf.predict(ax_b, ay_b, heading_deg)
        if i % int(ekf.imu_rate_hz) == 0:
            ekf.update_xy(ekf.x[0]+np.random.randn()*0.2,
                           ekf.x[1]+np.random.randn()*0.2)
    print("x,y:", ekf.x[:2])
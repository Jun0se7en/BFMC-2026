# ekf_imu_gps.py – Extended Kalman Filter fusing full‑IMU (acc + gyro + orientation)
# with absolute (x, y) fixes from indoor GPS / localization at lower rate.
# -----------------------------------------------------------------------------
# Inspired by: https://github.com/balamuruganky/EKF_IMU_GPS
# Author: ChatGPT – June 2025 | MIT Licence
# -----------------------------------------------------------------------------
"""
IMU rate   : 42 Hz  (≈ dt = 0.02381 s)
GPS rate   :  1 Hz  (irregular – call `update_position()` when available)

State & Model (planar)
----------------------
X = [x, y, vx, vy]ᵀ in world frame (ENU).

Predict step for each IMU sample:
    1. **Remove gravity** & rotate body‑frame acceleration to world frame
       using provided Euler angles (roll, pitch, yaw).
    2. **Rotate existing velocity vector** by Δyaw since last step – this
       implicitly adds the centripetal component when the robot turns at
       roughly constant speed.
    3. **Integrate** acceleration → velocity → position.

Update step (when GPS/local‑pos fix arrives):
    z = [x_pos, y_pos]ᵀ,  simple linear measurement H = [[1,0,0,0],[0,1,0,0]].

Notes
-----
* Gyroscope (gx, gy, gz) is **not** integrated explicitly here because yaw
  is provided directly.  If you prefer dead‑reckoning without yaw measurement
  you can adapt the algorithm by estimating yaw in the state vector.
* Acceleration standard deviation (`accel_std`) and GPS noise (`pos_std`) are
  the main parameters to tune.
* All angles in **degrees** on the API, cast to rad internally.
"""

from __future__ import annotations
import numpy as np
import math
from typing import Optional, Tuple

__all__ = ["EKFImuGps"]

G_STD = 9.80665   # m/s² – standard gravity

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def wrap_pi(rad: float) -> float:
    """Wrap angle into (‑π, π]."""
    return (rad + math.pi) % (2 * math.pi) - math.pi


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return 3×3 body→world rotation matrix for ZYX Euler (deg inputs)."""
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # Rz @ Ry @ Rx (world = R * body)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [ -sp ,            cp*sr ,            cp*cr ],
    ], dtype=float)

# ---------------------------------------------------------------------------
# EKF class
# ---------------------------------------------------------------------------

class EKFImuGps:
    """Extended Kalman Filter fusing 6‑axis IMU + attitude with 2‑D position.

    Parameters
    ----------
    dt : float
        Nominal IMU sampling period (s) – ≈ 1/42 = 0.02381 s.
    accel_std : float, default 0.30
        1‑σ noise of **linearised** acceleration (m/s²).
    pos_std : float, default 0.40
        1‑σ noise of (x, y) position fixes (m).
    init_vel : Tuple[float, float], default (0,0)
        Initial guess of velocity (m/s).  Set this if robot already moves at
        EKF start.
    """

    def __init__(
        self,
        dt: float = 1.0/42.0,
        accel_std: float = 0.30,
        pos_std: float = 0.40,
        init_vel: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.dt_nominal = dt
        self.prev_yaw_rad: Optional[float] = None  # for Δyaw

        # ---- state & covariance ---------------------------------------
        # X = [x, y, vx, vy]
        self.state = np.zeros(4)
        self.state[2:] = init_vel

        self.P = np.eye(4) * 1e-3

        # ---- noise matrices ------------------------------------------
        self._q_acc2 = accel_std ** 2       # (m/s²)²
        self.R = np.diag([0.00203383, 0.00396694])

    # ------------------------------------------------------------------
    # Predict step – call at IMU rate
    # ------------------------------------------------------------------
    def predict(
        self,
        ax: float,
        ay: float,
        az: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        dt: Optional[float] = None,
    ) -> None:
        """Propagate filter with one full IMU sample.

        All inputs in **sensor/body frame**.  Angles in degrees.
        """
        if dt is None:
            dt = self.dt_nominal

        # ----------------------------------------------------------------
        # 1. Get rotation & linear acceleration in world frame
        # ----------------------------------------------------------------
        R_bw = euler_to_rot(roll_deg, pitch_deg, yaw_deg)
        acc_body = np.array([ax, ay, az])
        acc_world = R_bw @ acc_body  # includes gravity
        acc_world[2] -= G_STD        # remove gravity (z‑axis)
        ax_w, ay_w = acc_world[0], acc_world[1]

        # ----------------------------------------------------------------
        # 2. Rotate existing velocity by Δyaw (centripetal effect)
        # ----------------------------------------------------------------
        yaw_rad = math.radians(yaw_deg)
        if self.prev_yaw_rad is None:
            dyaw = 0.0
        else:
            dyaw = wrap_pi(yaw_rad - self.prev_yaw_rad)

        if abs(dyaw) > 1e-9:
            c, s = math.cos(dyaw), math.sin(dyaw)
            vx, vy = self.state[2], self.state[3]
            self.state[2] = c * vx - s * vy
            self.state[3] = s * vx + c * vy

        self.prev_yaw_rad = yaw_rad

        # ----------------------------------------------------------------
        # 3. Integrate acceleration → velocity & position
        # ----------------------------------------------------------------
        self.state[0] += self.state[2] * dt + 0.5 * ax_w * dt * dt
        self.state[1] += self.state[3] * dt + 0.5 * ay_w * dt * dt
        self.state[2] += ax_w * dt
        self.state[3] += ay_w * dt

        # ----------------------------------------------------------------
        # 4. Propagate covariance (linearised)
        # ----------------------------------------------------------------
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        # Process noise (white‑noise acceleration model) – treat X & Y indep.
        q = self._q_acc2
        q11 = 0.25 * dt**4 * q
        q13 = 0.5  * dt**3 * q
        q33 =        dt**2 * q
        Q = np.array([[q11, 0.0, q13, 0.0],
                      [0.0, q11, 0.0, q13],
                      [q13, 0.0, q33, 0.0],
                      [0.0, q13, 0.0, q33]], dtype=float)

        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    # Update step – call at GPS/local‑pos rate (≤ 1 Hz)
    # ------------------------------------------------------------------
    def update_position(self, x_meas: float, y_meas: float) -> None:
        """Fuse absolute (x, y) measurement."""
        z = np.array([x_meas, y_meas])
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        y = z - H @ self.state                # innovation
        S = H @ self.P @ H.T + self.R

        # Mahalanobis distance²
        # d2 = y.T @ np.linalg.inv(S) @ y
        # if d2 > 9.0:
        #     # outlier → bỏ qua update
        #     return

        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    @property
    def position(self) -> Tuple[float, float]:
        return self.state[0], self.state[1]

    @property
    def velocity(self) -> Tuple[float, float]:
        return self.state[2], self.state[3]

    # ------------------------------------------------------------------
    # Reset helper
    # ------------------------------------------------------------------
    def reset(self, *, x=0.0, y=0.0, vx=0.0, vy=0.0):
        """Reset state & covariance (keeps previous yaw history)."""
        self.state[:] = (x, y, vx, vy)
        self.P[:] = np.eye(4) * 1e-3
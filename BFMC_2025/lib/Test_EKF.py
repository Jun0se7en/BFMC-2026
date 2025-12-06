# imu_ekf.py – Extended Kalman Filter fusing body‑frame IMU (ax, ay, heading)
# with optional absolute (x, y) fixes (GPS, vision, SLAM…).
# -----------------------------------------------------------------------------
# Author: ChatGPT – June 2025
# Licence: MIT
# -----------------------------------------------------------------------------
"""
Usage example
-------------
>>> from imu_ekf import IMUEKF
>>> ekf = IMUEKF(dt=0.02)                # 50 Hz loop
>>> while True:
...     ax_b, ay_b, heading_deg = read_imu()
...     ekf.predict(ax_b, ay_b, heading_deg)    # always call predict
...     if new_gps:
...         x_fix, y_fix = read_gps()
...         ekf.update_position(x_fix, y_fix)   # call when fix available
...     print(ekf.position)

Principle
---------
* **State**  X = [x, y, vx, vy]ᵀ  in *world* frame.
* **Predict**
    1. Rotate existing velocity (vx, vy) by Δheading to account for centripetal
       effect when the vehicle turns at roughly constant speed.
    2. Rotate body‑frame accelerations (ax_b, ay_b) to world frame using the
       *current* heading, then integrate (double‑integral) to update velocity
       and position.
* **Update** – simple linear measurement z = [x, y]ᵀ.

Why rotate velocity?
--------------------
When a vehicle drives a circle at constant speed, linear acceleration along the
body axes is nearly zero (centripetal acceleration appears in the lateral axis
but is small and often highly filtered).  If we integrate only acceleration,
the EKF would keep velocity pointing straight ahead and the predicted path
remains a straight line.  Rotating the velocity vector by the observed change
in heading adds the missing centripetal term implicitly, yielding an accurate
circular trajectory even with near‑zero accelerometer readings.
"""

from __future__ import annotations
import numpy as np
import math
from typing import Optional

__all__ = ["IMUEKF"]

# ----------------------------------------------------------------------------
# Helper ──────────────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------

def _wrap(rad: float) -> float:
    """Wrap angle into (‑π, π]."""
    return (rad + math.pi) % (2 * math.pi) - math.pi

# ----------------------------------------------------------------------------
# EKF class ───────────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------

class IMUEKF:
    """Extended Kalman Filter for planar vehicle using body‑frame IMU.

    Parameters
    ----------
    dt : float
        Nominal sampling period (s).  You may pass a different *dt* to
        :py:meth:`predict` each frame if the loop is not perfectly periodic.
    accel_std : float, default 0.25
        1‑σ standard deviation of linear acceleration noise **in body frame**
        (m/s²).
    pos_meas_std : float, default 0.50
        1‑σ of position fixes (m).
    init_vel : tuple[float, float], default (0, 0)
        Optional initial velocity in world frame (m/s).  For a vehicle that
        starts already in motion, set this to the known tangential speed to
        avoid the start‑up straight‑line artefact.
    """

    def __init__(
        self,
        dt: float = 0.1,
        # *,
        accel_std: float = 0.25,
        pos_meas_std: float = 0.50,
        init_vel: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.dt_nominal = dt
        self.prev_heading_rad: Optional[float] = None

        # state X = [x, y, vx, vy]
        self.state = np.zeros(4)
        self.state[2:] = init_vel

        # covariance P (start small but non‑zero)
        self.P = np.eye(4) * 1e-3

        self._accel_std2 = accel_std ** 2
        self._pos_r = np.diag([pos_meas_std**2, pos_meas_std**2])

    # ───────────────────────────────────────────────── predict ─────────────
    def predict(
        self,
        ax_body: float,
        ay_body: float,
        heading_deg: float,
        # *,
        dt: Optional[float] = None,
    ) -> None:
        """Propagate the filter with one IMU sample.

        Parameters
        ----------
        ax_body, ay_body : float
            Linear acceleration in *body* frame (m/s²).
        heading_deg : float
            Current yaw/heading angle (°) in world frame (NED/ENU consistent).
        dt : float, optional
            If omitted, uses the nominal dt passed at construction.
        """
        if dt is None:
            dt = self.dt_nominal

        # Δheading (rad) ---------------------------------------------------
        theta = math.radians(heading_deg)
        if self.prev_heading_rad is None:
            dtheta = 0.0
        else:
            dtheta = _wrap(theta - self.prev_heading_rad)

        # 1) rotate existing velocity by Δθ (centripetal effect) -----------
        if abs(dtheta) > 1e-9:
            c, s = math.cos(dtheta), math.sin(dtheta)
            vx, vy = self.state[2], self.state[3]
            self.state[2] = c * vx - s * vy
            self.state[3] = s * vx + c * vy

        self.prev_heading_rad = theta

        # 2) rotate body acceleration into world frame --------------------
        c_h, s_h = math.cos(theta), math.sin(theta)
        ax_w = c_h * ax_body - s_h * ay_body
        ay_w = s_h * ax_body + c_h * ay_body

        # 3) propagate state ----------------------------------------------
        # position update:  x += v*dt + 0.5*a*dt²
        self.state[0] += self.state[2] * dt + 0.5 * ax_w * dt * dt
        self.state[1] += self.state[3] * dt + 0.5 * ay_w * dt * dt
        # velocity update:  v += a*dt
        self.state[2] += ax_w * dt
        self.state[3] += ay_w * dt

        # 4) propagate covariance ----------------------------------------
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)

        # noise due to linear accel (white noise acceleration model)
        q = self._accel_std2
        q11 = 0.25 * dt**4 * q
        q13 = 0.5  * dt**3 * q
        q33 =        dt**2 * q
        Qx = np.array([[q11, q13],
                        [q13, q33]])
        Q = np.block([[Qx, np.zeros((2, 2))],
                      [np.zeros((2, 2)), Qx]])

        self.P = F @ self.P @ F.T + Q

    # ───────────────────────────────────────────────── update ─────────────
    def update_position(self, x_meas: float, y_meas: float) -> None:
        """Fuse absolute (x, y) measurement (e.g. GPS/Lidar SLAM)."""
        z = np.array([x_meas, y_meas])
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        y = z - H @ self.state                    # innovation
        S = H @ self.P @ H.T + self._pos_r   # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    # ───────────────────────────── handy getters ─────────────────────────
    @property
    def position(self):
        """Return (x, y) estimate."""
        return self.state[0], self.state[1]

    @property
    def velocity(self):
        """Return (vx, vy) estimate."""
        return self.state[2], self.state[3]

    # -------------------------------------------------------------------
    # Optional: reset / set state quickly (could be expanded as needed)
    # -------------------------------------------------------------------
    def reset(self, *, x=0.0, y=0.0, vx=0.0, vy=0.0):
        """Reset filter state and covariance (keeps previous heading)."""
        self.state[:] = (x, y, vx, vy)
        self.P[:] = np.eye(4) * 1e-3

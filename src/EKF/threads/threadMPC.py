import cv2
import threading
import base64
import time
import numpy as np
import os
import sys
import json
import random
import ctypes
import pandas as pd
from scipy.optimize import minimize

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Record,
    Config,
    Position,
)
from src.templates.threadwithstop import ThreadWithStop
import socket
import json
import math
import signal
from lib.Transform import Transform
import serial
import pynmea2
import struct
from scipy.spatial.transform import Rotation as R

# mpc_controller.py – MPC path‑tracking with SciPy SLSQP optimiser
# -----------------------------------------------------------------------------
# This version abandons CasADi / cvxpy and uses only **SciPy** so bạn không phải
# cài thêm solver phức tạp.  (pip install numpy pandas scipy matplotlib)
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Helper – kinematic bicycle model (simple, no slip)
# ──────────────────────────────────────────────────────────────────────────────

def kinematic_step(x: float, y: float, psi: float, v: float,
                   delta: float, a: float,
                   dt: float, Lf: float) -> Tuple[float, float, float, float]:
    """1‑step forward Euler integration."""
    beta = delta  # small‑angle approx (front‑wheel steer only)
    x += v * np.cos(psi) * dt
    y += v * np.sin(psi) * dt
    psi += v / Lf * beta * dt
    v += a * dt
    return x, y, psi, v

# ──────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────── I/O helpers

def load_route(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df[['x', 'y']].values.astype(float)
# ──────────────────────────────────────────────────────────────────────────────

class MPC:
    """Bare‑bones MPC for 2‑D path tracking using SciPy SLSQP."""

    def __init__(self,
                 route_points: np.ndarray,
                 horizon: int = 10,
                 dt: float = 0.1,
                 Lf: float = 2.5,
                 ref_v: float = 2.0,
                 steer_limit: float = 0.4,
                 accel_limit: float = 1.0,
                 w_cte: float = 1.0,
                 w_epsi: float = 1.0,
                 w_v: float = 0.1,
                 w_smooth_steer: float = 10.0,
                 w_smooth_acc: float = 1.0,
                 ) -> None:
        self.route = route_points  # shape (M,2)
        self.horizon = horizon
        self.dt = dt
        self.Lf = Lf
        self.ref_v = ref_v
        self.steer_limit = steer_limit
        self.accel_limit = accel_limit
        # weights
        self.w_cte = w_cte
        self.w_epsi = w_epsi
        self.w_v = w_v
        self.w_smooth_steer = w_smooth_steer
        self.w_smooth_acc = w_smooth_acc

        # previous controls (for smooth cost)
        self._prev_delta = 0.0
        self._prev_a = 0.0

    # ───────────────────────────────────────────────── nearest reference
    def _nearest_index(self, x: float, y: float) -> int:
        d2 = np.sum((self.route - np.array([x, y]))**2, axis=1)
        return int(np.argmin(d2))

    # ───────────────────────────────────────────────── cost function
    def _objective(self, u: np.ndarray, state0: Tuple[float, float, float, float], ref_slice: np.ndarray) -> float:
        """Compute cumulative cost over horizon.
           u = [δ0..δN-1, a0..aN-1] length 2N.
        """
        N = self.horizon
        deltas = u[:N]
        accels = u[N:]

        x, y, psi, v = state0
        cost = 0.0

        for k in range(N):
            # step vehicle
            x, y, psi, v = kinematic_step(x, y, psi, v, deltas[k], accels[k], self.dt, self.Lf)
            # ref point is kth element in ref_slice (cyclic within slice length)
            xr, yr = ref_slice[min(k, len(ref_slice)-1)]
            # cross‑track error (distance in plane)
            cte = np.hypot(x - xr, y - yr)
            # orientation error: angle between path tangent and heading (approx)
            if k < len(ref_slice)-1:
                dx = ref_slice[min(k+1, len(ref_slice)-1)][0] - xr
                dy = ref_slice[min(k+1, len(ref_slice)-1)][1] - yr
                path_psi = np.arctan2(dy, dx + 1e-9)
                epsi = np.abs(np.arctan2(np.sin(psi - path_psi), np.cos(psi - path_psi)))
            else:
                epsi = 0
            cost += self.w_cte * cte**2 + self.w_epsi * epsi**2 + self.w_v * (v - self.ref_v)**2
            # smoothness
            if k == 0:
                cost += self.w_smooth_steer * (deltas[k] - self._prev_delta)**2
                cost += self.w_smooth_acc * (accels[k] - self._prev_a)**2
            else:
                cost += self.w_smooth_steer * (deltas[k] - deltas[k-1])**2
                cost += self.w_smooth_acc * (accels[k] - accels[k-1])**2
        return cost

    # ───────────────────────────────────────────────── public solve()
    def solve(self,
              x: float, y: float, psi: float, v: float,
              delta_prev: float = 0.0) -> Tuple[float, float]:
        """Return optimal (steer, velocity_cmd)."""
        self._prev_delta = delta_prev
        self._prev_a = 0.0  # could be tracked if available

        # 1) pick reference slice starting from nearest point
        idx0 = self._nearest_index(x, y)
        ref_slice = self.route[idx0: idx0 + self.horizon]
        if ref_slice.shape[0] < self.horizon:  # wrap around if at tail
            extra = self.horizon - ref_slice.shape[0]
            ref_slice = np.vstack([ref_slice, self.route[:extra]])

        # 2) initial guess (zeros)
        N = self.horizon
        u0 = np.zeros(2*N)

        # 3) bounds
        bounds = [(-self.steer_limit, self.steer_limit)] * N + [(-self.accel_limit, self.accel_limit)] * N

        # 4) solve
        res = minimize(self._objective,
                       u0,
                       args=((x, y, psi, v), ref_slice),
                       method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-4, 'disp': True})

        if not res.success:
            # fallback: keep previous delta, zero accel
            return self._prev_delta, v
        u_opt = res.x
        steer_cmd = np.clip(u_opt[0], -self.steer_limit, self.steer_limit)
        accel_cmd = np.clip(u_opt[N], -self.accel_limit, self.accel_limit)
        vel_cmd = v + accel_cmd * self.dt
        return float(steer_cmd), float(vel_cmd)

class threadMPC(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, debugger):
        super(threadMPC, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.debugger = debugger
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        # self.client.send(self.speed, self.smooth_angle)
        self.subscribe()
        self.Configs()
        self.message = {}
        self.message_type = ""

        ################### MPC #####################
        self.route = load_route('./lib/route_points.csv')
        self.mpc = MPC(route_points=self.route, horizon=3, dt=0.1)
        self.v = 0
        self.delta_prev = 0.0
        self.x = 0
        self.y = 0

    # ================================= SUBCRIBE ===============================================
    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadMPC", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadMPC", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        self.ser.close()
        super(threadMPC, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()
    
    # =============================== MPC =================================================
    
    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. 
        It captures the image from camera and make the required modifies and then it send the data to process gateway."""   
        while self._running:
            # print("MPC Running!!!")
            if not self.queuesList[Position.Queue.value].empty():
                coords = self.queuesList[Position.Queue.value].get()["msgValue"]
                # print(coords)
                if self.x != coords["x"] or self.y != coords["y"]:
                    self.x = coords["x"]
                    self.y = coords["y"]
                    self.psi = coords["heading"]
                    steer, vel_cmd = self.mpc.solve(self.x, self.y, self.psi, self.v, self.delta_prev)
                    a_cmd = (vel_cmd - self.v) / self.mpc.dt

                    print(math.degrees(steer), vel_cmd, a_cmd)
                    self.delta_prev = steer

                # print(coords)


    # =============================== START ===============================================
    def start(self):
        super(threadMPC, self).start()

        

"""Shared helpers for the edge prototype: MQTT topics, array encoding, INA219 energy meter."""
from __future__ import annotations

import base64
import threading
import time

import numpy as np

T_SENSOR = "hcfl/sensor/#"
T_ROUND = "hcfl/round/start"          # server -> clients: round id + global model
T_UPDATE = "hcfl/round/update"        # clients -> server: DP update + weighting terms
T_STATE = "hcfl/counterpart/state"    # clients -> server: counterpart state summary


def enc(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()


def dec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).astype(np.float64)


class EnergyMeter:
    """Integrates INA219 power (mW) over time -> energy (mJ). Falls back to None without hardware.

    pip install adafruit-circuitpython-ina219 ; INA219 in series with the Pi supply, I2C addr 0x40.
    """

    def __init__(self, period_s=0.005):
        self.period = period_s
        self.ok = False
        try:
            import board          # noqa
            import busio          # noqa
            from adafruit_ina219 import INA219
            self.sensor = INA219(busio.I2C(board.SCL, board.SDA))
            self.ok = True
        except Exception:
            self.sensor = None
        self._run = False

    def _loop(self):
        last = time.perf_counter()
        while self._run:
            p = self.sensor.power * 1000.0       # W -> mW
            now = time.perf_counter()
            self.energy_mJ += p * (now - last)
            last = now
            time.sleep(self.period)

    def start(self):
        self.energy_mJ = 0.0
        if self.ok:
            self._run = True
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()

    def stop(self):
        if self.ok:
            self._run = False
            self._th.join()
            return self.energy_mJ
        return None

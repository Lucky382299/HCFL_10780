"""Raspberry Pi 4B edge client.

Roles (Section III-A of the manuscript): edge client, local preprocessing unit,
holographic-counterpart engine and local model-update generator.

  1. Subscribes to timestamped NodeMCU packets (hcfl/sensor/<node>) over MQTT.
  2. Aligns the node streams on a 1 s grid into one observation vector x_t.
  3. Runs the holographic counterpart (Eqs. 2-4) on every grid tick, with
     a_t = 0 when a packet is missing and Delta = synchronization age.
  4. On hcfl/round/start, trains the local MLP on the corrected states h,
     clips + Laplace-perturbs the update (Eq. 13) and publishes it with
     n_i, p~_i (Eqs. 1,5-7) and c_i.
  5. Logs inference latency, peak memory (psutil) and energy per round (INA219).

Run:  python edge/rpi_edge_client.py --client-id 0 --broker 127.0.0.1 --nodes node01 node02 node03
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.common import T_ROUND, T_SENSOR, T_STATE, T_UPDATE, EnergyMeter, dec, enc  # noqa: E402
from hcfl.config import load_config                    # noqa: E402
from hcfl.counterpart import HolographicCounterpart     # noqa: E402
from hcfl.models import make_model                      # noqa: E402
from hcfl.priority import adjusted_priority, initial_priority, network_condition, reliability  # noqa: E402
from hcfl.accountant import calibrate_sigma, epsilon     # noqa: E402
from hcfl.privacy import laplace_mechanism              # noqa: E402

# Feature order per node packet: [CO2-equivalent ppm (MQ-135), PM2.5, PM10, T (C), RH (%)]
PM25_THRESHOLD = 35.0     # ug/m3: "poor air quality" label for on-device training (edit to match manuscript)


class EdgeClient:
    def __init__(self, a):
        self.a = a
        self.cfg = load_config(a.config)
        self.nodes = a.nodes
        self.per_node = 5
        self.d = self.per_node * len(self.nodes)
        self.lock = threading.Lock()
        self.latest = {n: None for n in self.nodes}   # (recv_time, ts_ms, x)
        self.last_rx = {n: 0.0 for n in self.nodes}
        self.seen = {n: (0, 0) for n in self.nodes}   # (received, expected) for PER
        # running scaler (fixed after warm-up; only local data are used)
        self.buf_raw = deque(maxlen=a.buffer)
        self.H, self.Y, self.C, self.F, self.E = (deque(maxlen=a.buffer) for _ in range(5))
        cc = self.cfg.counterpart
        self.cp = HolographicCounterpart(self.d, cc.nu, cc.tau0, cc.kappa, cc.eps0, a.mode)
        self.model = None
        self.mu = self.sd = None
        self.rng = np.random.default_rng(a.client_id)
        self.cli = mqtt.Client(client_id=f"edge{a.client_id}")
        self.cli.on_message = self.on_msg
        self.cli.connect(a.broker, 1883)
        self.cli.subscribe(T_SENSOR); self.cli.subscribe(T_ROUND)
        self.meter = EnergyMeter()
        self.delays_ms = deque(maxlen=a.buffer)
        self.log = open(f"edge_client{a.client_id}_log.jsonl", "a")

    # -------------------------------------------------- MQTT
    def on_msg(self, cli, ud, msg):
        if msg.topic.startswith("hcfl/sensor/"):
            p = json.loads(msg.payload)
            n = p["node"]
            if n in self.latest:
                with self.lock:
                    self.latest[n] = (time.time(), p["ts_ms"], np.asarray(p["x"], float))
                    r, e = self.seen[n]; self.seen[n] = (r + 1, e)
        elif msg.topic == T_ROUND:
            threading.Thread(target=self.do_round, args=(json.loads(msg.payload),), daemon=True).start()

    # -------------------------------------------------- 1 Hz alignment + counterpart
    def tick(self):
        now = time.time()
        x = np.empty(self.d); received = True; ages = []
        with self.lock:
            for k, n in enumerate(self.nodes):
                r, e = self.seen[n]; self.seen[n] = (r, e + 1)
                item = self.latest[n]
                if item is None or now - item[0] > 1.5 * self.cfg.counterpart.sample_period_s:
                    received = False
                    ages.append(now - (item[1] / 1000 if item else now - 999))
                    x[k * 5:(k + 1) * 5] = np.nan
                else:
                    x[k * 5:(k + 1) * 5] = item[2]
                    ages.append(max(0.0, now - item[1] / 1000.0))   # delay incl. network latency
        if received and not np.isnan(x).any():
            self.buf_raw.append(x.copy())
        if self.mu is None:
            if len(self.buf_raw) >= self.a.warmup:   # fit z-score on local warm-up data only
                R = np.asarray(self.buf_raw); self.mu = R.mean(0); self.sd = R.std(0) + 1e-9
                self.cp.fill = np.zeros(self.d)
            return
        xn = None if not received or np.isnan(x).any() else (x - self.mu) / self.sd
        t0 = time.perf_counter()
        h, c, f, e = self.cp.step(xn, xn is not None, float(np.max(ages)))
        if xn is not None:
            self.delays_ms.append(1000 * float(np.max(ages)))
        self.cp_ms = 1000 * (time.perf_counter() - t0)
        if xn is not None:   # label from the physical observation (threshold rule)
            pm25 = x[1::5]
            self.H.append(h); self.Y.append(int(np.nanmax(pm25) > PM25_THRESHOLD))
            self.C.append(c); self.F.append(f); self.E.append(e)

    # -------------------------------------------------- one FL round
    def do_round(self, msg):
        if self.mu is None or len(self.H) < 32:
            return
        tc, pc = self.cfg.train, self.cfg.priority
        X = np.asarray(self.H); y = np.asarray(self.Y)
        if self.model is None:
            self.model = make_model(self.d, 2, tc, seed=self.a.client_id)
        theta = dec(msg["theta"])
        assert theta.size == self.model.num_params, "server --dim must equal 5 x number of nodes"
        self.model.set_flat(theta)
        self.meter.start()
        t0 = time.perf_counter()
        dpc = self.cfg.dp
        if dpc.enabled and dpc.mechanism == "dpsgd":
            n = len(y); q = round(min(1.0, dpc.batch_size / n), 6)
            max_steps = tc.rounds * dpc.local_epochs * int(np.ceil(n / dpc.batch_size))
            sigma = calibrate_sigma(q, max_steps, dpc.epsilon, dpc.delta)
            self.dp_steps = getattr(self, "dp_steps", 0) + self.model.train_dpsgd(
                X, y, dpc.local_epochs, dpc.batch_size, dpc.lr, dpc.clip_C, sigma,
                seed=msg["round"], noise_seed=int(self.rng.integers(2 ** 31)))
            u_t = self.model.get_flat() - theta
            eps_spent = epsilon(q, sigma, self.dp_steps, dpc.delta)
        else:
            self.model.train_local(X, y, tc.local_epochs, tc.batch_size, tc.lr, seed=msg["round"])
            u = self.model.get_flat() - theta
            u_t, _ = laplace_mechanism(u, dpc.clip_C, dpc.epsilon, self.rng) if dpc.enabled else (u, u)
            eps_spent = None
        train_s = time.perf_counter() - t0
        # inference latency (single sample, includes counterpart step)
        t1 = time.perf_counter()
        for _ in range(50):
            self.model.predict_proba(X[-1:])
        infer_ms = 1000 * (time.perf_counter() - t1) / 50 + getattr(self, "cp_ms", 0.0)
        energy = self.meter.stop()
        # priority terms
        resid = np.diff(X, n=2, axis=0)
        eta = float(np.clip(np.median(np.abs(resid)) / np.sqrt(6) / pc.eta_ref, 0, 1))
        r_, e_ = map(sum, zip(*self.seen.values()))
        per = 1.0 - r_ / max(e_, 1)
        Cc = float(y.mean()); Hh = 1.0 - per
        p = initial_priority(Cc, eta, Hh, pc)
        L = min(1.0, float(np.mean(self.delays_ms)) / self.cfg.channel.max_delay_ms) if self.delays_ms else 0.0
        N = network_condition(self.a.bandwidth, L, per, pc)
        pt = adjusted_priority(p, reliability(eta, pc), N, pc)
        c_mean = float(np.mean(self.C))
        payload = {"client": self.a.client_id, "round": msg["round"], "update": enc(u_t),
                   "n": int(len(y)), "p_tilde": pt, "c": c_mean}
        self.cli.publish(T_UPDATE, json.dumps(payload))
        state = {"client": self.a.client_id, "h": enc(self.cp.h1), "c": c_mean,
                 "f": float(np.mean(self.F)), "e": float(np.mean(self.E))}
        state_msg = json.dumps(state)
        if self.a.mode != "none":
            self.cli.publish(T_STATE, state_msg)
        rec = {"round": msg["round"], "train_s": train_s, "infer_ms": infer_ms,
               "peak_rss_MB": psutil.Process().memory_info().rss / 2 ** 20,
               "energy_mJ": energy, "state_KB": (len(state_msg) / 1024 if self.a.mode != "none" else 0.0),
               "update_KB": len(payload["update"]) / 1024, "c": c_mean, "PER": per, "eps_spent": eps_spent}
        self.log.write(json.dumps(rec) + "\n"); self.log.flush()
        print(rec, flush=True)

    def run(self):
        self.cli.loop_start()
        period = self.cfg.counterpart.sample_period_s
        nxt = time.time()
        while True:
            nxt += period
            self.tick()
            time.sleep(max(0.0, nxt - time.time()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-id", type=int, default=0)
    ap.add_argument("--broker", default="127.0.0.1")
    ap.add_argument("--nodes", nargs="+", default=["node01"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", default="full", choices=["full", "static", "none"])
    ap.add_argument("--bandwidth", type=float, default=1.0, help="normalized link bandwidth B in [0,1]")
    ap.add_argument("--buffer", type=int, default=3600)
    ap.add_argument("--warmup", type=int, default=120)
    EdgeClient(ap.parse_args()).run()

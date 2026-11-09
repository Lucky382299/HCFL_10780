"""Federated server for the hardware prototype (honest-but-curious aggregator).

Each round: broadcasts theta^t, collects privatized updates with (n_i, p~_i, c_i)
from the clients that respond before the deadline (asynchronous participation),
computes the counterpart-confidence weights (Eq. 8) and applies Eq. (14).

Run:  python edge/fl_server.py --broker 127.0.0.1 --rounds 12 --deadline 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.common import T_ROUND, T_UPDATE, dec, enc   # noqa: E402
from hcfl import aggregation as agg                     # noqa: E402
from hcfl.config import load_config                    # noqa: E402
from hcfl.models import make_model                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="127.0.0.1")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--deadline", type=float, default=60.0, help="seconds to wait for updates per round")
    ap.add_argument("--warmup", type=float, default=180.0, help="seconds before round 1 (counterpart warm-up)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dim", type=int, required=True, help="feature dimension d = 5 x number of NodeMCU nodes")
    a = ap.parse_args()
    cfg = load_config(a.config)

    inbox, lock = {}, threading.Lock()

    def on_msg(cli, ud, msg):
        p = json.loads(msg.payload)
        with lock:
            inbox.setdefault(p["round"], {})[p["client"]] = p

    cli = mqtt.Client(client_id="hcfl-server")
    cli.on_message = on_msg
    cli.connect(a.broker, 1883)
    cli.subscribe(T_UPDATE)
    cli.loop_start()
    time.sleep(a.warmup)

    theta = make_model(a.dim, 2, cfg.train, seed=0).get_flat()   # theta^0 broadcast to all clients
    prev_w = {}
    for t in range(1, a.rounds + 1):
        cli.publish(T_ROUND, json.dumps({"round": t, "theta": enc(theta)}))
        time.sleep(a.deadline)
        with lock:
            got = inbox.pop(t, {})
        if not got:
            print(f"round {t}: no updates"); continue
        ids = sorted(got)
        U = [dec(got[i]["update"]) for i in ids]
        t0 = time.perf_counter()
        w_bar = agg.preliminary_weights([got[i]["n"] for i in ids], [got[i]["p_tilde"] for i in ids],
                                        [got[i]["c"] for i in ids])
        w = agg.damped_weights(ids, w_bar, prev_w, cfg.counterpart.gamma)
        theta = agg.apply_update(theta, U, w)
        print(json.dumps({"round": t, "clients": ids, "weights": [round(float(x), 4) for x in w],
                          "agg_ms": 1000 * (time.perf_counter() - t0)}), flush=True)
    np.save("global_model.npy", theta)


if __name__ == "__main__":
    main()

# Holographic Counterpart–Enabled Privacy-Preserving Federated Learning for IoT Consumer Electronics

**Journal:** IEEE Latin America Transactions  
**Submission ID:** 10780  
**Authors:** Laxman Kisan Lonkar ([ORCID 0009-0007-5771-752X](https://orcid.org/0009-0007-5771-752X)) and B. Malarkodi ([ORCID 0000-0003-2867-6574](https://orcid.org/0000-0003-2867-6574))  
**Affiliation:** Department of Electronics and Communication Engineering, National Institute of Technology Tiruchirappalli, Tamil Nadu, India  
**Video abstract:** https://youtu.be/gWATonaW96c

This repository contains the code for the manuscript. HC-FL is a priority-adaptive federated learning framework in which lightweight holographic counterparts (virtual copies of physical IoT sensor nodes) track physical–virtual divergence and synchronization freshness, correct stale sensor states, and set the aggregation weights. Client updates can be protected with differential privacy (see the section on differential privacy below).

## Repository structure

| Path | Contents |
|---|---|
| `hcfl/` | Core library: counterpart, priority, aggregation, privacy, models, FL loop, channel simulation, data loading, metrics, plots |
| `experiments/run.py` | Single entry point that reproduces the tables and figures |
| `configs/` | Experiment settings for MIMIC-III, UCI Air Quality and a synthetic test set |
| `edge/` | Hardware prototype: NodeMCU firmware, Raspberry Pi edge client, FL server, overhead measurement |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |

## Map from manuscript to code

| Manuscript | Code |
|---|---|
| Eq. (1) contextual priority, Eq. (5) reliability, Eq. (6) network condition, Eq. (7) adjusted priority | `hcfl/priority.py` |
| Eq. (2) trend prediction, Eq. (3) divergence / freshness / confidence, Eq. (4) state correction | `hcfl/counterpart.py` |
| Eq. (8) confidence weights with damping, Eq. (14) global update | `hcfl/aggregation.py` |
| Eqs. (12)–(13) L1 clipping + Laplace DP, Eq. (15) PDI, sequential composition | `hcfl/privacy.py` |
| Eqs. (9)–(11) local objective, MLP 64-32-16, ReLU, dropout 0.25, Adam 1e-3, batch 16, 5 local epochs | `hcfl/models.py` |
| Algorithm 1, early stopping (12 rounds, patience 3), FedAvg / FedProx / SCAFFOLD / ablations | `hcfl/fl.py` |
| Delay, packet loss, disconnection, Gaussian noise 0.02, client heterogeneity | `hcfl/channel.py` |
| Dirichlet non-IID (alpha_D = 0.5) with quantity imbalance | `hcfl/partition.py` |
| MIMIC-III and UCI Air Quality preprocessing, splits, threshold labels | `hcfl/data.py` |
| Tables II, III, IV and Figs. 3, 4, 5 | `experiments/run.py` |
| Table I (Raspberry Pi overhead) | `edge/measure_overhead.py` |
| NodeMCU firmware (MQ-135, PMS5003, SHT25, MQTT) | `edge/nodemcu_sensor_node/` |
| Raspberry Pi edge client and federated server | `edge/rpi_edge_client.py`, `edge/fl_server.py` |

## Install

```bash
pip install -r requirements.txt           # workstation (PyTorch backend)
pip install numpy pandas scikit-learn pyyaml psutil paho-mqtt adafruit-circuitpython-ina219   # Raspberry Pi
```
`train.backend: auto` uses PyTorch when it is installed and otherwise a NumPy
implementation of the same network. Both expose the same flat-parameter interface.

## Data

* **MIMIC-III** (credentialed PhysioNet access). Point `data.path` in `configs/mimic3.yaml`
  to the folder holding `CHARTEVENTS.csv.gz`. The first run extracts hourly HR, SBP, DBP, RR,
  SpO2 and temperature (standard CareVue + MetaVision ITEMIDs) into `hcfl_vitals_hourly.csv`.
* **UCI Air Quality** (UCI id 360). Point `data.path` in `configs/uci_air.yaml` to `AirQualityUCI.csv`.
* **Synthetic** vital-sign streams (`configs/synthetic.yaml`) are for testing the pipeline only.

**Check the label thresholds before running.** `MIMIC_RANGES`, `UCI_LABEL_COL` and `UCI_THRESHOLD`
in `hcfl/data.py` must match the ranges stated in the manuscript.

## Run

```bash
python experiments/run.py all --config configs/mimic3.yaml --jobs 4
python experiments/run.py all --config configs/uci_air.yaml --jobs 4
```
Results go to `results/<dataset>/`. Each experiment writes per-seed CSVs, mean ± SD markdown
tables and 300-dpi figures.

| Command | Produces |
|---|---|
| `main` | Table II (7 methods), Table IV |
| `privacy` | Fig. 3 (macro-F1 and PDI for ε = 0.6, 0.8, 1.0, 1.2) |
| `sync` | Fig. 4 (delay 0/100/250/500 ms, loss 0/5/10/20 %, disconnection 5/10/20/30 s) |
| `ablation` | Table III (stress condition: 250 ms, 10 % loss, 10 s disconnection) |
| `scalability` | Fig. 5 (10/20/50 clients; 100/75/50 % availability) |

## Hardware prototype

1. Flash `edge/nodemcu_sensor_node/nodemcu_sensor_node.ino` to each NodeMCU (set Wi-Fi, broker IP, `NODE_ID`).
2. On the Raspberry Pi, run Mosquitto, then
   `python edge/rpi_edge_client.py --client-id 0 --nodes node01 node02 node03`.
3. On the server, run `python edge/fl_server.py --dim 15` (d = 5 × number of nodes).
4. For Table I, record a stream to CSV and run `python edge/measure_overhead.py --csv stream.csv` on the Pi.
   Energy needs an INA219 on the Pi supply rail. Without it, the energy column prints N/A.

## Implementation decisions 

* **Client streams.** The counterpart needs temporally ordered data, so the Dirichlet partition
  allocates contiguous blocks (one patient for MIMIC-III, one 24-h day for UCI) instead of single rows.
* **Round-level confidence.** c_i^t used in Eq. (8) is the mean per-sample confidence over the client's round window.
* **C, η, H.** C = local share of high-risk samples. η = robust noise estimate from second differences.
  H = 1 − fraction of the window spent disconnected.
* **Damping under partial participation.** Eq. (8) keeps Σω = 1 only if the previous weights of the
  current participants sum to one. The code renormalizes them over A_t first.
* **Baselines.** "Without counterpart" replaces lost packets by the training mean. "Static counterpart"
  holds the last received value. Both use priority weights (Eq. 8 with c = 1).
* **Recovery time.** Seconds after reconnection until the relative state error stays below
  2 × the pre-disconnection median error for 3 consecutive samples.
* **DP is off by default** (`dp.enabled: false`) except in the `privacy` experiment (Fig. 3).

## Differential privacy 

The R1 manuscript clipped each client update to an L1 bound C and added Laplace noise of scale 2C/ε (Eq. 13).
With P model parameters, ‖ξ‖₂ ≈ √(2P)·2C/ε while ‖ū‖₂ ≤ C, so PDI ≥ 2√(2P)/ε (≥ 262 at ε = 0.6 for
P = 3,090). That mechanism makes the model unusable. It is kept as `dp.mechanism: laplace` for comparison only.

The default is now **DP-SGD with Rényi-DP accounting** (`dp.mechanism: dpsgd`, `hcfl/accountant.py`):

* Poisson sampling with rate q = B/n_i, per-example L2 clipping to C, Gaussian noise N(0, σ²C²) per step.
* Each client's σ_i is calibrated before training so that participating in all T rounds stays within the
  **total** budget (ε, δ). ε in Fig. 3 is therefore a total per-client budget, not a per-round one.
* The accountant matches TF-Privacy on the standard MNIST setting (ε ≈ 3.01).
* DP settings: `dp.batch_size: 256`, `dp.local_epochs: 1`, `dp.lr: 0.01`, `dp.clip_C: 1.0`, `dp.delta: 1e-5`.
  Tune them on the validation partition, and keep δ below 1/n_i.
* PDI (Eq. 15) compares each private update with a clipped, noise-free twin run on the same mini-batches.
  The twin is for evaluation only and doubles the simulated training cost (`dp.compute_pdi`).
* DP-SGD needs reasonably large client datasets. On synthetic clients with about 21,000 samples each,
  macro-F1 was 84.1 % at ε = 0.6 and 86.0 % at ε = 1.2, against 93.8 % without DP.
  With about 2,800 samples per client the model collapsed to the majority class even at ε = 8.

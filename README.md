# BDA450-Elevator-Simulation-Cost-vs-Waiting-Time
> 🏢 SimPy-based discrete-event simulation to optimize elevator systems in commercial buildings. Python + SimPy. BDA450 Project @ Seneca Polytechnic.
---
# 🏢 ElevatorSim

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![SimPy](https://img.shields.io/badge/SimPy-4.0+-green.svg)](https://simpy.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *How many elevators does a 5-floor commercial building actually need — 1, 2, or 3?*

**ElevatorSim** is a **data-driven discrete-event simulation (DES)** built with Python and SimPy. It models real passenger traffic using **non-homogeneous Poisson processes** and evaluates elevator performance under a cooperative SCAN algorithm.

The objective: determine the optimal number of elevators by balancing **capital cost** and **service quality** (waiting time, trip time, fairness, and extreme delays).

---

## 🎯 What This Project Solves

| Question | Approach |
|----------|--------|
| How long do passengers wait? | Queue-to-board waiting time (mean, std, percentiles) |
| How long is the full trip? | Door-to-door system time |
| Are some floors underserved? | Pass-count fairness metric |
| When is demand highest? | 7 time-of-day buckets |
| What’s the cost vs. service trade-off? | Cost vs. mean waiting time analysis |

---

## 🏗️ Architecture
```
experiments_full.py (Entry point)
│
├── data_input_full.py → Traffic generation (Poisson arrivals)
├── elevator_model_full.py → SimPy DES core (SCAN algorithm)
└── analysis_full.py → Metrics + 7 visualizations

Inputs:

OnCounts.xlsx
OffCounts.xlsx

Outputs (results/):

summarytable.csv
allrecords.csv
plots (histograms, boxplots, heatmaps, etc.)
```

---

## 📁 Project Structure
```
elevatorsim/
├── experiments_full.py
├── data_input_full.py
├── elevator_model_full.py
├── analysis_full.py
├── OnCounts.xlsx
├── OffCounts.xlsx
├── requirements.txt
├── README.md
├── LICENSE
└── results/
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/yourusername/elevatorsim.git
cd elevatorsim
pip install -r requirements.txt
```

### Requirements
```
simpy>=4.0
numpy>=1.24
pandas>=2.0
openpyxl>=3.1
matplotlib>=3.7
seaborn>=0.12
```

### Running the Simulation

```bash
python experiments_full.py
```

This runs the **full factorial experiment**:
- **3 elevator counts:** 1, 2, 3
- **3 traffic scales:** 0.8×, 1.0×, 1.2×
- **20 replications per scenario**
- **Total: 180 simulation runs**

All results are saved to the `results/` folder automatically.

---

## 📊 Module Breakdown

### `data_input_full.py` — Data-Driven Passenger Generation

Reads `OnCounts.xlsx` and `OffCounts.xlsx` and generates per-passenger arrival events using a **non-homogeneous Poisson process** with piecewise-constant rates.

**Key features:**
- **Time-varying destination probabilities** — derived directly from each 15-min block's off-counts. Evening blocks naturally produce more downward trips because parking and ground-floor off-counts are higher.
- **Block-level Poisson arrivals** — within each 15-minute block, actual arrival count is drawn from `Poisson(expected × traffic_scale)`, with arrival times uniform within the block.
- **Fallback logic** — if a block has zero off-counts, the all-day aggregate is used, guaranteeing every passenger gets a valid destination.

### `elevator_model_full.py` — SimPy Discrete-Event Core

**Design choices:**
- **Cooperative SCAN algorithm** — every elevator sees every waiting passenger. No per-elevator call ownership. Any car that sweeps past a floor stops for same-direction waiters it can board.
- **Why no dispatcher?** In a small building, strict call ownership prevents free elevators from helping nearby passengers and creates long tails when an assigned elevator is delayed.

**Correctness guarantees:**
1. **Direction-flip re-check** — after reversing direction, the car immediately re-evaluates stopping criteria so turnaround-floor passengers aren't left behind.
2. **Pass-count tracking** — whenever a full car reaches a floor with same-direction waiters, each stranded passenger gets `passcount += 1` (used for fairness/starvation analysis).
3. **Idle wake-up** — all idle elevators share one `Building.wakeup` event. Any passenger arrival signals all waiting cars.

### `analysis_full.py` — Statistics & Visualization

7 time-of-day buckets covering the full 24-hour day (no passenger silently excluded):

| Bucket | Hours |
|--------|-------|
| Early | 00:00–06:00 |
| Morning | 06:00–09:00 |
| Mid-morning | 09:00–12:00 |
| Lunch | 12:00–14:00 |
| Afternoon | 14:00–17:00 |
| Evening | 17:00–20:00 |
| Night | 20:00–24:00 |

**Plot types generated:**
1. Waiting-time histogram per elevator count
2. Boxplots (waiting time & total trip time vs. elevators)
3. Floor fairness panels (waiting time + pass-count by floor)
4. Time-of-day comparison (waiting time + trip time by TOD bucket)
5. Mean waiting time timeline over the day (with rush hour annotations)
6. Floor × time-of-day heatmap (reveals which floors suffer at which times)
7. **Cost vs. service quality trade-off** (capital cost vs. mean wait, one line per traffic scale)

---

## 📈 Key Results

| Elevators | Mean Wait (s) | P95 Wait (s) | Mean Trip (s) | Capital Cost |
|-----------|---------------|--------------|---------------|--------------|
| 1 | ~182 | ~1410 | ~197 | $100,000 |
| 2 | ~28 | ~100 | ~43 | $200,000 |
| 3 | ~9 | ~30 | ~23 | $300,000 |

**The verdict:**
- **1 elevator is not enough** — some passengers waited over 23 minutes (P95 = 1410s)
- **3 elevators give the best service** — mean wait drops to just 9 seconds
- **2 elevators are the sweet spot** — ~85% improvement over 1 elevator at half the cost of 3

![Cost vs Service](results/costvsservice.png)

---

## 🔧 Customization

### Traffic Scale

```python
TRAFFICSCALES = [0.8, 1.0, 1.2]  # Lower, baseline, higher demand
```

### Elevator Parameters

```python
FLOORTRAVELTIME = 3.0    # seconds per floor
DOOROPENTIME = 2.0       # seconds
DOORCLOSETIME = 2.0      # seconds
ELEVATORCAPACITY = 8     # passengers
```

### Replications

```python
NREPS = 20  # replications per scenario (increase for tighter confidence intervals)
```

### Custom Traffic Data

Replace `OnCounts.xlsx` and `OffCounts.xlsx` with your own building's traffic counts. The format:

| Row 0 | None | time | 0 | 1 | 2 | 3 | 4 |
|-------|------|------|---|---|---|---|---|
| Row 1+ | | 0.0, 0.25 | count | count | count | count | count |

Time is in decimal hours (e.g., `8.25, 8.5` = 8:15 AM to 8:30 AM).

---

## 🎓 Course Context

**BDA450: Simulation and Modeling** — Winter 2026
**Seneca Polytechnic**

**Team:**
- Lionael Surya Dwitama
- Chaehoon Shin
- Margil Parekh
- Alif Hossain

---

## 📄 License

MIT License — feel free to fork, modify, and build on this.

---

## 🙏 Acknowledgments

Traffic count data provided as part of the BDA450 course project materials.


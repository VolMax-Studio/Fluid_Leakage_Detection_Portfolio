# Fluid Leakage Detection Portfolio

Non-intrusive leak detection and localization pipeline for water distribution networks (WDNs), inspired by the BattLeDIM (Battle of the Leakage Detection and Isolation Methods) international benchmark.

**Data source: synthetic hydraulic simulation. NOT real field data, NOT EPANET output.**

The network topology (12 nodes, 14 pipes, 6 pressure/flow sensors) is a simplified synthetic WDN built with Hazen-Williams hydraulics. The pipeline is designed to run on real SCADA pressure logs when they become available.

---

## Methods

Three complementary detection approaches:

**1. DWT Burst Transient Detection** (`src/dwt_detector.py`)  
Discrete Wavelet Transform (db4, level 3) on pressure time series. Level-1 detail coefficients localize abrupt step changes characteristic of sudden pipe bursts. Threshold: 3.5σ using MAD estimator (robust to sustained pressure shifts from active leaks).

**2. RF Baseline Predictor + Residual Alarm** (`src/leak_predictor.py`)  
Random Forest regression trained on the first 40% of data (leak-free period). Features: circular time encoding (hour/day), inlet flow, upstream pressure, 2-lag AR terms. Alarm triggers when residual < −2.5σ for ≥4 consecutive 15-min steps. Detects both burst and gradual leaks.

**3. Sensitivity Matrix Localization** (`src/leak_localizer.py`)  
Hydraulic sensitivity matrix S[i,j] = ΔP_sensor_i / Δq_leak_j [m/(L/s)], computed numerically by perturbing each node's demand. Leak localization: weighted least-squares fit between observed pressure residuals and S columns. Valid for small leaks (< ~20% of branch flow) in the linear hydraulic regime.

---

## Results (synthetic simulation)

7-day simulation, 15-min timestep, 2 injected events:
- Burst at N6, t=72h, 2.5 L/s (sudden pipe rupture)
- Gradual leak at N11, t=120h, 0.35 L/s (slow corrosion)

| Metric | Value | Notes |
|--------|-------|-------|
| Detection Precision | 1.00 | No false positives in detected window |
| Detection Recall | 0.50 | Burst detected; gradual 0.35 L/s not detected |
| F1-score | 0.67 | |
| Mean TTD (burst) | 7.5 h | Time from burst injection to first alarm |
| Localization (N6 burst) | Rank 3, hop dist. 1 | Top-1 incorrect; true node in top-3 |

**Honest limitations:**
- Gradual leak (0.35 L/s) falls below detection threshold at current noise level (±0.04 m H₂O). Would require longer observation window (CUSUM) or lower-noise sensors.
- Localization top-1 incorrect: N12 ranked above N6 because N12 is downstream of N6 and shows a stronger relative sensitivity signature. Correct node is rank 3, 1 hop away.
- Sensitivity matrix assumes linear hydraulics — accuracy degrades for large leaks (> ~1 L/s in branches with 80-100mm pipes).

---

## Quick start

```bash
pip install -r requirements.txt
python3 run_pipeline.py
```

Output: `results/leak_detection_isolation.png` and `results/performance_report.txt`

```bash
pytest tests/ -v    # 12 tests, all pass
```

---

## Usage as a library

```python
from src.network_simulator import generate_simulation, compute_sensitivity_matrix
from src.dwt_detector import scan_all_sensors
from src.leak_predictor import LeakPredictor
from src.leak_localizer import LeakLocalizer

# Generate or load pressure/flow time series
sim = generate_simulation(duration_hours=168, leak_events=[...])

# DWT burst detection
events = scan_all_sensors(sim['pressures'], sim['timestamps_h'])

# RF baseline detection
predictor = LeakPredictor()
predictor.fit(sim['timestamps_h'], sim['pressures'], sim['flows'])
alarms = predictor.predict_and_alarm(sim['timestamps_h'], sim['pressures'], sim['flows'])

# Localization
S, sensor_ids, node_ids = compute_sensitivity_matrix()
localizer = LeakLocalizer(S, sensor_ids, node_ids)
candidates = localizer.localize_over_window(sim['timestamps_h'], residuals, alarm_start_h)
```

---

## Domain connection

Same principle as NILM (Non-Intrusive Load Monitoring) applied to fluid systems: infer what is happening throughout a network from a small number of sensor points, without installing sensors at every node. The hydraulic sensitivity matrix is the fluid-domain analog of the power system impedance matrix used in grid monitoring.

Relevant to: water utility loss reduction, industrial process leak detection, district heating network monitoring.

---

## Roadmap

- [ ] Integrate real BattLeDIM L-Town dataset when available (open benchmark data)  
- [ ] CUSUM detector for gradual leak sensitivity below 0.5 L/s  
- [ ] Multiple simultaneous leaks (current assumption: one dominant leak per event)  
- [ ] Validation on open EPANET benchmark networks (Net1, Net3)

## License

MIT

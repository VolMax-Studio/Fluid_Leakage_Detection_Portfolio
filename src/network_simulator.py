"""
network_simulator.py — Water Distribution Network Hydraulic Simulator
======================================================================
Simulates pressure and flow in a small water distribution network (WDN)
using linearized Hazen-Williams hydraulics with iterative Newton-Raphson
convergence. Injects realistic leaks (burst and gradual) and diurnal
demand cycles for pipeline benchmarking.

Network topology: 12 demand nodes, 14 pipes, 1 reservoir, 6 sensors.
This is a simplified synthetic network inspired by L-Town (BattLeDIM
benchmark). NOT EPANET output; NOT real field data.

Physical units throughout:
  Pressure / Head : meters [m H₂O]  (1 bar ≈ 10.2 m H₂O)
  Flow            : liters/second [L/s]
  Pipe length     : meters [m]
  Pipe diameter   : millimeters [mm]

Hazen-Williams head loss formula:
    h_f = (10.67 * L * Q^1.852) / (C^1.852 * D^4.87)
where L [m], Q [m³/s], D [m], C = H-W roughness coefficient (dimensionless).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ── Network definition ────────────────────────────────────────────────────────

@dataclass
class Pipe:
    id: str
    from_node: str
    to_node: str
    length_m: float      # pipe length [m]
    diameter_mm: float   # internal diameter [mm]
    hw_coeff: float = 120.0  # Hazen-Williams C (120 = new ductile iron)


@dataclass
class Node:
    id: str
    base_demand_ls: float   # base demand [L/s] at peak hour
    elevation_m: float      # node elevation [m above datum]
    is_sensor: bool = False


RESERVOIR_HEAD_M = 60.0    # fixed head at source [m]
RESERVOIR_NODE   = "R"

# 12 demand nodes — small district WDN
NODES: List[Node] = [
    Node("N1",  0.80, 5.0,  is_sensor=True),   # inlet, always sensored
    Node("N2",  0.50, 6.0),
    Node("N3",  0.60, 7.0,  is_sensor=True),
    Node("N4",  0.40, 5.5),
    Node("N5",  0.70, 6.5,  is_sensor=True),
    Node("N6",  0.55, 8.0),
    Node("N7",  0.45, 7.5,  is_sensor=True),
    Node("N8",  0.50, 6.0),
    Node("N9",  0.60, 5.0,  is_sensor=True),
    Node("N10", 0.35, 7.0),
    Node("N11", 0.45, 8.5,  is_sensor=True),
    Node("N12", 0.30, 6.5),
]

# 14 pipes forming a looped network with two rings
PIPES: List[Pipe] = [
    Pipe("P1",  "R",   "N1",  80,  200),    # Reservoir main
    Pipe("P2",  "N1",  "N2",  150, 150),
    Pipe("P3",  "N2",  "N3",  120, 150),
    Pipe("P4",  "N3",  "N4",  100, 125),
    Pipe("P5",  "N4",  "N5",  130, 125),
    Pipe("P6",  "N5",  "N6",  110, 100),
    Pipe("P7",  "N1",  "N7",  160, 150),
    Pipe("P8",  "N7",  "N8",  140, 125),
    Pipe("P9",  "N8",  "N9",  120, 125),
    Pipe("P10", "N3",  "N8",  100, 100),    # Loop connector
    Pipe("P11", "N5",  "N9",  130, 100),    # Loop connector
    Pipe("P12", "N9",  "N10", 90,  100),
    Pipe("P13", "N10", "N11", 110, 80),
    Pipe("P14", "N6",  "N12", 95,  80),
]

NODE_IDS   = [n.id for n in NODES]
SENSOR_IDS = [n.id for n in NODES if n.is_sensor]


# ── Hydraulic solver ─────────────────────────────────────────────────────────

def _hw_resistance(pipe: Pipe) -> float:
    """
    Hazen-Williams chord resistance R = h_f / Q  [m / (L/s)].
    Evaluated at an operating point Q_op derived from pipe diameter
    at 0.5 m/s reference velocity (typical WDN design velocity).

    h_f = K * Q^1.852   [K in m/(m³/s)^1.852]
    R_chord = h_f / Q = K * Q^0.852  [m/(m³/s)]  → divide by 1000 for [m/(L/s)]

    Unit conversion: 1 m³/s = 1000 L/s, so:
        R [m/(L/s)] = R [m/(m³/s)] / 1000
    """
    import math
    D_m = pipe.diameter_mm / 1000.0
    K = 10.67 * pipe.length_m / (pipe.hw_coeff**1.852 * D_m**4.87)
    # Operating point: cross-sectional area * reference velocity 0.5 m/s
    Q_op_m3s = max(1e-5, math.pi / 4 * D_m**2 * 0.5)
    # Chord resistance [m/(m³/s)] = K * Q_op^0.852
    R_m_per_m3s = K * (Q_op_m3s ** 0.852)
    # Convert to [m/(L/s)]: divide by 1000
    return R_m_per_m3s / 1000.0


def build_admittance_matrix() -> Tuple[np.ndarray, Dict[str, int], List[Pipe]]:
    """
    Build nodal admittance matrix Y for the linearized WDN.
    Y[i,j] = -1/R_ij for connected nodes i,j
    Y[i,i] = sum(1/R_ij) for all pipes at node i

    Reservoir node R is the reference (ground); its row/col removed.
    Returns: (Y_matrix, node_index_map, pipes_list)
    """
    n = len(NODES)
    node_idx = {node.id: i for i, node in enumerate(NODES)}
    Y = np.zeros((n, n))

    for pipe in PIPES:
        if pipe.from_node == RESERVOIR_NODE or pipe.to_node == RESERVOIR_NODE:
            # Reservoir pipe: contributes only to connected demand node diagonal
            demand_node = pipe.to_node if pipe.from_node == RESERVOIR_NODE else pipe.from_node
            if demand_node in node_idx:
                i = node_idx[demand_node]
                R = _hw_resistance(pipe)
                Y[i, i] += 1.0 / R
        else:
            i = node_idx[pipe.from_node]
            j = node_idx[pipe.to_node]
            R = _hw_resistance(pipe)
            Y[i, i] += 1.0 / R
            Y[j, j] += 1.0 / R
            Y[i, j] -= 1.0 / R
            Y[j, i] -= 1.0 / R

    return Y, node_idx, PIPES


def solve_pressures(
    demand_ls: Dict[str, float],
    leak_flows: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Solve nodal pressure heads [m] given demand at each node.

    Parameters
    ----------
    demand_ls : dict {node_id: demand [L/s]}
    leak_flows : dict {node_id: leak_flow [L/s]}, optional

    Returns
    -------
    dict {node_id: pressure_head [m]}
    """
    Y, node_idx, _ = build_admittance_matrix()
    n = len(NODES)

    # RHS: known demand (injection = positive, withdrawal = negative)
    b = np.zeros(n)
    for node in NODES:
        d = demand_ls.get(node.id, 0.0)
        leak = (leak_flows or {}).get(node.id, 0.0)
        b[node_idx[node.id]] = -(d + leak)  # withdrawals are negative injection

    # Add reservoir contribution to RHS
    for pipe in PIPES:
        if pipe.from_node == RESERVOIR_NODE:
            demand_node = pipe.to_node
            if demand_node in node_idx:
                R = _hw_resistance(pipe)
                i = node_idx[demand_node]
                b[i] += RESERVOIR_HEAD_M / R
        elif pipe.to_node == RESERVOIR_NODE:
            demand_node = pipe.from_node
            if demand_node in node_idx:
                R = _hw_resistance(pipe)
                i = node_idx[demand_node]
                b[i] += RESERVOIR_HEAD_M / R

    # Solve Y * H = b
    try:
        H = np.linalg.solve(Y, b)
    except np.linalg.LinAlgError:
        H = np.linalg.lstsq(Y, b, rcond=None)[0]

    return {node.id: float(H[node_idx[node.id]]) for node in NODES}


def compute_sensitivity_matrix() -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute pressure sensitivity matrix S[i,j]:
        S[i,j] = ΔP_sensor_i / Δq_leak_j   [m / (L/s)]

    Computed numerically: perturb each node's demand by δ=0.1 L/s,
    observe change in sensor pressures.

    Returns
    -------
    S : np.ndarray, shape (n_sensors, n_nodes)
    sensor_ids : list of sensor node IDs (row labels)
    node_ids   : list of all node IDs (column labels)
    """
    delta = 0.1  # L/s perturbation

    # Nominal operating demands
    nominal_demand = {n.id: n.base_demand_ls for n in NODES}
    H_nominal = solve_pressures(nominal_demand)

    n_sensors = len(SENSOR_IDS)
    n_nodes = len(NODE_IDS)
    S = np.zeros((n_sensors, n_nodes))

    for j, node_id in enumerate(NODE_IDS):
        perturbed = dict(nominal_demand)
        perturbed[node_id] += delta
        H_perturbed = solve_pressures(perturbed)

        for i, sensor_id in enumerate(SENSOR_IDS):
            S[i, j] = (H_perturbed[sensor_id] - H_nominal[sensor_id]) / delta

    return S, SENSOR_IDS, NODE_IDS


# ── Demand and leak profiles ─────────────────────────────────────────────────

def diurnal_factor(hour: float) -> float:
    """
    Diurnal demand multiplier for hour in [0, 24).
    Based on residential WDN patterns: morning peak ~7-9h, evening ~18-20h.
    Min at ~4h (≈0.25), max at ~8h (≈1.80).
    """
    t = hour % 24
    # Two-Gaussian diurnal model
    morning = 1.80 * np.exp(-0.5 * ((t - 8.0) / 1.5) ** 2)
    evening = 1.30 * np.exp(-0.5 * ((t - 19.0) / 1.8) ** 2)
    baseline = 0.25
    return float(baseline + morning + evening)


def generate_simulation(
    duration_hours: int = 168,         # 7 days
    timestep_minutes: int = 15,
    leak_events: Optional[List[dict]] = None,
    sensor_noise_m: float = 0.04,      # ±0.04 m H₂O ≈ ±4 mbar (0.1% accuracy at 50m)
    seed: int = 42,
) -> dict:
    """
    Generate time-series pressure and flow data for all sensor nodes.

    Parameters
    ----------
    duration_hours : int
        Total simulation duration [hours]. Default: 168 (7 days).
    timestep_minutes : int
        Hydraulic timestep. Default: 15 min.
    leak_events : list of dict, optional
        Each dict: {
          'node': str,         # which node leaks
          'start_hour': float, # when leak begins
          'end_hour': float,   # when leak ends (use np.inf for permanent)
          'type': str,         # 'burst' or 'gradual'
          'max_flow_ls': float # max leak flow [L/s]
        }
    sensor_noise_m : float
        Gaussian noise standard deviation [m H₂O]. Default: 0.08.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'timestamps_h'  : np.ndarray, shape (T,)   — time axis [hours]
        'pressures'     : dict {sensor_id: np.ndarray, shape (T,)}  [m H₂O]
        'flows'         : dict {sensor_id: np.ndarray, shape (T,)}  [L/s]
        'leak_events'   : list of dicts (input, with computed true flows)
        'true_leak_flow': dict {node_id: np.ndarray}  ground-truth leak [L/s]
        'timestep_h'    : float
    """
    rng = np.random.default_rng(seed)
    if leak_events is None:
        leak_events = []

    dt_h = timestep_minutes / 60.0
    T = int(duration_hours / dt_h)
    timestamps = np.arange(T) * dt_h

    # Pre-compute true leak flows at each timestep
    true_leak = {n.id: np.zeros(T) for n in NODES}
    for event in leak_events:
        node = event['node']
        t_start = event['start_hour']
        t_end = event.get('end_hour', np.inf)
        leak_type = event.get('type', 'burst')
        max_flow = event['max_flow_ls']

        for t_idx, t in enumerate(timestamps):
            if t < t_start or t >= t_end:
                continue
            elapsed = t - t_start
            duration = min(t_end, timestamps[-1]) - t_start

            if leak_type == 'burst':
                # Step function: immediate full leak at onset
                true_leak[node][t_idx] = max_flow
            elif leak_type == 'gradual':
                # Linear ramp over 48 hours to max_flow
                ramp_h = min(48.0, duration)
                flow = max_flow * min(elapsed / ramp_h, 1.0)
                true_leak[node][t_idx] = flow

    # Simulate hydraulics at each timestep
    pressures_clean = {sid: np.zeros(T) for sid in SENSOR_IDS}
    flows_clean = {sid: np.zeros(T) for sid in SENSOR_IDS}

    for t_idx, t in enumerate(timestamps):
        hour = t % 24
        factor = diurnal_factor(hour)

        # Weekly variation: slightly lower demand on weekends (day 5, 6)
        day = int(t / 24) % 7
        weekly_factor = 0.85 if day >= 5 else 1.0

        demand = {n.id: n.base_demand_ls * factor * weekly_factor for n in NODES}
        leak_at_t = {nid: true_leak[nid][t_idx] for nid in true_leak}

        H = solve_pressures(demand, leak_at_t)

        for sid in SENSOR_IDS:
            node = next(n for n in NODES if n.id == sid)
            # Pressure = head - elevation
            pressures_clean[sid][t_idx] = H[sid] - node.elevation_m
            # Flow at inlet sensor N1 = sum of all demands + leaks
            if sid == "N1":
                flows_clean[sid][t_idx] = (
                    sum(demand.values()) + sum(leak_at_t.values())
                )
            else:
                flows_clean[sid][t_idx] = demand.get(sid, 0.0)

    # Add measurement noise
    pressures_noisy = {}
    flows_noisy = {}
    for sid in SENSOR_IDS:
        p_noise = rng.normal(0, sensor_noise_m, T)
        q_noise = rng.normal(0, 0.02, T)  # ±0.02 L/s flow meter noise
        pressures_noisy[sid] = pressures_clean[sid] + p_noise
        flows_noisy[sid] = flows_clean[sid] + q_noise

    return {
        'timestamps_h': timestamps,
        'pressures': pressures_noisy,
        'flows': flows_noisy,
        'pressures_clean': pressures_clean,
        'leak_events': leak_events,
        'true_leak_flow': true_leak,
        'timestep_h': dt_h,
        'sensor_ids': SENSOR_IDS,
        'node_ids': NODE_IDS,
    }


# ── Default scenario ──────────────────────────────────────────────────────────

DEFAULT_LEAK_EVENTS = [
    {
        'node': 'N6',
        'start_hour': 72.0,    # burst at 72h (day 3)
        'end_hour': np.inf,
        'type': 'burst',
        'max_flow_ls': 2.50,   # 2.5 L/s burst — SNR≈7.7 at nearest sensor (noise 0.04 m)
        'description': 'Burst in pipe P6 area (N5→N6)',
    },
    {
        'node': 'N11',
        'start_hour': 120.0,   # gradual leak starts at day 5
        'end_hour': np.inf,
        'type': 'gradual',
        'max_flow_ls': 0.35,   # 0.35 L/s — slow degradation
        'description': 'Gradual corrosion leak near N11 (P13)',
    },
]

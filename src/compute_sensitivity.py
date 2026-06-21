import os
import time
import numpy as np
import pandas as pd
import wntr

def compute_and_cache_sensitivity(inp_path: str, scada_pressures_path: str, out_path: str) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Computes the hydraulic sensitivity matrix S[sensor, junction] = dP_sensor / dq_junction [m / (L/s)]
    numerically using wntr, and caches the result.
    """
    if os.path.exists(out_path):
        print(f"Loading cached sensitivity matrix from {out_path}...")
        data = np.load(out_path, allow_pickle=True)
        return data['S'], list(data['sensor_ids']), list(data['node_ids'])

    print(f"Computing sensitivity matrix from {inp_path}...")
    t0 = time.time()
    
    # Load water network
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.duration = 0  # steady-state single-step simulation
    
    # Read sensor IDs from SCADA pressures CSV
    df_p = pd.read_csv(scada_pressures_path, sep=";", decimal=",", nrows=1)
    sensor_ids = [col.lower() for col in df_p.columns[1:]] # L-Town nodes are lowercase in wntr
    
    junctions = wn.junction_name_list
    n_sensors = len(sensor_ids)
    n_nodes = len(junctions)
    
    # Run nominal simulation
    sim = wntr.sim.EpanetSimulator(wn)
    res_nominal = sim.run_sim()
    p_nominal = res_nominal.node['pressure'].loc[0, sensor_ids].to_numpy()
    
    S = np.zeros((n_sensors, n_nodes))
    delta = 2.0  # L/s perturbation
    delta_m3s = delta / 1000.0  # convert L/s to m3/s for EPANET
    
    for j_idx, j_name in enumerate(junctions):
        node = wn.get_node(j_name)
        # Add perturbation
        node.demand_timeseries_list.append((delta_m3s, None, 'leak_perturb'))
        
        # Run simulation
        sim = wntr.sim.EpanetSimulator(wn)
        res_perturbed = sim.run_sim()
        p_perturbed = res_perturbed.node['pressure'].loc[0, sensor_ids].to_numpy()
        
        # Remove perturbation
        node.demand_timeseries_list.pop()
        
        # Calculate sensitivity: dP / dq
        dp = p_perturbed - p_nominal
        S[:, j_idx] = dp / delta
        
        if (j_idx + 1) % 100 == 0 or j_idx == n_nodes - 1:
            print(f"  Processed {j_idx + 1}/{n_nodes} junctions...")
            
    # Make sure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, S=S, sensor_ids=sensor_ids, node_ids=junctions)
    
    t1 = time.time()
    print(f"Sensitivity matrix computation completed in {t1 - t0:.2f} seconds. Saved to {out_path}")
    return S, sensor_ids, junctions

if __name__ == "__main__":
    inp = "/home/volmax-studio/volmax-projects/iot2/PORTFOLIO/Fluid_Leakage_Detection_Portfolio/battledim_data/L-TOWN.inp"
    scada = "/home/volmax-studio/volmax-projects/iot2/PORTFOLIO/Fluid_Leakage_Detection_Portfolio/battledim_data/2018_SCADA_Pressures.csv"
    out = "/home/volmax-studio/volmax-projects/iot2/PORTFOLIO/Fluid_Leakage_Detection_Portfolio/results/sensitivity_matrix_ltown.npz"
    compute_and_cache_sensitivity(inp, scada, out)

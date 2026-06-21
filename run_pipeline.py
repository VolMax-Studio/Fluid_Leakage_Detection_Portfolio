"""
run_pipeline.py — Fluid Leakage Detection Pipeline
====================================================
Generates synthetic WDN simulation, runs detection and localization,
outputs performance report and diagnostic plots.

Usage:
    python3 run_pipeline.py

Output:
    results/leak_detection_isolation.png
    results/performance_report.txt
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from src.network_simulator import (
    generate_simulation, compute_sensitivity_matrix,
    DEFAULT_LEAK_EVENTS, PIPES, NODE_IDS, SENSOR_IDS
)
from src.dwt_detector import scan_all_sensors
from src.leak_predictor import LeakPredictor
from src.leak_localizer import LeakLocalizer, build_adjacency_from_pipes

os.makedirs("results", exist_ok=True)


def run(mode="synthetic", resample="15min", detector="consecutive", cusum_drift=0.2, localizer_method="wls", lasso_alpha=0.01):
    print("=" * 62)
    print("  Fluid Leakage Detection Portfolio")
    if mode == "real":
        print(f"  Real L-Town SCADA Dataset — BattLeDIM Benchmark Pipeline ({detector.upper()} + {localizer_method.upper()})")
    else:
        print(f"  Synthetic WDN Simulation — BattLeDIM-inspired pipeline ({detector.upper()} + {localizer_method.upper()})")
    print("=" * 62)

    # ── Step 1: Ingest / Generate data ────────────────────────────
    if mode == "real":
        print(f"\n[1/5] Loading real L-Town 2018 SCADA data (resample={resample})...")
        from battledim_loader import load_battledim, to_pipeline_dict
        data = load_battledim("./battledim_data", year=2018, resample=resample)
        sim = to_pipeline_dict(data)
        leak_events = sim['leak_events']
        T = len(sim['timestamps_h'])
        sensor_ids = sim['sensor_ids']
        print(f"      {T} timesteps, {len(sensor_ids)} sensors")
        print(f"      Labelled leaks: {len(leak_events)}")
        # Print top 5 leaks for summary
        for ev in leak_events[:5]:
            print(f"        → Node {ev['node']}: {ev['type']} at t={ev['start_hour']:.0f}h "
                  f"({ev['max_flow_ls']:.2f} L/s)")
        if len(leak_events) > 5:
            print(f"        ... and {len(leak_events) - 5} more leaks.")
    else:
        print("\n[1/5] Generating 7-day WDN simulation (15-min timestep)...")
        sim = generate_simulation(
            duration_hours=168,
            timestep_minutes=15,
            leak_events=DEFAULT_LEAK_EVENTS,
            sensor_noise_m=0.08,
            seed=42,
        )
        leak_events = DEFAULT_LEAK_EVENTS
        T = len(sim['timestamps_h'])
        sensor_ids = SENSOR_IDS
        print(f"      {T} timesteps, {len(SENSOR_IDS)} sensors, {len(NODE_IDS)} nodes")
        print(f"      Injected leaks: {len(DEFAULT_LEAK_EVENTS)}")
        for ev in DEFAULT_LEAK_EVENTS:
            print(f"        → Node {ev['node']}: {ev['type']} at t={ev['start_hour']:.0f}h "
                  f"({ev['max_flow_ls']:.2f} L/s) — {ev['description']}")

    # ── Step 2: DWT burst detection ──────────────────────────────
    print("\n[2/5] DWT burst transient detection...")
    dwt_events = scan_all_sensors(
        sim['pressures'],
        sim['timestamps_h'],
        wavelet='db4',
        threshold_sigma=3.5,
    )
    # Filter out NaNs from magnitude
    dwt_events = [e for e in dwt_events if not np.isnan(e['magnitude'])]
    print(f"      Alarms triggered: {sum(1 for e in dwt_events if e['alarm'])}"
          f" / {len(dwt_events)} sensors")
    # Print top 5 sensor alarms
    for ev in dwt_events[:5]:
        status = f"ALARM at t={ev['onset_h']:.1f}h" if ev['alarm'] else "no alarm"
        print(f"        Sensor {ev['sensor']}: {status} (magnitude={ev['magnitude']:.3f})")

    # ── Step 3: RF baseline predictor ────────────────────────────
    # For L-Town real dataset, first 40% (Jan - May) is mostly leak-free by design
    print(f"\n[3/5] Training RF baseline predictor (first 40% = training)...")
    predictor = LeakPredictor(
        alarm_sigma_multiplier=2.5,
        alarm_consecutive_steps=4,
        detector_type=detector,
        cusum_drift=cusum_drift,
    )
    predictor.fit(
        sim['timestamps_h'],
        sim['pressures'],
        sim['flows'],
        training_fraction=0.40,
    )
    alarm_results = predictor.predict_and_alarm(
        sim['timestamps_h'],
        sim['pressures'],
        sim['flows'],
    )
    metrics = predictor.detection_summary(
        alarm_results, leak_events, sim['timestamps_h']
    )

    print(f"      Precision: {metrics['precision']:.2f}")
    print(f"      Recall:    {metrics['recall']:.2f}")
    print(f"      F1:        {metrics['f1']:.2f}")
    print(f"      Mean TTD:  {metrics['mean_ttd_hours']:.1f} h")
    print(f"      Total alarms: {metrics['n_alarms']}")

    # ── Step 4: Leak localization ────────────────────────────────
    print("\n[4/5] Computing sensitivity matrix + leak localization...")
    if mode == "real":
        # Load precomputed sensitivity matrix
        s_data = np.load("results/sensitivity_matrix_ltown.npz", allow_pickle=True)
        S = s_data['S']
        sensor_ids = list(s_data['sensor_ids'])
        node_ids = list(s_data['node_ids'])
        
        # Build L-Town network adjacency from wntr
        import wntr
        wn = wntr.network.WaterNetworkModel("battledim_data/L-TOWN.inp")
        adjacency = {}
        for pipe_name in wn.pipe_name_list:
            pipe = wn.get_link(pipe_name)
            f = pipe.start_node_name.lower()
            t = pipe.end_node_name.lower()
            adjacency.setdefault(f, []).append(t)
            adjacency.setdefault(t, []).append(f)
    else:
        S, sensor_ids, node_ids = compute_sensitivity_matrix()
        adjacency = build_adjacency_from_pipes(PIPES)

    localizer = LeakLocalizer(S, sensor_ids, node_ids, lasso_alpha=lasso_alpha)

    localization_results = []
    # For L-Town, we only evaluate leaks that start after the training period (first 40%)
    t_threshold = sim['timestamps_h'][int(T * 0.40)]
    eval_leaks = [ev for ev in leak_events if ev['start_hour'] >= t_threshold]
    
    print(f"      Evaluating localization on {len(eval_leaks)} post-training leak events...")
    for event in eval_leaks[:10]:
        if mode == "real":
            try:
                pipe = wn.get_link(event['node'])
                true_nodes = [pipe.start_node_name.lower(), pipe.end_node_name.lower()]
            except Exception:
                true_nodes = [event['node'].lower()]
        else:
            true_nodes = [event['node'].lower()]

        # Find first alarm time from RF predictor after the leak starts
        alarm_times = []
        for r in alarm_results.values():
            mask = r['alarm_mask']
            # Find times where alarm is active and is after leak start
            active_times = sim['timestamps_h'][mask & (sim['timestamps_h'] >= event['start_hour'])]
            active_times = active_times[active_times < event['start_hour'] + 168.0] # within 7 days
            if len(active_times) > 0:
                alarm_times.append(active_times[0])
        if not alarm_times:
            # We did not detect this leak -> cannot localize
            continue
        alarm_h = min(alarm_times)

        # Use residuals from all sensors
        sensor_residuals = {sid: alarm_results[sid]['residuals'] for sid in sensor_ids if sid in alarm_results}
        candidates = localizer.localize_over_window(
            sim['timestamps_h'], sensor_residuals,
            alarm_start_h=alarm_h, window_hours=24.0,
            method=localizer_method
        )
        acc = LeakLocalizer.localization_accuracy(candidates, true_nodes, node_ids, adjacency)
        localization_results.append({
            'event': event,
            'candidates': candidates,
            'accuracy': acc,
            'alarm_h': alarm_h,
        })
        print(f"      True leak: {event['node'].lower()} | Top-1 guess: "
              f"{candidates[0]['node'] if candidates else 'N/A'} "
              f"(rank of true: {acc['rank_of_true']}, "
              f"hop dist: {acc['hop_distance']})")

    # ── Step 5: Plot ─────────────────────────────────────────────
    print("\n[5/5] Generating diagnostic plots...")
    _plot_results(sim, alarm_results, leak_events, localization_results, mode, detector, localizer_method)
    print(f"      Saved: results/{mode}_{detector}_{localizer_method}_leak_detection_isolation.png")

    # ── Performance report ────────────────────────────────────────
    _write_report(metrics, localization_results, mode, detector, localizer_method, sensor_ids)
    print(f"      Saved: results/{mode}_{detector}_{localizer_method}_performance_report.txt")

    print("\n" + "=" * 62)
    print("  PIPELINE COMPLETE")
    print(f"  Detection Precision: {metrics['precision']:.2f}  "
          f"Recall: {metrics['recall']:.2f}  "
          f"F1: {metrics['f1']:.2f}")
    print(f"  Mean Time-to-Detection: {metrics['mean_ttd_hours']:.1f} hours")
    if localization_results:
        n_top1 = sum(1 for r in localization_results if r['accuracy']['top1_correct'])
        print(f"  Localization Top-1:   {n_top1}/{len(localization_results)} events")
    else:
        print("  Localization Top-1:   0/0 events (no leaks detected/evaluated)")
    print("=" * 62)

    return metrics, localization_results


def _plot_results(sim, alarm_results, leak_events, loc_results, mode="synthetic", detector="consecutive", localizer_method="wls"):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    t = sim['timestamps_h']
    colors = plt.cm.tab10.colors

    # Panel 1: Pressure at inlet + far sensor
    ax1 = fig.add_subplot(gs[0, :])
    plot_sensors = ["N1", "N9"] if mode == "synthetic" else ["n1", "n163"]
    for idx, sid in enumerate(plot_sensors):
        if sid not in sim['pressures']:
            continue
        ax1.plot(t, sim['pressures'][sid], lw=0.8, alpha=0.7,
                 color=colors[idx], label=f"Sensor {sid} (measured)")
        if sid in alarm_results and alarm_results[sid]['predicted'] is not None:
            ax1.plot(t, alarm_results[sid]['predicted'], '--',
                     lw=1.2, color=colors[idx], alpha=0.9, label=f"Sensor {sid} (predicted)")
    for ev in leak_events[:5]: # only label first 5 leaks to prevent overlap
        ax1.axvline(ev['start_hour'], color='red', lw=1.5, ls='--', alpha=0.8)
        ax1.text(ev['start_hour'] + 1, ax1.get_ylim()[0] + (ax1.get_ylim()[1]-ax1.get_ylim()[0])*0.1,
                 f"Leak\n{ev['node']}", fontsize=7, color='red')
    ax1.set_xlabel("Time [hours]")
    ax1.set_ylabel("Pressure [m H₂O]")
    ax1.set_title("Nodal Pressure — Measured vs. RF Baseline Prediction")
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Residuals
    ax2 = fig.add_subplot(gs[1, 0])
    residual_sensors = ["N5", "N7"] if mode == "synthetic" else ["n54", "n114"]
    for idx, sid in enumerate(residual_sensors):
        if sid not in alarm_results:
            continue
        res = alarm_results[sid]['residuals']
        ax2.plot(t, res, lw=0.8, color=colors[idx], alpha=0.8, label=f"Residual {sid}")
        thr = alarm_results[sid]['threshold']
        ax2.axhline(thr, color=colors[idx], ls=':', lw=1.2)
    for ev in leak_events[:5]:
        ax2.axvline(ev['start_hour'], color='red', lw=1.2, ls='--', alpha=0.7)
    # Mark alarms
    for sid in residual_sensors:
        if sid not in alarm_results:
            continue
        mask = alarm_results[sid]['alarm_mask']
        if mask.any():
            ax2.scatter(t[mask], alarm_results[sid]['residuals'][mask],
                        marker='v', s=20, color='red', zorder=5, alpha=0.6)
    ax2.set_xlabel("Time [hours]")
    ax2.set_ylabel("Residual [m H₂O]")
    ax2.set_title("Pressure Residuals (Measured − Predicted)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3: DWT detail coefficients
    ax3 = fig.add_subplot(gs[1, 1])
    try:
        from src.dwt_detector import dwt_detail_coefficients
        p_node = 'N3' if mode == 'synthetic' else 'n31'
        p_series = sim['pressures'].get(p_node, sim['pressures'][list(sim['pressures'].keys())[0]])
        details = dwt_detail_coefficients(p_series, wavelet='db4', level=3)
        cD1 = details[0]
        # remove NaNs for robust stats
        cD1_clean = cD1[~np.isnan(cD1)]
        t_cD1 = t[:len(cD1) * 2 : 2][:len(cD1)]
        ax3.plot(t_cD1[:len(cD1)], cD1, lw=0.8, color='steelblue', label=f'DWT cD1 ({p_node})')
        mad = np.median(np.abs(cD1_clean - np.median(cD1_clean)))
        sigma = mad / 0.6745
        ax3.axhline(3.5 * sigma, color='red', ls='--', lw=1, label='Threshold +3.5σ')
        ax3.axhline(-3.5 * sigma, color='red', ls='--', lw=1)
        for ev in leak_events[:5]:
            if ev['type'] == 'burst':
                ax3.axvline(ev['start_hour'], color='orange', lw=1.5, ls='--')
    except Exception as e:
        ax3.text(0.5, 0.5, f"DWT plot unavailable: {e}", ha='center', va='center',
                 transform=ax3.transAxes)
    ax3.set_xlabel("Time [hours]")
    ax3.set_ylabel("Detail Coefficient")
    ax3.set_title("DWT Level-1 Detail — Burst Transient Detection")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Sensitivity matrix heatmap
    ax4 = fig.add_subplot(gs[2, 0])
    if mode == "real":
        s_data = np.load("results/sensitivity_matrix_ltown.npz", allow_pickle=True)
        S = s_data['S']
        s_ids = list(s_data['sensor_ids'])
        n_ids = list(s_data['node_ids'])
    else:
        from src.network_simulator import compute_sensitivity_matrix
        S, s_ids, n_ids = compute_sensitivity_matrix()
    
    im = ax4.imshow(S, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.0)
    # Downsample ticks for real network to avoid black text block
    if mode == "real":
        ax4.set_xticks(range(0, len(n_ids), 40))
        ax4.set_xticklabels([n_ids[i] for i in range(0, len(n_ids), 40)], rotation=90, fontsize=6)
        ax4.set_yticks(range(0, len(s_ids), 3))
        ax4.set_yticklabels([s_ids[i] for i in range(0, len(s_ids), 3)], fontsize=6)
    else:
        ax4.set_xticks(range(len(n_ids)))
        ax4.set_xticklabels(n_ids, rotation=90, fontsize=7)
        ax4.set_yticks(range(len(s_ids)))
        ax4.set_yticklabels(s_ids, fontsize=8)
    ax4.set_title("Hydraulic Sensitivity Matrix S[sensor, node]")
    plt.colorbar(im, ax=ax4, fraction=0.046)

    # Panel 5: Localization bar chart
    ax5 = fig.add_subplot(gs[2, 1])
    if loc_results:
        lr = loc_results[0]  # first event
        cands = lr['candidates'][:8]
        nodes = [c['node'] for c in cands]
        scores = [c['score'] for c in cands]
        bar_colors = ['#2ecc71' if n.lower() == lr['event']['node'].lower() else '#3498db' for n in nodes]
        ax5.barh(nodes, scores, color=bar_colors)
        ax5.set_xlabel("Localization Score (R²)")
        ax5.set_title(f"Leak Localization — Event: {lr['event']['node']}")
        ax5.set_xlim([0, 1.05])
        ax5.grid(True, axis='x', alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No localization results", ha='center', va='center',
                 transform=ax5.transAxes)

    plt.suptitle(
        f"Fluid Leakage Detection Portfolio — {mode.capitalize()} WDN Evaluation ({detector.upper()} + {localizer_method.upper()})\n"
        f"(Sensors: {len(sim['pressures'])} | Data: {mode})",
        fontsize=10, y=0.98
    )
    plt.savefig(f"results/{mode}_{detector}_{localizer_method}_leak_detection_isolation.png", dpi=150, bbox_inches='tight')
    plt.close()


def _write_report(metrics, loc_results, mode="synthetic", detector="consecutive", localizer_method="wls", sensor_ids=None):
    n_sensors = len(sensor_ids) if sensor_ids is not None else 'N/A'
    lines = [
        "=" * 56,
        f"  FLUID LEAKAGE DETECTION — PERFORMANCE REPORT ({mode.upper()} — {detector.upper()} + {localizer_method.upper()})",
        "=" * 56,
        "",
        f"DATA SOURCE: {mode.capitalize()} benchmark dataset",
        f"Alarm Method: {detector.upper()}",
        f"Localization Method: {localizer_method.upper()}",
        f"Pressure Sensors: {n_sensors}",
        "",
        "DETECTION METRICS (RF Baseline Residual Method)",
        "-" * 40,
        f"  Precision:          {metrics['precision']:.3f}",
        f"  Recall:             {metrics['recall']:.3f}",
        f"  F1-score:           {metrics['f1']:.3f}",
        f"  Mean TTD (hours):   {metrics['mean_ttd_hours']:.1f} hours",
        f"  Total alarms fired: {metrics['n_alarms']}",
        f"  True positives:     {metrics['true_positives']}",
        f"  False positives:    {metrics['false_positives']}",
        f"  Ground truth events:{metrics['n_ground_truth_events']}",
        "",
        "LOCALIZATION RESULTS (Sensitivity Matrix Method)",
        "-" * 40,
    ]
    for lr in loc_results:
        acc = lr['accuracy']
        ev = lr['event']
        lines += [
            f"  Event: {ev['node']} ({ev['type']}, {ev['max_flow_ls']:.2f} L/s)",
            f"    Top-1 correct:  {acc['top1_correct']}",
            f"    Top-3 correct:  {acc['top3_correct']}",
            f"    Rank of true:   {acc['rank_of_true']}",
            f"    Hop distance:   {acc['hop_distance']}",
        ]
    lines += [
        "",
        "METHODOLOGY NOTES",
        "-" * 40,
        "  Detection: Random Forest regressor trained on first 40%",
        "  of data (assumed leak-free). Alarm = residual < -2.5σ",
        "  for ≥4 consecutive steps.",
        "",
        "  Localization: Linearized hydraulic sensitivity matrix",
        "  S[i,j] = ΔP_sensor_i / Δq_leak_j (numerical Jacobian).",
        "  Weighted LS fit ranks candidate leak nodes.",
        "=",
    ]
    with open(f"results/{mode}_{detector}_{localizer_method}_performance_report.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--resample", default="15min")
    ap.add_argument("--detector", default="consecutive", choices=["consecutive", "cusum"])
    ap.add_argument("--cusum-drift", type=float, default=0.2)
    ap.add_argument("--localizer", default="wls", choices=["wls", "lasso"])
    ap.add_argument("--lasso-alpha", type=float, default=0.01)
    args = ap.parse_args()
    run(
        mode=args.mode,
        resample=args.resample,
        detector=args.detector,
        cusum_drift=args.cusum_drift,
        localizer_method=args.localizer,
        lasso_alpha=args.lasso_alpha
    )

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


def run():
    print("=" * 62)
    print("  Fluid Leakage Detection Portfolio")
    print("  Synthetic WDN Simulation — BattLeDIM-inspired pipeline")
    print("=" * 62)

    # ── Step 1: Generate simulation ──────────────────────────────
    print("\n[1/5] Generating 7-day WDN simulation (15-min timestep)...")
    sim = generate_simulation(
        duration_hours=168,
        timestep_minutes=15,
        leak_events=DEFAULT_LEAK_EVENTS,
        sensor_noise_m=0.08,
        seed=42,
    )
    T = len(sim['timestamps_h'])
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
    print(f"      Alarms triggered: {sum(1 for e in dwt_events if e['alarm'])}"
          f" / {len(dwt_events)} sensors")
    for ev in dwt_events:
        status = f"ALARM at t={ev['onset_h']:.1f}h" if ev['alarm'] else "no alarm"
        print(f"        Sensor {ev['sensor']}: {status} (magnitude={ev['magnitude']:.3f})")

    # ── Step 3: RF baseline predictor ────────────────────────────
    print("\n[3/5] Training RF baseline predictor (first 40% = leak-free)...")
    predictor = LeakPredictor(
        alarm_sigma_multiplier=2.5,
        alarm_consecutive_steps=4,
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
        alarm_results, DEFAULT_LEAK_EVENTS, sim['timestamps_h']
    )

    print(f"      Precision: {metrics['precision']:.2f}")
    print(f"      Recall:    {metrics['recall']:.2f}")
    print(f"      F1:        {metrics['f1']:.2f}")
    print(f"      Mean TTD:  {metrics['mean_ttd_hours']:.1f} h")
    print(f"      Total alarms: {metrics['n_alarms']}")

    # ── Step 4: Leak localization ────────────────────────────────
    print("\n[4/5] Computing sensitivity matrix + leak localization...")
    S, sensor_ids, node_ids = compute_sensitivity_matrix()
    localizer = LeakLocalizer(S, sensor_ids, node_ids)
    adjacency = build_adjacency_from_pipes(PIPES)

    localization_results = []
    for event in DEFAULT_LEAK_EVENTS:
        true_node = event['node']
        # Find first alarm time from RF predictor
        alarm_times = [
            r['first_alarm_h'] for r in alarm_results.values()
            if r['first_alarm_h'] is not None
               and r['first_alarm_h'] >= event['start_hour']
               and r['first_alarm_h'] < event['start_hour'] + 48
        ]
        if not alarm_times:
            print(f"      Event at {true_node}: not detected → cannot localize")
            continue
        alarm_h = min(alarm_times)

        # Use residuals from all sensors
        sensor_residuals = {sid: alarm_results[sid]['residuals'] for sid in sensor_ids}
        candidates = localizer.localize_over_window(
            sim['timestamps_h'], sensor_residuals,
            alarm_start_h=alarm_h, window_hours=24.0
        )
        acc = LeakLocalizer.localization_accuracy(candidates, true_node, node_ids, adjacency)
        localization_results.append({
            'event': event,
            'candidates': candidates,
            'accuracy': acc,
            'alarm_h': alarm_h,
        })
        print(f"      True leak: {true_node} | Top-1 guess: "
              f"{candidates[0]['node'] if candidates else 'N/A'} "
              f"(rank of true: {acc['rank_of_true']}, "
              f"hop dist: {acc['hop_distance']})")

    # ── Step 5: Plot ─────────────────────────────────────────────
    print("\n[5/5] Generating diagnostic plots...")
    _plot_results(sim, alarm_results, DEFAULT_LEAK_EVENTS, localization_results)
    print("      Saved: results/leak_detection_isolation.png")

    # ── Performance report ────────────────────────────────────────
    _write_report(metrics, localization_results)
    print("      Saved: results/performance_report.txt")

    print("\n" + "=" * 62)
    print("  PIPELINE COMPLETE")
    print(f"  Detection Precision: {metrics['precision']:.2f}  "
          f"Recall: {metrics['recall']:.2f}  "
          f"F1: {metrics['f1']:.2f}")
    print(f"  Mean Time-to-Detection: {metrics['mean_ttd_hours']:.1f} hours")
    n_top1 = sum(1 for r in localization_results if r['accuracy']['top1_correct'])
    print(f"  Localization Top-1:   {n_top1}/{len(localization_results)} events")
    print("=" * 62)

    return metrics, localization_results


def _plot_results(sim, alarm_results, leak_events, loc_results):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    t = sim['timestamps_h']
    colors = plt.cm.tab10.colors

    # Panel 1: Pressure at N1 and N9 (inlet + far sensor)
    ax1 = fig.add_subplot(gs[0, :])
    for idx, sid in enumerate(["N1", "N9"]):
        ax1.plot(t, sim['pressures'][sid], lw=0.8, alpha=0.7,
                 color=colors[idx], label=f"Sensor {sid} (measured)")
        if sid in alarm_results:
            ax1.plot(t, alarm_results[sid]['predicted'], '--',
                     lw=1.2, color=colors[idx], alpha=0.9, label=f"Sensor {sid} (predicted)")
    for ev in leak_events:
        ax1.axvline(ev['start_hour'], color='red', lw=1.5, ls='--', alpha=0.8)
        ax1.text(ev['start_hour'] + 1, ax1.get_ylim()[0] + 1,
                 f"Leak\n{ev['node']}", fontsize=7, color='red')
    ax1.set_xlabel("Time [hours]")
    ax1.set_ylabel("Pressure [m H₂O]")
    ax1.set_title("Nodal Pressure — Measured vs. RF Baseline Prediction")
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Residuals N5
    ax2 = fig.add_subplot(gs[1, 0])
    for idx, sid in enumerate(["N5", "N7"]):
        if sid not in alarm_results:
            continue
        res = alarm_results[sid]['residuals']
        ax2.plot(t, res, lw=0.8, color=colors[idx], alpha=0.8, label=f"Residual {sid}")
        thr = alarm_results[sid]['threshold']
        ax2.axhline(thr, color=colors[idx], ls=':', lw=1.2)
    for ev in leak_events:
        ax2.axvline(ev['start_hour'], color='red', lw=1.2, ls='--', alpha=0.7)
    # Mark alarms
    for sid in ["N5", "N7"]:
        if sid not in alarm_results:
            continue
        mask = alarm_results[sid]['alarm_mask']
        ax2.scatter(t[mask], alarm_results[sid]['residuals'][mask],
                    marker='v', s=20, color='red', zorder=5, alpha=0.6)
    ax2.set_xlabel("Time [hours]")
    ax2.set_ylabel("Residual [m H₂O]")
    ax2.set_title("Pressure Residuals (Measured − Predicted)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3: DWT detail coefficients (N3)
    ax3 = fig.add_subplot(gs[1, 1])
    try:
        from src.dwt_detector import dwt_detail_coefficients
        p_n3 = sim['pressures'].get('N3', sim['pressures']['N1'])
        details = dwt_detail_coefficients(p_n3, wavelet='db4', level=3)
        cD1 = details[0]
        t_cD1 = t[:len(cD1) * 2 : 2][:len(cD1)]  # approximate time axis
        ax3.plot(t_cD1, cD1, lw=0.8, color='steelblue', label='DWT cD1 (N3)')
        mad = np.median(np.abs(cD1 - np.median(cD1)))
        sigma = mad / 0.6745
        ax3.axhline(3.5 * sigma, color='red', ls='--', lw=1, label='Threshold +3.5σ')
        ax3.axhline(-3.5 * sigma, color='red', ls='--', lw=1)
        for ev in leak_events:
            if ev['type'] == 'burst':
                ax3.axvline(ev['start_hour'], color='orange', lw=1.5, ls='--')
    except Exception:
        ax3.text(0.5, 0.5, "DWT plot unavailable", ha='center', va='center',
                 transform=ax3.transAxes)
    ax3.set_xlabel("Time [hours]")
    ax3.set_ylabel("Detail Coefficient")
    ax3.set_title("DWT Level-1 Detail — Burst Transient Detection")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Sensitivity matrix heatmap
    ax4 = fig.add_subplot(gs[2, 0])
    from src.network_simulator import compute_sensitivity_matrix
    S, s_ids, n_ids = compute_sensitivity_matrix()
    im = ax4.imshow(S, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.0)
    ax4.set_xticks(range(len(n_ids)))
    ax4.set_xticklabels(n_ids, rotation=90, fontsize=7)
    ax4.set_yticks(range(len(s_ids)))
    ax4.set_yticklabels(s_ids, fontsize=8)
    ax4.set_title("Hydraulic Sensitivity Matrix S[sensor, node]\n[m H₂O / (L/s)]")
    plt.colorbar(im, ax=ax4, fraction=0.046)

    # Panel 5: Localization bar chart
    ax5 = fig.add_subplot(gs[2, 1])
    if loc_results:
        lr = loc_results[0]  # first event
        cands = lr['candidates'][:8]
        nodes = [c['node'] for c in cands]
        scores = [c['score'] for c in cands]
        bar_colors = ['#2ecc71' if n == lr['event']['node'] else '#3498db' for n in nodes]
        ax5.barh(nodes, scores, color=bar_colors)
        ax5.set_xlabel("Localization Score (R²)")
        ax5.set_title(f"Leak Localization — Event: {lr['event']['node']} "
                      f"(green = true leak)")
        ax5.set_xlim([0, 1.05])
        ax5.grid(True, axis='x', alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No localization results", ha='center', va='center',
                 transform=ax5.transAxes)

    plt.suptitle(
        "Fluid Leakage Detection Portfolio — Synthetic WDN Simulation\n"
        "(Network topology: 12 nodes, 14 pipes, 6 sensors | Data: synthetic)",
        fontsize=10, y=0.98
    )
    plt.savefig("results/leak_detection_isolation.png", dpi=150, bbox_inches='tight')
    plt.close()


def _write_report(metrics, loc_results):
    lines = [
        "=" * 56,
        "  FLUID LEAKAGE DETECTION — PERFORMANCE REPORT",
        "=" * 56,
        "",
        "DATA SOURCE: Synthetic simulation (not real field data)",
        "Network: 12 nodes, 14 pipes, 6 pressure/flow sensors",
        "Duration: 7 days, 15-min timestep (672 timesteps)",
        "Noise: ±0.08 m H₂O Gaussian pressure sensor noise",
        "",
        "DETECTION METRICS (RF Baseline Residual Method)",
        "-" * 40,
        f"  Precision:          {metrics['precision']:.3f}",
        f"  Recall:             {metrics['recall']:.3f}",
        f"  F1-score:           {metrics['f1']:.3f}",
        f"  Mean TTD (burst):   {metrics['mean_ttd_hours']:.1f} hours",
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
        "  for ≥4 consecutive 15-min steps.",
        "",
        "  Localization: Linearized hydraulic sensitivity matrix",
        "  S[i,j] = ΔP_sensor_i / Δq_leak_j (numerical Jacobian).",
        "  Weighted LS fit ranks candidate leak nodes.",
        "",
        "  DWT detection: db4 wavelet, level-1 detail coefficients,",
        "  MAD-based 3.5σ threshold for burst onset localization.",
        "=" * 56,
    ]
    with open("results/performance_report.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()

# Release history

## v2.0.1 — 2026-11-11 — Corrective audit release

**Status: published numerical results reproduce, but validated leak detection/localization is NOT DEMONSTRATED.**

This release corrects the *public interpretation and documentation* of v2.0.0. It is **not** a new model or a repaired detection pipeline. A reliable scorer, leak-free training/evaluation design, and CUSUM state handling remain outstanding.

- Adds the full November 2026 methodological audit (`AUDIT_2026-11.md`), executable reproduction script (`scripts/audit_evaluation.py`) and frozen raw output.
- Documents that 94.51% of the original training time steps fall within labeled leak intervals.
- Documents evaluator flaws (first alarm per sensor; event TP vs sensor FP; training leaks in recall denominator), near-continuous CUSUM occupancy, and full seven-leak localization distances.
- Corrects the median-hop attribution: the 55-hop WLS and 43-hop Lasso medians were reconstructed using CUSUM-triggered windows; baseline consecutive localization was not independently re-evaluated.
- Replaces unsupported `real-world` and `production-ready` claims; adds `LICENSE`, `pandas` and `wntr` requirements, and clarifying code comments.
- Adds an audit notice without changing the original published numerical result fields, algorithms, or historical figures.

**Verification:** 12/12 existing Python tests pass; C++ builds and replays 35,038 steps (17 CUSUM alarms, 134 Haar detail alarms); the audit reproduction exactly matches the frozen textual output (64/64 lines). These tests do not validate field performance.

**Archive policy:** v2.0.0 ([10.5281/zenodo.20786241](https://doi.org/10.5281/zenodo.20786241)) remains immutable. v2.0.1 is intended as the next linked Zenodo version; its version-specific DOI must be confirmed from Zenodo and must not be invented.

## v2.0.0 — 2026-06-21 — Historical original

Original L-Town 2018 pipeline, tests and result reports. Its headline performance claims are superseded in interpretation by the November 2026 audit. Historical files and numbers are preserved for provenance.

# Zenodo: linked corrective version v2.0.1

**Not proof of Zenodo publication.** The actual Zenodo upload requires record-owner authorization and the new version DOI must be taken from Zenodo after publication.

- Historical version: https://doi.org/10.5281/zenodo.20786241 (v2.0.0)
- Existing all-versions (concept) DOI: https://doi.org/10.5281/zenodo.20786240
- Release archive to attach: `Fluid_Leakage_Detection_Portfolio_v2.0.1.zip` (available from the GitHub v2.0.1 release)

## New version metadata

Title: Fluid Leakage Detection and Sparse Localization for Water Distribution Networks — Corrective Evaluation Audit
Type: Software
Version: 2.0.1
Publication date: 2026-11-11
Creator: Ivan Nestorov
License: MIT

Description:

**CORRECTIVE AUDIT RELEASE — VALIDATED LEAK DETECTION AND LOCALIZATION NOT DEMONSTRATED.**
This v2.0.1 archive preserves the 2018 BattLeDIM L-Town RF/CUSUM/Lasso experimental pipeline while correcting the interpretation of v2.0.0 performance claims. The original reported F1 values (0.095 and 0.278) are reproducible from the original code, but the scorer uses only the first alarm per sensor and mixes event-level true positives with sensor-level false positives. The original 40%-of-2018 training period is not leak-free: 94.51% of its time steps fall in labeled leak intervals; the CUSUM alarm is nearly continuously active at network level after the training boundary. No validated field detection or localization performance is claimed.
The complete evaluation audit, independent reproducibility script, frozen output, and all seven localization errors are included. The previous v2.0.0 record (DOI 10.5281/zenodo.20786241) is historical and remains available. A new event-level evaluator, leak-free experimental design and fresh benchmark results are future work, not part of this release.

## Record-owner workflow

1. Sign in to the Zenodo account that owns v2.0.0. Open https://zenodo.org/records/20786241 and select **New version** (not New upload).
2. Replace the previous-version ZIP with the tagged v2.0.1 release ZIP. Set version, description and publication date above, verify it is linked under the same concept DOI.
3. Publish the new version and capture the **new version-specific DOI**, then put that DOI into the latest repository's citation metadata. Do not reuse the v2.0.0 version-specific DOI.
4. Optionally edit the older v2.0.0 record metadata to add a prominent historical-correction note and link to the new version, without changing original archived files.

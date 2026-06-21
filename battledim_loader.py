#!/usr/bin/env mode-python
"""
battledim_loader.py — BattLeDIM L-Town benchmark loader for the
VolMax Fluid Leakage Detection pipeline (v2).

Loads the real, public BattLeDIM 2020 dataset (Vrachimis et al., Zenodo
10.5281/zenodo.4017659): the L-Town water-distribution network with SCADA
pressure / flow / demand time series and labelled leak ground truth.

This replaces the v1 synthetic 12-node simulator as the data source, so the
pipeline's detection / localization numbers become comparable to the published
BattLeDIM leaderboard instead of testing the code against its own simulator.

Run:
    python battledim_loader.py --data-dir ./battledim_data --year 2018 --plot

Output:
    - prints a dataset summary (span, sensors, leak breakdown)
    - results/battledim_<year>_sanity.png   (pressure traces + shaded leaks)
    - a BattLeDIMData object ready for the pipeline (to_pipeline_frame)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# --------------------------------------------------------------------------- #
# Robust CSV reading  [T1]
# --------------------------------------------------------------------------- #
def _sniff_csv(path: Path, n_probe: int = 5) -> tuple[str, str]:
    """Auto-detect (delimiter, decimal) for a SCADA CSV.

    BattLeDIM CSVs ship in both US (',' sep, '.' decimal) and EU
    (';' sep, ',' decimal) flavours depending on the mirror, so we probe the
    header rather than assume.
    """
    head = path.read_text(errors="replace").splitlines()[:n_probe]
    if not head:
        raise ValueError(f"{path} is empty")
    header = head[0]
    if header.count(";") >= header.count(",") and header.count(";") > 0:
        return ";", ","          # EU flavour pairs ';' with ',' decimals
    return ",", "."              # US flavour


def load_scada_csv(path: Path, resample: str | None = None) -> pd.DataFrame:
    """Load a SCADA CSV (pressures/flows/demands/levels) -> timestamp-indexed,
    one column per sensor."""
    delim, decimal = _sniff_csv(path)
    df = pd.read_csv(path, sep=delim, decimal=decimal)
    ts_col = df.columns[0]                                  # first col = time
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.isna().all():                                    # retry EU date order
        ts = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)
    df = df.drop(columns=[ts_col])
    df.index = ts
    df = df[df.index.notna()].sort_index()
    df.index.name = "timestamp"
    df = df.apply(pd.to_numeric, errors="coerce")          # stray strings -> NaN
    if resample:
        df = df.resample(resample).mean()
    print(f"  loaded {path.name}: {df.shape[0]} rows x {df.shape[1]} sensors "
          f"[{df.index.min()} -> {df.index.max()}]  (sep='{delim}', dec='{decimal}')")
    return df


# --------------------------------------------------------------------------- #
# Leak ground truth  [T2]
# --------------------------------------------------------------------------- #
# canonical name -> possible header spellings across BattLeDIM mirrors
ALIASES = {
    "leak_id":   ["leakid", "id", "leak", "leaknr", "leakno"],
    "link_id":   ["linkid", "pipeid", "pipe", "link", "leakpipe", "location", "node"],
    "start":     ["starttime", "start", "leakstart", "tstart", "leakstarttime"],
    "end":       ["endtime", "end", "leakend", "tend", "leakendtime",
                  "repairtime", "leakrepairtime"],
    "type":      ["leaktype", "type", "kind"],
    "peak_flow": ["peakflow", "leakflow", "flow", "maxflow", "leakpeak",
                  "peak", "leakdiameter", "diameter"],
}


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def load_leaks(path: Path) -> pd.DataFrame:
    """Load + normalise the leak ground-truth table.

    Supports both:
    1. A metadata table format (columns: link_id, start, end, etc.)
    2. A SCADA time-series format (columns: Timestamp, p31, p158, ... with flow rates)
    """
    delim, decimal = _sniff_csv(path)
    raw = pd.read_csv(path, sep=delim, decimal=decimal)

    # Check if this is a time-series flow file (e.g. 2018_Leakages.csv)
    first_col = raw.columns[0]
    if len(raw.columns) > 5 and any(raw.columns[1].startswith(prefix) for prefix in ('p', 'P', 'link', 'pipe', 'leak')):
        print(f"  detected time-series leak format in {path.name}. Extracting events...")
        ts = pd.to_datetime(raw[first_col], errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(raw[first_col], errors="coerce", dayfirst=True)
        raw = raw.drop(columns=[first_col])
        raw.index = ts
        raw = raw[raw.index.notna()].sort_index()
        raw = raw.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        leaks_list = []
        for col in raw.columns:
            series = raw[col]
            active = series > 1e-4
            if not active.any():
                continue
            diff = active.astype(int).diff()
            starts = list(raw.index[diff == 1])
            if active.iloc[0]:
                starts.insert(0, raw.index[0])
            ends = list(raw.index[diff == -1])
            if active.iloc[-1]:
                ends.append(raw.index[-1])

            for s, e in zip(starts, ends):
                segment = series.loc[s:e]
                max_val = float(segment.max())
                # Classify type based on time-to-peak
                t_max = segment.idxmax()
                duration_to_peak_h = (t_max - s).total_seconds() / 3600.0
                leak_type = 'burst' if duration_to_peak_h < 24.0 else 'gradual'
                leaks_list.append({
                    'leak_id': f"{col}_{s.strftime('%m%d')}",
                    'link_id': col,
                    'start': s,
                    'end': e,
                    'type': leak_type,
                    'peak_flow': max_val
                })
        return pd.DataFrame(leaks_list)

    # Otherwise fallback to metadata table
    norm_map = {_norm(c): c for c in raw.columns}
    out: dict[str, pd.Series] = {}
    for canon, options in ALIASES.items():
        hit = next((norm_map[o] for o in options if o in norm_map), None)
        if hit is not None:
            out[canon] = raw[hit]
    leaks = pd.DataFrame(out)
    missing = [k for k in ("link_id", "start", "end") if k not in leaks.columns]
    if missing:
        print(f"  [T2] WARNING: could not map {missing} in {path.name}")
        print(f"       actual header: {list(raw.columns)}")
        print( "       -> add the real spelling to ALIASES and re-run.")
    for tcol in ("start", "end"):
        if tcol in leaks.columns:
            parsed = pd.to_datetime(leaks[tcol], errors="coerce")
            if parsed.isna().all():
                parsed = pd.to_datetime(leaks[tcol], errors="coerce", dayfirst=True)
            leaks[tcol] = parsed
    if "type" not in leaks.columns:
        leaks["type"] = "unknown"
    print(f"  loaded {path.name}: {len(leaks)} leak events")
    return leaks


def build_leak_labels(index: pd.DatetimeIndex, leaks: pd.DataFrame) -> pd.Series:
    """Per-timestamp boolean ground truth (True where >=1 leak active).
    This is what detection precision / recall / TTD get scored against."""
    labels = pd.Series(False, index=index, name="leak_active")
    if not {"start", "end"}.issubset(leaks.columns):
        return labels
    for _, lk in leaks.iterrows():
        s, e = lk.get("start"), lk.get("end")
        if pd.isna(s):
            continue
        e = index.max() if pd.isna(e) else e
        labels.loc[(index >= s) & (index <= e)] = True
    return labels


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass
class BattLeDIMData:
    year: int
    pressures: pd.DataFrame
    flows: pd.DataFrame | None
    demands: pd.DataFrame | None
    leaks: pd.DataFrame
    leak_labels: pd.Series

    @property
    def sensors(self) -> list[str]:
        return list(self.pressures.columns)

    def summary(self) -> None:
        p = self.pressures
        print(f"\n=== BattLeDIM L-Town {self.year} ===")
        print(f"  span      : {p.index.min()} -> {p.index.max()}")
        step = (p.index[1] - p.index[0]) if len(p.index) > 1 else None
        print(f"  step      : {step}   rows: {len(p)}")
        print(f"  pressure  : {p.shape[1]} sensors")
        print(f"  flows     : {0 if self.flows is None else self.flows.shape[1]} sensors")
        print(f"  leaks     : {len(self.leaks)} events")
        if "type" in self.leaks.columns and len(self.leaks):
            for t, n in self.leaks['type'].value_counts().items():
                print(f"            : {n}x {t}")
        print(f"  leak-active timestamps: {100.0 * self.leak_labels.mean():.1f}% of record")


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def _find(data_dir: Path, year: int, kind: str) -> Path | None:
    for pat in (f"*{year}*{kind.capitalize()}*.csv",
                f"*{year}*{kind}*.csv",
                f"*{kind.capitalize()}*{year}*.csv"):
        hits = sorted(data_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _find_leaks(data_dir: Path, year: int) -> Path | None:
    for pat in (f"*{year}*eak*.csv", "*eakage*.csv", "*eaks*.csv"):
        hits = sorted(data_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def load_battledim(data_dir: str | Path, year: int = 2018,
                   resample: str | None = None) -> BattLeDIMData:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        sys.exit(f"data dir not found: {data_dir}\n"
                 "Download BattLeDIM from Zenodo 10.5281/zenodo.4017659 and "
                 "unzip the CSVs there.")
    print(f"[load] BattLeDIM L-Town {year} from {data_dir}")
    p_path = _find(data_dir, year, "pressures")
    if p_path is None:
        avail = "\n  ".join(f.name for f in sorted(data_dir.glob('*.csv'))) or "(none)"
        sys.exit(f"no {year} pressure CSV found. CSVs present:\n  {avail}")
    pressures = load_scada_csv(p_path, resample=resample)

    f_path = _find(data_dir, year, "flows")
    d_path = _find(data_dir, year, "demands")
    flows = load_scada_csv(f_path, resample=resample) if f_path else None
    demands = load_scada_csv(d_path, resample=resample) if d_path else None

    l_path = _find_leaks(data_dir, year)
    if l_path is None:
        print("  [T2] no leak ground-truth CSV found -> labels all-False.")
        leaks = pd.DataFrame(columns=["link_id", "start", "end", "type", "peak_flow"])
    else:
        leaks = load_leaks(l_path)

    labels = build_leak_labels(pressures.index, leaks)
    return BattLeDIMData(year, pressures, flows, demands, leaks, labels)


# --------------------------------------------------------------------------- #
# Sanity plot
# --------------------------------------------------------------------------- #
def sanity_plots(data: BattLeDIMData, out: Path, n_sensors: int = 4) -> None:
    if not _HAS_MPL:
        print("  matplotlib not available -> skipping plot")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    p = data.pressures
    cols = p.var(numeric_only=True).sort_values(ascending=False).head(n_sensors).index
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 2.2 * len(cols)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        ax.plot(p.index, p[c], lw=0.5, color="#1f77b4")
        ax.set_ylabel(f"{c}\n[m]", fontsize=8)
        for _, lk in data.leaks.iterrows():
            s, e = lk.get("start"), lk.get("end")
            if pd.isna(s):
                continue
            e = p.index.max() if pd.isna(e) else e
            ax.axvspan(s, e, color="red", alpha=0.12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[0].set_title(f"BattLeDIM L-Town {data.year} — pressure traces "
                      "(red = labelled leak windows)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# [T3] adapter to the v1 pipeline
# --------------------------------------------------------------------------- #
def to_pipeline_frame(data: BattLeDIMData) -> pd.DataFrame:
    """Flatten into the wide frame the v1 detectors consume: pressure columns
    prefixed 'P_<sensor>', flow columns 'Q_<sensor>', plus the 'leak_active'
    label for scoring."""
    frame = data.pressures.add_prefix("P_")
    if data.flows is not None:
        frame = frame.join(data.flows.add_prefix("Q_"), how="left")
    frame["leak_active"] = data.leak_labels
    return frame


def to_pipeline_dict(data: BattLeDIMData) -> dict:
    """Convert L-Town data to the dictionary-of-numpy-arrays format
    expected by the v1 detector and predictor classes."""
    t0 = data.pressures.index[0]
    timestamps = (data.pressures.index - t0).total_seconds() / 3600.0
    
    pressures_dict = {col: data.pressures[col].to_numpy() for col in data.pressures.columns}
    flows_dict = {}
    if data.flows is not None:
        flows_dict = {col: data.flows[col].to_numpy() for col in data.flows.columns}
        
    leak_events = []
    for _, row in data.leaks.iterrows():
        start_hour = (row['start'] - t0).total_seconds() / 3600.0
        end_hour = (row['end'] - t0).total_seconds() / 3600.0 if not pd.isna(row['end']) else np.inf
        leak_events.append({
            'node': row['link_id'],
            'start_hour': start_hour,
            'end_hour': end_hour,
            'type': row['type'],
            'max_flow_ls': row['peak_flow'] if not pd.isna(row['peak_flow']) else 0.0,
            'description': f"Leak at {row['link_id']} ({row['type']})"
        })
        
    dt_h = (data.pressures.index[1] - data.pressures.index[0]).total_seconds() / 3600.0
    
    return {
        'timestamps_h': timestamps.to_numpy(),
        'pressures': pressures_dict,
        'flows': flows_dict,
        'leak_events': leak_events,
        'timestep_h': dt_h,
        'sensor_ids': list(data.pressures.columns),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Load BattLeDIM L-Town for the v2 leak pipeline")
    ap.add_argument("--data-dir", default="./battledim_data")
    ap.add_argument("--year", type=int, default=2018, choices=[2018, 2019])
    ap.add_argument("--resample", default=None,
                    help="pandas offset to resample to, e.g. 15min")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="results/battledim_sanity.png")
    args = ap.parse_args()

    data = load_battledim(args.data_dir, args.year, resample=args.resample)
    data.summary()
    if args.plot:
        out = Path(args.out)
        if out.name == "battledim_sanity.png":
            out = out.with_name(f"battledim_{args.year}_sanity.png")
        sanity_plots(data, out)
    _ = to_pipeline_frame(data)
    print("\n[ok] data ready. feed to_pipeline_frame(data) or to_pipeline_dict(data) into the detectors.")


if __name__ == "__main__":
    main()

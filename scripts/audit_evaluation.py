"""
audit_evaluation.py — read-only audit of the published L-Town 2018 evaluation.

Reproduces the published detection numbers with the repository's own code,
then decomposes them: training-window leak contamination, how the evaluator
arrives at TP/FP, alarm duty cycle, localization inputs, and an exploratory
2018 -> 2019 run. Writes nothing to disk; prints to stdout.

Usage (BattLeDIM CSVs + L-TOWN.inp must be in ./battledim_data):
    python scripts/audit_evaluation.py .  > audit/audit_output.txt

Sections D and F are post-hoc / exploratory and carry no verdict weight.
"""
import sys, os, io, contextlib, json
sys.dont_write_bytecode = True
REPO = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from battledim_loader import load_battledim, to_pipeline_dict
from src.leak_predictor import LeakPredictor
from src.leak_localizer import LeakLocalizer, _bfs_hop_distance

DD = os.path.join(REPO, "battledim_data")
def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)

print("== A. TRAINING CONTAMINATION ==")
d5 = quiet(load_battledim, DD, 2018, None)
n = len(d5.pressures); nt = int(n * 0.40)
lab = d5.leak_labels.to_numpy()
print("5min rows", n, "train rows", nt, "boundary", d5.pressures.index[nt])
print("leak-active in train: %d / %d = %.2f%%" % (lab[:nt].sum(), nt, 100 * lab[:nt].mean()))
print("leak-active in rest : %.2f%%" % (100 * lab[nt:].mean()))
lk = d5.leaks.sort_values("start")
bnd = d5.pressures.index[nt]
for _, r in lk.iterrows():
    print("  %-5s %s -> %s  loader_type=%-7s peak=%.2f  %s" % (r.link_id, r.start, r.end, r.type, r.peak_flow,
          "PRE-BOUNDARY" if r.start < bnd else "post"))
raw = pd.read_csv(os.path.join(DD, "2018_Leakages.csv"), sep=None, engine="python", decimal=",")
tot = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").sum(axis=1).to_numpy()
print("total leak flow (csv units): mean train %.2f | mean rest %.2f | max %.2f" % (tot[:nt].mean(), tot[nt:].mean(), tot.max()))
fl = d5.flows.sum(axis=1).to_numpy()
print("sum of 3 flow sensors (csv units): mean %.2f  -> leak share of inflow in train %.1f%%" % (fl.mean(), 100 * tot[:nt].mean() / fl[:nt].mean()))
# Erratum: PUMP_1 (n54 -> T1) is an internal transfer; true inflow is p227 + p235.
src = (d5.flows["p227"] + d5.flows["p235"]).to_numpy()
print("true inflow p227+p235 (csv units): mean %.2f  -> leak share of inflow in train %.1f%% | whole year %.1f%%" % (src.mean(), 100 * tot[:nt].mean() / src[:nt].mean(), 100 * tot.mean() / src.mean()))

print("\n== B. REPRODUCE PUBLISHED METRICS (15min) ==")
d = quiet(load_battledim, DD, 2018, "15min")
sim = to_pipeline_dict(d)
t = sim["timestamps_h"]; T = len(t); tb = t[int(T * 0.40)]
ev = sim["leak_events"]
res = {}
for det in ("consecutive", "cusum"):
    p = LeakPredictor(alarm_sigma_multiplier=2.5, alarm_consecutive_steps=4, detector_type=det, cusum_drift=0.5)
    p.fit(t, sim["pressures"], sim["flows"], training_fraction=0.40)
    ar = p.predict_and_alarm(t, sim["pressures"], sim["flows"])
    m = p.detection_summary(ar, ev, t)
    res[det] = (p, ar)
    print(det, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})

print("\n== C. DECOMPOSE CUSUM SCORE ==")
p, ar = res["cusum"]
starts = {e["node"]: e["start_hour"] for e in ev}
print("train boundary h=%.2f ; events with start+48h < boundary (undetectable by construction under CUSUM): %s" % (
    tb, [k for k, v in starts.items() if v + 48 < tb]))
per_event = {}
nomatch = []
for sid, r in ar.items():
    a = r["first_alarm_h"]
    if a is None: continue
    hit = [k for k, v in starts.items() if v <= a <= v + 48]
    if hit: per_event.setdefault(hit[0], []).append((sid, round(a - starts[hit[0]], 1)))
    else: nomatch.append((sid, round(a, 1)))
    if len(hit) > 1: print("  ambiguous match", sid, hit)
for k, v in per_event.items(): print("  event", k, "matched by", len(v), "sensors; delays h:", sorted(x[1] for x in v))
print("  unmatched first alarms:", len(nomatch), sorted(x[1] for x in nomatch))
print("  first alarms inside training period:", sum(1 for r in ar.values() if r["first_alarm_h"] is not None and r["first_alarm_h"] < tb))
post = t >= tb
duty = {sid: float(r["alarm_mask"][post].mean()) for sid, r in ar.items()}
edges = {sid: int((np.diff(r["alarm_mask"].astype(int)) == 1).sum()) for sid, r in ar.items()}
dv = np.array(list(duty.values()))
print("  per-sensor alarm duty cycle after training: median %.1f%% max %.1f%% ; sensors >50%%: %d" % (100 * np.median(dv), 100 * dv.max(), (dv > 0.5).sum()))
print("  per-sensor rising edges (alarm episodes) whole year: total %d, median %d, max %d" % (sum(edges.values()), np.median(list(edges.values())), max(edges.values())))
union = np.zeros(T, bool)
for r in ar.values(): union |= r["alarm_mask"]
ue = np.where(np.diff(union.astype(int)) == 1)[0] + 1
ue_t = t[ue]
print("  network-level (any sensor) alarm on %.1f%% of post-training time; network episodes: %d" % (100 * union[post].mean(), len(ue_t)))

print("\n== D. EPISODE-LEVEL RESCORE (exploratory, post-hoc) ==")
def rescore(edge_times, events, t0, t1, win=48.0):
    evs = [e for e in events if e["start_hour"] >= t0]
    det = [e["node"] for e in evs if ((edge_times >= e["start_hour"]) & (edge_times <= e["start_hour"] + win)).any()]
    fp = sum(1 for a in edge_times if a >= t0 and not any(e["start_hour"] <= a <= e["start_hour"] + win for e in events))
    tp_ep = sum(1 for a in edge_times if a >= t0 and any(e["start_hour"] <= a <= e["start_hour"] + win for e in events))
    grid = np.arange(t0, t1 - win, 0.25)
    es = np.sort(edge_times)
    lo = np.searchsorted(es, grid, "left"); hi = np.searchsorted(es, grid + win, "right")
    chance = float((hi > lo).mean())
    return dict(events=len(evs), detected=det, fp_episodes=fp, tp_episodes=tp_ep, chance_hit_per_48h_window=round(chance, 3))
print("  network episodes:", rescore(ue_t, ev, tb, t[-1]))
alle = np.sort(np.concatenate([t[np.where(np.diff(r["alarm_mask"].astype(int)) == 1)[0] + 1] for r in ar.values()]))
print("  all per-sensor episodes:", rescore(alle, ev, tb, t[-1]))
fa = np.array([r["first_alarm_h"] for r in ar.values() if r["first_alarm_h"] is not None])
print("  first-alarm-per-sensor (as published):", rescore(fa, ev, tb, t[-1]))

print("\n== E. LOCALIZATION ==")
adjp, adjall = {}, {}
sec = None
for line in open(os.path.join(DD, "L-TOWN.inp"), errors="replace"):
    s = line.strip()
    if s.startswith("["): sec = s.strip("[]").upper(); continue
    if not s or s.startswith(";"): continue
    if sec in ("PIPES", "VALVES", "PUMPS"):
        f = s.split()
        a, b = f[1].lower(), f[2].lower()
        adjall.setdefault(a, []).append(b); adjall.setdefault(b, []).append(a)
        if sec == "PIPES":
            adjp.setdefault(a, []).append(b); adjp.setdefault(b, []).append(a)
    if sec == "PIPES": pass
pipes = {}
sec = None
for line in open(os.path.join(DD, "L-TOWN.inp"), errors="replace"):
    s = line.strip()
    if s.startswith("["): sec = s.strip("[]").upper(); continue
    if sec == "PIPES" and s and not s.startswith(";"):
        f = s.split(); pipes[f[0].lower()] = (f[1].lower(), f[2].lower())
def comp(adj):
    c = {}; i = 0
    for n0 in adj:
        if n0 in c: continue
        st = [n0]; c[n0] = i
        while st:
            x = st.pop()
            for y in adj[x]:
                if y not in c: c[y] = i; st.append(y)
        i += 1
    return c, i
cp, ncp = comp(adjp)
print("pipe-only graph components:", ncp, "sizes", sorted(pd.Series(cp).value_counts().tolist(), reverse=True)[:6])
sd = np.load(os.path.join(REPO, "results", "sensitivity_matrix_ltown.npz"), allow_pickle=True)
S, sids, nids = sd["S"], list(sd["sensor_ids"]), list(sd["node_ids"])
for meth, alpha in (("wls", 0.01), ("lasso", 1e-4)):
    loc = LeakLocalizer(S, sids, nids, lasso_alpha=alpha)
    for e in [x for x in ev if x["start_hour"] >= tb]:
        tn = list(pipes[e["node"].lower()])
        at = []
        for r in ar.values():
            a = t[r["alarm_mask"] & (t >= e["start_hour"])]; a = a[a < e["start_hour"] + 168]
            if len(a): at.append(a[0])
        if not at: print(" ", meth, e["node"], "no alarm within 7d"); continue
        ah = min(at)
        cands = loc.localize_over_window(t, {s_: ar[s_]["residuals"] for s_ in sids if s_ in ar}, ah, 24.0, method=meth)
        g = cands[0]["node"].lower()
        hp = min([x for x in (_bfs_hop_distance(g, n_, adjp) for n_ in tn) if x is not None], default=None)
        ha = min([x for x in (_bfs_hop_distance(g, n_, adjall) for n_ in tn) if x is not None], default=None)
        already = bool(any(r["alarm_mask"][np.searchsorted(t, e["start_hour"]) - 1] for r in ar.values()))
        print("  %-5s %-5s alarm_h-start=%6.2fh alarm_already_on_before_leak=%s guess=%s hop(pipes)=%s hop(pipes+valves+pumps)=%s same_pipe_component=%s" % (
            meth, e["node"], ah - e["start_hour"], already, g, hp, ha, cp.get(g) == cp.get(tn[0])))

print("\n== F. OFFICIAL SETUP: fit 2018 (100%), score 2019 (exploratory) ==")
d19 = quiet(load_battledim, DD, 2019, "15min")
s19 = to_pipeline_dict(d19)
p2 = LeakPredictor(detector_type="cusum", cusum_drift=0.5)
p2.fit(t, sim["pressures"], sim["flows"], training_fraction=1.0)
t19 = s19["timestamps_h"] + 8760.0
ev19 = [dict(e, start_hour=e["start_hour"] + 8760.0) for e in s19["leak_events"]]
a19 = p2.predict_and_alarm(t19, s19["pressures"], s19["flows"])
m19 = p2.detection_summary(a19, ev19, t19)
print("2019 leaks:", len(ev19), "| published-style evaluator:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m19.items()})
u19 = np.zeros(len(t19), bool)
for r in a19.values(): u19 |= r["alarm_mask"]
e19 = t19[np.where(np.diff(u19.astype(int)) == 1)[0] + 1]
print("2019 network alarm on %.1f%% of year; episodes %d; rescore:" % (100 * u19.mean(), len(e19)), rescore(e19, ev19, t19[0], t19[-1]))
print("DONE")

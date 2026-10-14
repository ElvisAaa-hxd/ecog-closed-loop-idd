# -*- coding: utf-8 -*-
"""Offline calibration harness for the threshold-adjustable closed-loop algorithm.

Replicates Stream_the_data-Version2_FFT_python_judge.py logic exactly
(same sample order, same window/lock semantics), so chosen constants can be
written back 1:1. Ground truth is extracted from figure_closed-loop processing.xlsx.

Usage:
  python offline_calibration.py --filter    # Step 1: bandpass candidates
  python offline_calibration.py --grid      # Step 2+3: SEIZ_D x THRESH_LOCK_S
  python offline_calibration.py --fit       # Step 4: SEIZ_R least-squares fit
  python offline_calibration.py --final     # Step 5: validate final constants
"""
import argparse
import os
import zipfile
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
CSV = os.path.join(ROOT, "data", "EPILEPSY.csv")
XLSX = os.path.join(ROOT, "data", "figure_closed-loop processing.xlsx")

# ---------------- data ----------------

def load_csv():
    with open(CSV, "r") as f:
        lines = f.read().strip().split("\n")[3:]
    t = np.array([float(l.split(",")[0]) for l in lines])
    x = np.array([float(l.split(",")[1]) for l in lines])
    return t, x


def load_ground_truth():
    z = zipfile.ZipFile(XLSX)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    T, C, D = [], [], []
    for r in sheet.findall(".//" + ns + "row"):
        if int(r.get("r")) <= 43:
            continue
        cells = {
            c.get("r")[0]: (c.find(ns + "v").text if c.find(ns + "v") is not None else "0")
            for c in r.findall(ns + "c")
        }
        if "A" not in cells:
            continue
        T.append(float(cells["A"]))
        C.append(float(cells.get("C", "0")))
        D.append(float(cells.get("D", "0")))
    T = np.array(T)
    ther = _edges(T, np.array(D))          # therapy windows [(start, end)]
    puls = _edges(T, np.array(C), edges_only=True)  # detection pulses [t]
    return ther, puls


def _edges(T, col, edges_only=False):
    out = []
    for i in range(1, len(T)):
        if col[i] > 50 and col[i - 1] <= 50:
            out.append([T[i], None])
        if col[i] <= 50 and col[i - 1] > 50 and out and out[-1][1] is None:
            out[-1][1] = T[i]
    if edges_only:
        return [o[0] for o in out]
    return [(a, b if b is not None else T[-1]) for a, b in out]


# ---------------- signal chain ----------------

def bandpass(x, fs, f0, Q):
    w0 = 2.0 * np.pi * f0 / fs
    c0, s0 = np.cos(w0), np.sin(w0)
    alpha = s0 / (2.0 * Q)
    a0 = 1.0 + alpha
    b0, b2 = alpha / a0, -alpha / a0
    a1, a2 = (-2.0 * c0) / a0, (1.0 - alpha) / a0
    y = np.empty(len(x))
    z1 = z2 = 0.0
    for i, v in enumerate(x):
        yy = b0 * v + z1
        z1 = -a1 * yy + z2
        z2 = b2 * v - a2 * yy
        y[i] = yy
    return y


def simulate(t, x, fs, D=3.0, T_need=3, R=0.01, SHORT=20, LONG=1000,
             LOCK=10.0, F0=21.0, Q=2.0):
    """Exact replica of Stream_the_data-Version2_FFT_python_judge.py."""
    y = bandpass(x, fs, F0, Q)
    sb = deque(maxlen=SHORT)
    lb = deque(maxlen=LONG)
    wstart, count = 0.0, 0
    locked, lock_until, frozen = False, 0.0, 0.0
    cure, cure_until = 0, 0.0
    windows, pulses = [], []
    first_cross, frozen_val = None, None
    S_tr = np.full(len(x), np.nan)
    th_tr = np.full(len(x), np.nan)
    for i in range(len(x)):
        sb.append(y[i]); lb.append(y[i])
        if t[i] - wstart >= 1.0:
            count = 0; wstart = t[i]
        if len(sb) == SHORT and len(lb) == LONG:
            if locked and t[i] >= lock_until:
                locked = False
            th_live = D * float(np.var(lb))
            th = frozen if locked else th_live
            S = float(np.var(sb))
            S_tr[i] = S; th_tr[i] = th
            if S > th:
                if first_cross is None:
                    first_cross = t[i]
                if not locked:
                    frozen = th_live; locked = True; lock_until = t[i] + LOCK
                    if frozen_val is None:
                        frozen_val = th_live
                if cure == 0:
                    count += 1
                    if count >= T_need:
                        dur = R * S * SHORT
                        dur = 0.1 if dur < 0.1 else (10.0 if dur > 10.0 else dur)
                        cure = 1; cure_until = t[i] + dur; count = 0
                        windows.append([t[i], None, dur, S])
                        pulses.append(t[i])
        if cure == 1 and t[i] >= cure_until:
            cure = 0; count = 0
            if windows and windows[-1][1] is None:
                windows[-1][1] = t[i]
    if windows and windows[-1][1] is None:
        windows[-1][1] = t[-1]
    return windows, pulses, first_cross, frozen_val, S_tr, th_tr


def simulate_old_fixed(t, x, thresh, win_s=1.0, need=3, W=20):
    """stream-v2 pseudo-variance fixed-threshold logic (reference only)."""
    buf = deque(maxlen=W)
    wstart, count = t[0], 0
    dets = []
    for i in range(len(x)):
        buf.append(x[i])
        if len(buf) == W:
            b = np.asarray(buf)
            var = float((b * b).mean() - (np.abs(b).mean()) ** 2)
            if var > thresh:
                count += 1
            if t[i] - wstart >= win_s:
                if count >= need:
                    dets.append(t[i])
                wstart = t[i]; count = 0
    return dets


# ---------------- scoring ----------------

def score(windows, gt, t_end=6.5, t_max=43.2):
    n = len(windows)
    onsets = [w[0] for w in windows]
    errs, matched = [], []
    used = set()
    for g0, g1 in gt:
        best, bi = None, None
        for j, o in enumerate(onsets):
            if j in used:
                continue
            e = abs(o - g0)
            if best is None or e < best:
                best, bi = e, j
        if bi is not None and best < 0.5:
            used.add(bi); errs.append(best); matched.append((gt.index((g0, g1)), bi))
    false_base = sum(1 for o in onsets if o < t_end)
    false_late = n - len(used) - false_base
    return dict(n=n, matched=len(used), onset_err=errs,
                max_onset_err=max(errs) if errs else None,
                false_base=false_base, false_other=false_late)


def fmt_windows(windows):
    return ["(%.3f, dur %.3f, S %.3f)" % (w[0], w[2], w[3]) for w in windows]


# ---------------- steps ----------------

def step_filter(t, x, fs, gt):
    print("=== Step 1: bandpass candidates (beta 12.5-21 Hz target) ===")
    print("GT therapy starts:", ["%.3f" % g[0] for g in gt])
    for f0, q in [(21.0, 2.0), (16.75, 2.0), (16.75, 1.5), (16.2, 2.0), (17.0, 1.8)]:
        y = bandpass(x, fs, f0, q)
        sb = deque(maxlen=20)
        S = np.full(len(x), np.nan)
        for i, v in enumerate(y):
            sb.append(v)
            if len(sb) == 20:
                S[i] = float(np.var(sb))
        base = np.nanmax(S[t < 6.5])
        row = []
        for g0, g1 in gt:
            i = np.argmin(np.abs(t - g0))
            row.append(S[i])
        late1 = np.nanmax(S[(t >= 21.0) & (t <= 21.6)])
        late2 = np.nanmax(S[(t >= 26.0) & (t <= 26.7)])
        print("f0=%.2f Q=%.1f | band~%.1f-%.1f Hz | baseSmax=%.3f | S@starts=%s | late21=%.3f late26=%.3f"
              % (f0, q, f0 - f0 / q / 2, f0 + f0 / q / 2, base,
                 " ".join("%.2f" % v for v in row), late1, late2))


def step_grid(t, x, fs, gt, f0, q):
    print("=== Step 2+3: SEIZ_D x THRESH_LOCK_S grid (F0=%.2f, Q=%.1f) ===" % (f0, q))
    for lock in [10.0, 20.0, 25.0]:
        for D in [4.0, 5.0, 6.0, 7.0, 8.0]:
            w, p, fc, fv, _, _ = simulate(t, x, fs, D=D, LOCK=lock, F0=f0, Q=q)
            s = score(w, gt)
            print("D=%.1f lock=%.0f | nwin=%d matched=%d maxerr=%s baseFP=%d otherFP=%d | first=%s frozen=%.3f"
                  % (D, lock, s["n"], s["matched"],
                     ("%.3f" % s["max_onset_err"]) if s["max_onset_err"] is not None else "-",
                     s["false_base"], s["false_other"],
                     ("%.3f" % fc) if fc else "-", fv if fv else 0.0))


def step_fit(t, x, fs, gt, D, lock, f0, q, R0=0.01):
    print("=== Step 4: SEIZ_R fit (D=%.1f lock=%.0f F0=%.2f Q=%.1f) ===" % (D, lock, f0, q))
    w, p, fc, fv, S_tr, th_tr = simulate(t, x, fs, D=D, R=R0, LOCK=lock, F0=f0, Q=q)
    print("windows(R=%.4f): %s" % (R0, " ".join(fmt_windows(w))))
    print("score:", score(w, gt))
    # pair simulated windows to GT in order
    Ss = np.array([wi[3] for wi in w])
    durs = np.array([gt[i][1] - gt[i][0] for i in range(min(len(gt), len(w)))])
    Ss = Ss[:len(durs)]
    R = float((durs @ Ss) / (20.0 * (Ss @ Ss)))
    print("least-squares R = %.6f" % R)
    for k in range(len(durs)):
        pred = min(10.0, max(0.1, 20 * R * Ss[k]))
        print("  win %d: GT dur=%.3f  S=%.3f  pred=%.3f  resid=%+.3f" % (k, durs[k], Ss[k], pred, pred - durs[k]))
    return R


def step_final(t, x, fs, gt, gt_pulses, D, T_need, R, lock, f0, q, SHORT=20, LONG=1000):
    print("=== Step 5: final validation D=%.1f T=%d R=%.5f LOCK=%.0f F0=%.2f Q=%.1f SHORT=%d LONG=%d ==="
          % (D, T_need, R, lock, f0, q, SHORT, LONG))
    w, p, fc, fv, S_tr, th_tr = simulate(t, x, fs, D=D, T_need=T_need, R=R,
                                         LOCK=lock, F0=f0, Q=q, SHORT=SHORT, LONG=LONG)
    print("therapy windows:")
    for k, wi in enumerate(w):
        print("  %d: start=%.3f end=%.3f dur=%.3f S=%.3f" % (k, wi[0], wi[1], wi[2], wi[3]))
    print("detection pulses:", " ".join("%.3f" % pi for pi in p))
    print("first crossing: %.3f | frozen threshold: %.4f" % (fc, fv))
    s = score(w, gt)
    print("score:", s)
    # duration comparison
    for k in range(min(len(gt), len(w))):
        gd = gt[k][1] - gt[k][0]
        print("  dur GT=%.3f sim=%.3f err=%+.3f" % (gd, w[k][2], w[k][2] - gd))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(t, x, lw=0.5, color="0.4")
    for g0, g1 in gt:
        ax[0].axvspan(g0, g1, color="red", alpha=0.25)
    for wi in w:
        ax[0].axvspan(wi[0], wi[1], color="blue", alpha=0.2)
    ax[0].set_ylabel("raw signal")
    ax[0].legend(["data", "GT therapy", "sim therapy"], loc="upper right")
    ax[1].plot(t, S_tr, lw=0.6, color="green", label="S (short win var)")
    ax[1].plot(t, th_tr, lw=0.8, color="orange", label="threshold (adaptive/frozen)")
    for g0, g1 in gt:
        ax[1].axvspan(g0, g1, color="red", alpha=0.15)
    ax[1].set_ylabel("variance (filtered)")
    ax[1].legend(loc="upper right")
    ax[2].plot(t, x, lw=0.4, color="0.7")
    for tp in gt_pulses:
        ax[2].axvline(tp, color="red", lw=0.8)
    for tp in p:
        ax[2].axvline(tp, color="blue", lw=0.8, ls="--")
    ax[2].set_ylabel("events")
    ax[2].set_xlabel("time (s)")
    ax[2].legend(["data", "GT pulses (33)", "sim pulses"], loc="upper right")
    out = os.path.join(ROOT, "calibration_overlay.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("overlay saved:", out)
    return w, p, fc, fv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--oldscan", action="store_true")
    ap.add_argument("--f0", type=float, default=16.75)
    ap.add_argument("--q", type=float, default=2.0)
    ap.add_argument("--D", type=float, default=22.0)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--lock", type=float, default=30.0)
    ap.add_argument("--R", type=float, default=None)
    a = ap.parse_args()

    t, x = load_csv()
    fs = 1.0 / float(np.median(np.diff(t)))
    gt, gt_pulses = load_ground_truth()
    print("data: n=%d fs=%.1f Hz | GT windows=%d pulses=%d" % (len(x), fs, len(gt), len(gt_pulses)))

    if a.oldscan:
        print("=== old fixed-threshold reference scan (pseudo-var, 1s data window) ===")
        for th in [500, 400, 300, 200, 100, 50, 20, 10, 5, 3]:
            d = simulate_old_fixed(t, x, th)
            print("th=%-4g n=%-2d first=%s" % (th, len(d), ("%.2f" % d[0]) if d else "-"))
    if a.filter:
        step_filter(t, x, fs, gt)
    if a.grid:
        step_grid(t, x, fs, gt, a.f0, a.q)
    if a.fit:
        R = step_fit(t, x, fs, gt, a.D, a.lock, a.f0, a.q)
        print("FIT_R=%.6f" % R)
    if a.final:
        R = a.R if a.R is not None else 0.0012
        step_final(t, x, fs, gt, gt_pulses, a.D, a.T, R, a.lock, a.f0, a.q)


if __name__ == "__main__":
    main()

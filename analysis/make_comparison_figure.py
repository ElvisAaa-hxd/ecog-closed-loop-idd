# -*- coding: utf-8 -*-
"""Old (fixed-threshold) vs new (adaptive-threshold) closed-loop output comparison.

Panel a: original figure data (s-score, seizure points, therapy windows).
Panel b: recalibrated adaptive algorithm on the same recording
(constants parsed from Stream_the_data-Version2_FFT_python_judge.py,
 simulation logic identical to that script).
"""
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
except ImportError:  # optional QA helper; skip alignment gate when absent
    def require_matplotlib_panel_alignment(fig, **kwargs):
        return None

import offline_calibration as oc  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

OUT = os.path.join(os.path.dirname(BASE), "figure_old_vs_new")


def load_old():
    z = zipfile.ZipFile(oc.XLSX)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    T, B, C, D = [], [], [], []
    for r in sheet.findall(".//" + ns + "row"):
        if int(r.get("r")) <= 43:
            continue
        cells = {c.get("r")[0]: (c.find(ns + "v").text if c.find(ns + "v") is not None else "0")
                 for c in r.findall(ns + "c")}
        if "A" not in cells:
            continue
        T.append(float(cells["A"])); B.append(float(cells.get("B", "0")))
        C.append(float(cells.get("C", "0"))); D.append(float(cells.get("D", "0")))
    return np.array(T), np.array(B), np.array(C), np.array(D)


def main():
    t_old, s_old, c_old, d_old = load_old()
    t, x = oc.load_csv()
    fs = 1.0 / float(np.median(np.diff(t)))

    src = open(os.path.join(BASE, "stream_fft_judge.py"), encoding="utf-8").read()
    g = lambda name: float(re.search(r"^%s\s*=\s*([0-9.]+)" % name, src, re.M).group(1))
    w_new, p_new, fc, fv, S_tr, th_tr = oc.simulate(
        t, x, fs, D=g("SEIZ_D"), T_need=int(g("SEIZ_T")), R=g("SEIZ_R"),
        LOCK=g("THRESH_LOCK_S"), F0=g("BP_F0"), Q=g("BP_Q"),
        SHORT=int(g("SHORT_WINDOW")), LONG=int(g("LONG_WINDOW")))

    gt, _ = oc.load_ground_truth()

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True,
                             gridspec_kw=dict(hspace=0.32))
    raw_col = "0.75"
    old_col = "#C44E52"
    new_col = "#4C72B0"
    thr_col = "#DD8452"

    for ax in axes:
        ax.set_xlim(0, 43.5)
        ax.set_ylim(-100, 100)
        ax.set_ylabel("ECoG (mV)", fontsize=7)
        ax.tick_params(labelsize=6.5, width=0.8, length=2.5)

    # ---- panel a: original fixed-threshold output ----
    ax = axes[0]
    ax.plot(t, x, color=raw_col, lw=0.4, zorder=1)
    axr = ax.twinx()
    axr.set_ylim(0, 4600)
    axr.set_ylabel("s-score", fontsize=7)
    axr.tick_params(labelsize=6.5, width=0.8, length=2.5)
    axr.spines["right"].set_visible(True)
    axr.plot(t_old, s_old, color="#B7A132", lw=0.5, zorder=2)
    for g0, g1 in gt:
        ax.axvspan(g0, g1, color=old_col, alpha=0.22, zorder=3)
    for i in range(1, len(t_old)):
        if c_old[i] > 50 and c_old[i - 1] <= 50:
            ax.plot([t_old[i], t_old[i]], [84, 98], color=old_col, lw=0.9,
                    alpha=0.8, zorder=4, solid_capstyle="butt")
    ax.set_title("Fixed threshold (original data)", fontsize=8, loc="left", pad=3)

    # ---- panel b: recalibrated adaptive-threshold output ----
    ax = axes[1]
    ax.plot(t, x, color=raw_col, lw=0.4, zorder=1)
    axr = ax.twinx()
    axr.set_ylim(0, 2600)
    axr.set_ylabel("Variance (mV²)", fontsize=7)
    axr.tick_params(labelsize=6.5, width=0.8, length=2.5)
    axr.spines["right"].set_visible(True)
    axr.plot(t, S_tr, color="#55A868", lw=0.5, zorder=2, label="S (20-pt variance)")
    axr.plot(t, th_tr, color=thr_col, lw=0.9, zorder=3,
             label="threshold (adaptive, then locked)")
    for g0, g1 in gt:
        axr.plot([g0, g0, g1, g1], [0, 2520, 2520, 0],
                 color=old_col, lw=0.5, alpha=0.5, zorder=4)
    for wi in w_new:
        ax.axvspan(wi[0], wi[1], color=new_col, alpha=0.25, zorder=5)
    for tp in p_new:
        ax.plot([tp, tp], [84, 98], color=new_col, lw=1.0, alpha=0.9,
                zorder=6, solid_capstyle="butt")
    ax.set_title("Adaptive threshold (recalibrated)", fontsize=8, loc="left", pad=3)
    ax.set_xlabel("Time (s)", fontsize=7)

    axes[0].text(-0.09, 1.04, "a", transform=axes[0].transAxes,
                 fontsize=9, fontweight="bold", va="bottom", ha="left")
    axes[1].text(-0.09, 1.04, "b", transform=axes[1].transAxes,
                 fontsize=9, fontweight="bold", va="bottom", ha="left")

    leg = axr.legend(loc="upper right", fontsize=6, handlelength=1.6,
                     borderaxespad=0.2, labelspacing=0.35)
    for l in leg.get_lines():
        l.set_linewidth(1.2)

    require_matplotlib_panel_alignment(
        fig, json_out=OUT + ".alignment.json", overlay_svg=OUT + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(OUT + ".svg", bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    print("saved:", OUT + ".{svg,pdf,tiff,png}")


if __name__ == "__main__":
    main()

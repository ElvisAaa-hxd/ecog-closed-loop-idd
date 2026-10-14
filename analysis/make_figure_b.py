# -*- coding: utf-8 -*-
"""Figure b (adaptive-threshold algorithm) redrawn in the original figure style.

Style matched to the user's existing Origin panel: white background, thin black
frame, ~6.9:1 wide strip, light-pink therapy windows, magenta seizure-point
blocks (0.1 s), thin green s-score trace near the baseline, no axis labels.

Data: figure_data_new.xlsx (identical 4-column layout to the original figure
data; s-score column is identical to the original by construction).
"""
import os
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
except ImportError:  # optional QA helper; skip alignment gate when absent
    def require_matplotlib_panel_alignment(fig, **kwargs):
        return None

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
})

XLSX = os.path.join(ROOT, "data", "figure_data_new.xlsx")
OUT = os.path.join(ROOT, "figure_b_adaptive_oldstyle_v2")

# style constants (tweak here if the Origin colours differ)
C_SPAN = "#F3B8CB"   # light pink therapy windows
C_PULSE = "#C51B7D"  # magenta seizure-point blocks
C_LINE = "#339933"   # green s-score trace
LINE_FRAC = 0.30     # s-score max as a fraction of panel height


def load_new_data():
    z = zipfile.ZipFile(XLSX)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    T, B, C, D = [], [], [], []
    for r in sheet.findall(".//" + ns + "row"):
        if int(r.get("r")) == 1:
            continue
        cells = {c.get("r")[0]: (c.find(ns + "v").text if c.find(ns + "v") is not None else "0")
                 for c in r.findall(ns + "c")}
        if "A" not in cells:
            continue
        T.append(float(cells["A"])); B.append(float(cells.get("B", "0")))
        C.append(float(cells.get("C", "0"))); D.append(float(cells.get("D", "0")))
    return np.array(T), np.array(B), np.array(C), np.array(D)


def edges(T, col):
    e = []
    for i in range(1, len(T)):
        if col[i] > 50 and col[i - 1] <= 50:
            e.append([T[i], None])
        if col[i] <= 50 and col[i - 1] > 50 and e and e[-1][1] is None:
            e[-1][1] = T[i]
    return [(a, b if b is not None else T[-1]) for a, b in e]


def main():
    t, s, c, d = load_new_data()
    t = t - 3.0  # shift time axis: original 3 s becomes 0 s
    spans = edges(t, d)
    pulses = edges(t, c)

    # threshold trace from the shipped algorithm (filtered-variance units)
    import re
    import offline_calibration as oc
    src = open(os.path.join(BASE, "stream_fft_judge.py"), encoding="utf-8").read()
    g = lambda name: float(re.search(r"^%s\s*=\s*([0-9.]+)" % name, src, re.M).group(1))
    _, _, _, _, S_tr, th_tr = oc.simulate(
        t, oc.load_csv()[1], 1.0 / float(np.median(np.diff(t))),
        D=g("SEIZ_D"), T_need=int(g("SEIZ_T")), R=g("SEIZ_R"),
        LOCK=g("THRESH_LOCK_S"), F0=g("BP_F0"), Q=g("BP_Q"),
        SHORT=int(g("SHORT_WINDOW")), LONG=int(g("LONG_WINDOW")))

    fig, ax = plt.subplots(figsize=(10.35, 1.90))
    fig.subplots_adjust(left=0.055, right=0.988, top=0.86, bottom=0.22)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1000)
    ax.set_xticks(np.arange(0, 41, 5))
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.tick_params(labelsize=6.5, width=0.8, length=2.5)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Variance (mV²)", fontsize=7)
    ax.set_title("Adaptive threshold (new algorithm)", fontsize=8, loc="left", pad=3)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_color("black")

    for g0, g1 in spans:
        ax.axvspan(g0, g1, color=C_SPAN, alpha=0.45, lw=0, zorder=1)
    for g0, g1 in pulses:
        ax.axvspan(g0, g1, color=C_PULSE, alpha=0.9, lw=0, zorder=2)
    ax.plot(t, S_tr, color=C_LINE, lw=0.6, zorder=3)
    ax.plot(t, th_tr, color="#DD8452", lw=1.1, zorder=4)
    ax.text(19.6, 150, "threshold", fontsize=6, color="#DD8452",
            ha="left", va="bottom")

    require_matplotlib_panel_alignment(
        fig, json_out=OUT + ".alignment.json", overlay_svg=OUT + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(OUT + ".png", dpi=300)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".svg")
    fig.savefig(OUT + ".tiff", dpi=600)
    print("saved:", OUT + ".{png,pdf,svg,tiff}")
    print("spans:", ["%.3f-%.3f" % (a, b) for a, b in spans])
    print("pulses:", ["%.3f-%.3f" % (a, b) for a, b in pulses])


if __name__ == "__main__":
    main()

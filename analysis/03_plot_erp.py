# erp_plot_from_summary_and_stats_compact_leftlegends.py
# Compact layout + CNV plot-time re-baselining + legends ONLY on the left (first-half) panels.

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from config import OUTPUT_DIR

# ---------------- CONFIG ------------------------------------------------------
DATA_DIR      = OUTPUT_DIR
SUMMARY_PATH  = DATA_DIR / "grand_erp_summary.pkl"
CLUSTERS_PATH = DATA_DIR / "erp_clusters.pkl"

ALPHA_SEM     = 0.25

# Cluster-span shading
CLUST_ALPHA    = 0.18
COLOR_CONTRAST = "#2ca02c"                          # CS+ − CS−  (P3, CNV/SPN)
COLOR_FRN      = {"CS+": "#1f77b4", "CS-": "#d62728"}  # Exp − Unexp per cue

# ERP component windows (ms)
P3_WINDOW        = (200, 500)
CNV_WINDOW       = (300, 700)
SPN_WINDOW       = (900, 1000)
FRN_WINDOW_COMP  = (200, 300)   # shaded FRN component only
FRN_WINDOW_SIG   = (200, 500)   # significance band extent

ERP_WINDOW_COLOR  = "0.9"
ERP_WINDOW_ALPHA  = 0.9

X_P3   = (-300, 500)
X_CNV  = (-300, 1000)
X_FRN  = (-300, 500)

PANEL_CHS = {
    "P3":  ["CPz", "Pz"],
    "CNV": ["FCz", "Cz"],
    "SPN": ["Fz", "Cz"],
    "FRN": ["Fz", "FCz"],
}

P3_LABELS  = ["CS- presentation", "CS+ presentation"]
CNV_LABELS = ["CNV- fixation", "CNV+ fixation"]
FRN_LABELS = ["FRN+ Expected", "FRN+ Unexpected", "FRN- Expected", "FRN- Unexpected"]
HALVES     = ["first", "second"]

# Names of contrasts
CONTRAST_P3       = "CS+ vs CS-"
CONTRAST_CNV      = "CS+ vs CS-"
CONTRAST_FRN_CSP  = "CS+ Unexp vs Exp"
CONTRAST_FRN_CSM  = "CS- Unexp vs Exp"

# ---- CNV/SPN re-baselining at plot time (ms) ----
BASELINE_CNV_MS = (-200, 0)

# ---------------- COLOR & STYLE SYSTEM ---------------------------------------
def color_for(label: str) -> str:
    """One colour per cue: CS+ -> blue, CS- -> red. Outcome shown via linestyle."""
    L = label.lower()
    if ("cs+" in L) or ("frn+" in L) or ("cnv+ " in L):
        return "#1f77b4"
    if ("cs-" in L) or ("frn-" in L) or ("cnv- " in L):
        return "#d62728"
    return "#7f7f7f"


def linestyle_for(label: str) -> str:
    """Outcome mapping: Reward -> solid, No-reward -> dotted.

      CS+ Expected   = reward      CS+ Unexpected = no-reward
      CS- Unexpected = reward      CS- Expected   = no-reward
    """
    L = label.lower()
    unexpected = "unexpected" in L
    is_plus = ("frn+" in L) or ("cs+" in L) or ("cnv+ " in L)
    reward = (not unexpected) if is_plus else unexpected
    return "-" if reward else ":"


def pretty_label(panel, raw):
    if panel == "P3":
        return ""
    if panel in {"CNV", "SPN"}:
        return "CS+" if "+" in raw else "CS−"
    # FRN: map Expected/Unexpected -> Reward/No-reward by cue
    #   CS+ Expected = Reward,    CS+ Unexpected = No-reward
    #   CS- Expected = No-reward, CS- Unexpected = Reward
    val = "CS+" if "FRN+" in raw else "CS−"
    unexpected = "Unexpected" in raw
    if "FRN+" in raw:
        outcome = "No-reward" if unexpected else "Reward"
    else:
        outcome = "Reward" if unexpected else "No-reward"
    return f"{val} {outcome}"


# ---------------- HELPERS -----------------------------------------------------
def force_100ms_ticks(ax, xmin, xmax):
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax + 0.1, 100))


def neg_up_axis_labels(ax, ylabel="µV (neg ↑)"):
    # Avoid matplotlib warning by fixing ticks before setting labels
    yt = ax.get_yticks()
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{-v:g}" for v in yt])
    ax.set_ylabel(ylabel)


def shade_window(ax, window):
    ax.axvspan(window[0], window[1], color=ERP_WINDOW_COLOR, alpha=ERP_WINDOW_ALPHA, zorder=0)


def shade_clusters(ax, cluster_list, color, alpha=CLUST_ALPHA, label_p=True,
                   label_y=0.97):
    """Shade each significant cluster (t0_ms, t1_ms, p) as a vertical span.

    p-value labels sit at axes-fraction height `label_y`; pass different
    label_y per contrast to keep overlapping contrasts on separate rows.
    Adjacent labels within a contrast are nudged down to avoid collision.
    """
    if not cluster_list:
        return
    prev_x = None
    for (t0, t1, p) in sorted(cluster_list, key=lambda r: r[0]):
        ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0, zorder=0.4)
        if label_p:
            xc = (t0 + t1) / 2.0
            yy = label_y
            if prev_x is not None and abs(xc - prev_x) < 90:
                yy = label_y - 0.07          # second of a close pair drops a row
            ax.text(xc, yy, f"p={p:.3f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, color=color, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.65))
            prev_x = xc


def annotate_interaction(ax, items, y=0.03):
    """Text summary of the Condition×Half interaction cluster test(s).

    items: list of (prefix, color, cluster_list).
    """
    lines = []
    colors = []
    for prefix, color, cl in items:
        if cl:
            spans = ", ".join(f"{int(t0)}–{int(t1)} ms (p={p:.3f})"
                              for t0, t1, p in cl)
            lines.append(f"{prefix} × Half: {spans}")
        else:
            lines.append(f"{prefix} × Half: n.s.")
        colors.append(color)

    yy = y
    for line, color in zip(lines[::-1], colors[::-1]):   # stack upward
        ax.text(0.98, yy, line, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
        yy += 0.075


def apply_baseline_to_curve(t_ms, mean, sem, baseline_ms):
    """
    Plot-time baseline correction by subtracting the mean amplitude in baseline_ms.
    SEM is unchanged by subtracting a constant.
    """
    t = np.asarray(t_ms)
    m = np.asarray(mean)
    s = np.asarray(sem)

    b0, b1 = baseline_ms
    mask = (t >= b0) & (t <= b1)
    if not np.any(mask):
        return m, s

    offset = float(np.mean(m[mask]))
    return m - offset, s


# ---------------- PLOTTING ----------------------------------------------------
def plot_figure(summary, clusters):
    data = summary["data"]

    # Compact, readable defaults
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })

    fig, axes = plt.subplots(
        3, 2,
        figsize=(14.5, 7.2),
        gridspec_kw={"height_ratios": [1.0, 1.15, 1.15]},
        constrained_layout=True,
    )
    fig.suptitle("Grand-average ERPs — First vs. Second half\n"
                 "Shaded = cluster-corrected p < .05 (CS contrast); "
                 "× Half = Condition×Half interaction cluster test",
                 fontsize=13)

    for j, half in enumerate(HALVES):
        left = (j == 0)

        # ========================= P3 ==========================
        ax = axes[0, j]
        shade_window(ax, P3_WINDOW)

        for lab in P3_LABELS:
            df = data["P3"][half][lab]
            if df.empty:
                continue
            t, m, s = df["t_ms"], df["mean"], df["sem"]
            c = color_for(lab)
            ax.fill_between(t, m - s, m + s, color=c, alpha=ALPHA_SEM, linewidth=0)
            ax.plot(t, m, lw=2.2, color=c)

        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        force_100ms_ticks(ax, *X_P3)
        ax.set_title(f"P3 ({', '.join(PANEL_CHS['P3'])}) — {half.capitalize()}")
        ax.set_ylabel("µV")
        ax.grid(False)

        shade_clusters(ax, clusters["P3"][half], COLOR_CONTRAST)
        if not left:
            annotate_interaction(ax, [("CS+ − CS−", COLOR_CONTRAST,
                                       clusters["P3"]["interaction"])])

        # ====================== CNV / SPN ========================
        ax = axes[1, j]
        shade_window(ax, CNV_WINDOW)
        shade_window(ax, SPN_WINDOW)

        handles, labels = [], []
        for lab in CNV_LABELS:
            df = data["CNV"][half][lab]
            if df.empty:
                continue
            t, m, s = df["t_ms"], df["mean"], df["sem"]

            # Plot-time CNV/SPN re-baselining
            m, s = apply_baseline_to_curve(t, m, s, BASELINE_CNV_MS)

            c = color_for(lab)
            lbl = pretty_label("CNV", lab)
            h = ax.plot(t, m, lw=2.2, color=c, label=lbl)[0]
            ax.fill_between(t, m - s, m + s, color=c, alpha=ALPHA_SEM, linewidth=0)
            handles.append(h); labels.append(lbl)

        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        force_100ms_ticks(ax, *X_CNV)
        cnvspn_chs = list(dict.fromkeys(PANEL_CHS["CNV"]))   # channels actually plotted/tested
        ax.set_title(f"CNV / SPN ({', '.join(cnvspn_chs)}) — {half.capitalize()}")
        neg_up_axis_labels(ax)
        ax.grid(False)

        # Legend ONLY on the left CNV panel
        if left and handles:
            uniq_h, uniq_l = [], []
            for h, l in zip(handles, labels):
                if l not in uniq_l:
                    uniq_l.append(l)
                    uniq_h.append(h)
            ax.legend(uniq_h, uniq_l, frameon=True, loc="upper left")

        shade_clusters(ax, clusters["CNV"][half], COLOR_CONTRAST)
        if not left:
            annotate_interaction(ax, [("CS+ − CS−", COLOR_CONTRAST,
                                       clusters["CNV"]["interaction"])])

        # ======================== FRN ===========================
        ax = axes[2, j]
        shade_window(ax, FRN_WINDOW_COMP)

        handles, labels = [], []
        for lab in FRN_LABELS:
            df = data["FRN"][half][lab]
            if df.empty:
                continue
            t, m, s = df["t_ms"], df["mean"], df["sem"]
            c  = color_for(lab)
            ls = linestyle_for(lab)
            lbl = pretty_label("FRN", lab)

            h = ax.plot(t, m, lw=2.0, color=c, linestyle=ls, label=lbl)[0]
            ax.fill_between(t, m - s, m + s, color=c, alpha=ALPHA_SEM, linewidth=0)
            handles.append(h); labels.append(lbl)

        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        force_100ms_ticks(ax, *X_FRN)
        ax.set_title(f"FRN ({', '.join(PANEL_CHS['FRN'])}) — {half.capitalize()}")
        ax.set_ylim(-4, 6)          # neg-up display: −6 (top) to 4 (bottom), both halves
        neg_up_axis_labels(ax)
        ax.set_xlabel("Time (ms)")
        ax.grid(False)

        # Legend ONLY on the left FRN panel
        if left and handles:
            uniq_h, uniq_l = [], []
            for h, l in zip(handles, labels):
                if l not in uniq_l:
                    uniq_l.append(l)
                    uniq_h.append(h)
            ax.legend(uniq_h, uniq_l, frameon=True, loc="upper left")

        # FRN clusters: Reward − No-reward, per cue type (labels on separate rows)
        shade_clusters(ax, clusters["FRN"][half]["CS+"], COLOR_FRN["CS+"], label_y=0.97)
        shade_clusters(ax, clusters["FRN"][half]["CS-"], COLOR_FRN["CS-"], label_y=0.88)
        if not left:
            annotate_interaction(ax, [
                ("CS+ (R−NR)", COLOR_FRN["CS+"],
                 clusters["FRN"]["interaction"]["CS+"]),
                ("CS− (R−NR)", COLOR_FRN["CS-"],
                 clusters["FRN"]["interaction"]["CS-"]),
            ])

    out = DATA_DIR / "GrandAverage_P3_CNVSPN_FRN_byHalf_clusters.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print("[INFO] Saved:", out)


# ---------------- RUN ---------------------------------------------------------
if __name__ == "__main__":
    with open(SUMMARY_PATH, "rb") as f:
        summary = pickle.load(f)
    with open(CLUSTERS_PATH, "rb") as f:
        clusters = pickle.load(f)["clusters"]

    plot_figure(summary, clusters)

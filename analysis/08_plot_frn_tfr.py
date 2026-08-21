# Plot_FRN_TF_3way_composite.py
#
# Composite 3×3 figure: CS × Outcome × Half interaction in feedback-locked TFR.
#
# Rows : CS+ (R−NR)  |  CS− (R−NR)  |  Interaction
# Cols : First half  |  Second half  |  Collapsed
#
# Row labels on the LEFT as rotated y-axis text.
# One colourbar per row in a dedicated 4th column (no overlap with labels).

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import mne
import os, re

from config import EPOCHS_DIR, EXCLUDED_SUBJECTS, OUTPUT_DIR, TFR_DIR

mne.set_log_level("WARNING")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = EPOCHS_DIR
OUT_BASE = OUTPUT_DIR
MASK_DIR  = OUT_BASE / "cluster_masks" / "FRN_CNV_noanova" / "FRN"

EXCLUDE_SUBJECTS = EXCLUDED_SUBJECTS

FRN_FMIN  = 4.0
PAD_START = 0.1
PAD_END   = 0.01
CMAP      = "RdBu_r"
ROW_VMAX  = [0.55, 0.55, 0.70]   # per-row colour limits

COL_LABELS = ["First half", "Second half"]
ROW_LABELS = [
    r"$CS^+$ (R$-$NR)",
    r"$CS^-$ (R$-$NR)",
    "Interaction\n" + r"($CS^+$ $-$ $CS^-$)",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_subjects():
    pat = re.compile(r"sub-(\d+)")
    subs = {pat.search(f).group(1) for f in os.listdir(TFR_DIR) if pat.search(f)}
    return [s for s in sorted(subs, key=int) if s not in EXCLUDE_SUBJECTS]

def read_tfr(path):
    res = mne.time_frequency.read_tfrs(str(path))
    return res[0] if isinstance(res, list) else res

def load_cond(subj, cond):
    f = TFR_DIR / f"sub-{subj}_FRN_{cond}-tfr.h5"
    if not f.exists():
        return None, None, None
    try:
        tfr = read_tfr(f)
    except Exception:
        return None, None, None
    return tfr.freqs, tfr.times, np.mean(tfr.data, axis=0)

def select_tf(X, freqs, times):
    tmask = (times >= times[0] + PAD_START) & (times <= times[-1] - PAD_END)
    fmask = freqs >= FRN_FMIN
    return X[:, fmask][:, :, tmask], times[tmask], freqs[fmask]

def load_mask(name):
    p = MASK_DIR / f"{name}_mask.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["tf_mask"].astype(bool)


# ── GRAND AVERAGES ────────────────────────────────────────────────────────────
def build_grand_averages(subjects):
    CONDS = [
        "CSplus_Expected_first",   "CSplus_Unexpected_first",
        "CSminus_Expected_first",  "CSminus_Unexpected_first",
        "CSplus_Expected_second",  "CSplus_Unexpected_second",
        "CSminus_Expected_second", "CSminus_Unexpected_second",
    ]

    arrays = {c: [] for c in CONDS}
    times_ref = freqs_ref = None

    for subj in subjects:
        loaded = {}
        ok = True
        for c in CONDS:
            f_arr, t_arr, d = load_cond(subj, c)
            if d is None:
                ok = False; break
            loaded[c] = (f_arr, t_arr, d)
        if not ok:
            continue
        if times_ref is None:
            times_ref = loaded[CONDS[0]][1]
            freqs_ref = loaded[CONDS[0]][0]
        for c in CONDS:
            arrays[c].append(loaded[c][2])

    if times_ref is None:
        raise RuntimeError("No data loaded.")

    stacked = {}
    for c in CONDS:
        X = np.array(arrays[c])
        Xs, t_sel, f_sel = select_tf(X, freqs_ref, times_ref)
        stacked[c] = Xs

    def diff_mean(A, B):
        return (stacked[A] - stacked[B]).mean(axis=0)

    # per-half difference maps
    cp1 = diff_mean("CSplus_Expected_first",    "CSplus_Unexpected_first")
    cm1 = diff_mean("CSminus_Unexpected_first", "CSminus_Expected_first")
    cp2 = diff_mean("CSplus_Expected_second",   "CSplus_Unexpected_second")
    cm2 = diff_mean("CSminus_Unexpected_second","CSminus_Expected_second")

    # collapsed
    cp_coll = (cp1 + cp2) / 2
    cm_coll = (cm1 + cm2) / 2

    # interactions
    int1    = cp1 - cm1
    int2    = cp2 - cm2
    int_col = cp_coll - cm_coll

    # masks
    ga = {
        # (data, mask)
        "csplus_first":      (cp1,    None),
        "csplus_second":     (cp2,    None),
        "csplus_collapsed":  (cp_coll,None),
        "csminus_first":     (cm1,    None),
        "csminus_second":    (cm2,    None),
        "csminus_collapsed": (cm_coll, load_mask("FRN_CSminus_collapsed")),
        "int_first":         (int1,   load_mask("FRN_INT_first")),
        "int_second":        (int2,   load_mask("FRN_INT_second")),
        "int_collapsed":     (int_col,load_mask("FRN_INT_collapsed")),
    }
    return ga, t_sel, f_sel


# ── PLOTTING ──────────────────────────────────────────────────────────────────
def plot_panel(ax, data, times, freqs, mask, vmax,
               show_xlabel, show_ylabel, show_yticks):
    im = ax.imshow(
        data,
        origin="lower", aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=CMAP, vmin=-vmax, vmax=vmax,
        interpolation="bilinear",
    )
    if mask is not None:
        ax.contour(times, freqs, mask.astype(float),
                   levels=[0.5], colors="black", linewidths=1.4)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(round(x*1000))}"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    if not show_xlabel:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (ms)", fontsize=10)

    if not show_yticks:
        ax.set_yticklabels([])

    # significance annotation
    sig_text = "p < .05*" if mask is not None else "n.s."
    sig_col  = "black"    if mask is not None else "0.45"
    ax.text(0.03, 0.97, sig_text, transform=ax.transAxes,
            fontsize=8, va="top", color=sig_col,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

    return im


def run():
    subjects = get_subjects()
    print(f"[INFO] {len(subjects)} subjects")
    ga, t_sel, f_sel = build_grand_averages(subjects)
    print("[INFO] Grand averages computed")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})

    # GridSpec: 3 rows × (2 panels + 1 narrow cbar column)
    fig = plt.figure(figsize=(9.5, 8.5))
    gs  = GridSpec(
        3, 3,
        figure=fig,
        width_ratios=[1, 1, 0.055],
        hspace=0.35,
        wspace=0.10,
        left=0.14, right=0.95, top=0.91, bottom=0.09,
    )

    axes     = [[fig.add_subplot(gs[ri, ci]) for ci in range(2)] for ri in range(3)]
    cbar_axs = [fig.add_subplot(gs[ri, 2])  for ri in range(3)]

    ROW_KEYS = ["csplus", "csminus", "int"]
    COL_KEYS = ["first",  "second"]

    last_im = [None] * 3

    for ri, row_key in enumerate(ROW_KEYS):
        for ci, col_key in enumerate(COL_KEYS):
            ax   = axes[ri][ci]
            key  = f"{row_key}_{col_key}"
            data, mask = ga[key]
            vmax = ROW_VMAX[ri]

            im = plot_panel(
                ax, data, t_sel, f_sel, mask, vmax,
                show_xlabel = (ri == 2),
                show_ylabel = (ci == 0),
                show_yticks = (ci == 0),
            )
            last_im[ri] = im

            # column titles (top row only)
            if ri == 0:
                ax.set_title(COL_LABELS[ci], fontsize=12,
                             fontweight="bold", pad=5)

        # row label as rotated text to the left of the leftmost panel
        axes[ri][0].text(
            -0.13, 0.5, ROW_LABELS[ri],
            transform=axes[ri][0].transAxes,
            fontsize=10, ha="center", va="center",
            rotation=90, multialignment="center",
            clip_on=False,
        )

        # colourbar in dedicated column
        cb = fig.colorbar(last_im[ri], cax=cbar_axs[ri])
        cb.set_ticks([-ROW_VMAX[ri], 0, ROW_VMAX[ri]])
        cb.ax.tick_params(labelsize=8)
        cb.set_label("dB", fontsize=9, labelpad=4)

    # shared frequency label for middle and right columns
    fig.text(0.05, 0.50, "Frequency (Hz)", ha="center", va="center",
             rotation=90, fontsize=11)

    fig.suptitle(
        "Feedback-locked time-frequency power\n"
        "Black contour = cluster-corrected p\u202f<\u202f.05",
        fontsize=12,
    )

    out = OUT_BASE / "FRN_TF_3way_composite.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"[INFO] Saved: {out}")


if __name__ == "__main__":
    run()

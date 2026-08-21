# Plot_CNV_TF_composite.py
#
# CS+ − CS− difference TFR: First half | Second half
# Cluster contour on the significant panel (second half).
#
# Style matches Plot_FRN_TF_3way_composite.py exactly.

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
MASK_DIR  = OUT_BASE / "cluster_masks" / "FRN_CNV_noanova" / "CNV"

EXCLUDE_SUBJECTS = EXCLUDED_SUBJECTS

CNV_FMIN  = 4.0
PAD_START = 0.1
PAD_END   = 0.01
CMAP      = "RdBu_r"
VMAX      = 0.55

COL_LABELS = ["First half (CS⁺ − CS⁻)",
              "Second half (CS⁺ − CS⁻)",
              "Second − First (CS⁺ − CS⁻)"]
ROW_LABEL  = r"$CS^+$ $-$ $CS^-$"


# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_subjects():
    pat = re.compile(r"sub-(\d+)")
    subs = {pat.search(f).group(1) for f in os.listdir(TFR_DIR) if pat.search(f)}
    return [s for s in sorted(subs, key=int) if s not in EXCLUDE_SUBJECTS]


def read_tfr(path):
    res = mne.time_frequency.read_tfrs(str(path))
    return res[0] if isinstance(res, list) else res


def load_cond(subj, cond):
    f = TFR_DIR / f"sub-{subj}_CNV_{cond}-tfr.h5"
    if not f.exists():
        return None, None, None
    try:
        tfr = read_tfr(f)
    except Exception:
        return None, None, None
    return tfr.freqs, tfr.times, np.mean(tfr.data, axis=0)


def select_tf(X, freqs, times):
    tmask = (times >= times[0] + PAD_START) & (times <= times[-1] - PAD_END)
    fmask = freqs >= CNV_FMIN
    return X[:, fmask][:, :, tmask], times[tmask], freqs[fmask]


def load_mask(name):
    p = MASK_DIR / f"{name}_mask.npz"
    if not p.exists():
        return None, None, None
    d = np.load(p)
    freqs = d["freqs"] if "freqs" in d else None
    times = d["times"] if "times" in d else None
    return d["tf_mask"].astype(bool), freqs, times


# ── GRAND AVERAGES ────────────────────────────────────────────────────────────
def build_grand_averages(subjects):
    CONDS = ["CSplus_first", "CSplus_second", "CSminus_first", "CSminus_second"]
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

    diff1 = (stacked["CSplus_first"]  - stacked["CSminus_first"]).mean(axis=0)
    diff2 = (stacked["CSplus_second"] - stacked["CSminus_second"]).mean(axis=0)
    # Condition x Half interaction: (CS+ - CS-)_second - (CS+ - CS-)_first
    diff_int = diff2 - diff1

    mask2, mask2_freqs, mask2_times = load_mask("CNV_CSplus_minus_CSminus_second")
    mask_int, mask_int_freqs, mask_int_times = load_mask(
        "CNV_CSplus_minus_CSminus_second_minus_first")

    return (
        [(diff1,    None,     None,           None),
         (diff2,    mask2,    mask2_freqs,    mask2_times),
         (diff_int, mask_int, mask_int_freqs, mask_int_times)],
        t_sel, f_sel
    )


# ── PANEL PLOTTING ────────────────────────────────────────────────────────────
def plot_panel(ax, data, times, freqs, mask, mask_freqs, mask_times, show_yticks):
    im = ax.imshow(
        data,
        origin="lower", aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=CMAP, vmin=-VMAX, vmax=VMAX,
        interpolation="bilinear",
    )
    if mask is not None:
        ct = mask_times if mask_times is not None else times
        cf = mask_freqs if mask_freqs is not None else freqs
        ax.contour(ct, cf, mask.astype(float),
                   levels=[0.5], colors="black", linewidths=1.4)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(round(x*1000))}"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlabel("Time (ms)", fontsize=10)

    if not show_yticks:
        ax.set_yticklabels([])

    sig_text = "p < .05*" if mask is not None else "n.s."
    sig_col  = "black"    if mask is not None else "0.45"
    ax.text(0.03, 0.97, sig_text, transform=ax.transAxes,
            fontsize=8, va="top", color=sig_col,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

    return im


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run():
    subjects = get_subjects()
    print(f"[INFO] {len(subjects)} subjects")
    panels, t_sel, f_sel = build_grand_averages(subjects)
    print("[INFO] Grand averages computed")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})

    # GridSpec: 2 × 2 panels (+ narrow cbar column)
    #   top-left    = First half
    #   top-right   = Second half
    #   bottom-left = Second − First
    #   bottom-right = reserved for topomap (added separately)
    fig = plt.figure(figsize=(10, 8.5))
    gs  = GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1, 1, 0.05],
        height_ratios=[1, 1],
        hspace=0.30,
        wspace=0.10,
        left=0.10, right=0.92, top=0.88, bottom=0.09,
    )

    # panel -> (row, col, show_yticks); left column shows the frequency ticks
    positions = [(0, 0, True), (0, 1, False), (1, 0, True)]

    last_im = None
    for (data, mask, mask_freqs, mask_times), (r, c, show_y), label in zip(
            panels, positions, COL_LABELS):
        ax = fig.add_subplot(gs[r, c])
        im = plot_panel(ax, data, t_sel, f_sel, mask, mask_freqs, mask_times,
                        show_yticks=show_y)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=5)
        last_im = im

    # bottom-right quadrant reserved for the topomap (add it here)
    ax_topo = fig.add_subplot(gs[1, 1])
    ax_topo.axis("off")

    # colourbar spanning both rows
    cbar_ax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_ticks([-VMAX, 0, VMAX])
    cb.ax.tick_params(labelsize=8)
    cb.set_label("dB", fontsize=9, labelpad=4)

    # shared y-axis label, moved close to the axis
    fig.text(0.055, 0.5, "Frequency (Hz)", ha="center", va="center",
             rotation=90, fontsize=11)

    fig.suptitle(
        "Anticipatory time-frequency power\n"
        "Black contour = cluster-corrected p\u202f<\u202f.05",
        fontsize=12,
    )

    out = OUT_BASE / "CNV_TF_composite.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"[INFO] Saved: {out}")


if __name__ == "__main__":
    run()

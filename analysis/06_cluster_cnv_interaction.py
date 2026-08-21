# -*- coding: utf-8 -*-
"""
Standalone cluster-permutation test for the CNV Condition x Half interaction
in time-frequency power:

    (CS+ - CS-)_second  -  (CS+ - CS-)_first

Mirrors Running_cluster_29_01.py exactly (same exclude list, channel average
over FCz/Cz, fmin=1.0, pad_start=0.1, pad_end=0.01, 1000 permutations, two-sided).

Saves ONLY the new mask:
    cluster_masks/FRN_CNV_noanova/CNV/CNV_CSplus_minus_CSminus_second_minus_first_mask.npz

It does NOT touch any existing mask, so the already-reported per-half clusters
are left untouched.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.stats import spatio_temporal_cluster_1samp_test, combine_adjacency

from config import EPOCHS_DIR, EXCLUDED_SUBJECTS, OUTPUT_DIR, TFR_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ---------------------------------------------------------------------
# PATHS  (identical to Running_cluster_29_01.py)
# ---------------------------------------------------------------------
DATA_PATH = EPOCHS_DIR
OUT_BASE = OUTPUT_DIR

OUT_DIR  = OUT_BASE / "clusters"      / "FRN_CNV_noanova" / "CNV"
MASK_DIR = OUT_BASE / "cluster_masks" / "FRN_CNV_noanova" / "CNV"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# SUBJECTS  (identical to Running_cluster_29_01.py)
# ---------------------------------------------------------------------
EXCLUDE_SUBJECTS = EXCLUDED_SUBJECTS

CNV_CONDS = ["CSplus_first", "CSminus_first", "CSplus_second", "CSminus_second"]

# cluster params (identical to the CNV runner)
FMIN      = 1.0
PAD_START = 0.1
PAD_END   = 0.01
N_PERM    = 1000
SEED      = 42


def find_subjects(root):
    pat = re.compile(r"sub-(\d+)")
    subs = set()
    for f in os.listdir(root):
        m = pat.search(f)
        if m:
            subs.add(m.group(1))
    return sorted(subs, key=lambda s: int(s))


def get_subjects():
    subs = [s for s in find_subjects(TFR_DIR) if s not in EXCLUDE_SUBJECTS]
    print(f"[INFO] Using {len(subs)} subjects")
    return subs


def read_one_tfr(path):
    tfr_list = mne.time_frequency.read_tfrs(str(path))
    return tfr_list[0] if isinstance(tfr_list, list) else tfr_list


def tfr_array_from_files(subjects, component, conds):
    arrays, keep_lists = [], []
    times = freqs = None
    for c in conds:
        X, keep = [], []
        for s in subjects:
            f = TFR_DIR / f"sub-{s}_{component}_{c}-tfr.h5"
            if not f.exists():
                continue
            t = read_one_tfr(f)
            X.append(np.mean(t.data, axis=0))   # channel average -> (freqs, times)
            times, freqs = t.times, t.freqs
            keep.append(s)
        arrays.append(np.array(X))
        keep_lists.append(keep)
    return arrays, times, freqs, keep_lists


def intersect_arrays(arrs, keep_lists):
    common = set(keep_lists[0])
    for k in keep_lists[1:]:
        common &= set(k)
    common = sorted(common, key=lambda x: int(x))
    out = []
    for X, k in zip(arrs, keep_lists):
        idx = [k.index(s) for s in common]
        out.append(X[idx])
    return out, common


def select_tf(X, times, freqs, fmin, pad_start, pad_end):
    tmask = (times >= times[0] + pad_start) & (times <= times[-1] - pad_end)
    fmask = freqs >= fmin
    return X[:, fmask][:, :, tmask], times[tmask], freqs[fmask]


def main():
    subjects = get_subjects()
    Xs, times, freqs, keep = tfr_array_from_files(subjects, "CNV", CNV_CONDS)
    Xs, subs = intersect_arrays(Xs, keep)
    print(f"[INFO] CNV interaction: complete subjects: {len(subs)}")
    if len(subs) == 0:
        return

    # indices: 0 CS+ first, 1 CS- first, 2 CS+ second, 3 CS- second
    # interaction = (CS+ - CS-)_second - (CS+ - CS-)_first
    interaction = (Xs[2] - Xs[3]) - (Xs[0] - Xs[1])

    Xsel, t, f = select_tf(interaction, times, freqs, FMIN, PAD_START, PAD_END)
    adjacency = combine_adjacency(len(f), len(t))

    Tobs, clusters, pvals, _ = spatio_temporal_cluster_1samp_test(
        Xsel,
        adjacency=adjacency,
        n_permutations=N_PERM,
        tail=0,
        out_type="mask",
        seed=SEED,
        verbose=False,
    )

    sig = [clusters[i] for i, p in enumerate(pvals) if p < 0.05]
    name = "CNV_CSplus_minus_CSminus_second_minus_first"

    if sig:
        mask = np.zeros_like(sig[0])
        for s in sig:
            mask |= s
        np.savez(MASK_DIR / f"{name}_mask.npz", tf_mask=mask, times=t, freqs=f)
        sig_ps = sorted(p for p in pvals if p < 0.05)
        print(f"[INFO] {len(sig)} significant cluster(s); p = {sig_ps}")
        print(f"[INFO] Saved mask: {MASK_DIR / (name + '_mask.npz')}")
    else:
        minp = float(np.min(pvals)) if len(pvals) else float("nan")
        print(f"[INFO] No significant cluster (min p = {minp:.3f}); no mask saved.")

    # diagnostic figure
    plt.figure(figsize=(7, 4))
    im = plt.imshow(np.mean(Xsel, 0), origin="lower", aspect="auto",
                    extent=[t[0], t[-1], f[0], f[-1]], cmap="RdBu_r")
    for c in sig:
        plt.contour(t, f, c, colors="black")
    plt.colorbar(im, label="dB")
    plt.title("CNV: (CS+ - CS-) second - first")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {OUT_DIR / (name + '.png')}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Cluster-based permutation tests (NO ANOVAs)

FRN (8 cells: CS± × Expected/Unexpected × half):
Per half + collapsed:
- CS+ (Reward − No-Reward)
- CS− (Reward − No-Reward)
- Interaction: (CS+ (R−NR)) − (CS− (R−NR))

CNV (4 cells: CS± × half):
Per half + collapsed:
- CS+ − CS−
Also runs (optional but included): CS+ vs 0 and CS− vs 0 per half + collapsed.

File patterns on disk (tfr_dB):
FRN: sub-XXXX_FRN_CSplus_Expected_first-tfr.h5 etc.
CNV: sub-XXXX_CNV_CSplus_first-tfr.h5 etc.
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

SEED = 42

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
DATA_PATH = EPOCHS_DIR
OUT_BASE = OUTPUT_DIR

OUT_ROOT = OUT_BASE / "clusters" / "FRN_CNV_noanova"
MASK_ROOT = OUT_BASE / "cluster_masks" / "FRN_CNV_noanova"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
MASK_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# SUBJECTS
# ---------------------------------------------------------------------
EXCLUDE_SUBJECTS = EXCLUDED_SUBJECTS

def find_subjects(root):
    pat = re.compile(r"sub-(\d+)")
    subs = set()
    for f in os.listdir(root):
        m = pat.search(f)
        if m:
            subs.add(m.group(1))
    return sorted(subs, key=lambda s: int(s))

def get_subjects():
    subs = find_subjects(TFR_DIR)
    subs = [s for s in subs if s not in EXCLUDE_SUBJECTS]
    print(f"[INFO] Using {len(subs)} subjects")
    return subs

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def read_one_tfr(path: Path):
    tfr_list = mne.time_frequency.read_tfrs(str(path))
    return tfr_list[0] if isinstance(tfr_list, list) else tfr_list

def tfr_array_from_files(subjects, component, conds):
    """
    conds: list of condition strings exactly matching filename middle part, e.g.
      FRN cond: 'CSplus_Expected_first'
      CNV cond: 'CSplus_first'
    Returns:
      arrays: list of X per cond (subjects, freqs, times)
      times, freqs
      keep_lists: list of subject IDs kept per cond
    """
    arrays, keep_lists = [], []
    times = freqs = None

    for c in conds:
        X, keep = [], []
        for s in subjects:
            f = TFR_DIR / f"sub-{s}_{component}_{c}-tfr.h5"
            if not f.exists():
                continue
            t = read_one_tfr(f)
            X.append(np.mean(t.data, axis=0))  # channel average -> (freqs, times)
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

def select_tf(X, times, freqs, fmin=1.0, pad_start=0.1, pad_end=0.01):
    tmask = (times >= times[0] + pad_start) & (times <= times[-1] - pad_end)
    fmask = freqs >= fmin
    return X[:, fmask][:, :, tmask], times[tmask], freqs[fmask]

def save_mask(mask_dir, name, mask, times, freqs):
    np.savez(mask_dir / f"{name}_mask.npz", tf_mask=mask, times=times, freqs=freqs)

def plot_tf(cluster_dir, data, clusters, times, freqs, title, fname, cmap="RdBu_r"):
    plt.figure(figsize=(7, 4))
    im = plt.imshow(
        data, origin="lower", aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=cmap
    )
    for c in clusters:
        plt.contour(times, freqs, c, colors="black")
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(cluster_dir / fname, dpi=300)
    plt.close()

def run_cluster_1samp(X, times, freqs, name, title, cluster_dir, mask_dir,
                     fmin=1.0, pad_start=0.1, pad_end=0.01,
                     n_permutations=1000, seed=SEED):
    Xs, t, f = select_tf(X, times, freqs, fmin=fmin, pad_start=pad_start, pad_end=pad_end)
    adjacency = combine_adjacency(len(f), len(t))

    Tobs, clusters, pvals, _ = spatio_temporal_cluster_1samp_test(
        Xs,
        adjacency=adjacency,
        n_permutations=n_permutations,
        tail=0,
        out_type="mask",
        seed=seed,
        verbose=False
    )

    sig = [clusters[i] for i, p in enumerate(pvals) if p < 0.05]
    sig_ps = sorted(float(p) for p in pvals if p < 0.05)
    minp = float(np.min(pvals)) if len(pvals) else float("nan")
    print(f"[PVAL] {name}: sig p={sig_ps if sig_ps else 'none'}  (min cluster p={minp:.4f})")
    if sig:
        mask = np.zeros_like(sig[0])
        for s in sig:
            mask |= s
        save_mask(mask_dir, name, mask, t, f)

    plot_tf(cluster_dir, np.mean(Xs, 0), sig, t, f, title, f"{name}.png")

# ---------------------------------------------------------------------
# FRN RUNNER
# ---------------------------------------------------------------------
FRN_CONDS = [
    "CSplus_Expected_first",
    "CSplus_Unexpected_first",
    "CSminus_Expected_first",
    "CSminus_Unexpected_first",
    "CSplus_Expected_second",
    "CSplus_Unexpected_second",
    "CSminus_Expected_second",
    "CSminus_Unexpected_second",
]

def run_frn(subjects):
    cluster_dir = OUT_ROOT / "FRN"
    mask_dir = MASK_ROOT / "FRN"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    Xs, times, freqs, keep = tfr_array_from_files(subjects, "FRN", FRN_CONDS)
    Xs, subs = intersect_arrays(Xs, keep)
    print(f"[INFO] FRN: Complete subjects: {len(subs)}")
    if len(subs) == 0:
        return

    # indices:
    # 0 CS+ Exp  first (R)
    # 1 CS+ Unexp first (NR)
    # 2 CS− Exp  first (NR)
    # 3 CS− Unexp first (R)
    # 4 CS+ Exp  second (R)
    # 5 CS+ Unexp second (NR)
    # 6 CS− Exp  second (NR)
    # 7 CS− Unexp second (R)

    # Per-half simple effects
    csplus_first   = (Xs[0] - Xs[1])     # CS+ (R-NR)
    csminus_first  = (Xs[3] - Xs[2])     # CS- (R-NR)
    int_first      = csplus_first - csminus_first

    csplus_second  = (Xs[4] - Xs[5])
    csminus_second = (Xs[7] - Xs[6])
    int_second     = csplus_second - csminus_second

    # Collapsed
    csplus_coll  = ((Xs[0] + Xs[4]) / 2) - ((Xs[1] + Xs[5]) / 2)
    csminus_coll = ((Xs[3] + Xs[7]) / 2) - ((Xs[2] + Xs[6]) / 2)
    int_coll     = csplus_coll - csminus_coll

    params = dict(fmin=4.0, pad_start=0.1, pad_end=0.01, n_permutations=1000)

    # Interaction
    run_cluster_1samp(int_first, times, freqs, "FRN_INT_first",
                      "FRN: (CS+ (R−NR)) − (CS− (R−NR)) (first half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(int_second, times, freqs, "FRN_INT_second",
                      "FRN: (CS+ (R−NR)) − (CS− (R−NR)) (second half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(int_coll, times, freqs, "FRN_INT_collapsed",
                      "FRN: (CS+ (R−NR)) − (CS− (R−NR)) (collapsed across halves)",
                      cluster_dir, mask_dir, **params)

    # CS+
    run_cluster_1samp(csplus_first, times, freqs, "FRN_CSplus_first",
                      "FRN: CS+ (R−NR) (first half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(csplus_second, times, freqs, "FRN_CSplus_second",
                      "FRN: CS+ (R−NR) (second half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(csplus_coll, times, freqs, "FRN_CSplus_collapsed",
                      "FRN: CS+ (R−NR) (collapsed across halves)",
                      cluster_dir, mask_dir, **params)

    # CS-
    run_cluster_1samp(csminus_first, times, freqs, "FRN_CSminus_first",
                      "FRN: CS− (R−NR) (first half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(csminus_second, times, freqs, "FRN_CSminus_second",
                      "FRN: CS− (R−NR) (second half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(csminus_coll, times, freqs, "FRN_CSminus_collapsed",
                      "FRN: CS− (R−NR) (collapsed across halves)",
                      cluster_dir, mask_dir, **params)

# ---------------------------------------------------------------------
# CNV RUNNER (4 files: CS± × half)
# ---------------------------------------------------------------------
CNV_CONDS = [
    "CSplus_first",
    "CSminus_first",
    "CSplus_second",
    "CSminus_second",
]

def run_cnv(subjects):
    cluster_dir = OUT_ROOT / "CNV"
    mask_dir = MASK_ROOT / "CNV"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    Xs, times, freqs, keep = tfr_array_from_files(subjects, "CNV", CNV_CONDS)
    Xs, subs = intersect_arrays(Xs, keep)
    print(f"[INFO] CNV: Complete subjects: {len(subs)}")
    if len(subs) == 0:
        return

    # indices:
    # 0 CS+ first
    # 1 CS- first
    # 2 CS+ second
    # 3 CS- second

    diff_first  = Xs[0] - Xs[1]   # CS+ - CS-
    diff_second = Xs[2] - Xs[3]
    diff_coll   = ((Xs[0] + Xs[2]) / 2) - ((Xs[1] + Xs[3]) / 2)

    # Also allow simple effects vs 0 (optional, but consistent)
    csplus_coll  = (Xs[0] + Xs[2]) / 2
    csminus_coll = (Xs[1] + Xs[3]) / 2

    params = dict(fmin=1.0, pad_start=0.1, pad_end=0.01, n_permutations=1000)

    # CS+ - CS-
    run_cluster_1samp(diff_first, times, freqs, "CNV_CSplus_minus_CSminus_first",
                      "CNV: CS+ − CS− (first half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(diff_second, times, freqs, "CNV_CSplus_minus_CSminus_second",
                      "CNV: CS+ − CS− (second half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(diff_coll, times, freqs, "CNV_CSplus_minus_CSminus_collapsed",
                      "CNV: CS+ − CS− (collapsed across halves)",
                      cluster_dir, mask_dir, **params)

    # CS+ and CS- vs 0 (if you do not want these, delete these calls)
    run_cluster_1samp(Xs[0], times, freqs, "CNV_CSplus_first",
                      "CNV: CS+ (first half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(Xs[2], times, freqs, "CNV_CSplus_second",
                      "CNV: CS+ (second half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(csplus_coll, times, freqs, "CNV_CSplus_collapsed",
                      "CNV: CS+ (collapsed across halves)",
                      cluster_dir, mask_dir, **params)

    run_cluster_1samp(Xs[1], times, freqs, "CNV_CSminus_first",
                      "CNV: CS− (first half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(Xs[3], times, freqs, "CNV_CSminus_second",
                      "CNV: CS− (second half)",
                      cluster_dir, mask_dir, **params)
    run_cluster_1samp(csminus_coll, times, freqs, "CNV_CSminus_collapsed",
                      "CNV: CS− (collapsed across halves)",
                      cluster_dir, mask_dir, **params)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    subjects = get_subjects()
    run_frn(subjects)
    run_cnv(subjects)

if __name__ == "__main__":
    main()

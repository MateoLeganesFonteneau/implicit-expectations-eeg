# Cluster_ERP_waveforms.py
#
# Cluster-based permutation tests on the ERP waveforms (mass-univariate over
# time), replacing the point-wise paired t-tests / significance bars.
#
# For each component the per-subject ROI-mean difference wave is tested with
# mne.stats.permutation_cluster_1samp_test over the POST-ONSET window (0..end):
#
#   P3   : CS+ - CS-                  (CPz, Pz)         0..500 ms
#   CNV  : CS+ - CS-                  (FCz, Cz)         0..1000 ms  (covers SPN too)
#   FRN  : Reward - No-reward         (Fz,  FCz)        0..500 ms
#          tested separately for CS+ and CS-
#
# Half factor (per-half + interaction):
#   - "first"  : contrast within the first half
#   - "second" : contrast within the second half
#   - "interaction": (contrast)_second - (contrast)_first   (difference of differences)
#
# Subjects are aligned by ID; only subjects present in every cell of a given
# test are included. Exclusion list matches the 44-subject ERP/TF analyses.
#
# Saves: erp_clusters.pkl
#   {
#     "clusters": {
#        "P3":  {"first":[(t0,t1,p),...], "second":[...], "interaction":[...]},
#        "CNV": {"first":[...], "second":[...], "interaction":[...]},
#        "FRN": {"first":  {"CS+":[...], "CS-":[...]},
#                "second": {"CS+":[...], "CS-":[...]},
#                "interaction": {"CS+":[...], "CS-":[...]}},
#     },
#     "times": {"P3":..., "CNV":..., "FRN":...},   # ms, full epoch
#     "win":   {"P3":(0,500), ...},
#     "n":     {...},                              # subjects per test
#   }

from pathlib import Path
import pickle
import re
import numpy as np
import mne
from mne.stats import permutation_cluster_1samp_test

from config import EPOCHS_DIR, EXCLUDED_SUBJECTS, OUTPUT_DIR

mne.set_log_level("WARNING")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = EPOCHS_DIR
OUT_DIR = OUTPUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_SUBJECTS = EXCLUDED_SUBJECTS

HALVES = ["first", "second"]
N_PERM = 1000
SEED   = 42
ALPHA  = 0.05

# Two-component contrasts (CS+ - CS-)
PAIR_DEF = {
    "P3":  dict(chs=["CPz", "Pz"], plus="CS+ presentation",
                minus="CS- presentation", win=(0, 500),  baseline=None),
    "CNV": dict(chs=["FCz", "Cz"], plus="CNV+ fixation",
                minus="CNV- fixation",    win=(0, 1000), baseline=(-200, 0)),
}

# FRN: Reward - No-reward, per cue type.
# Outcome mapping (from the Fig.5 caption):
#   CS+ Expected   = reward     CS+ Unexpected = no-reward
#   CS- Unexpected = reward     CS- Expected   = no-reward
# pairs are (reward_label, noreward_label) so pair_diff gives Reward - No-reward.
FRN_DEF = dict(
    chs=["Fz", "FCz"], win=(0, 500), baseline=None,
    pairs={
        "CS+": ("FRN+ Expected",   "FRN+ Unexpected"),   # reward, no-reward
        "CS-": ("FRN- Unexpected", "FRN- Expected"),     # reward, no-reward
    },
)


# ── SUBJECT DISCOVERY ─────────────────────────────────────────────────────────
def get_subjects():
    pat = re.compile(r"^sub-(\d+)_")
    subs = {pat.match(f.name).group(1)
            for f in DATA_DIR.glob("sub-*-epo.fif")
            if pat.match(f.name)}
    subs = sorted(subs - EXCLUDE_SUBJECTS, key=int)
    print(f"[INFO] {len(subs)} subjects after exclusion")
    return subs


# ── LOAD ONE SUBJECT ROI-MEAN WAVE ────────────────────────────────────────────
def load_wave(subj, label, half, channels, baseline_ms=None):
    """Return (times_ms, roi_mean_uV) or (None, None)."""
    fname = DATA_DIR / f"sub-{subj}_{label}_{half}-epo.fif"
    if not fname.exists():
        return None, None
    try:
        epo = mne.read_epochs(str(fname), preload=True, verbose=False)
    except Exception as e:
        print(f"[WARN] {fname.name}: {e}")
        return None, None

    evoked = epo.average()
    picks  = mne.pick_channels(evoked.ch_names, include=channels, ordered=False)
    if len(picks) == 0:
        return None, None

    roi = (evoked.data[picks, :] * 1e6).mean(axis=0)   # µV, (n_times,)
    t_ms = evoked.times * 1000.0

    if baseline_ms is not None:
        b0, b1 = baseline_ms
        m = (t_ms >= b0) & (t_ms <= b1)
        if np.any(m):
            roi = roi - roi[m].mean()

    return t_ms, roi


def load_component_waves(subjects, labels, channels, baseline_ms=None):
    """waves[half][label][subj] = 1d array; plus shared times."""
    waves = {h: {l: {} for l in labels} for h in HALVES}
    times = None
    for subj in subjects:
        for half in HALVES:
            for label in labels:
                t_ms, roi = load_wave(subj, label, half, channels, baseline_ms)
                if roi is None:
                    continue
                if times is None:
                    times = t_ms
                waves[half][label][subj] = roi
    return waves, times


# ── CLUSTER TEST (1D over time) ───────────────────────────────────────────────
def cluster_1d(diff_mat, times, win):
    """diff_mat: (n_subj, n_times). Returns list of (t0_ms, t1_ms, p) for p<ALPHA."""
    if diff_mat.shape[0] < 3:
        return []
    mask = (times >= win[0]) & (times <= win[1])
    X = diff_mat[:, mask]
    t = times[mask]

    T_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        X, n_permutations=N_PERM, tail=0, seed=SEED,
        out_type="indices", verbose=False,
    )
    out = []
    for cl, p in zip(clusters, cluster_pv):
        if p < ALPHA:
            # out_type="indices" -> cluster is a tuple of index arrays (here 1D)
            idx = cl[0] if isinstance(cl, tuple) else cl
            idx = np.atleast_1d(np.asarray(idx)).ravel()
            if idx.size == 0:
                continue
            out.append((float(t[idx.min()]), float(t[idx.max()]), float(p)))
    if len(cluster_pv):
        print(f"      [diagnostic: min candidate cluster p = {np.min(cluster_pv):.3f} "
              f"of {len(cluster_pv)} candidate cluster(s)]")
    return sorted(out, key=lambda r: r[0])


# ── DIFFERENCE-WAVE BUILDERS (subject-aligned) ────────────────────────────────
def pair_diff(waves, half, plus, minus):
    """(contrast)_half per subject -> (n_subj, n_times), and subject list."""
    subs = sorted(set(waves[half][plus]) & set(waves[half][minus]), key=int)
    if not subs:
        return np.empty((0,)), subs
    D = np.vstack([waves[half][plus][s] - waves[half][minus][s] for s in subs])
    return D, subs


def interaction_diff(waves, plus, minus):
    """(contrast)_second - (contrast)_first per subject; subjects in all 4 cells."""
    subs = sorted(
        set(waves["first"][plus]) & set(waves["first"][minus]) &
        set(waves["second"][plus]) & set(waves["second"][minus]),
        key=int,
    )
    if not subs:
        return np.empty((0,)), subs
    D = np.vstack([
        (waves["second"][plus][s] - waves["second"][minus][s]) -
        (waves["first"][plus][s]  - waves["first"][minus][s])
        for s in subs
    ])
    return D, subs


def collapsed_diff(waves, plus, minus):
    """Contrast averaged across both halves per subject; subjects in all 4 cells."""
    subs = sorted(
        set(waves["first"][plus]) & set(waves["first"][minus]) &
        set(waves["second"][plus]) & set(waves["second"][minus]),
        key=int,
    )
    if not subs:
        return np.empty((0,)), subs
    D = np.vstack([
        ((waves["second"][plus][s] - waves["second"][minus][s]) +
         (waves["first"][plus][s]  - waves["first"][minus][s])) / 2.0
        for s in subs
    ])
    return D, subs


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run():
    subjects = get_subjects()
    clusters = {}
    times_out = {}
    win_out = {}
    n_out = {}

    # ---- P3 and CNV (CS+ - CS-) ----
    for comp, d in PAIR_DEF.items():
        labels = [d["plus"], d["minus"]]
        waves, times = load_component_waves(subjects, labels, d["chs"], d["baseline"])
        times_out[comp] = times
        win_out[comp] = d["win"]
        clusters[comp] = {}
        n_out[comp] = {}

        for half in HALVES:
            D, subs = pair_diff(waves, half, d["plus"], d["minus"])
            clusters[comp][half] = cluster_1d(D, times, d["win"]) if D.size else []
            n_out[comp][half] = len(subs)
            print(f"[{comp}] {half}: N={len(subs)} -> "
                  f"{len(clusters[comp][half])} cluster(s) "
                  f"{clusters[comp][half]}")

        Dint, subs_i = interaction_diff(waves, d["plus"], d["minus"])
        clusters[comp]["interaction"] = cluster_1d(Dint, times, d["win"]) if Dint.size else []
        n_out[comp]["interaction"] = len(subs_i)
        print(f"[{comp}] interaction: N={len(subs_i)} -> "
              f"{clusters[comp]['interaction']}")

        Dcol, subs_c = collapsed_diff(waves, d["plus"], d["minus"])
        clusters[comp]["collapsed"] = cluster_1d(Dcol, times, d["win"]) if Dcol.size else []
        n_out[comp]["collapsed"] = len(subs_c)
        print(f"[{comp}] collapsed (both halves): N={len(subs_c)} -> "
              f"{clusters[comp]['collapsed']}")

    # ---- FRN (Expected - Unexpected, per cue type) ----
    all_frn_labels = [lab for pair in FRN_DEF["pairs"].values() for lab in pair]
    waves, times = load_component_waves(subjects, all_frn_labels,
                                        FRN_DEF["chs"], FRN_DEF["baseline"])
    times_out["FRN"] = times
    win_out["FRN"] = FRN_DEF["win"]
    clusters["FRN"] = {"first": {}, "second": {}, "interaction": {}, "collapsed": {}}
    n_out["FRN"] = {"first": {}, "second": {}, "interaction": {}, "collapsed": {}}

    for cue, (lab_r, lab_nr) in FRN_DEF["pairs"].items():
        for half in HALVES:
            D, subs = pair_diff(waves, half, lab_r, lab_nr)
            clusters["FRN"][half][cue] = cluster_1d(D, times, FRN_DEF["win"]) if D.size else []
            n_out["FRN"][half][cue] = len(subs)
            print(f"[FRN {cue} R-NR] {half}: N={len(subs)} -> {clusters['FRN'][half][cue]}")

        Dint, subs_i = interaction_diff(waves, lab_r, lab_nr)
        clusters["FRN"]["interaction"][cue] = (
            cluster_1d(Dint, times, FRN_DEF["win"]) if Dint.size else [])
        n_out["FRN"]["interaction"][cue] = len(subs_i)
        print(f"[FRN {cue} R-NR] interaction: N={len(subs_i)} -> "
              f"{clusters['FRN']['interaction'][cue]}")

        Dcol, subs_c = collapsed_diff(waves, lab_r, lab_nr)
        clusters["FRN"]["collapsed"][cue] = (
            cluster_1d(Dcol, times, FRN_DEF["win"]) if Dcol.size else [])
        n_out["FRN"]["collapsed"][cue] = len(subs_c)
        print(f"[FRN {cue} R-NR] collapsed (both halves): N={len(subs_c)} -> "
              f"{clusters['FRN']['collapsed'][cue]}")

    out = {"clusters": clusters, "times": times_out, "win": win_out, "n": n_out}
    path = OUT_DIR / "erp_clusters.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"\n[INFO] Saved: {path}")


if __name__ == "__main__":
    run()

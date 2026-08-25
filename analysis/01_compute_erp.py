# Compute_ERP_grand_averages.py
#
# Loads single-trial epochs (.fif), averages per subject, then computes
# grand-average ERPs (mean ± SEM across subjects) and point-wise paired
# t-tests for P3, CNV/SPN, and FRN.
#
# Saves:
#   grand_erp_summary.pkl  – grand-average waveforms
#   grand_erp_stats.pkl    – point-wise t-test results
#
# Participant selection matches the final ERP and time-frequency analyses.

from pathlib import Path
import pickle
import re
import numpy as np
import pandas as pd
import mne
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

from config import EPOCHS_DIR, EXCLUDED_SUBJECTS, OUTPUT_DIR

mne.set_log_level("WARNING")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = EPOCHS_DIR
OUT_DIR = OUTPUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_SUBJECTS = EXCLUDED_SUBJECTS

# Channels averaged per component
PANEL_CHS = {
    "P3":  ["CPz", "Pz"],
    "CNV": ["FCz", "Cz"],
    "FRN": ["Fz",  "FCz"],
}

# Condition labels per component
COMP_LABELS = {
    "P3":  ["CS- presentation", "CS+ presentation"],
    "CNV": ["CNV- fixation",    "CNV+ fixation"],
    "FRN": ["FRN+ Expected", "FRN+ Unexpected",
            "FRN- Expected", "FRN- Unexpected"],
}

HALVES = ["first", "second"]

# Mean-amplitude windows for RM ANOVAs (ms)
ANOVA_WINDOWS = {
    "P3":  (200, 500),
    "CNV": (300, 700),
    "SPN": (900, 1000),   # same epochs as CNV
}


# ── SUBJECT DISCOVERY ─────────────────────────────────────────────────────────
def get_subjects():
    pat = re.compile(r"^sub-(\d+)_")
    subs = {pat.match(f.name).group(1)
            for f in DATA_DIR.glob("sub-*-epo.fif")
            if pat.match(f.name)}
    subs = sorted(subs - EXCLUDE_SUBJECTS, key=int)
    print(f"[INFO] {len(subs)} subjects after exclusion")
    return subs


# ── LOAD ONE EVOKED (mean amplitude at ROI channels per subject) ───────────────
def load_subject_evoked(subj, label, half, channels):
    """
    Load epoch file, average across trials, pick channels, return
    (times_ms, data_1d) where data_1d is mean across the ROI channels.
    Returns (None, None) if file missing.
    """
    fname = DATA_DIR / f"sub-{subj}_{label}_{half}-epo.fif"
    if not fname.exists():
        return None, None
    try:
        epo = mne.read_epochs(str(fname), preload=True, verbose=False)
    except Exception as e:
        print(f"[WARN] Could not load {fname.name}: {e}")
        return None, None

    evoked  = epo.average()
    picks   = mne.pick_channels(evoked.ch_names, include=channels,
                                ordered=False)
    if len(picks) == 0:
        print(f"[WARN] Channels {channels} not found in {fname.name}")
        return None, None

    data_uv = evoked.data[picks, :] * 1e6        # V → µV
    roi_mean = data_uv.mean(axis=0)               # (n_times,)
    times_ms = evoked.times * 1000                # s → ms
    return times_ms, roi_mean


# ── BUILD PER-SUBJECT ARRAYS ──────────────────────────────────────────────────
def build_arrays(subjects):
    """
    Returns:
      arrays[component][half][label] = np.array shape (n_valid_subjects, n_times)
      times[component]               = np.array (ms)
    """
    arrays = {}
    times  = {}

    for comp, labels in COMP_LABELS.items():
        channels = PANEL_CHS[comp]
        arrays[comp] = {h: {l: [] for l in labels} for h in HALVES}
        times[comp]  = None

        for subj in subjects:
            for half in HALVES:
                for label in labels:
                    t_ms, roi = load_subject_evoked(subj, label, half, channels)
                    if roi is None:
                        continue
                    if times[comp] is None:
                        times[comp] = t_ms
                    arrays[comp][half][label].append(roi)

        # Convert to arrays
        for half in HALVES:
            for label in labels:
                lst = arrays[comp][half][label]
                arrays[comp][half][label] = (np.vstack(lst)
                                             if lst else np.empty((0,)))

    return arrays, times


# ── GRAND AVERAGES ────────────────────────────────────────────────────────────
def compute_grand_averages(arrays, times_dict):
    """
    Returns summary["data"][comp][half][label] = DataFrame(t_ms, mean, sem)
    """
    data = {}
    for comp, labels in COMP_LABELS.items():
        t_ms = times_dict[comp]
        data[comp] = {}
        for half in HALVES:
            data[comp][half] = {}
            for label in labels:
                X = arrays[comp][half][label]   # (n_subs, n_times)
                if X.ndim < 2 or X.shape[0] == 0:
                    data[comp][half][label] = pd.DataFrame(
                        {"t_ms": t_ms if t_ms is not None else [],
                         "mean": np.nan, "sem": np.nan})
                    continue
                mean = X.mean(axis=0)
                sem  = X.std(axis=0, ddof=1) / np.sqrt(X.shape[0])
                data[comp][half][label] = pd.DataFrame(
                    {"t_ms": t_ms, "mean": mean, "sem": sem})
                print(f"  {comp} {half} '{label}': N={X.shape[0]}")
    return data


# ── POINT-WISE PAIRED T-TESTS ─────────────────────────────────────────────────
def run_stats(arrays):
    """
    Returns stats[comp][half][contrast] = {"t_ms": ..., "p": ..., "t_stat": ...}
    Only includes subjects present in BOTH conditions being contrasted.
    """
    stats = {}

    for comp in COMP_LABELS:
        stats[comp] = {}
        for half in HALVES:
            stats[comp][half] = {}
            d = arrays[comp][half]

            # ── P3 & CNV: CS+ vs CS− ────────────────────────────────────────
            if comp in ("P3", "CNV"):
                label_p = [l for l in COMP_LABELS[comp] if "+" in l][0]
                label_m = [l for l in COMP_LABELS[comp] if "-" in l][0]
                Xp = d[label_p]
                Xm = d[label_m]
                n  = min(len(Xp), len(Xm))
                if n >= 3:
                    t_stat, p = ttest_rel(Xp[:n], Xm[:n], axis=0)
                    stats[comp][half]["CS+ vs CS-"] = {
                        "t_ms":   None,   # filled below per-component
                        "p":      p,
                        "t_stat": t_stat,
                    }

            # ── FRN: CS+ Expected vs Unexpected ─────────────────────────────
            if comp == "FRN":
                for prefix, label_E, label_U in [
                    ("CS+", "FRN+ Expected", "FRN+ Unexpected"),
                    ("CS-", "FRN- Expected", "FRN- Unexpected"),
                ]:
                    XE = d[label_E]
                    XU = d[label_U]
                    n  = min(len(XE), len(XU))
                    if n >= 3:
                        t_stat, p = ttest_rel(XE[:n], XU[:n], axis=0)
                        key = f"{prefix} Unexp vs Exp"
                        stats[comp][half][key] = {
                            "t_ms":   None,
                            "p":      p,
                            "t_stat": t_stat,
                        }

    return stats


def attach_times(stats, times_dict):
    """Fill in the t_ms arrays now that we have them."""
    for comp in stats:
        t_ms = times_dict.get(comp)
        if t_ms is None:
            continue
        for half in stats[comp]:
            for contrast in stats[comp][half]:
                stats[comp][half][contrast]["t_ms"] = t_ms
    return stats


# ── RM ANOVAs ON MEAN WINDOW AMPLITUDES ──────────────────────────────────────
def mean_in_window(arr_1d, times_ms, win):
    """Mean amplitude of arr_1d within win=(t0,t1) ms."""
    mask = (times_ms >= win[0]) & (times_ms <= win[1])
    return arr_1d[mask].mean()


def run_anovas(arrays, times_dict, subjects):
    """
    P3  : CS Type × Half  (2×2 RM ANOVA on mean 200–500 ms)
    CNV : CS Type × Half  (2×2 RM ANOVA on mean 300–700 ms)
    SPN : CS Type × Half  (2×2 RM ANOVA on mean 900–1000 ms, same CNV epochs)
    """
    print("\n── RM ANOVAs ─────────────────────────────────────────────────────")

    results = {}
    anova_tables = []

    for comp_key, (comp_epoch, win) in {
        "P3":  ("P3",  ANOVA_WINDOWS["P3"]),
        "CNV": ("CNV", ANOVA_WINDOWS["CNV"]),
        "SPN": ("CNV", ANOVA_WINDOWS["SPN"]),   # SPN uses CNV epochs
    }.items():
        t_ms = times_dict[comp_epoch]
        if t_ms is None:
            print(f"[WARN] No time axis for {comp_key}, skipping ANOVA.")
            continue

        label_p = [l for l in COMP_LABELS[comp_epoch] if "+" in l][0]
        label_m = [l for l in COMP_LABELS[comp_epoch] if "-" in l][0]

        rows = []
        for hi, half in enumerate(HALVES):
            Xp = arrays[comp_epoch][half][label_p]
            Xm = arrays[comp_epoch][half][label_m]
            n  = min(len(Xp), len(Xm))
            for i in range(n):
                rows.append({
                    "subject":  i,
                    "half":     half,
                    "cs_type":  "CS+",
                    "amp":      mean_in_window(Xp[i], t_ms, win),
                })
                rows.append({
                    "subject":  i,
                    "half":     half,
                    "cs_type":  "CS-",
                    "amp":      mean_in_window(Xm[i], t_ms, win),
                })

        df = pd.DataFrame(rows)
        if df.empty or df["subject"].nunique() < 3:
            print(f"[WARN] Not enough data for {comp_key} ANOVA.")
            continue

        try:
            aovrm = AnovaRM(df, depvar="amp", subject="subject",
                            within=["cs_type", "half"])
            res   = aovrm.fit()
            print(f"\n{comp_key} RM ANOVA  (window {win[0]}–{win[1]} ms, "
                  f"N={df['subject'].nunique()}):")
            print(res.summary())
            results[comp_key] = res

            table = res.anova_table.reset_index()
            table.columns = ["effect", "F", "df_num", "df_den", "p"]
            table.insert(0, "component", comp_key)
            table.insert(1, "window_start_ms", win[0])
            table.insert(2, "window_end_ms", win[1])
            table["partial_eta_squared"] = (
                table["F"] * table["df_num"]
                / (table["F"] * table["df_num"] + table["df_den"])
            )
            anova_tables.append(table)
        except Exception as e:
            print(f"[ERROR] {comp_key} ANOVA failed: {e}")

    # ── Wide-format CSVs (updated with correct exclusions) ──────────────────
    print("\n── Saving wide-format CSVs ────────────────────────────────────")
    for comp_key, (comp_epoch, win) in {
        "P3":  ("P3",  ANOVA_WINDOWS["P3"]),
        "SPN": ("CNV", ANOVA_WINDOWS["SPN"]),
    }.items():
        t_ms   = times_dict[comp_epoch]
        label_p = [l for l in COMP_LABELS[comp_epoch] if "+" in l][0]
        label_m = [l for l in COMP_LABELS[comp_epoch] if "-" in l][0]

        rows = []
        for hi, half in enumerate(HALVES):
            Xp = arrays[comp_epoch][half][label_p]
            Xm = arrays[comp_epoch][half][label_m]
            n  = min(len(Xp), len(Xm))
            for i in range(n):
                rows.append({
                    "subject":       subjects[i] if i < len(subjects) else i,
                    f"{comp_key}_{half}_CS+": mean_in_window(Xp[i], t_ms, win),
                    f"{comp_key}_{half}_CS-": mean_in_window(Xm[i], t_ms, win),
                })

        # Merge halves on subject
        df_first  = pd.DataFrame([r for r in rows
                                   if f"{comp_key}_first_CS+" in r])
        df_second = pd.DataFrame([r for r in rows
                                   if f"{comp_key}_second_CS+" in r])
        if not df_first.empty and not df_second.empty:
            df_wide = pd.merge(df_first, df_second, on="subject")
        elif not df_first.empty:
            df_wide = df_first
        else:
            df_wide = df_second

        out = OUT_DIR / f"df_{comp_key}_MEAN_WIDE.csv"
        df_wide.to_csv(out, index=False)
        print(f"  Saved: {out}  (N={len(df_wide)})")

    if anova_tables:
        anova_path = OUT_DIR / "ERP_window_ANOVAs.csv"
        pd.concat(anova_tables, ignore_index=True).to_csv(anova_path, index=False)
        print(f"  Saved: {anova_path}")

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run():
    subjects = get_subjects()

    print("\n[STEP 1] Loading epochs and building per-subject arrays ...")
    arrays, times_dict = build_arrays(subjects)

    print("\n[STEP 2] Computing grand averages ...")
    grand_data = compute_grand_averages(arrays, times_dict)

    print("\n[STEP 3] Running point-wise t-tests ...")
    stats = run_stats(arrays)
    stats = attach_times(stats, times_dict)

    print("\n[STEP 4] Running RM ANOVAs on mean window amplitudes ...")
    run_anovas(arrays, times_dict, subjects)

    # ── Save ─────────────────────────────────────────────────────────────────
    summary = {
        "data":           grand_data,
        "subjects":       subjects,
        "n_participants": len(subjects),
        "exclude":        sorted(EXCLUDE_SUBJECTS),
    }

    summary_path = OUT_DIR / "grand_erp_summary.pkl"
    stats_path   = OUT_DIR / "grand_erp_stats.pkl"

    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)
    print(f"\n[INFO] Saved: {summary_path}")

    with open(stats_path, "wb") as f:
        pickle.dump({"stats": stats}, f)
    print(f"[INFO] Saved: {stats_path}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n── Significance summary (uncorrected p < .05) ──")
    for comp in stats:
        for half in HALVES:
            for contrast, res in stats[comp][half].items():
                p = res["p"]
                if p is not None and len(p) > 0:
                    n_sig = np.sum(p < 0.05)
                    print(f"  {comp} {half:6s} '{contrast}': "
                          f"{n_sig}/{len(p)} timepoints p<.05")


if __name__ == "__main__":
    run()

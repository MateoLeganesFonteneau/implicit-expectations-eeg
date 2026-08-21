# FRN_reward_vs_noreward.py
#
# Grand-average FRN analysis grouped by OUTCOME (reward / no-reward)
# and contrasted by CS type (CS+ vs CS-) within each outcome.
#
# Condition re-mapping from epoched files:
#   FRN+ Expected   → CS+ Reward      (CS+ received the expected reward)
#   FRN- Unexpected → CS- Reward      (CS- received the unexpected reward)
#   FRN+ Unexpected → CS+ No Reward   (CS+ did not receive the expected reward)
#   FRN- Expected   → CS- No Reward   (CS- received the expected no-reward)
#
# Two contrasts, split by first / second half:
#   1. CS+ Reward vs CS- Reward
#   2. CS+ No Reward vs CS- No Reward
#
# Outputs:
#   Figure 1 : grand-average ERPs (2 rows × 2 cols)
#   Figure 2 : difference waves CS+ − CS− per outcome (1 row × 2 cols)
#   CSV      : mean amplitude in FRN window, t, p, Cohen's d per contrast × half
#
# Channels : Fz, FCz
# Stats    : pointwise paired t-test + BH-FDR  |  windowed t-test (200–300 ms)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from scipy import stats
from statsmodels.stats.multitest import multipletests
import mne

from config import EXCLUDED_SUBJECTS, FRN_EPOCHS_DIR, OUTPUT_DIR

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
ANALYSIS_DIR = FRN_EPOCHS_DIR
OUT_DIR = OUTPUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRN_CHANNELS    = ["Fz", "FCz"]
HALVES          = ["first", "second"]

EXCLUDE_SUBJECTS = EXCLUDED_SUBJECTS
FRN_WINDOW      = (200, 300)    # component window for windowed stats & shading
FRN_WINDOW_SIG  = (200, 500)    # span for pointwise FDR correction
X_FRN           = (-300, 500)
X_DIFF          = (100, 400)    # tighter x range for difference wave plot

ALPHA_SEM = 0.25
ALPHA_SIG = 0.05

# Two contrasts: (contrast name, CS+ file label, CS- file label)
CONTRASTS = [
    ("Reward",    "FRN+ Expected",   "FRN- Unexpected"),
    ("No Reward", "FRN+ Unexpected", "FRN- Expected"),
]

# Colors for grand-average ERP figure
COLOR_ERP = {
    ("Reward",    "plus"):  "#1f77b4",   # dark blue
    ("Reward",    "minus"): "#d62728",   # dark red
    ("No Reward", "plus"):  "#9ecae1",   # light blue
    ("No Reward", "minus"): "#f4a582",   # light red
}

# Colors for difference wave figure (one line per contrast)
COLOR_DIFF = {
    "Reward":    "#1f77b4",   # blue
    "No Reward": "#d62728",   # red
}

# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
def find_subjects(analysis_dir):
    fifs = list(analysis_dir.glob("sub-*_FRN+*-epo.fif"))
    all_subs = sorted({f.name.split("_")[0] for f in fifs})
    return [s for s in all_subs if s.replace("sub-", "") not in EXCLUDE_SUBJECTS]


def load_subject_erp(subj, file_label, half, analysis_dir, channels):
    """
    Try both naming conventions:
      Old (1001-1034): sub-XXXX_FRN+ Expected_first-epo.fif
      New (1035+):     sub-XXXX_FRN+_Expected_first-half-epo.fif
    Returns (times_ms, data_1d) or (None, None).
    """
    ul = file_label.replace(" ", "_")
    candidates = [
        analysis_dir / f"{subj}_{file_label}_{half}-epo.fif",
        analysis_dir / f"{subj}_{ul}_{half}-half-epo.fif",
    ]
    fname = next((p for p in candidates if p.exists()), None)
    if fname is None:
        return None, None
    try:
        epochs = mne.read_epochs(str(fname), preload=True, verbose=False)
    except Exception:
        return None, None
    available = [ch for ch in channels if ch in epochs.ch_names]
    if not available or len(epochs) == 0:
        return None, None
    epochs.pick(available)
    evoked = epochs.average()
    return evoked.times * 1000.0, evoked.data.mean(axis=0) * 1e6   # V → µV


def collect_contrast_data(subjects, half, file_plus, file_minus, analysis_dir, channels):
    """
    Single pass: load CS+ and CS- ERPs for every subject that has both.
    Returns: times_ms, A (n×T), B (n×T), valid_subjects
    """
    A, B, valid = [], [], []
    times_ms = None

    for subj in subjects:
        tp, dp = load_subject_erp(subj, file_plus,  half, analysis_dir, channels)
        tm, dm = load_subject_erp(subj, file_minus, half, analysis_dir, channels)
        if dp is None or dm is None:
            continue
        if times_ms is None:
            times_ms = tp
        # align if lengths differ
        if len(dp) != len(times_ms):
            dp = np.interp(times_ms, tp, dp)
        if len(dm) != len(times_ms):
            dm = np.interp(times_ms, tm, dm)
        A.append(dp)
        B.append(dm)
        valid.append(subj)

    if not valid:
        return None, None, None, []
    return times_ms, np.array(A), np.array(B), valid


# --------------------------------------------------------------------------- #
# STATS HELPERS
# --------------------------------------------------------------------------- #
def pointwise_fdr_ttest(A, B, times_ms, window):
    """Paired t-test at every time point; FDR corrected within window."""
    t_vals, p_vals = stats.ttest_rel(A, B, axis=0)
    p_corr = p_vals.copy()
    mask = (times_ms >= window[0]) & (times_ms <= window[1])
    if mask.any():
        _, p_fdr, _, _ = multipletests(p_vals[mask], method="fdr_bh")
        p_corr[mask] = p_fdr
    return t_vals, p_corr


def build_anova_df(ga_data, times_ref, window):
    """
    Build a long-format DataFrame for repeated-measures ANOVA.
    Factors: CS_type (CS+/CS-) × Outcome (Reward/No Reward) × Half (first/second)
    DV: mean amplitude in window (µV).
    Only subjects present in ALL 4 condition × half combinations are included.
    """
    win_mask = (times_ref >= window[0]) & (times_ref <= window[1])

    rows = []
    for half in HALVES:
        t_r,  A_r,  B_r,  subs_r  = ga_data["Reward"][half]
        t_nr, A_nr, B_nr, subs_nr = ga_data["No Reward"][half]

        # subjects present in both contrasts for this half
        common = sorted(set(subs_r) & set(subs_nr))

        idx_r  = {s: i for i, s in enumerate(subs_r)}
        idx_nr = {s: i for i, s in enumerate(subs_nr)}

        for subj in common:
            ir  = idx_r[subj]
            inr = idx_nr[subj]
            rows.append({
                "subject":  subj,
                "half":     half,
                "outcome":  "Reward",
                "cs_type":  "CS+",
                "amp":      float(A_r[ir,  win_mask].mean()),
            })
            rows.append({
                "subject":  subj,
                "half":     half,
                "outcome":  "Reward",
                "cs_type":  "CS-",
                "amp":      float(B_r[ir,  win_mask].mean()),
            })
            rows.append({
                "subject":  subj,
                "half":     half,
                "outcome":  "No Reward",
                "cs_type":  "CS+",
                "amp":      float(A_nr[inr, win_mask].mean()),
            })
            rows.append({
                "subject":  subj,
                "half":     half,
                "outcome":  "No Reward",
                "cs_type":  "CS-",
                "amp":      float(B_nr[inr, win_mask].mean()),
            })

    return pd.DataFrame(rows)


def run_posthoc(ga_data, times_ref, window, out_dir):
    """
    Post-hoc one-sample t-tests on the CS+ − CS− difference score vs zero.

    Follows up the significant CS × Half interaction:
      - Difference score collapsed across outcomes, per half (+ both halves collapsed)

    Also reports cells for CS × Outcome (per half + collapsed) for completeness.

    Bonferroni correction applied separately within each follow-up family.
    """
    win_mask = (times_ref >= window[0]) & (times_ref <= window[1])

    # ---- helper: get per-subject diff scores for one contrast × half -------- #
    def get_diff(cname, half):
        t, A, B, valid = ga_data[cname][half]
        if A is None:
            return None, None
        return valid, (A - B)[:, win_mask].mean(axis=1)

    # ---- 1. Follow-up CS × Half: collapse across outcomes ------------------- #
    # For each subject and half: average diff across both outcomes
    rows_cs_half = []
    half_data = {}   # half → (subjects, diff_array collapsed across outcomes)

    for half in HALVES:
        by_outcome = {}
        for cname, _, _ in CONTRASTS:
            subs, diff = get_diff(cname, half)
            if subs is not None:
                by_outcome[cname] = {s: d for s, d in zip(subs, diff)}

        if len(by_outcome) < 2:
            continue
        common = sorted(set(by_outcome[CONTRASTS[0][0]]) & set(by_outcome[CONTRASTS[1][0]]))
        diff_avg = np.array([(by_outcome[CONTRASTS[0][0]][s] +
                              by_outcome[CONTRASTS[1][0]][s]) / 2
                             for s in common])
        half_data[half] = (common, diff_avg)
        t_val, p_raw = stats.ttest_1samp(diff_avg, 0)
        rows_cs_half.append({
            "contrast": "CS type (collapsed across outcomes)",
            "half": half, "n": len(common),
            "mean_diff": float(diff_avg.mean()),
            "sd_diff":   float(diff_avg.std(ddof=1)),
            "t": float(t_val), "p_raw": float(p_raw),
        })

    # collapsed across both halves
    if "first" in half_data and "second" in half_data:
        subs_f, diff_f = half_data["first"]
        subs_s, diff_s = half_data["second"]
        common = sorted(set(subs_f) & set(subs_s))
        idx_f = {s: i for i, s in enumerate(subs_f)}
        idx_s = {s: i for i, s in enumerate(subs_s)}
        diff_col = np.array([(diff_f[idx_f[s]] + diff_s[idx_s[s]]) / 2 for s in common])
        t_val, p_raw = stats.ttest_1samp(diff_col, 0)
        rows_cs_half.insert(0, {
            "contrast": "CS type (collapsed across outcomes)",
            "half": "collapsed", "n": len(common),
            "mean_diff": float(diff_col.mean()),
            "sd_diff":   float(diff_col.std(ddof=1)),
            "t": float(t_val), "p_raw": float(p_raw),
        })

    ph1 = pd.DataFrame(rows_cs_half)
    _, p_bonf, _, _ = multipletests(ph1["p_raw"], method="bonferroni")
    ph1["p_bonf"] = p_bonf

    # ---- 2. CS × Outcome cells: per outcome × half + collapsed -------------- #
    rows_cs_out = []
    for cname, _, _ in CONTRASTS:
        out_data = {}
        for half in HALVES:
            subs, diff = get_diff(cname, half)
            if subs is not None:
                out_data[half] = (subs, diff)

        # collapsed
        if "first" in out_data and "second" in out_data:
            subs_f, diff_f = out_data["first"]
            subs_s, diff_s = out_data["second"]
            common = sorted(set(subs_f) & set(subs_s))
            idx_f = {s: i for i, s in enumerate(subs_f)}
            idx_s = {s: i for i, s in enumerate(subs_s)}
            diff_col = np.array([(diff_f[idx_f[s]] + diff_s[idx_s[s]]) / 2
                                 for s in common])
            t_val, p_raw = stats.ttest_1samp(diff_col, 0)
            rows_cs_out.append({
                "contrast": cname, "half": "collapsed", "n": len(common),
                "mean_diff": float(diff_col.mean()),
                "sd_diff":   float(diff_col.std(ddof=1)),
                "t": float(t_val), "p_raw": float(p_raw),
            })

        for half in HALVES:
            if half not in out_data:
                continue
            subs, diff = out_data[half]
            t_val, p_raw = stats.ttest_1samp(diff, 0)
            rows_cs_out.append({
                "contrast": cname, "half": half, "n": len(diff),
                "mean_diff": float(diff.mean()),
                "sd_diff":   float(diff.std(ddof=1)),
                "t": float(t_val), "p_raw": float(p_raw),
            })

    ph2 = pd.DataFrame(rows_cs_out)
    _, p_bonf, _, _ = multipletests(ph2["p_raw"], method="bonferroni")
    ph2["p_bonf"] = p_bonf

    # ---- combine and save --------------------------------------------------- #
    ph_all = pd.concat([ph1, ph2], ignore_index=True)
    for col in ["mean_diff", "sd_diff", "t", "p_raw", "p_bonf"]:
        ph_all[col] = ph_all[col].round(4)

    print("\n[POST-HOC] Follow-up: CS × Half (collapsed across outcomes)")
    print(ph1[["half","n","mean_diff","sd_diff","t","p_raw","p_bonf"]].round(4).to_string(index=False))
    print("\n[POST-HOC] CS × Outcome cells (for completeness)")
    print(ph2[["contrast","half","n","mean_diff","sd_diff","t","p_raw","p_bonf"]].round(4).to_string(index=False))

    path = out_dir / "FRN_reward_noreward_posthoc.csv"
    ph_all.to_csv(path, index=False)
    print(f"\n[INFO] Post-hoc saved: {path}")
    return ph_all


def run_anova(ga_data, times_ref, window, out_dir):
    """
    3-way repeated-measures ANOVA: CS_type × Outcome × Half (statsmodels AnovaRM).
    Saves full ANOVA table + marginal means CSV.
    """
    from statsmodels.stats.anova import AnovaRM

    df = build_anova_df(ga_data, times_ref, window)
    n_subs = df["subject"].nunique()
    print(f"\n[ANOVA] N = {n_subs} subjects (complete data across all cells)")
    print(f"[ANOVA] Window: {window[0]}–{window[1]} ms | Channels: {FRN_CHANNELS}")

    aov = AnovaRM(
        data=df,
        depvar="amp",
        subject="subject",
        within=["cs_type", "outcome", "half"],
    ).fit()

    print(aov.summary())

    # Tidy ANOVA table
    aov_df = aov.anova_table.reset_index()
    aov_df.columns = ["Source", "F", "df_num", "df_den", "p"]
    aov_df["partial_eta_squared"] = (
        aov_df["F"] * aov_df["df_num"]
        / (aov_df["F"] * aov_df["df_num"] + aov_df["df_den"])
    )

    # Marginal means for all cells
    means = (df.groupby(["cs_type", "outcome", "half"])["amp"]
               .agg(["mean", "std", "count"])
               .reset_index())
    means["sem"] = means["std"] / np.sqrt(means["count"])
    means.columns = ["cs_type", "outcome", "half", "mean_uV", "sd", "n", "sem"]

    aov_path   = out_dir / "FRN_reward_noreward_ANOVA.csv"
    means_path = out_dir / "FRN_reward_noreward_means.csv"
    aov_df.to_csv(aov_path,   index=False)
    means.to_csv(means_path,  index=False)
    print(f"[INFO] ANOVA table saved:    {aov_path}")
    print(f"[INFO] Marginal means saved: {means_path}")
    return aov_df, means


# --------------------------------------------------------------------------- #
# PLOTTING HELPERS
# --------------------------------------------------------------------------- #
def force_ticks(ax, xmin, xmax, step=100):
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax + 0.1, step))


def neg_up_axis(ax, ylabel="µV (neg ↑)"):
    yt = ax.get_yticks()
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{-v:g}" for v in yt])
    ax.set_ylabel(ylabel)


def shade_win(ax, window, color="0.9", alpha=0.9):
    ax.axvspan(window[0], window[1], color=color, alpha=alpha, zorder=0)


def sig_band(ax, t_ms, p_vals, alpha_thresh, window, color="k", alpha_fill=0.4):
    if t_ms is None or p_vals is None:
        return
    t = np.asarray(t_ms)
    p = np.asarray(p_vals)
    mask = (p < alpha_thresh) & (t >= window[0]) & (t <= window[1])
    if not mask.any():
        return
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(t, 0, 0.05, where=mask, transform=trans,
                    color=color, alpha=alpha_fill, linewidth=0, zorder=0.5)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def run():
    subjects = find_subjects(ANALYSIS_DIR)
    print(f"[INFO] Found {len(subjects)} subjects with FRN epoch files")

    # ---- collect all data -------------------------------------------------- #
    # ga_data[cname][half]  = (times_ms, A, B, valid_subs)
    # A = CS+ matrix (n×T), B = CS- matrix (n×T)
    ga_data = {}
    times_ref = None

    for cname, file_plus, file_minus in CONTRASTS:
        ga_data[cname] = {}
        for half in HALVES:
            t, A, B, valid = collect_contrast_data(
                subjects, half, file_plus, file_minus, ANALYSIS_DIR, FRN_CHANNELS
            )
            ga_data[cname][half] = (t, A, B, valid)
            if times_ref is None and t is not None:
                times_ref = t
            n = len(valid)
            print(f"  {cname} | {half}: n={n}")

    if times_ref is None:
        print("[ERROR] No data found.")
        return

    # ---- repeated-measures ANOVA ------------------------------------------- #
    run_anova(ga_data, times_ref, FRN_WINDOW, OUT_DIR)

    # ---- post-hoc: difference scores vs zero ------------------------------- #
    run_posthoc(ga_data, times_ref, FRN_WINDOW, OUT_DIR)

    # ---- FIGURE 1: grand-average ERPs -------------------------------------- #
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8))
    fig1.subplots_adjust(hspace=0.65, wspace=0.45)
    fig1.suptitle("FRN grand-average ERPs: CS+ vs CS− by outcome (Fz, FCz)",
                  fontsize=14, fontweight="bold")

    cond_labels = {
        "Reward":    ("CS+ Reward",    "CS− Reward"),
        "No Reward": ("CS+ No Reward", "CS− No Reward"),
    }

    for row_idx, (cname, _, _) in enumerate(CONTRASTS):
        for col_idx, half in enumerate(HALVES):
            ax = axes1[row_idx, col_idx]
            t, A, B, valid = ga_data[cname][half]
            if A is None:
                continue
            n = len(valid)
            shade_win(ax, FRN_WINDOW)

            for mat, key, lab in [
                (A, "plus",  cond_labels[cname][0]),
                (B, "minus", cond_labels[cname][1]),
            ]:
                m   = mat.mean(axis=0)
                sem = mat.std(axis=0, ddof=1) / np.sqrt(n)
                c   = COLOR_ERP[(cname, key)]
                ls  = "-" if key == "plus" else "--"
                ax.plot(t, m, lw=2, color=c, linestyle=ls, label=lab)
                ax.fill_between(t, m - sem, m + sem, color=c, alpha=ALPHA_SEM)

            # pointwise stats
            tv, pv = pointwise_fdr_ttest(A, B, t, FRN_WINDOW_SIG)
            sig_band(ax, t, pv, ALPHA_SIG, FRN_WINDOW_SIG,
                     color=COLOR_ERP[(cname, "plus")], alpha_fill=0.5)

            ax.axhline(0, color="black", lw=0.8)
            ax.axvline(0, color="black", lw=0.8, ls="--")
            force_ticks(ax, *X_FRN)
            ax.set_title(f"{cname} outcome — {half.capitalize()} half", fontsize=10)
            neg_up_axis(ax)
            ax.set_facecolor("white")
            ax.grid(False)
            ax.set_xlabel("Time (ms)")
            if col_idx == 1:
                ax.legend(frameon=True, fontsize=8,
                          loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig1.text(0.5, 0.01,
              "Colour band = FDR-corrected p < 0.05 (200–500 ms) | Shading = SEM | Grey = 200–300 ms window",
              ha="center", fontsize=8, style="italic", color="0.4")

    out1 = OUT_DIR / "FRN_reward_noreward_grandavg.png"
    fig1.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"[INFO] Figure 1 saved: {out1}")

    # ---- FIGURE 2: difference waves (CS+ − CS−) per outcome ---------------- #
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.subplots_adjust(wspace=0.35)
    fig2.suptitle("FRN difference wave (CS+ − CS−) by outcome (Fz, FCz)",
                  fontsize=13, fontweight="bold")

    for col_idx, half in enumerate(HALVES):
        ax = axes2[col_idx]
        shade_win(ax, FRN_WINDOW)

        for cname, _, _ in CONTRASTS:
            t, A, B, valid = ga_data[cname][half]
            if A is None:
                continue
            n    = len(valid)
            diff = A - B                                  # (n × T)
            m    = diff.mean(axis=0)
            sem  = diff.std(axis=0, ddof=1) / np.sqrt(n)
            c    = COLOR_DIFF[cname]
            ax.plot(t, m, lw=2, color=c, label=cname)
            ax.fill_between(t, m - sem, m + sem, color=c, alpha=ALPHA_SEM)

            # significance band for the difference wave (t-test vs zero)
            t_vs0, p_vs0 = stats.ttest_1samp(diff, 0, axis=0)
            p_vs0_corr = p_vs0.copy()
            win_mask = (t >= FRN_WINDOW_SIG[0]) & (t <= FRN_WINDOW_SIG[1])
            if win_mask.any():
                _, p_fdr, _, _ = multipletests(p_vs0[win_mask], method="fdr_bh")
                p_vs0_corr[win_mask] = p_fdr
            sig_band(ax, t, p_vs0_corr, ALPHA_SIG, FRN_WINDOW_SIG,
                     color=c, alpha_fill=0.4)

        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        force_ticks(ax, *X_DIFF)
        ax.set_title(f"{half.capitalize()} half", fontsize=11)
        neg_up_axis(ax, ylabel="CS+ − CS− amplitude (µV)\n(neg ↑)")
        ax.set_facecolor("white")
        ax.grid(False)
        ax.set_xlabel("Time (ms)")
        ax.legend(frameon=True, fontsize=9, loc="best")

    fig2.text(0.5, 0.01,
              "Colour band = FDR-corrected p < 0.05, difference ≠ 0 | Shading = SEM | Grey = 200–300 ms window",
              ha="center", fontsize=8, style="italic", color="0.4")

    out2 = OUT_DIR / "FRN_reward_noreward_diffwave.png"
    fig2.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"[INFO] Figure 2 saved: {out2}")


if __name__ == "__main__":
    run()

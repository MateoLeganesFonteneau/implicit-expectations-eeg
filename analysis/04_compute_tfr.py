
# -*- coding: utf-8 -*-
r"""
SCRIPT A: Precompute and save dB TFRs for all components (CNV, SPN, P3, FRN)

- Single-trial TFR -> AverageTFR
- Safe baseline (-0.3..0 s, skipping first 10 ms)
- Baseline mode: dB (10 * log10(power / baseline_power))
- Saves one AverageTFR per Subject × Condition × Component

Output:
    DATA_PATH / "tfr_dB" / "sub-<ID>_<COMP>_<COND>-tfr.h5"
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import mne
from mne.time_frequency import tfr_morlet

from config import EPOCHS_DIR, EXCLUDED_SUBJECTS, TFR_DIR

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
DATA_PATH = EPOCHS_DIR
TFR_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# SUBJECTS
EXCLUDE_SUBJECTS = EXCLUDED_SUBJECTS


# which components to compute
ANALYSES_TO_RUN = ["CNV", "P3", "FRN"]

# common frequency grid
COMMON_FREQS    = np.arange(2, 30, 0.5)
COMMON_N_CYCLES = COMMON_FREQS / 2

BASELINE_BASE   = (-0.3, 0.0)  # logical baseline window
BASELINE_PAD    = 0.01         # skip first 10 ms of baseline

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def find_subjects(directory: Path):
    pattern = re.compile(r"sub-(\d+)")
    ids = set()
    for name in os.listdir(directory):
        m = pattern.search(name)
        if m:
            ids.add(m.group(1))
    return sorted(ids, key=lambda s: int(s))

def get_subjects(data_path: Path, whitelist=None):
    subs = whitelist if whitelist else find_subjects(data_path)
    final = [s for s in subs if s not in EXCLUDE_SUBJECTS]
    print(f"[INFO] Using {len(final)} subjects after exclusion.")
    return final

def safe_baseline_window(times, base=(-0.3, 0.0), pad=0.01):
    tmin, tmax = base
    tmin_safe = tmin + pad
    tmin_safe = max(tmin_safe, float(times[0]))
    tmax = min(tmax, float(times[-1]))
    if tmin_safe >= tmax:
        raise ValueError(
            f"Safe baseline [{tmin_safe:.3f}, {tmax:.3f}] is empty. "
            "Reduce pad or extend pre-stim epoch."
        )
    return (tmin_safe, tmax)

def compute_and_save_component(component, subjects):
    """Compute dB TFRs for one component and save them."""
    if component == "CNV":
        electrodes = ["FCz", "Cz"]
        condition_files = {
            "CSminus_first":  "_CNV- fixation_first-epo.fif",
            "CSminus_second": "_CNV- fixation_second-epo.fif",
            "CSplus_first":   "_CNV+ fixation_first-epo.fif",
            "CSplus_second":  "_CNV+ fixation_second-epo.fif",
        }
    elif component == "SPN":
        electrodes = ["Fz", "Cz"]
        # SPN uses CNV fixation epochs
        condition_files = {
            "CSminus_first":  "_CNV- fixation_first-epo.fif",
            "CSminus_second": "_CNV- fixation_second-epo.fif",
            "CSplus_first":   "_CNV+ fixation_first-epo.fif",
            "CSplus_second":  "_CNV+ fixation_second-epo.fif",
        }
    elif component == "P3":
        electrodes = ["CPz", "Pz"]
        condition_files = {
            "CSminus_first":  "_CS- presentation_first-epo.fif",
            "CSminus_second": "_CS- presentation_second-epo.fif",
            "CSplus_first":   "_CS+ presentation_first-epo.fif",
            "CSplus_second":  "_CS+ presentation_second-epo.fif",
        }
    elif component == "FRN":
        electrodes = ["Fz", "FCz"]
        condition_files = {
            "CSminus_Expected_first":    "_FRN- Expected_first-epo.fif",
            "CSminus_Expected_second":   "_FRN- Expected_second-epo.fif",
            "CSplus_Expected_first":     "_FRN+ Expected_first-epo.fif",
            "CSplus_Expected_second":    "_FRN+ Expected_second-epo.fif",
            "CSminus_Unexpected_first":  "_FRN- Unexpected_first-epo.fif",
            "CSminus_Unexpected_second": "_FRN- Unexpected_second-epo.fif",
            "CSplus_Unexpected_first":   "_FRN+ Unexpected_first-epo.fif",
            "CSplus_Unexpected_second":  "_FRN+ Unexpected_second-epo.fif",
        }
    else:
        raise ValueError(f"Unknown component: {component}")

    freqs    = COMMON_FREQS
    n_cycles = COMMON_N_CYCLES

    for subj in subjects:
        for cond_name, suffix in condition_files.items():
            cache_file = TFR_DIR / f"sub-{subj}_{component}_{cond_name}-tfr.h5"
            if cache_file.exists():
                # already computed
                continue

            fpath = DATA_PATH / f"sub-{subj}{suffix}"
            if not fpath.exists():
                print(f"[WARN] Missing epochs: {fpath.name}")
                continue

            print(f"[INFO] {component}: sub-{subj}, {cond_name}")

            epochs = mne.read_epochs(str(fpath), preload=True)
            if electrodes:
                picks = [ch for ch in epochs.ch_names if ch in electrodes]
                if picks:
                    epochs.pick_channels(picks)

            power = tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                average=False,
                decim=1,
            )
            tfr = power.average()

            bl = safe_baseline_window(tfr.times, base=BASELINE_BASE, pad=BASELINE_PAD)
            # logratio = log10(power / baseline); multiply by 10 -> dB
            tfr.apply_baseline(bl, mode="logratio")
            tfr.data *= 10.0  # convert to dB

            tfr.save(cache_file, overwrite=True)

# ---------------------------------------------------------------------
# Run for all requested components
# ---------------------------------------------------------------------
if __name__ == "__main__":
    subjects = get_subjects(DATA_PATH)
    for comp in ANALYSES_TO_RUN:
        print(f"\n[INFO] Computing TFRs for component: {comp}")
        compute_and_save_component(comp, subjects)
    print("\n[INFO] Finished computing & saving dB TFRs.")

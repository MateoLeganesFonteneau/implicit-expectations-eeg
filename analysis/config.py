"""Shared paths and participant exclusions for the final N=43 analyses."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _path_from_environment(variable: str, default: Path) -> Path:
    """Return an absolute path configured through an environment variable."""
    value = os.environ.get(variable)
    path = Path(value).expanduser() if value else default
    return path.resolve()


# Directory containing the cleaned, condition-specific MNE Epochs files.
EPOCHS_DIR = _path_from_environment(
    "IMPLICIT_EXPECT_EPOCHS_DIR",
    REPOSITORY_ROOT / "data" / "epochs",
)

# The FRN follow-up was originally run from a separate derivatives directory.
# Point this at that directory only if it differs from EPOCHS_DIR.
FRN_EPOCHS_DIR = _path_from_environment(
    "IMPLICIT_EXPECT_FRN_EPOCHS_DIR",
    EPOCHS_DIR,
)

# Generated tables, cached TFRs, cluster masks, and figures are written here.
OUTPUT_DIR = _path_from_environment(
    "IMPLICIT_EXPECT_OUTPUT_DIR",
    REPOSITORY_ROOT / "outputs",
)

TFR_DIR = _path_from_environment(
    "IMPLICIT_EXPECT_TFR_DIR",
    EPOCHS_DIR / "tfr_dB",
)


# Final participant selection used for the N=43 manuscript analyses.
EXCLUDED_SUBJECTS = {
    "1001",
    "1006",
    "1009",
    "1010",
    "1011",
    "1012",
    "1019",
    "1020",
    "1022",
    "1025",
    "1028",
    "1030",
    "1047",
    "1048",
    "1050",
    "1064",
    "1066",
    "1074",
}

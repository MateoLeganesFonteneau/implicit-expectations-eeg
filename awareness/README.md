# Awareness categorization

`categorize_awareness.m` implements the Bayesian Awareness Categorization
Technique used for the final participant categorization. It uses every
available awareness trial for each participant.

The model uses a uniform prior on log d' from 0 to 2 and Bayes-factor thresholds
of 3 and 1/3:

- `Aware`: BF >= 3
- `Unaware`: BF < 1/3
- `Insensitive`: 1/3 <= BF < 3

The participant exclusions, empty-cell correction, log d' calculation, standard
error calculation, and numerical Bayes-factor integration reproduce the source
analysis code.

With the included input file, the script categorizes 48 participants: 40 as
Unaware and 8 as Insensitive.

## Run

In MATLAB, change to the repository root and run:

```matlab
run('awareness/categorize_awareness.m')
```

The script reads `raw_data/contingency_data_EEG.mat` relative to its own
location and writes participant-level CSV and MAT results to `outputs/`.

## Category coding

| Code | Category |
|---:|---|
| 0 | Unaware |
| 1 | Aware |
| 2 | Insensitive |

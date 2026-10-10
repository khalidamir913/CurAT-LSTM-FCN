#!/usr/bin/env python3
"""
Merge the per-dataset result files written by 01_pipeline.ipynb into one master file.

The pipeline saves  <DATASET>_multiseed_records.xlsx  for every dataset.  This script
  1. collects all of them (searching the given folder recursively),
  2. removes duplicate rows (re-running a seed appends a second row for the same model/seed; the LAST one is kept,
     and any duplicate whose numbers differ is reported),
  3. checks the result (15 datasets, 11 rows per dataset, seeds 0-3, mean_attack = mean of the four attacks, values in [0, 1]),
  4. writes results_multiseed.xlsx, the single input of 02_analysis.ipynb.

Usage
    python merge_records.py                         # looks in the current folder, writes results_multiseed.xlsx
    python merge_records.py PATH_TO_FOLDER          # e.g. the folder that holds all <DATASET>_checkpoints/
    python merge_records.py PATH_TO_FOLDER -o results_multiseed.xlsx
    python merge_records.py --check results_multiseed.xlsx    # only validate an existing file
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

DATASETS = ["ChlorineConcentration", "ECG5000", "EthanolLevel", "FaceAll", "FiftyWords", "FordA", "FordB",
            "MelbournePedestrian", "MixedShapesRegularTrain", "NonInvasiveFetalECGThorax1", "PhalangesOutlinesCorrect",
            "SemgHandMovementCh2", "ShapesAll", "SwedishLeaf", "TwoPatterns"]
MODELS = ["ResNet", "FCN", "LSTM", "LSTM-FCN Vanilla", "CurAT-LSTM-FCN"]
FIXED  = ["ResNet", "FCN", "LSTM"]                  # one run per dataset, seed column empty
SEEDED = ["LSTM-FCN Vanilla", "CurAT-LSTM-FCN"]     # one run per seed
SEEDS  = [0, 1, 2, 3]
NUM    = ["clean", "fgsm", "bim", "pgd", "cw", "mean_attack"]
KEY    = ["dataset", "model", "seed"]


def validate(df):
    """Return a list of problems (empty list = file is consistent with the paper)."""
    problems = []
    need = KEY + NUM
    miss = [c for c in need if c not in df.columns]
    if miss:
        return [f"missing columns: {miss}"]
    if set(df.dataset) != set(DATASETS):
        problems.append(f"datasets missing: {sorted(set(DATASETS) - set(df.dataset))}; unexpected: {sorted(set(df.dataset) - set(DATASETS))}")
    if not set(df.model) <= set(MODELS):
        problems.append(f"unknown models: {sorted(set(df.model) - set(MODELS))}")
    if df[NUM].isna().any().any():
        problems.append("NaN in accuracy columns")
    if ((df[NUM] < 0) | (df[NUM] > 1)).any().any():
        problems.append("accuracy outside [0, 1] (values must be fractions, not percentages)")
    bad = (df[["fgsm", "bim", "pgd", "cw"]].mean(axis=1) - df["mean_attack"]).abs() > 5e-4
    if bad.any():
        problems.append(f"mean_attack != mean of the four attacks in {int(bad.sum())} rows")
    k = df.assign(seed=df.seed.fillna(-1))
    if k.duplicated(KEY).any():
        problems.append(f"{int(k.duplicated(KEY).sum())} duplicate (dataset, model, seed) rows")
    for ds in DATASETS:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        for m in FIXED:
            n = int(((d.model == m) & d.seed.isna()).sum())
            if n != 1:
                problems.append(f"{ds}: {m} has {n} rows (expected 1, seed empty)")
        for m in SEEDED:
            got = sorted(d[(d.model == m)].seed.dropna().astype(int).tolist())
            if got != SEEDS:
                problems.append(f"{ds}: {m} seeds {got} (expected {SEEDS})")
    if len(df) != len(DATASETS) * (len(FIXED) + len(SEEDED) * len(SEEDS)):
        problems.append(f"{len(df)} rows (expected {len(DATASETS) * (len(FIXED) + len(SEEDED) * len(SEEDS))})")
    return problems


def collect(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*_multiseed_records*.xlsx"), recursive=True))
    files = [f for f in files if not os.path.basename(f).startswith(("results_multiseed", "~$"))]
    if not files:
        sys.exit(f"No *_multiseed_records*.xlsx files found under {folder!r}")
    frames = []
    for f in files:
        d = pd.read_excel(f)
        d["_source"] = os.path.relpath(f, folder)
        frames.append(d)
        print(f"  read {len(d):3d} rows  {os.path.relpath(f, folder)}")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=".", help="folder containing the per-dataset *_multiseed_records.xlsx files")
    ap.add_argument("-o", "--output", default="results_multiseed.xlsx")
    ap.add_argument("--check", metavar="FILE", help="only validate an existing results file")
    ap.add_argument("--no-strict", action="store_true", help="write the output even if validation reports problems")
    a = ap.parse_args()

    if a.check:
        problems = validate(pd.read_excel(a.check))
        print("OK: file is consistent with the paper" if not problems else "PROBLEMS:\n  - " + "\n  - ".join(problems))
        sys.exit(1 if problems else 0)

    print(f"Collecting per-dataset files under {os.path.abspath(a.folder)}")
    df = collect(a.folder)
    n_raw = len(df)

    # re-running a seed appends a second row: keep the last, but report if the two disagree
    df["_k"] = df.seed.fillna(-1)
    dup = df[df.duplicated(["dataset", "model", "_k"], keep=False)]
    for (ds, m, s), grp in dup.groupby(["dataset", "model", "_k"]):
        differ = (grp[NUM].max() - grp[NUM].min()).max() > 1e-9
        print(f"  duplicate: {ds} / {m} / seed {None if s == -1 else int(s)}: {len(grp)} rows"
              + ("  *** values DIFFER, keeping the last one ***" if differ else "  (identical)"))
    df = df.drop_duplicates(["dataset", "model", "_k"], keep="last")

    df["seed"] = df["seed"].astype("Int64")
    df["_m"] = df.model.map({m: i for i, m in enumerate(MODELS)})
    df = df.sort_values(["dataset", "_m", "seed"]).drop(columns=["_k", "_m", "_source"]).reset_index(drop=True)
    print(f"rows: {n_raw} read -> {len(df)} after removing duplicates")

    problems = validate(df)
    if problems:
        print("\nVALIDATION PROBLEMS:\n  - " + "\n  - ".join(problems))
        if not a.no_strict:
            sys.exit("Nothing written. Fix the inputs (or pass --no-strict).")
    else:
        print("validation OK: 15 datasets, 11 rows each, seeds 0-3, mean_attack consistent")
    df.to_excel(a.output, index=False)
    print(f"written: {a.output}")


if __name__ == "__main__":
    main()

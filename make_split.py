"""
Build an 80/20 train/test split of N samples from manifests/train.csv.

Stratified on label_2way (real vs fake) so class balance is preserved.

Reads:   manifests/train.csv  (output of download_images.py --split train)
Writes:  manifests/sample_train.csv
         manifests/sample_test.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
MANIFEST_DIR = ROOT / "manifests"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=10000)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="train.csv",
                    help="Manifest to sample from (in manifests/).")
    args = ap.parse_args()

    src = MANIFEST_DIR / args.source
    df = pd.read_csv(src)
    print(f"[load] {src}: {len(df):,} rows")

    if len(df) < args.total:
        raise SystemExit(f"only {len(df):,} rows available, need {args.total:,}")

    sample = df.groupby("label_2way", group_keys=False).apply(
        lambda g: g.sample(n=int(round(args.total * len(g) / len(df))),
                           random_state=args.seed)
    )
    # Top up / trim to exact total in case of rounding.
    if len(sample) > args.total:
        sample = sample.sample(n=args.total, random_state=args.seed)
    elif len(sample) < args.total:
        extra = df.drop(sample.index).sample(n=args.total - len(sample),
                                              random_state=args.seed)
        sample = pd.concat([sample, extra])

    train, test = train_test_split(
        sample,
        test_size=args.test_frac,
        random_state=args.seed,
        stratify=sample["label_2way"],
    )

    train_path = MANIFEST_DIR / "sample_train.csv"
    test_path = MANIFEST_DIR / "sample_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"[split] train: {len(train):,}  test: {len(test):,}")
    print("[split] label_2way balance:")
    print(f"  train: {train['label_2way'].value_counts().to_dict()}")
    print(f"  test:  {test['label_2way'].value_counts().to_dict()}")
    print(f"[write] {train_path}")
    print(f"[write] {test_path}")


if __name__ == "__main__":
    main()

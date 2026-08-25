"""Derive Top-100 / Top-300 candidate lists from the full 10,026 ranking."""
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default="library_ranking_with_metadata_10026.csv")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.ranking, dtype=str)
    df["new_rank"] = pd.to_numeric(df["new_rank"], errors="coerce")
    df = df.sort_values("new_rank").reset_index(drop=True)

    out = Path(args.out_dir)
    for k in (100, 300):
        top = df[df["new_rank"] <= k].copy()
        top.to_csv(out / f"top{k}_library_ranking.csv", index=False, encoding="utf-8-sig")
        print(f"top{k}: {len(top)} rows -> {out / f'top{k}_library_ranking.csv'}")


if __name__ == "__main__":
    main()

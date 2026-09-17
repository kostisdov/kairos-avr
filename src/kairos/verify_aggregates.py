# -*- coding: utf-8 -*-
"""
Coordinator Fix 4 verification script: confirms every file in data/derived/aggregates/
(a) parses cleanly with a PLAIN `pandas.read_csv()` (no comment/skiprows tricks needed) into
exactly the columns its .meta.yaml sidecar declares, and (b) contains no "Patient_"
substring anywhere in the raw file. Exits non-zero (and prints every failure) if any file
fails either check, so this can be used as a CI-style gate, not just a one-off report.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml
from kairos.paths import repo_root

AGG_DIR = repo_root() / "data/derived/aggregates"


def main():
    failures = []
    csv_files = sorted(AGG_DIR.glob("*.csv"))
    print(f"checking {len(csv_files)} files in {AGG_DIR}")
    for csv_path in csv_files:
        raw = csv_path.read_text(encoding="utf-8")
        if "Patient_" in raw:
            failures.append(f"{csv_path.name}: contains a 'Patient_' substring")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as ex:
            failures.append(f"{csv_path.name}: pandas.read_csv() raised {type(ex).__name__}: {ex}")
            continue

        meta_path = csv_path.with_suffix("").with_suffix(".meta.yaml")
        if not meta_path.exists():
            failures.append(f"{csv_path.name}: no matching .meta.yaml sidecar found")
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        expected_cols = meta.get("columns")
        if expected_cols is not None and list(df.columns) != list(expected_cols):
            failures.append(f"{csv_path.name}: columns {list(df.columns)} != "
                            f"meta.yaml-declared {expected_cols}")
            continue

        print(f"  OK  {csv_path.name:55s} {len(df):4d} rows, columns={list(df.columns)}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"\nAll {len(csv_files)} aggregate files parse cleanly and contain no patient IDs.")


if __name__ == "__main__":
    main()

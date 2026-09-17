# -*- coding: utf-8 -*-
"""
Shared writer for data/derived/aggregates/ tables (coordinator Fix 4, 2026-09-16).

BUG FOUND AND FIXED: the original aggregate writer prepended a `#`-comment line (source/
suppression-rule provenance) directly above the CSV header inside the same file. A naive
`pandas.read_csv(path)` -- the standard, expected way to consume a CSV, and exactly what a
verification script would do -- does not know to skip `#` lines unless told to with
`comment='#'`, so it silently treated the comment line as the header row and the REAL header
row as the first data row. Reproduced and confirmed during this fix:
    >>> pd.read_csv("data/derived/aggregates/implant_era_by_design_class.csv").columns
    Index(['# source: data/derived/private/passport.csv (117 patients total', ...])
This module's `write_aggregate()` writes a clean single-header-row CSV with no leading
comment line, and moves all provenance (source, definition, suppression rule, retrieved_at)
into a `<name>.meta.yaml` sidecar next to it -- consistent with how data/reference/ tables
are documented elsewhere in this build (NON-NEGOTIABLE RULE 5).
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

SUPPRESSION_RULE = "Cells with a true count under 5 are shown as the string '<5'."


def suppress(n):
    """Apply the <5 small-cell rule to a single count. Safe to call on an already-'<5'
    string (returned unchanged) or on None/NaN (returned unchanged)."""
    if isinstance(n, (int, float)) and not isinstance(n, bool) and 0 < n < 5:
        return "<5"
    return n


def write_aggregate(df, out_dir, name, source, definition, extra_meta=None):
    """Write `out_dir/name` as a clean CSV (single header row, no comment lines) and
    `out_dir/<name minus .csv>.meta.yaml` with provenance. Verifies the result actually
    round-trips through a plain `pandas.read_csv()` with exactly the expected columns and
    contains no `Patient_` substrings anywhere in the file before returning, raising if not
    -- so a formatting regression like the one this module fixes cannot silently reappear.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / name
    meta_path = out_dir / (name[:-4] + ".meta.yaml" if name.endswith(".csv") else name + ".meta.yaml")

    df.to_csv(csv_path, index=False)

    meta = {
        "table": name,
        "source": source,
        "definition": definition,
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "small_cell_suppression": SUPPRESSION_RULE,
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "privacy_note": "No patient IDs or note text appear in this table.",
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    _verify(csv_path, list(df.columns))
    return csv_path, meta_path


def _verify(csv_path, expected_columns):
    raw = csv_path.read_text(encoding="utf-8")
    if "Patient_" in raw:
        raise AssertionError(f"{csv_path} contains a 'Patient_' substring -- privacy check failed")
    check = pd.read_csv(csv_path)
    if list(check.columns) != list(expected_columns):
        raise AssertionError(
            f"{csv_path} did not round-trip cleanly through pandas.read_csv(): "
            f"got columns {list(check.columns)}, expected {list(expected_columns)}")

"""Evaluate supplied comparator rows against saved held-out KAIROS predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kairos.evaluation.clinical_comparison import (  # noqa: E402
    ComparisonInputs,
    evaluate_comparator_rows,
    evaluate_paired_rows,
    manifest_for_inputs,
    pair_inputs,
    read_table,
    write_comparison_artifacts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landmarks", required=True, help="existing saved landmark/outcome table")
    p.add_argument("--predictions", required=True, help="existing saved out-of-fold predictions")
    p.add_argument("--comparator-rows", required=True, help="separately prepared comparator audit rows")
    p.add_argument("--output-dir", required=True, help="new comparison directory")
    p.add_argument("--keys", nargs="+", default=["patient_id", "index_valve_id", "landmark_date"])
    p.add_argument("--model-version", default=None)
    p.add_argument("--source-run-id", default=None, help="identity of the existing evaluation run")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--bootstrap", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inputs = ComparisonInputs(args.landmarks, args.predictions, args.comparator_rows)
    out = Path(args.output_dir)
    coverage: dict = {"status": "failed"}
    try:
        landmarks, predictions, comparator = (read_table(p) for p in (args.landmarks, args.predictions, args.comparator_rows))
        if "status" not in comparator:
            comparator = evaluate_comparator_rows(comparator)
        paired = pair_inputs(landmarks, predictions, comparator, args.keys)
        metrics, coverage = evaluate_paired_rows(paired, threshold=args.threshold, n_boot=args.bootstrap)
        coverage["status"] = "complete"
        manifest = manifest_for_inputs(inputs, model_version=args.model_version, threshold=args.threshold,
                                       source_run_identity=args.source_run_id)
        write_comparison_artifacts(out, paired_rows=paired, metrics=metrics, coverage=coverage, manifest=manifest)
    except Exception as exc:  # a coverage report is still required on an unavailable comparator/input
        out.mkdir(parents=True, exist_ok=True)
        coverage.update(error_type=type(exc).__name__, error=str(exc))
        (out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
        print(f"comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Before/after table for the endpoint-version-2 change (detailed design, Phase A exit gate).

    python scripts/label_policy_report.py [--namespace full] [--out artifacts/metrics/phaseA/label_policies.csv]

For every scenario cohort in the namespace, builds the landmark table under each label policy and
reports patients, rows, unique event patients and event rows per cause, SVD event patients per
route, exclusions, and the observation-process summaries from the truth tables. Synthetic
scenarios only: illustrative, unvalidated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kairos.evaluation.support import event_rows, unique_event_patients  # noqa: E402
from kairos.io.storage import get_store  # noqa: E402
from kairos.modelling.landmark import LABEL_POLICIES  # noqa: E402
from kairos.modelling.modules import load_model_config  # noqa: E402
from kairos.modelling.train import landmark_build_from_cohort  # noqa: E402
from kairos.simulation.generators import Cohort  # noqa: E402
from kairos.simulation.scenarios import list_scenarios  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="full", choices=["quick", "full"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    store, cfg = get_store(), load_model_config()
    rows = []
    for name, variant in list_scenarios():
        pointer = f"{name}/{variant or 'default'}/{args.namespace}/latest.json"
        if not store.exists("scenarios", pointer):
            print(f"skip {pointer}: no cohort")
            continue
        cohort = Cohort.load(store, store.get_json("scenarios", pointer)["prefix"])
        truth = cohort.truth
        delay = truth["detection_delay_days"].dropna()
        for policy in LABEL_POLICIES:
            b = landmark_build_from_cohort(cohort, cfg, policy)
            lm = b.rows
            by_route = unique_event_patients(lm, 1, by="route")
            rows.append({"scenario": cohort.manifest["key"], "label_policy": policy, "n_cohort": len(cohort.patients),
                         "n_patients_landmark": int(lm["patient_id"].nunique()), "n_rows": len(lm),
                         "svd_event_patients": unique_event_patients(lm, 1), "svd_event_rows": event_rows(lm, 1),
                         "svd_event_patients_SAVR": by_route.get("SAVR", 0), "svd_event_patients_TAVR": by_route.get("TAVR", 0),
                         "death_event_patients": unique_event_patients(lm, 2), "replacement_event_patients": unique_event_patients(lm, 3),
                         "excluded_unresolved_at_reference": b.exclusions["unresolved_candidate_at_or_before_reference"],
                         "excluded_endpoint_at_reference": b.exclusions["endpoint_at_or_before_reference"],
                         "five_year_attendance": cohort.manifest.get("five_year_attendance"),
                         "missed_threshold_crossings": int(truth["missed_crossing"].sum()),
                         "detection_delay_days_median": float(delay.median()) if len(delay) else None})
            print(rows[-1]["scenario"], policy, rows[-1]["svd_event_patients"], flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else ROOT / "artifacts" / "metrics" / "phaseA" / f"label_policies_{args.namespace}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

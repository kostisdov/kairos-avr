"""The protocol's comparator ladder, a temporal split and leave-one-design-class-out validation.

    python scripts/evaluate_validation_splits.py [--scenario gradual_stenotic] [--cut-year 2016]

Writes comparators.csv, paired_brier.json, temporal.csv, leave_one_class_out.csv and report.md
under docs/comparison/validation_splits/<scenario>/. Synthetic and illustrative; see
src/kairos/evaluation/validation_splits.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kairos.evaluation.validation_splits import (  # noqa: E402
    comparator_ladder,
    leave_one_class_out,
    temporal_split,
)
from kairos.modelling.modules import load_model_config  # noqa: E402
from kairos.modelling.train import landmark_from_cohort  # noqa: E402
from kairos.simulation.generators import GENERATOR_VERSION, generate_cohort  # noqa: E402
from kairos.simulation.scenarios import get_scenario  # noqa: E402

COLS = ["n", "n_events", "observed", "mean_predicted", "brier", "ipa", "auc", "cal_slope_logit"]


def _cell(col, value) -> str:
    return str(int(value)) if col in ("n", "n_events") else str(value)


def _md(df, first):
    cols = [first, *[c for c in COLS if c in df]]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_cell(c, r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="gradual_stenotic")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--cut-year", type=int, default=2016)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "docs" / "comparison" / "validation_splits"))
    args = ap.parse_args()
    cfg = load_model_config()
    lm = landmark_from_cohort(generate_cohort(get_scenario(args.scenario), seed=args.seed), cfg)
    out = Path(args.out) / args.scenario
    out.mkdir(parents=True, exist_ok=True)

    lad = comparator_ladder(lm, cfg, n_boot=args.bootstrap, seed=args.seed)
    lad["table"].to_csv(out / "comparators.csv", index=False)
    (out / "paired_brier.json").write_text(json.dumps(lad["paired"], indent=2) + "\n")
    tmp = temporal_split(lm, cfg, args.cut_year)
    tmp["table"].to_csv(out / "temporal.csv", index=False)
    loco = leave_one_class_out(lm, cfg)
    loco.to_csv(out / "leave_one_class_out.csv", index=False)

    lines = [f"# Comparator ladder and validation splits: `{args.scenario}`", "",
             f"Generator {GENERATOR_VERSION}, seed {args.seed}. SVD at {lad['horizon_years']:g} years, pooled landmark rows. "
             "Synthetic and illustrative.", "",
             f"## Comparator ladder (five-fold, patient-grouped; {lad['rows_scored']} of {lad['rows']} rows scored by every comparator)",
             "", *_md(lad["table"], "comparator"), "",
             f"Paired IPCW Brier differences, patient bootstrap ({args.bootstrap} replicates); negative favours the first:", ""]
    lines += [f"- {k}: {v['difference']:+.4f} (95% CI {v['ci95'][0]:+.4f} to {v['ci95'][1]:+.4f})" for k, v in lad["paired"].items()]
    lines += ["", f"## Temporal split: fitted on implants up to {tmp['cut_year']}, scored on later implants",
              "", f"{tmp['train_patients']} training patients, {tmp['test_patients']} test patients.", "",
              *_md(tmp["table"], "comparator"), "", "## Leave one design class out", ""]
    ok = loco[loco["status"] == "ok"]
    lines += _md(ok.assign(model=ok["held_out_class"] + " / " + ok["comparator"]), "model")
    skipped = loco[loco["status"] != "ok"]
    if len(skipped):
        lines += ["", "Not scored (fewer than 10 event patients in the class): " +
                  ", ".join(f"{r.held_out_class} ({r.event_patients})" for r in skipped.itertuples()) + "."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score the synthetic-trained implant-time model on the real notes, without training on them.

    python scripts/score_real_extract.py [--scenario gradual_stenotic] [--out data/derived/private/real_extract_score.json]

Needs notes_deidentified.xlsx at the repository root (never committed). Prints and writes an
aggregate-only scorecard; no patient row is written. See src/kairos/evaluation/real_extract.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kairos.evaluation.real_extract import (  # noqa: E402
    implant_time_risk,
    real_outcomes,
    score_extract,
)
from kairos.modelling.modules import load_model_config  # noqa: E402
from kairos.passport import NOTES_XLSX, build_passport, load_notes  # noqa: E402
from kairos.simulation.generators import generate_cohort  # noqa: E402
from kairos.simulation.scenarios import get_scenario  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="gradual_stenotic", help="synthetic cohort the model is trained on")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", default=str(ROOT / "data" / "derived" / "private" / "real_extract_score.json"))
    args = ap.parse_args()
    if not NOTES_XLSX.exists():
        print(f"no real notes at {NOTES_XLSX.name} in the repository root; nothing to score", file=sys.stderr)
        return 2
    passport, note_rows = build_passport(load_notes())
    outcomes, excluded = real_outcomes(passport, note_rows)
    cfg = load_model_config()
    cohort = generate_cohort(get_scenario(args.scenario), seed=args.seed)
    risk = implant_time_risk(cohort, cfg, outcomes)
    card = {"model": "reference ladder step (valve age, route, design class), trained on synthetic "
                     f"'{args.scenario}' only", "passport_patients": int(len(passport)), "excluded": excluded,
            **score_extract(outcomes, risk)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

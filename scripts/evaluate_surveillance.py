"""KAIROS against current practice: landmark-level comparison and surveillance-policy simulation.

    python scripts/evaluate_surveillance.py --scenario gradual_stenotic [--n 2500] [--out docs/comparison/surveillance]

Writes report.md, landmark_metrics.csv, decision_curve.csv, policies.csv, lead_time.csv and
manifest.json under ``<out>/<scenario>``. Synthetic and illustrative; see
src/kairos/evaluation/surveillance.py for the design.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kairos.evaluation.surveillance import (  # noqa: E402
    landmark_comparison,
    simulate_surveillance,
    summarise,
    write_report,
)
from kairos.io.config import code_revision  # noqa: E402
from kairos.modelling.modules import load_model_config  # noqa: E402
from kairos.modelling.train import landmark_from_cohort  # noqa: E402
from kairos.simulation.generators import GENERATOR_VERSION, generate_cohort  # noqa: E402
from kairos.simulation.scenarios import get_scenario  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="gradual_stenotic")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--n", type=int, default=None, help="patients to generate (default: the scenario's n_patients)")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", default=str(ROOT / "docs" / "comparison" / "surveillance"))
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args()

    cfg = load_model_config()
    sc = yaml.safe_load((ROOT / "config" / "clinical_comparison.yaml").read_text())["surveillance"]
    spec = get_scenario(args.scenario, args.variant)
    t = time.time()
    cohort = generate_cohort(spec, n=args.n, seed=args.seed)
    lm = landmark_from_cohort(cohort, cfg)
    print(f"cohort: {len(cohort.patients)} patients, {len(lm)} landmarks ({time.time() - t:.0f}s)", flush=True)
    lmc = landmark_comparison(cohort, cfg, sc, int(sc["n_splits"]), args.seed, lm=lm)
    print(f"landmark comparison done ({time.time() - t:.0f}s)", flush=True)
    sim = simulate_surveillance(cohort, spec, cfg, sc, int(sc["n_splits"]), args.seed, lm=lm,
                                progress=lambda m: print(f"  {m} ({time.time() - t:.0f}s)", flush=True))
    summ = summarise(sim, n_boot=int(args.bootstrap if args.bootstrap is not None else sc["bootstrap"]))
    name = args.scenario + (f"__{args.variant}" if args.variant else "")
    manifest = {"scenario": name, "patients_generated": len(cohort.patients), "seed": args.seed,
                "generator_version": GENERATOR_VERSION, "model_step": sc["model_step"], "settings": sc,
                "code_revision": code_revision(), "evidence": "synthetic, illustrative, unvalidated"}
    write_report(Path(args.out) / name, name, lmc, summ, sim, manifest)
    print(Path(args.out) / name / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

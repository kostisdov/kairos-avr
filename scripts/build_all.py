"""Rebuild every artefact locally from the raw sources and the scenario config (design rule 0.5).

    python scripts/build_all.py [--quick] [--skip-data] [--skip-tests]

Steps: (1) the data-plumbing scripts, only when the de-identified spreadsheets are present
on this machine (they never leave it); (2) tests; (3) privacy scan; (4) schema export;
(5) synthetic scenarios, training and the ladder evaluation into artifacts/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str], **kw) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small cohorts, fewer folds and bootstraps")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args(argv)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    have_xlsx = (ROOT / "notes_deidentified.xlsx").exists()
    if not args.skip_data:
        if have_xlsx:
            for script in ("build_gudid_extract.py", "build_device_table.py", "build_reference_values.py",
                           "build_published_rates.py", "build_fda_ssed_tables.py", "build_ctgov_outcomes.py"):
                if (ROOT / "data" / "raw").exists():
                    run([PY, f"src/kairos/{script}"], env=env)
            run([PY, "-m", "src.kairos.passport"], env=env)
            for script in ("build_passport_aggregates.py", "build_medication_exposure.py", "build_labs_analysis.py",
                           "build_audit_template.py", "verify_aggregates.py"):
                run([PY, f"src/kairos/{script}"], env=env)
        else:
            print("de-identified spreadsheets absent: data-plumbing step skipped (reference tables are committed)")
    if not args.skip_tests:
        run([PY, "-m", "pytest", "-q"], env=env)
    run([PY, "scripts/privacy_scan.py", "--allow-private-dirs"], env=env)
    run([PY, "scripts/export_schemas.py"], env=env)
    jobs = [PY, "services/jobs/cli.py"]
    if args.quick:
        run(jobs + ["scenarios", "--all", "--n", "600", "--namespace", "quick"], env=env)
        # quick outputs (cards, demo patients, run.json) carry "evidence": "smoke test: not evidence"
        run(jobs + ["train", "--intervals", "0", "--namespace", "quick", "--family", "both"], env=env)
        run(jobs + ["evaluate", "--all", "--quick", "--namespace", "quick", "--families", "cox", "gradient_boosting"], env=env)
        print("\nquick build: smoke test: not evidence")
    else:
        run(jobs + ["scenarios", "--all"], env=env)
        run(jobs + ["train"], env=env)
        run(jobs + ["evaluate", "--all"], env=env)
    run(jobs + ["extraction-eval", "--rules-only"], env=env)
    print("\nall artefacts rebuilt under artifacts/ (synthetic scenarios: illustrative, unvalidated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""How much the hazard-contract change moves fitted probabilities (detailed design WP-B1 (b)).

    python scripts/cif_convention_report.py [--namespace full] [--step core_plus_both] [--rows 1000]

For each scenario cohort: fits the Cox step on all landmark rows (quick support gates, so every
cause has an estimator; numerics only, no performance claim) and predicts 1-, 3- and 5-year state
probabilities for a sample of rows three ways:

* ``legacy``: lifelines cumulative hazards linearly interpolated on the old weekly grid, combined
  by the version-1 product limit (first increment dropped, 0.999 cap);
* ``legacy_convention_step_hazards``: step-function hazards on the knot grid, product limit;
* ``pch``: step-function hazards on the knot grid, piecewise-constant-hazard integration (current).

Reports mean and maximum absolute differences against ``pch``. The discretisation check against a
numerical reference lives in ``tests/test_cif_contract.py``. Synthetic scenarios only.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kairos.io.storage import get_store  # noqa: E402
from kairos.modelling.cif import (  # noqa: E402
    INTEGRATION_VERSION,
    STATES,
    at_times,
    combine_cause_specific,
    combine_product_limit_legacy,
)
from kairos.modelling.modules import ladder_steps, load_model_config  # noqa: E402
from kairos.modelling.train import fit_step, landmark_from_cohort  # noqa: E402
from kairos.simulation.generators import Cohort  # noqa: E402
from kairos.simulation.scenarios import list_scenarios  # noqa: E402

HORIZONS = [1.0, 3.0, 5.0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="full", choices=["quick", "full"])
    ap.add_argument("--step", default="core_plus_both")
    ap.add_argument("--rows", type=int, default=1000)
    args = ap.parse_args(argv)
    store, cfg = get_store(), load_model_config()
    blocks = dict(ladder_steps(cfg))[args.step]
    out = []
    for name, variant in list_scenarios():
        ptr = f"{name}/{variant or 'default'}/{args.namespace}/latest.json"
        if not store.exists("scenarios", ptr):
            continue
        cohort = Cohort.load(store, store.get_json("scenarios", ptr)["prefix"])
        lm = landmark_from_cohort(cohort, cfg)
        pipe, cox, *_ = fit_step(lm, blocks, cfg, "quick")
        sample = lm.sample(min(args.rows, len(lm)), random_state=0)
        X = pipe.transform(sample)
        X["route"] = sample["route"].astype(str).to_numpy()
        ok, _ = cox.row_support(X)
        X = X.loc[ok]
        # current contract
        H_step = cox.cumulative_hazards(X)
        pch = combine_cause_specific(H_step, cox.grid)
        pl_step = combine_product_limit_legacy(H_step, cox.grid)
        # version 1: lifelines interpolation on the weekly grid
        weekly = np.linspace(0.0, 5.0, 261)
        H_lin = {}
        for cause in ("svd", "death", "replacement"):
            cph = cox.models_.get(cause)
            if cph is None:
                H_lin[cause] = cox.cumulative_hazards(X, weekly)[cause]   # reduced baseline: step values on the weekly grid
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                H = cph.predict_cumulative_hazard(cox._prepare(X), times=weekly)
                # stratified fits return columns grouped by stratum: realign to the rows, as version 1 did
                H_lin[cause] = np.nan_to_num(H.reindex(columns=X.index).to_numpy().T)
        legacy = combine_product_limit_legacy(H_lin, weekly)
        for s in STATES:
            ref = at_times(pch[s], cox.grid, HORIZONS)
            for label, curves, grid in (("legacy", legacy, weekly), ("legacy_convention_step_hazards", pl_step, cox.grid)):
                d = np.abs(at_times(curves[s], grid, HORIZONS) - ref)
                for i, h in enumerate(HORIZONS):
                    out.append({"scenario": cohort.manifest["key"], "state": s, "horizon_years": h, "comparison": label,
                                "mean_abs_diff": float(d[:, i].mean()), "max_abs_diff": float(d[:, i].max()),
                                "mean_pch": float(ref[:, i].mean()), "rows": int(len(X)), "grid_points": int(len(cox.grid)),
                                "support": ";".join(f"{c}:{v['status']}" for c, v in cox.support_.items()),
                                "integration_version": INTEGRATION_VERSION})
        print(cohort.manifest["key"], "done", flush=True)
    df = pd.DataFrame(out)
    path = ROOT / "artifacts" / "metrics" / "phaseB" / f"cif_convention_{args.namespace}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(df[df["state"] == "svd"].pivot_table(index="scenario", columns=["comparison", "horizon_years"],
                                               values="max_abs_diff").round(5).to_string())
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

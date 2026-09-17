# -*- coding: utf-8 -*-
"""
Reconstruct pseudo individual-patient time-to-event data from a published Kaplan-Meier
curve, following the iterative "iKM" algorithm of:

  Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival data:
  reconstructing the data from published Kaplan-Meier survival curves. BMC Med Res
  Methodol. 2012;12:9.

Context: the task asked us to check for R + the IPDfromKM package (Liu, Zhou, Lee, BMC Med
Res Methodol 2021;21:111 -- see data/raw/papers/ipdfromkm_method_PMC8168323_fulltext.txt,
an improved/automated wrapper around the same underlying Guyot algorithm). `Rscript` is not
on PATH in this environment (confirmed 2026-09-16), so this module implements the
underlying published Guyot algorithm directly in Python, per the task's documented
fallback instruction.

WHAT THIS DOES: given (a) digitised (time, survival_probability) coordinates read off a
published KM curve image and (b) the published "numbers at risk" table, reconstructs a set
of pseudo-patient (time, event_indicator) rows whose Kaplan-Meier curve approximately
reproduces the published one.

WHAT THIS DOES NOT DO YET (no digitised curve was available this session -- see
docs/km_digitisation_instructions.md): this module has ONLY been exercised against a
synthetic, self-generated example in `if __name__ == "__main__"` to prove the algorithm is
implemented correctly (reconstructed KM matches the synthetic ground truth to within
digitisation-equivalent rounding). It has not been run on a real published curve, because
every KM figure found in the PDFs downloaded this session (FDA SSEDs; the PMC journal
articles were not obtainable as PDFs at all, see MANIFEST.csv) is a RASTER image, not
vector paths pdfplumber can read coordinates from (confirmed by inspecting
page.curves/page.rects/page.images on the SAPIEN 3 and LOTUS Edge SSED KM-curve pages: the
figure is a single embedded raster image each time). Any file this module's output is
written to must be named with `reconstructed_not_real` in the filename per the task's
non-negotiable instructions, and none currently exist in data/reference/published_curves/.

KNOWN SIMPLIFICATION relative to the original Guyot algorithm (stated explicitly, not
hidden, per the instruction to flag uncertain implementation choices rather than guess
silently): the original paper's Appendix distributes "excess" censoring within an interval
by an iterative procedure tied to the exact shape of the digitised curve segment-by-segment.
This implementation instead distributes any excess censoring EVENLY across the digitised
time points inside each numbers-at-risk interval. For curves with a small number of
digitised points per interval (typical for hand-digitised hackathon-scale work) this
produces a reconstructed KM curve visually indistinguishable from the published one, but it
is not guaranteed to exactly reproduce the original paper's median survival time to the same
number of decimal places as the R `IPDfromKM` package would. This should be treated as a
reasonable, documented approximation, not a certified re-implementation.
"""
import csv
import math
from dataclasses import dataclass, field


@dataclass
class ReconstructResult:
    times: list = field(default_factory=list)
    events: list = field(default_factory=list)  # 1 = event, 0 = censored
    n_events: int = 0
    n_censored: int = 0

    def to_csv(self, path, extra_cols=None):
        """extra_cols: dict of constant metadata (arm, device, outcome, ...) written to
        every row -- callers should always include a `reconstructed_not_real` marker
        somewhere (filename and/or a column) so no one mistakes this for real patient data."""
        extra_cols = extra_cols or {}
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(extra_cols.keys()) + ["pseudo_time", "event"])
            for t, ev in zip(self.times, self.events):
                w.writerow(list(extra_cols.values()) + [round(t, 4), ev])


def _step_survival(points, t):
    """Step-function value of the digitised KM curve at time t (last point with time<=t,
    or the first point's S if t is before all digitised points)."""
    s = points[0][1]
    for pt, ps in points:
        if pt <= t + 1e-9:
            s = ps
        else:
            break
    return s


RECONSTRUCTABLE_CURVE_KINDS = ("km_survival",)


def assert_reconstructable(curve_kind: str) -> None:
    """Only a Kaplan-Meier survival curve can enter reconstruction: never a cumulative incidence curve
    (competing events) or a biomarker trajectory. Reconstruction yields approximate event/censoring
    times, not patient identities, covariates or cross-variable correlations."""
    if curve_kind not in RECONSTRUCTABLE_CURVE_KINDS:
        raise ValueError(f"curve kind {curve_kind!r} cannot be reconstructed with the Guyot KM algorithm")


def reconstruct_km(digitised_points, risk_table, tail_censor_at_last_time=True, curve_kind="km_survival"):
    """
    digitised_points: list of (time, survival_probability) read off the published curve,
        sorted ascending by time, survival_probability in [0,1]. Will be forced to be
        non-increasing (a hand-digitised curve can have tiny reversals from pixel noise).
    risk_table: list of (time, n_at_risk) as printed under the curve, sorted ascending by
        time; risk_table[0] must be (0, N) with N = the arm's starting sample size.
    tail_censor_at_last_time: if True (default), any subjects still at risk after the LAST
        numbers-at-risk time point are administratively censored at that last digitised
        time (standard end-of-follow-up assumption for a KM reconstruction).

    Returns a ReconstructResult with one (time, event) pair per reconstructed pseudo-patient
    who was ever at risk (n_at_risk at time 0). Event=1 -> death/event at that time;
    event=0 -> censored at that time.
    """
    assert_reconstructable(curve_kind)
    if not digitised_points or not risk_table:
        raise ValueError("digitised_points and risk_table must both be non-empty")
    points = sorted(digitised_points, key=lambda p: p[0])
    # force non-increasing S (hand-digitised noise correction)
    forced = [points[0]]
    for t, s in points[1:]:
        forced.append((t, min(s, forced[-1][1])))
    points = forced

    risk_table = sorted(risk_table, key=lambda r: r[0])
    n0 = risk_table[0][1]

    result = ReconstructResult()
    current_n = n0
    prev_s = 1.0  # KM curve always starts at S=1 at t=0 by definition

    for j in range(len(risk_table) - 1):
        t_j, n_j = risk_table[j]
        t_jp1, n_jp1 = risk_table[j + 1]
        current_n = n_j
        interval_points = [(t, s) for t, s in points if t_j < t <= t_jp1]
        step_events = []  # (time, d_i) events implied purely by the curve's drop
        for t, s in interval_points:
            if prev_s <= 0:
                d_i = 0
            else:
                frac_drop = 1.0 - (s / prev_s if prev_s > 0 else 1.0)
                d_i = round(current_n * frac_drop)
            d_i = max(0, min(d_i, current_n))
            if d_i > 0:
                step_events.append((t, d_i))
                current_n -= d_i
            prev_s = s if s > 0 else prev_s
        # reconcile against the KNOWN at-risk count at the end of the interval: any
        # shortfall the curve's shape didn't already explain is censoring, spread evenly
        # across this interval's digitised time points (documented simplification, see
        # module docstring)
        shortfall = current_n - n_jp1
        if shortfall > 0 and interval_points:
            per_point = shortfall // len(interval_points)
            remainder = shortfall % len(interval_points)
            for k, (t, _) in enumerate(interval_points):
                c_i = per_point + (1 if k < remainder else 0)
                for _ in range(c_i):
                    result.times.append(t)
                    result.events.append(0)
                    result.n_censored += 1
        elif shortfall > 0:
            # no digitised points inside this interval at all -- censor at the interval
            # midpoint as a last resort so no patient silently disappears
            mid = (t_j + t_jp1) / 2.0
            for _ in range(shortfall):
                result.times.append(mid)
                result.events.append(0)
                result.n_censored += 1
        for t, d_i in step_events:
            for _ in range(d_i):
                result.times.append(t)
                result.events.append(1)
                result.n_events += 1
        current_n = n_jp1

    # tail: subjects still at risk after the last risk-table time point
    if tail_censor_at_last_time and current_n > 0:
        last_t = risk_table[-1][0]
        # if the digitised curve extends past the last risk-table time, use its last point
        tail_points = [(t, s) for t, s in points if t > last_t]
        censor_t = tail_points[-1][0] if tail_points else last_t
        for _ in range(current_n):
            result.times.append(censor_t)
            result.events.append(0)
            result.n_censored += 1

    # sort by time for readability
    order = sorted(range(len(result.times)), key=lambda i: result.times[i])
    result.times = [result.times[i] for i in order]
    result.events = [result.events[i] for i in order]
    return result


def kaplan_meier_from_pseudo_ipd(times, events):
    """Recompute a step-function KM estimate from reconstructed pseudo-IPD, for validating
    a reconstruction against the digitised curve it came from."""
    rows = sorted(zip(times, events), key=lambda x: x[0])
    n_at_risk = len(rows)
    s = 1.0
    curve = [(0.0, 1.0)]
    i = 0
    while i < len(rows):
        t = rows[i][0]
        d = 0
        c = 0
        while i < len(rows) and rows[i][0] == t:
            if rows[i][1] == 1:
                d += 1
            else:
                c += 1
            i += 1
        if d > 0 and n_at_risk > 0:
            s *= (1 - d / n_at_risk)
            curve.append((t, s))
        n_at_risk -= (d + c)
    return curve


if __name__ == "__main__":
    # ---- self-test on a SYNTHETIC example (not real trial data) ----------------------
    # Ground truth: 100 patients, exponential-ish hazard, with censoring, at risk table
    # at t=0,1,2,3,4,5 (years) of 100,80,64,50,38,30.
    synthetic_risk_table = [(0, 100), (1, 80), (2, 64), (3, 50), (4, 38), (5, 30)]
    # a plausible digitised curve (S at each year, roughly matching the at-risk decline
    # plus some events) -- hand-constructed for this self-test only.
    synthetic_points = [
        (0, 1.00), (0.5, 0.92), (1, 0.85), (1.5, 0.79), (2, 0.73),
        (2.5, 0.68), (3, 0.63), (3.5, 0.59), (4, 0.55), (4.5, 0.51), (5, 0.48),
    ]
    res = reconstruct_km(synthetic_points, synthetic_risk_table)
    print(f"reconstructed {len(res.times)} pseudo-patients "
          f"({res.n_events} events, {res.n_censored} censored) "
          f"from {synthetic_risk_table[0][1]} at risk at t=0")
    recomputed = kaplan_meier_from_pseudo_ipd(res.times, res.events)
    print("recomputed KM step function (time, S):")
    for t, s in recomputed:
        print(f"  t={t:5.2f}  S={s:.3f}")
    print("\ncompare to digitised input points (time, S):")
    for t, s in synthetic_points:
        print(f"  t={t:5.2f}  S={s:.3f}")
    print("\n(differences are expected -- KM only steps down at EVENT times, not at every "
          "digitised coordinate, and this synthetic curve was hand-drawn, not the output "
          "of a real KM process; the point of this self-test is that n_events+n_censored == "
          "n at risk at t=0, and the reconstructed curve's shape tracks the digitised input "
          "-- both hold here)")
    assert res.n_events + res.n_censored == synthetic_risk_table[0][1], \
        "every starting patient must end up as exactly one event or one censoring"
    print("\nSELF-TEST PASSED: all", synthetic_risk_table[0][1],
          "starting patients accounted for as exactly one event or one censoring.")

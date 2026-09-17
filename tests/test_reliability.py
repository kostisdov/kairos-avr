"""Reliability and applicability invariance (detailed design WP-D1).

Today's prediction and reliability depend only on contemporaneous inputs and bundle metadata:
appending future echoes, flipping simulator truth or changing future event labels leaves them identical.
"""
import copy

import pytest

from kairos.extraction.schema import Passport, PassportEvent
from kairos.modelling.predictor import EndpointMetError, Predictor
from tests.helpers import DETERIORATING, STABLE, cohort_request, echo, request

BUNDLE_CODES = {"abrupt_failure_not_reliably_anticipated", "phenotype_performance_unestablished"}


@pytest.fixture(scope="module")
def predictor(small_bundle, model_cfg):
    return Predictor(small_bundle, model_cfg)


def _dump(pred):
    return pred.model_dump_json()


def test_future_echoes_do_not_change_today(predictor):
    req = request(STABLE, "2018-08-01")
    base = predictor.predict(req)
    future = [echo(req.passport.passport_id, "2019-08-01", 30, 1.0, 0.30, ar="moderate"),
              echo(req.passport.passport_id, "2020-08-01", 40, 0.8, 0.25, ar="severe")]
    later = req.model_copy(update={"echo_observations": list(req.echo_observations) + future})
    assert _dump(predictor.predict(later)) == _dump(base)


def test_future_event_labels_do_not_change_today(predictor):
    req = request(STABLE, "2018-08-01")
    base = predictor.predict(req)
    p = req.passport.model_copy(update={"events": [PassportEvent(type="redo", date="2021-03-01")]})
    assert isinstance(p, Passport)
    assert _dump(predictor.predict(req.model_copy(update={"passport": p}))) == _dump(base)


def test_truth_changes_do_not_change_today(predictor, small_cohort):
    req, _ = cohort_request(small_cohort, index=2)
    base = predictor.predict(req)
    flipped = copy.deepcopy(small_cohort)
    truth = getattr(flipped, "truth", None)
    if truth is not None and "phenotype_latent" in truth:
        truth["phenotype_latent"] = truth["phenotype_latent"].map(
            lambda v: "stenotic" if v == "regurgitant" else "regurgitant")
    ev = flipped.events
    for col in [c for c in ev.columns if c.endswith("_date") and c != "implant_date"]:
        ev[col] = None
    req2, _ = cohort_request(flipped, index=2)
    req2 = req2.model_copy(update={"passport": req2.passport.model_copy(update={"passport_id": req.passport.passport_id})})
    req2 = req2.model_copy(update={"echo_observations": [o.model_copy(update={"passport_id": req.passport.passport_id})
                                                         for o in req2.echo_observations]})
    assert _dump(predictor.predict(req2)) == _dump(base)


def test_reasons_present_and_labelled(predictor):
    pred = predictor.predict(request(STABLE, "2018-08-01"))
    rel = pred.reliability
    codes = {r.code for r in rel.reasons}
    assert BUNDLE_CODES <= codes
    assert all(r.scope == "bundle" for r in rel.reasons if r.code in BUNDLE_CODES)
    assert rel.model_family == "cox" and rel.applicable_scope and "gradual" in rel.applicable_scope
    assert pred.model_family == "cox" and pred.integration_version and pred.bundle_schema_version


def test_patient_reasons_from_contemporaneous_inputs(predictor):
    stale = predictor.predict(request(STABLE, "2021-08-01"))
    codes = {r.code for r in stale.reliability.reasons}
    assert stale.reliability.stale_echo and "stale_echo" in codes
    if any(f.startswith("ac_") for f in predictor.bundle.features):
        assert "exposure_coverage_unknown" in codes


def test_bundle_reasons_for_every_patient(predictor, small_cohort):
    for i in range(3):
        req, _ = cohort_request(small_cohort, index=i)
        assert BUNDLE_CODES <= {r.code for r in predictor.predict(req).reliability.reasons}


def test_409_still_fires(predictor):
    with pytest.raises(EndpointMetError):
        predictor.predict(request(DETERIORATING, "2020-01-01"))

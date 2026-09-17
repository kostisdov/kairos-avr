"""Adjudication framework: reference study, candidates, confirmation, uncertain class."""
from datetime import date

from kairos.adjudication.framework import (
    EchoPoint,
    adjudicate,
    candidate_flags,
    select_reference,
    thrombosis_windows_from_exposures,
)
from kairos.extraction.schema import ExposureEpisode, PassportEvent


def pt(d, g, eoa, dvi, ar=0, i=0):
    return EchoPoint(date=d, mean_gradient=g, eoa=eoa, dvi=dvi, ar_ordinal=ar, index=i)


REF = pt(date(2015, 4, 1), 12, 1.8, 0.5, 0, 0)


def test_reference_window_30_to_180_days():
    implant = date(2015, 1, 15)
    early = pt(date(2015, 1, 30), 12, 1.8, 0.5)   # 15 days: too early
    ok = pt(date(2015, 4, 1), 12, 1.8, 0.5)        # 76 days
    assert select_reference([early, ok], implant) is ok
    inadequate = pt(date(2015, 3, 1), 12, None, None)  # no EOA or DVI
    assert select_reference([inadequate, ok], implant) is ok
    assert select_reference([early], implant) is None
    assert select_reference([ok], None) is None


def test_confirmed_stenotic_endpoint():
    pts = [REF, pt(date(2016, 4, 1), 14, 1.7, 0.48, 0, 1), pt(date(2018, 4, 1), 26, 1.2, 0.35, 0, 2),
           pt(date(2019, 4, 1), 30, 1.1, 0.33, 0, 3)]
    d = adjudicate(pts, REF)
    assert d.met and d.date == date(2018, 4, 1) and d.stage == "2" and d.mechanism == "svd"
    assert d.phenotype == "stenotic" and d.confidence == "high" and d.establishing_index == 2


def test_transient_finding_is_uncertain_not_endpoint():
    pts = [REF, pt(date(2017, 4, 1), 25, 1.2, 0.35, 0, 1), pt(date(2017, 10, 1), 13, 1.7, 0.49, 0, 2)]
    d = adjudicate(pts, REF)
    assert not d.met and d.uncertain_dates == [date(2017, 4, 1)]


def test_single_unconfirmed_study_is_moderate_confidence():
    pts = [REF, pt(date(2018, 4, 1), 26, 1.2, 0.35, 0, 1)]
    d = adjudicate(pts, REF)
    assert d.met and d.confidence == "moderate"


def test_regurgitant_endpoint_without_area_criteria():
    pts = [REF, pt(date(2018, 4, 1), 13, 1.8, 0.5, 3, 1), pt(date(2018, 8, 1), 14, 1.8, 0.5, 3, 2)]
    d = adjudicate(pts, REF)
    assert d.met and d.stage == "3" and d.phenotype == "regurgitant"


def test_endocarditis_mention_overrides_mechanism():
    pts = [REF, pt(date(2018, 4, 1), 26, 1.2, 0.35, 0, 1), pt(date(2018, 8, 1), 27, 1.1, 0.33, 0, 2)]
    d = adjudicate(pts, REF, events=[PassportEvent(type="endocarditis", date="2018-03-15")])
    assert not d.met and d.mechanism == "endocarditis"


def test_resolved_treated_thrombosis_returns_patient_to_risk_set():
    pts = [REF, pt(date(2017, 4, 1), 25, 1.2, 0.35, 0, 1), pt(date(2017, 12, 1), 13, 1.7, 0.49, 0, 2),
           pt(date(2019, 4, 1), 27, 1.1, 0.33, 0, 3), pt(date(2020, 4, 1), 29, 1.0, 0.31, 0, 4)]
    episodes = [ExposureEpisode(**{"class": "VKA", "indication": "suspected_valve_thrombosis", "start": "2017-04-20", "stop": "2017-11-01"})]
    windows = thrombosis_windows_from_exposures(episodes, pts, REF)
    assert windows and windows[0][0] <= date(2017, 4, 1) <= windows[0][1]
    d = adjudicate(pts, REF, thrombosis_windows=windows)
    assert d.thrombosis_attributed_dates == [date(2017, 4, 1)]
    assert d.met and d.date == date(2019, 4, 1)


def test_no_reference_means_uncertain_everywhere():
    pts = [pt(date(2018, 4, 1), 26, 1.2, 0.35, 0, 1)]
    assert candidate_flags(pts, None) == []
    assert not adjudicate(pts, None).met

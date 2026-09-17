"""Builders shared by the service and pipeline tests."""
from datetime import date

from kairos.extraction.schema import (
    EchoObservation,
    Passport,
    PassportEvent,
    PassportSource,
    PatientStatic,
    PredictRequest,
)


def passport(implant="2016-06-01", route="SAVR", model="Trifecta", size=23, events=()):
    return Passport(source=PassportSource(note_ref="test", note_type="operative", date=implant), route=route,
                    canonical_model=model, design_class="externally mounted pericardial", size_mm=size,
                    implant_date=implant, events=[PassportEvent(**e) for e in events])


def echo(pid, d, grad, eoa, dvi, ar="none", lvef=58, svi=40):
    return EchoObservation(passport_id=pid, date=d, mean_gradient_mmhg=grad, eoa_cm2=eoa, dvi=dvi, ar_grade=ar,
                           lvef_pct=lvef, svi_ml_m2=svi, native_vs_prosthetic="prosthetic", source="manual")


def request(echoes, prediction_time, p=None, static=None):
    p = p or passport()
    obs = [echo(p.passport_id, *e) for e in echoes]
    return PredictRequest(passport=p, echo_observations=obs, prediction_time=prediction_time,
                          static=static or PatientStatic(age_at_implant=66, sex="F", bsa_m2=1.8, diabetes=True, egfr_ml_min=60))


STABLE = [("2016-08-01", 10, 1.8, 0.50), ("2017-08-01", 11, 1.8, 0.49), ("2018-08-01", 12, 1.7, 0.48)]
DETERIORATING = [("2016-08-01", 10, 1.8, 0.50), ("2017-08-01", 12, 1.7, 0.47), ("2018-08-01", 24, 1.2, 0.35),
                 ("2019-08-01", 27, 1.1, 0.33)]


def cohort_request(cohort, index=0, min_echoes=3):
    """A synthetic patient with at least ``min_echoes`` echoes and no adjudicated endpoint."""
    counts = cohort.echoes.groupby("patient_id").size()
    ev = cohort.events.set_index("patient_id")["svd_adjudicated_date"]
    pid = [p for p, n in counts.items() if n >= min_echoes and ev.get(p) is None or (n >= min_echoes and ev.get(p) != ev.get(p))][index]
    pt = cohort.patients[cohort.patients.patient_id == pid].iloc[0]
    ech = cohort.echoes[cohort.echoes.patient_id == pid].sort_values("date")
    p = Passport(source=PassportSource(note_ref=pid, note_type="operative", date=pt.implant_date.isoformat()), route=pt.route,
                 canonical_model=pt.canonical_model, design_class=pt.design_class, size_mm=int(pt.size_mm),
                 implant_date=pt.implant_date.isoformat())
    obs = [echo(p.passport_id, r.date.isoformat(), float(r.mean_gradient), float(r.eoa), float(r.dvi), r.ar_grade, float(r.lvef), float(r.svi))
           for r in ech.itertuples()]
    static = PatientStatic(age_at_implant=float(pt.age_at_implant), sex=pt.sex, bsa_m2=float(pt.bsa), diabetes=bool(pt.diabetes),
                           egfr_ml_min=float(pt.egfr0), dialysis=bool(pt.dialysis), atrial_fibrillation=bool(pt.af))
    return PredictRequest(passport=p, echo_observations=obs, static=static, prediction_time=ech.iloc[-1].date.isoformat()), ech


__all__ = ["date", "passport", "echo", "request", "STABLE", "DETERIORATING", "cohort_request"]

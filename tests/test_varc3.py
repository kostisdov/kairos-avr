# -*- coding: utf-8 -*-
"""Unit tests for src/kairos/varc3.py against hand-computed VARC-3 staging examples."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kairos.varc3 import Echo, stage_hvd


def test_stage3_gradient_and_eoa():
    ref = Echo(mean_gradient_mmHg=15, eoa_cm2=1.8, dvi=0.5, regurg_grade=0)
    fu = Echo(mean_gradient_mmHg=36, eoa_cm2=0.8, dvi=0.3, regurg_grade=0)
    r = stage_hvd(fu, ref)
    assert r.stage == "3"


def test_stage2_gradient_and_eoa():
    ref = Echo(mean_gradient_mmHg=10, eoa_cm2=2.0, dvi=0.5)
    fu = Echo(mean_gradient_mmHg=22, eoa_cm2=1.4, dvi=0.45)
    r = stage_hvd(fu, ref)
    assert r.stage == "2"


def test_stage2_gradient_and_dvi_only():
    # EOA missing on follow-up, but DVI fall of >=20% present -> still stage 2
    ref = Echo(mean_gradient_mmHg=10, dvi=0.55)
    fu = Echo(mean_gradient_mmHg=21, dvi=0.4)  # DVI fall = 0.15/0.55 = 27% >= 20%
    r = stage_hvd(fu, ref)
    assert r.stage == "2"


def test_stage3_severe_regurg_overrides_gradient():
    ref = Echo(mean_gradient_mmHg=10, eoa_cm2=1.8, regurg_grade=0)
    fu = Echo(mean_gradient_mmHg=11, eoa_cm2=1.75, regurg_grade=3)  # gradient barely moved
    r = stage_hvd(fu, ref)
    assert r.stage == "3"
    assert "regurgitation" in r.reason.lower()


def test_stage2_new_moderate_regurg_overrides_gradient():
    ref = Echo(mean_gradient_mmHg=10, eoa_cm2=1.8, regurg_grade=0)
    fu = Echo(mean_gradient_mmHg=11, eoa_cm2=1.75, regurg_grade=2)
    r = stage_hvd(fu, ref)
    assert r.stage == "2"


def test_stage0_no_change():
    ref = Echo(mean_gradient_mmHg=10, eoa_cm2=1.8, dvi=0.5, regurg_grade=0)
    fu = Echo(mean_gradient_mmHg=9, eoa_cm2=1.85, dvi=0.52, regurg_grade=0)
    r = stage_hvd(fu, ref)
    assert r.stage == "0"


def test_stage1_mild_worsening_below_stage2_threshold():
    ref = Echo(mean_gradient_mmHg=10, eoa_cm2=1.8, dvi=0.5, regurg_grade=0)
    fu = Echo(mean_gradient_mmHg=13, eoa_cm2=1.7, dvi=0.48, regurg_grade=0)  # small rise only
    r = stage_hvd(fu, ref)
    assert r.stage == "1"


def test_uncertain_no_reference_at_all():
    fu = Echo(mean_gradient_mmHg=25, eoa_cm2=1.0)
    r = stage_hvd(fu, None)
    assert r.stage == "uncertain"


def test_uncertain_eoa_and_dvi_both_missing():
    ref = Echo(mean_gradient_mmHg=10)
    fu = Echo(mean_gradient_mmHg=22)
    r = stage_hvd(fu, ref)
    assert r.stage == "uncertain"


def test_uncertain_missing_gradient():
    ref = Echo(mean_gradient_mmHg=None, eoa_cm2=1.8)
    fu = Echo(mean_gradient_mmHg=22, eoa_cm2=1.4)
    r = stage_hvd(fu, ref)
    assert r.stage == "uncertain"


def test_reference_fallback_used_and_flagged():
    fallback = Echo(mean_gradient_mmHg=11, eoa_cm2=1.7, dvi=0.5)
    fu = Echo(mean_gradient_mmHg=25, eoa_cm2=1.1, dvi=0.35)
    r = stage_hvd(fu, None, reference_fallback=fallback)
    assert r.stage == "2"
    assert r.reference_source == "ASE 2024 table"
    assert r.confidence == "low"


def test_reference_fallback_not_used_when_real_reference_present():
    real_ref = Echo(mean_gradient_mmHg=10, eoa_cm2=1.8, dvi=0.5)
    fallback = Echo(mean_gradient_mmHg=5, eoa_cm2=2.5, dvi=0.7)  # would change the answer if used
    fu = Echo(mean_gradient_mmHg=9, eoa_cm2=1.85, dvi=0.51)
    r = stage_hvd(fu, real_ref, reference_fallback=fallback)
    assert r.stage == "0"
    assert r.reference_source == "patient baseline echo"
    assert r.confidence == "high"

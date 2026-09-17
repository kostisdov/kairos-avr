"""Contract tests for the payload schemas (design section 3)."""
import pytest
from pydantic import ValidationError

from kairos import CONDITIONING, HORIZONS_YEARS, ILLUSTRATIVE_LABEL
from kairos.extraction.schema import (
    PAYLOAD_MODELS,
    EchoObservation,
    ExposureEpisode,
    ExposureTimeline,
    Messages,
    Passport,
    PassportSource,
    Prediction,
    Reliability,
    parse_date,
)


def _prediction(**over):
    base = dict(passport_id="p", prediction_time="2020-01-01", p_svd_before_death=[0.02, 0.06, 0.11],
                p_death_before_svd=[0.04, 0.12, 0.21], p_alive_intact=[0.94, 0.82, 0.68],
                p_replaced_non_svd=[0.0, 0.0, 0.0], p_svd_12m=0.02,
                reliability=Reliability(device_evidence="class-level", data_completeness=0.8, stale_echo=False),
                messages=Messages(current_abnormality=False, earlier_assessment=False, overdue_surveillance=True),
                model_version="0.1.0+test", scenario_set="gradual_stenotic:abc")
    base.update(over)
    return Prediction(**base)


def test_prediction_valid_and_labelled():
    p = _prediction()
    assert p.horizons_years == list(HORIZONS_YEARS)
    assert p.conditioning == CONDITIONING
    assert ILLUSTRATIVE_LABEL in p.reliability.label


def test_prediction_probabilities_must_sum_to_one():
    with pytest.raises(ValidationError):
        _prediction(p_alive_intact=[0.90, 0.82, 0.68])


def test_prediction_rejects_wrong_horizons():
    with pytest.raises(ValidationError):
        _prediction(horizons_years=[1, 2, 5])


def test_reliability_label_cannot_be_dropped():
    with pytest.raises(ValidationError):
        Reliability(device_evidence="none", data_completeness=0.5, stale_echo=False, label="validated model")


def test_dates_accept_year_or_day():
    PassportSource(note_ref="x", note_type="progress", date="2019")
    PassportSource(note_ref="x", note_type="progress", date="2019-02-28")
    with pytest.raises(ValidationError):
        PassportSource(note_ref="x", note_type="progress", date="02/28/2019")
    assert parse_date("2019").isoformat() == "2019-01-01"


def test_exposure_class_alias_round_trip():
    tl = ExposureTimeline(passport_id="p", episodes=[ExposureEpisode(**{"class": "VKA", "agent": "warfarin", "indication": "AF",
                                                                        "start": "2016-01-01", "stop": None})])
    dumped = tl.model_dump(by_alias=True)
    assert dumped["episodes"][0]["class"] == "VKA"
    assert ExposureTimeline.model_validate(dumped).episodes[0].class_ == "VKA"


def test_echo_observation_ranges():
    with pytest.raises(ValidationError):
        EchoObservation(passport_id="p", date="2019-01-01", mean_gradient_mmhg=400)


def test_passport_ids_are_uuid_like_and_schemas_render():
    p = Passport(source=PassportSource(note_ref="x", note_type="operative", date="2019"))
    assert len(p.passport_id) == 36
    for name, model in PAYLOAD_MODELS.items():
        schema = model.model_json_schema(by_alias=True)
        assert "properties" in schema, name

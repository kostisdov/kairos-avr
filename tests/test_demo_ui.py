"""The Streamlit patient model runs end to end with the in-process backend (milestone M5)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "services" / "demo" / "app.py"
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_notes" / "02_savr_op_trifecta_hashsize.txt"


@pytest.fixture()
def app(tmp_path, monkeypatch, small_bundle):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from kairos.io.config import reset_settings_cache

    small_bundle.save(tmp_path / "models" / "latest" / "bundle.joblib")
    monkeypatch.setenv("KAIROS_LOCAL_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("KAIROS_DEMO_BACKEND", "inprocess")
    monkeypatch.setenv("KAIROS_OPENAI_ENDPOINT", "")
    reset_settings_cache()
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    yield at
    st.cache_resource.clear()
    st.cache_data.clear()
    reset_settings_cache()


def _errors(at):
    return [e.value for e in at.exception]


def test_form_loads_the_first_example(app):
    assert not _errors(app)
    assert app.title[0].value == "Patient model"
    assert len(app.image) >= 1
    assert app.radio(key="v_route").value == "SAVR" and app.selectbox(key="v_model").value == "Trifecta"
    assert app.selectbox(key="v_size").value == 21 and app.number_input(key="p_age").value == 66.0
    assert any("Time zero" in s.value for s in app.success)
    assert any("inputs are complete" in s.value for s in app.success)


def test_prediction_risk_over_time_and_what_if(app):
    app.button(key="run_predict").click().run()
    assert not _errors(app)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["SVD before death, 12 months"].endswith("%") and "Alive, valve in place, SVD-free, 5 years" in metrics
    assert any("No current abnormality" in s.value for s in app.success)  # stage 1 at the latest echo
    app.button(key="run_traj").click().run()
    assert not _errors(app)
    app.button(key="run_whatif").click().run()
    assert not _errors(app)
    assert any("Hypothetical echo on" in m.value and "stage **2**" in m.value for m in app.markdown)


def test_a_current_finding_is_shown_as_a_finding(app):
    app.selectbox(key="preset_choice").set_value("Surgical Perimount 23 mm, new severe regurgitation (58 y)").run()
    app.button(key="load_preset").click().run()
    app.button(key="run_predict").click().run()
    assert not _errors(app)
    assert any("Needs clinical assessment now" in e.value for e in app.error)


def test_editing_inputs_and_loading_another_example(app):
    app.toggle(key="p_dialysis").set_value(True).run()
    app.button(key="run_predict").click().run()
    assert not _errors(app)
    app.selectbox(key="preset_choice").set_value("Transcatheter SAPIEN 3 Ultra 26 mm, stable, AF on apixaban (76 y)").run()
    app.button(key="load_preset").click().run()
    assert not _errors(app)
    assert app.radio(key="v_route").value == "TAVR" and app.selectbox(key="v_model").value == "SAPIEN 3 Ultra"
    assert app.toggle(key="p_af").value is True
    app.button(key="run_predict").click().run()
    assert not _errors(app)
    assert "SVD before death, 5 years" in {m.label for m in app.metric}


def test_changing_the_route_keeps_the_valve_consistent(app):
    app.radio(key="v_route").set_value("TAVR").run()
    assert not _errors(app)
    assert "TAVR" in app.selectbox(key="v_class").value
    assert app.selectbox(key="v_size").value in (20, 23, 26, 29, 34, 21, 24, 27, 31, 25)


def test_family_selector_is_present(app):
    assert not _errors(app)
    selector = app.selectbox(key="model_family")
    assert selector.label == "Model family" and len(selector.options) >= 1
    assert selector.value in ("cox", "gradient_boosting")
    assert [t.label for t in app.tabs][0] == "Patient Summary"
    assert "Compare families" in [t.label for t in app.tabs]


def test_reliability_reasons_are_rendered(app):
    app.button(key="run_predict").click().run()
    assert not _errors(app)
    assert any(m.value == "**Why this estimate may be limited**" for m in app.markdown)
    assert any(m.value == "**Excluded modules**" for m in app.markdown)
    body = app.session_state["res_predict"]["body"]
    shown = " ".join(str(e.value) for group in (app.warning, app.error, app.info) for e in group)
    for reason in body["reliability"].get("reasons", []):
        assert reason["message"] in shown
    if not body["reliability"].get("reasons"):
        assert any("No reliability reasons reported" in c.value for c in app.caption)


def test_single_family_comparison_state_is_explicit(app):
    assert not _errors(app)
    assert "res_compare" not in app.session_state
    assert app.button(key="run_family_compare").disabled is True
    assert any("two-family comparison has not been run" in i.value for i in app.info)


def test_summary_requires_update_and_does_not_call_llm_on_open(app):
    assert not _errors(app)
    assert "summary_binding" not in app.session_state and "summary_draft" not in app.session_state
    app.button(key="update_summary").click().run()
    assert not _errors(app)
    assert app.session_state["summary_binding"]["comparator"]["comparator_id"] == "varc3_hvd_comparator_v1"
    assert "summary_draft" not in app.session_state
    assert app.button(key="generate_findings").label == "Generate findings"


def test_fill_from_note_applies_the_extracted_valve(app):
    app.text_area(key="note_text").input(FIXTURE.read_text(encoding="utf-8")).run()
    app.selectbox(key="note_type").set_value("operative").run()
    app.button(key="run_extract").click().run()
    assert not _errors(app)
    assert {m.label: m.value for m in app.metric}["Valve model"] == "Trifecta"
    app.selectbox(key="preset_choice").set_value("New patient: one reference echo only").run()
    app.button(key="load_preset").click().run()
    assert app.selectbox(key="v_model").value == "Magna Ease"
    app.button(key="apply_valve").click().run()
    assert not _errors(app)
    assert app.selectbox(key="v_model").value == "Trifecta" and app.selectbox(key="v_size").value == 21

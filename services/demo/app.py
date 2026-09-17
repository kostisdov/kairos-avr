"""kairos-demo: interactive patient model (design milestone M5).

Every input the prototype model uses can be set for one patient: the valve, patient
characteristics, the echo series, biomarker laboratory values and the antithrombotic exposure
history. The page then shows the three probabilities at 1, 3 and 5 years with the 12-month
risk, the main drivers, a reliability statement and the three separated messages beside the
unchanged guideline schedule; the risk at every echo; a what-if for the next echo; and filling
the form from a note through the extract service.

Backends: ``KAIROS_DEMO_BACKEND=http`` (default, used in Azure) calls the extract and predict
services over HTTP; ``KAIROS_DEMO_BACKEND=inprocess`` runs the same two FastAPI apps inside this
process for local use and tests. Every model number is trained on synthetic scenarios:
illustrative, unvalidated.
"""
from __future__ import annotations

import html
import importlib.util
import json
import os
import sys
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kairos import ILLUSTRATIVE_LABEL, __version__  # noqa: E402
from kairos import demo_patient as dp  # noqa: E402
from kairos.io.config import get_settings  # noqa: E402
from kairos.io.storage import get_store  # noqa: E402
from kairos.summary.llm import PatientSummaryWriter  # noqa: E402
from kairos.summary.patient_summary import (  # noqa: E402
    assemble_patient_summary,
    comparator_for_request,
    deterministic_summary,
    export_summary,
    snapshot_hash,
)

settings = get_settings()
BACKEND = os.environ.get("KAIROS_DEMO_BACKEND", "http").strip().lower()
PUBLISHED = "Synthetic cohort patient published by the train job"
HORIZONS = ["1 year", "3 years", "5 years"]
OUTCOME_COLORS = ["#c0392b", "#7f8c8d", "#d68910", "#1e8449"]
MIN_DATE, MAX_DATE = date(1995, 1, 1), date(2040, 12, 31)
BANNER = ("Research prototype. Every prediction on this page comes from a model trained on synthetic scenarios: "
          "illustrative, unvalidated. Nothing here is a clinical claim, a treatment recommendation or a change to "
          "guideline surveillance.")
SAMPLE_NOTE = ("PROGRESS NOTE (SYNTHETIC)\n\nSurveillance visit after surgical aortic valve replacement with a #23 Magna Ease "
               "bioprosthesis.\n\nEchocardiographic Findings: the prosthetic valve mean gradient is 16 mmHg, the effective "
               "orifice area (EOA) is 1.5 cm2 and the dimensionless valve index is 0.44. Mild prosthetic regurgitation. "
               "LVEF 57%.\n\nPlan: continue annual surveillance echocardiography.")

st.set_page_config(page_title="KAIROS patient model", page_icon="🫀", layout="wide")
ss = st.session_state


# --- backend --------------------------------------------------------------------------------------
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # pydantic resolves forward references through sys.modules
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner="Starting the extract and predict services in-process")
def _inprocess() -> dict:
    from fastapi.testclient import TestClient

    extract_mod = _load_module("kairos_demo_extract_app", ROOT / "services" / "extract" / "app.py")
    predict_mod = _load_module("kairos_demo_predict_app", ROOT / "services" / "predict" / "app.py")
    bundle, bundle_error = None, None
    try:
        bundle = predict_mod.load_bundle_from_store(settings)
    except Exception as exc:  # noqa: BLE001 - shown in the sidebar; predict then answers 503
        bundle_error = f"{type(exc).__name__}: {exc}"
    return {"extract": TestClient(extract_mod.create_app(settings=settings, persist=False)),
            "predict": TestClient(predict_mod.create_app(settings=settings, bundle=bundle, persist=False)),
            "bundle_error": bundle_error}


def call(service: str, method: str, path: str, **kwargs):
    """Returns (response, error message); never raises."""
    try:
        if BACKEND == "inprocess":
            return _inprocess()[service].request(method, path, **kwargs), None
        base = (settings.extract_url if service == "extract" else settings.predict_url).rstrip("/")
        with httpx.Client(timeout=180) as client:  # a reasoning-model extraction can take tens of seconds
            return client.request(method, f"{base}{path}", **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, (f"The {service} service is not reachable ({type(exc).__name__}). Start it, or run the demo "
                      "with KAIROS_DEMO_BACKEND=inprocess.")


def detail(response) -> str:
    """The error message: ``detail.message`` of the structured error contract, else the plain string."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    if not isinstance(body, dict):
        return response.text
    value = body.get("detail", body)
    if isinstance(value, dict):
        message = value.get("message")
        if message:
            return str(message) + (f" [{value['code']}]" if value.get("code") else "")
    return str(value)


def result(response, error) -> dict:
    if error:
        return {"error": error}
    ok = response.status_code == 200
    return {"status": response.status_code, "body": response.json() if ok else None, "detail": "" if ok else detail(response)}


@st.cache_data(ttl=120, show_spinner=False)
def model_card(family: str | None = None) -> dict | None:
    response, error = call("predict", "GET", "/model", params={"family": family} if family else None)
    if response is None or response.status_code != 200:
        return None
    card = response.json()
    actual = card.get("family") if isinstance(card, dict) else None
    if family and actual != family:
        return None
    return card


DEFAULT_FAMILIES = ["cox", "gradient_boosting"]
FAMILY_LABELS = {"cox": "Cox", "gradient_boosting": "Gradient boosting"}


def parse_models(body) -> dict[str, dict]:
    """Normalise ``GET /models`` (a list of entries or a dict keyed by family) to {family: info}."""
    if isinstance(body, dict):
        for key in ("families", "models", "loaded"):
            if key in body and isinstance(body[key], (list, dict)):
                return parse_models(body[key])
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in body.items()
                if isinstance(v, dict) or v is None}
    out: dict[str, dict] = {}
    for entry in body if isinstance(body, list) else []:
        if isinstance(entry, str):
            out[entry] = {}
        elif isinstance(entry, dict) and (entry.get("family") or entry.get("model_family")):
            out[str(entry.get("family") or entry.get("model_family"))] = entry
    return out


@st.cache_data(ttl=120, show_spinner=False)
def model_inventory() -> dict:
    response, error = call("predict", "GET", "/models")
    if response is None or response.status_code != 200:
        return {"verified": False, "default_family": None, "namespace": None, "models": {},
                "error": error or (detail(response) if response is not None else "discovery failed")}
    try:
        body = response.json()
        models = parse_models(body)
        default = body.get("default_family") if isinstance(body, dict) else None
        namespace = body.get("namespace") if isinstance(body, dict) else None
        return {"verified": True, "default_family": default, "namespace": namespace,
                "models": models, "error": None}
    except ValueError:
        return {"verified": False, "default_family": None, "namespace": None, "models": {},
                "error": "the family inventory was not valid JSON"}


def loaded_models() -> dict[str, dict]:
    """Only service-listed families explicitly compatible with the service default."""
    inventory = model_inventory()
    return {family: info for family, info in inventory["models"].items()
            if info.get("compatible_with_default", True) is True}


def family_label(family: str | None) -> str:
    return FAMILY_LABELS.get(family or "", family or "unknown family")


def stamp(p: dict, requested: str | None = None) -> str:
    """Family plus version, written next to every number."""
    family = p.get("model_family") or p.get("reliability", {}).get("model_family")
    shown = family_label(family) if family else f"{family_label(requested)} requested; family not reported"
    return f"{shown}, model {p.get('model_version', 'unknown')}"


@st.cache_data(ttl=600, show_spinner=False)
def guideline(jurisdiction: str) -> dict | None:
    response, error = call("predict", "GET", f"/guideline/{jurisdiction}")
    return response.json() if (response is not None and response.status_code == 200) else None


def published_patient() -> dict | None:
    try:
        return get_store(settings).get_json("models", f"{settings.model_blob_prefix}/demo_patient.json")
    except Exception:  # noqa: BLE001
        return None


# --- form state -----------------------------------------------------------------------------------
def put_form(form: dict) -> None:
    """Load a form into the widgets. Call before the widgets render (initialisation or callbacks)."""
    valve, patient = form["valve"], form["patient"]
    ss["v_route"] = valve.get("route") or "SAVR"
    ss["v_class"] = valve.get("design_class")
    ss["v_model"] = valve.get("canonical_model")
    ss["v_size"] = int(valve["size_mm"]) if dp.to_num(valve.get("size_mm")) is not None else None
    ss["v_implant"] = dp.to_date(valve.get("implant_date")) or date.today()
    ss["p_age"] = float(patient["age_at_implant"])
    ss["p_sex"] = patient["sex"]
    ss["p_bsa"] = float(patient["bsa_m2"])
    ss["p_bmi"] = float(patient["bmi"])
    ss["p_egfr"] = float(patient["egfr_ml_min"])
    ss["p_diabetes"] = bool(patient["diabetes"])
    ss["p_diab_years"] = float(patient.get("diabetes_duration_years") or 0.0)
    ss["p_dialysis"] = bool(patient["dialysis"])
    ss["p_af"] = bool(patient["atrial_fibrillation"])
    ss["p_bicuspid"] = bool(patient["bicuspid_native_valve"])
    ss["p_lipid"] = bool(patient["lipid_lowering"])
    ss["echo_df"] = ss["echo_now"] = dp.echo_frame(form["echoes"])
    ss["labs_df"] = ss["labs_now"] = dp.labs_frame(form["labs"])
    ss["ac_df"] = ss["ac_now"] = dp.episodes_frame(form["episodes"])
    ss["editor_v"] = ss.get("editor_v", 0) + 1
    ss["prediction_time"] = dp.to_date(form.get("prediction_time")) or date.today()
    ss["jurisdiction"] = form.get("jurisdiction") or "ESC_EACTS"
    ss["source_provenance"] = form.get("source_provenance") or "synthetic"
    for key in ("res_predict", "res_traj", "res_whatif", "summary_binding", "summary_draft"):
        ss.pop(key, None)


def form_from_state() -> dict:
    return {"valve": {"route": ss.get("v_route"), "design_class": ss.get("v_class"), "canonical_model": ss.get("v_model"),
                      "size_mm": ss.get("v_size"), "implant_date": ss.get("v_implant")},
            "patient": {"age_at_implant": ss.get("p_age"), "sex": ss.get("p_sex"), "bsa_m2": ss.get("p_bsa"),
                        "bmi": ss.get("p_bmi"), "egfr_ml_min": ss.get("p_egfr"), "diabetes": ss.get("p_diabetes"),
                        "diabetes_duration_years": ss.get("p_diab_years"), "dialysis": ss.get("p_dialysis"),
                        "atrial_fibrillation": ss.get("p_af"), "bicuspid_native_valve": ss.get("p_bicuspid"),
                        "lipid_lowering": ss.get("p_lipid")},
            "echoes": dp.echo_records(ss["echo_now"]), "labs": dp.lab_records(ss["labs_now"]),
            "episodes": dp.episode_records(ss["ac_now"]),
            "prediction_time": ss.get("prediction_time"), "jurisdiction": ss.get("jurisdiction")}


def ensure_option(key: str, options: list) -> None:
    if options and ss.get(key) not in options:
        ss[key] = options[0]


def cb_load_example() -> None:
    choice = ss.get("preset_choice")
    if choice == PUBLISHED:
        story = published_patient()
        if story is None:
            ss["notice"] = ("warning", "No synthetic patient has been published yet. Run the train job first.")
            return
        put_form(dp.form_from_request(story["request"]))
        ss["notice"] = ("info", f"Loaded synthetic patient {story.get('patient_id')} from the {story.get('scenario')} scenario.")
        return
    put_form(dp.PRESETS[choice]())
    ss["notice"] = ("info", f"Loaded the synthetic example: {choice}.")


def cb_apply_extracted_valve() -> None:
    body = (ss.get("res_extract") or {}).get("body") or {}
    form, notes = dp.apply_passport(form_from_state(), body.get("passport", {}))
    put_form(form)
    ss["source_provenance"] = body.get("source_kind") or "unknown"
    ss["notice"] = ("info", "Extracted valve fields applied." + (" Note: " + "; ".join(notes) + "." if notes else ""))


def cb_add_extracted_echoes() -> None:
    body = (ss.get("res_extract") or {}).get("body") or {}
    form, added = dp.add_echo_observations(form_from_state(), body.get("echo_observations", []))
    put_form(form)
    ss["source_provenance"] = body.get("source_kind") or "unknown"
    ss["notice"] = ("info", f"Added {added} prosthetic echo observation(s) to the series.")


if "initialised" not in ss:
    first = next(iter(dp.PRESETS))
    put_form(dp.PRESETS[first]())
    ss["preset_choice"] = first
    ss["note_text"] = SAMPLE_NOTE
    ss["note_date"] = date.today()
    ss["initialised"] = True


# --- rendering helpers ------------------------------------------------------------------------------
def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def highlight(text: str, evidence: list[dict]) -> str:
    spans = sorted({(e["span"][0], e["span"][1], e["field"]) for e in evidence if e.get("span")}, key=lambda s: s[0])
    out, pos = [], 0
    for start, end, field in spans:
        if start < pos:
            continue
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark title="{html.escape(field)}">{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return ('<div style="white-space:pre-wrap;font-family:monospace;font-size:0.85em;border:1px solid #ddd;'
            'border-radius:6px;padding:10px">' + "".join(out) + "</div>")


def outcome_frame(p: dict) -> pd.DataFrame:
    return pd.DataFrame({"SVD before death": p["p_svd_before_death"], "Death before SVD": p["p_death_before_svd"],
                         "Replaced for another cause": p["p_replaced_non_svd"],
                         "Alive, valve in place, SVD-free": p["p_alive_intact"]}, index=HORIZONS) * 100


def render_messages(p: dict) -> None:
    st.subheader("Three separate messages")
    flags, notes = p["messages"], p.get("message_explanations", {})
    specs = [("Finding: current abnormality", "current_abnormality", "Needs clinical assessment now", "No current abnormality", st.error),
             ("Prediction: earlier assessment", "earlier_assessment", "Predicted 12-month risk would bring the next assessment forward",
              "Predicted 12-month risk is below the threshold", st.warning),
             ("Reminder: surveillance", "overdue_surveillance", "The scheduled echo is overdue", "Surveillance is on schedule", st.warning)]
    for column, (title, key, on_text, off_text, alert) in zip(st.columns(3, gap="medium"), specs):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            (alert if flags[key] else st.success)(on_text if flags[key] else off_text)
            st.caption(notes.get(key, ""))


def render_guideline(jurisdiction: str) -> None:
    g = guideline(jurisdiction)
    with st.expander("Guideline surveillance schedule (displayed unchanged)", expanded=True):
        if g is None:
            st.caption("The guideline schedule is unavailable.")
            return
        st.write(g.get("description", ""))
        if isinstance(g.get("interval_months"), dict):
            st.write("Imaging interval: " + "; ".join(f"{route} every {months} months" for route, months in g["interval_months"].items()))
        if isinstance(g.get("SAVR"), dict):
            st.write(f"Surgical valves: echo at {' and '.join(str(y) for y in g['SAVR']['milestones_years'])} years, "
                     f"then yearly from year {g['SAVR']['yearly_after_years']}.")
        if isinstance(g.get("TAVR"), dict):
            st.write(f"Transcatheter valves: echo every {g['TAVR']['interval_months']} months.")
        st.caption(g.get("note", ""))


ABRUPT_CODE = "abrupt_failure_not_reliably_anticipated"
ABRUPT_FALLBACK = "Abrupt (regurgitant) valve failure is not reliably anticipated by this model."
SEVERITY_ALERT = {"blocking": "error", "warning": "warning", "info": "info"}


def reasons_of(p: dict) -> list[dict]:
    reasons = (p.get("reliability") or {}).get("reasons") or []
    return [r for r in reasons if isinstance(r, dict)]


def render_abrupt_sentence(p: dict) -> None:
    """The concise abrupt-failure sentence, placed directly beside the numbers."""
    reason = next((r for r in reasons_of(p) if r.get("code") == ABRUPT_CODE), None)
    if reason is not None:
        st.warning(reason.get("message") or ABRUPT_FALLBACK, icon="⚡")
    else:
        st.caption("The service reported no abrupt-failure limitation for this prediction.")


def render_drivers(p: dict) -> None:
    """Grouped sensitivities: the change in 12-month probability when present, else the direction only."""
    rows = []
    for d in p.get("drivers") or []:
        row = {"feature": d.get("feature"), "value": d.get("value"),
               "effect on SVD risk": "raises" if d.get("direction") == "up" else "lowers"}
        if d.get("delta_probability") is not None:
            row["change in 12-month SVD risk"] = f"{100 * float(d['delta_probability']):+.1f} points"
        if d.get("method"):
            row["method"] = d["method"]
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Model sensitivities, not treatment effects. {stamp(p)}.")
    else:
        st.caption("No driver stands out for this patient.")


def render_reason_list(p: dict, *, heading: bool = True) -> None:
    """One structured reason renderer shared by prediction, summary and comparison."""
    if heading:
        st.markdown("**Why this estimate may be limited**")
    rel = p.get("reliability")
    if not isinstance(rel, dict) or "reasons" not in rel:
        st.caption("Reasons not supplied.")
        return
    reasons = reasons_of(p)
    if not reasons:
        st.caption("No reliability reasons reported.")
        return
    order = {"blocking": 0, "warning": 1, "info": 2}
    for index, reason in enumerate(sorted(reasons, key=lambda r: order.get(r.get("severity"), 3))):
        severity = reason.get("severity", "info")
        alert = getattr(st, SEVERITY_ALERT.get(severity, "info"))
        alert(reason.get("message") or "Unspecified reliability reason")
        with st.expander(f"Reason details: {reason.get('code', 'unknown')}", expanded=False):
            st.write(f"Code: `{reason.get('code', 'unknown')}`")
            st.write(f"Scope: `{reason.get('scope', 'unknown')}`; severity: `{severity}`")
            if reason.get("detail"):
                st.json(reason["detail"])


def render_reliability(p: dict) -> None:
    st.subheader("Reliability")
    rel = p.get("reliability") or {}
    st.write(f"Device evidence: **{rel.get('device_evidence', 'unknown')}**")
    completeness = float(rel.get("data_completeness") or 0.0)
    st.progress(min(1.0, max(0.0, completeness)), text=f"Data completeness {completeness:.0%}")
    st.write("Latest echo older than 18 months: **yes**" if rel.get("stale_echo") else "Latest echo older than 18 months: no")
    if rel.get("applicable_scope"):
        st.caption(f"Applicable scope: {rel['applicable_scope']}.")
    render_reason_list(p)
    st.markdown("**Excluded modules**")
    excluded_rows = []
    for name, info in (rel.get("modules") or {}).items():
        info = info if isinstance(info, dict) else {}
        excluded = info.get("excluded") or {}
        if info.get("status") != "full" or excluded:
            excluded_rows.append({"module": name, "status": info.get("status", "unknown"),
                                  "included": ", ".join(info.get("included") or []) or "none",
                                  "excluded": "; ".join(f"{k}: {v}" for k, v in excluded.items()) if isinstance(excluded, dict)
                                  else ", ".join(map(str, excluded))})
    if excluded_rows:
        st.dataframe(pd.DataFrame(excluded_rows), hide_index=True, width="stretch")
    else:
        st.caption("No module excluded." if rel.get("modules") else "The service did not report module status.")
    if rel.get("excluded_features"):
        st.caption("Excluded features: " + ", ".join(map(str, rel["excluded_features"])) + ".")
    st.caption(f"{stamp(p)}, trained on synthetic scenario set {p.get('scenario_set', 'unknown')}.")


def render_compare(results: dict[str, dict]) -> None:
    """Verified family results side by side, retaining partial success and structured reasons."""
    if not results:
        st.info("Complete the patient inputs first.")
        return
    ok = {fam: res["body"] for fam, res in results.items() if res.get("status") == 200 and res.get("body")}
    for fam, res in results.items():
        if fam in ok:
            continue
        if res.get("error"):
            st.error(f"{family_label(fam)}: {res['error']}")
        elif res.get("status") == 409:
            st.warning(f"{family_label(fam)}: no prediction: {res['detail']}")
        else:
            st.error(f"{family_label(fam)}: the predict service answered {res.get('status')}: {res.get('detail')}")
    if not ok:
        return
    outcomes = [("SVD before death", "p_svd_before_death"), ("Death before SVD", "p_death_before_svd"),
                ("Replaced for another cause", "p_replaced_non_svd"), ("Alive, valve in place, SVD-free", "p_alive_intact")]
    rows = [{"outcome": "SVD before death", "horizon": "12 months",
             **{family_label(fam): pct(p["p_svd_12m"]) for fam, p in ok.items()}}]
    for label, key in outcomes:
        for i, horizon in enumerate(HORIZONS):
            rows.append({"outcome": label, "horizon": horizon,
                         **{family_label(fam): pct(p[key][i]) for fam, p in ok.items()}})
    st.caption("Assessment date: " + str(next(iter(ok.values())).get("prediction_time", "unknown")))
    for column, (fam, p) in zip(st.columns(len(ok), gap="large"), ok.items()):
        with column:
            info = models_info.get(fam) or {}
            st.markdown(f"**{family_label(fam)} — {p.get('model_version', 'unknown')}**")
            st.caption(f"Ladder step {info.get('ladder_step', 'unknown')}; scenario {p.get('scenario_set', 'unknown')}.")
            num_col, note_col = st.columns([1, 1])
            num_col.metric("SVD before death, 12 months", pct(p["p_svd_12m"]), border=True)
            num_col.caption("Earlier assessment: " + ("yes" if p.get("messages", {}).get("earlier_assessment") else "no"))
            with note_col:
                render_abrupt_sentence(p)
            render_reason_list(p)
            st.markdown("**What influenced this estimate**")
            render_drivers(p)
    if len(ok) >= 2:
        families = list(ok)[:2]
        left, right = ok[families[0]], ok[families[1]]
        cards = {fam: model_card(fam) or {} for fam in families}
        endpoint_versions = [((cards[f].get("training_summary") or {}).get("dataset") or {}).get("endpoint_version") for f in families]
        comparable = (all(p.get("integration_version") for p in (left, right))
                      and left.get("integration_version") == right.get("integration_version")
                      and all(p.get("bundle_schema_version") is not None for p in (left, right))
                      and left.get("bundle_schema_version") == right.get("bundle_schema_version")
                      and left.get("horizons_years") == right.get("horizons_years")
                      and all(endpoint_versions) and len(set(endpoint_versions)) == 1)
        if comparable:
            gb = next((p for fam, p in ok.items() if fam == "gradient_boosting"), None)
            cox = next((p for fam, p in ok.items() if fam == "cox"), None)
            if gb and cox:
                st.metric("GB minus Cox", f"{100 * (gb['p_svd_12m'] - cox['p_svd_12m']):+.1f} percentage points",
                          help="Arithmetic patient-level difference, not an improvement score.")
        else:
            st.warning("Different model configurations or uncertain provenance — the arithmetic family delta is suppressed.")
        fingerprints = {}
        for fam in families:
            fingerprints[fam] = {json.dumps({k: r.get(k) for k in ("code", "scope", "severity", "message", "detail")},
                                                 sort_keys=True, default=str): r for r in reasons_of(ok[fam])}
        common = set(fingerprints[families[0]]) & set(fingerprints[families[1]])
        if common:
            st.markdown("**Shared reliability reasons (apply to both families)**")
            for key in sorted(common):
                reason = fingerprints[families[0]][key]
                getattr(st, SEVERITY_ALERT.get(reason.get("severity"), "info"))(reason.get("message", ""))
    st.markdown("**Outcome probability matrix**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_prediction(res: dict | None) -> None:
    if not res:
        st.info("Set the inputs, then run the prediction.")
        return
    if res.get("error"):
        st.error(res["error"])
        return
    if res["status"] == 409:
        st.warning(f"No prediction: {res['detail']}")
        st.caption("Move the prediction time before the confirmed finding, or edit the echo series.")
        return
    if res["status"] != 200:
        st.error(f"The predict service answered {res['status']}: {res['detail']}")
        return
    p = res["body"]
    st.caption(f"Prediction at {p['prediction_time']}, for a patient who is {p['conditioning']} at that time. "
               f"{p['reliability']['label'].capitalize()}. {stamp(p, res.get('family'))}.")
    cols = st.columns([1, 1, 1, 1, 1.4])
    cols[0].metric("SVD before death, 12 months", pct(p["p_svd_12m"]), border=True,
                   help="Near-term risk: the only horizon that can bring the next assessment forward.")
    cols[1].metric("SVD before death, 5 years", pct(p["p_svd_before_death"][2]), border=True)
    cols[2].metric("Death before SVD, 5 years", pct(p["p_death_before_svd"][2]), border=True)
    cols[3].metric("Alive, valve in place, SVD-free, 5 years", pct(p["p_alive_intact"][2]), border=True)
    with cols[4]:
        render_abrupt_sentence(p)
    frame = outcome_frame(p)
    chart_col, table_col = st.columns([3, 2], gap="large")
    chart_col.bar_chart(frame, horizontal=True, stack=True, color=OUTCOME_COLORS, height=280)
    table_col.dataframe(frame.map(lambda v: f"{v:.1f}%"), width="stretch")
    table_col.caption("The four outcomes add up to 100% at each horizon. One minus the SVD probability is not the "
                      "probability of a functioning valve, because it includes patients who died first.")
    drivers_col, reliability_col = st.columns([3, 2], gap="large")
    with drivers_col:
        st.subheader("Main drivers")
        render_drivers(p)
    with reliability_col:
        render_reliability(p)
    render_dp_ucmgp(p.get("dp_ucmgp"))
    render_messages(p)
    render_guideline(ss.get("jurisdiction") or "ESC_EACTS")


DP_UCMGP_STATUS = {"used": "used in this prediction", "not_measured": "not measured", "stale": "measured, but stale and not used",
                   "not_in_model": "measured, but this model has no dp-ucMGP substudy model"}


def render_dp_ucmgp(status: dict | None) -> None:
    st.subheader("Vitamin K status (dp-ucMGP)")
    if not status:
        st.caption("The predict service did not report dp-ucMGP use.")
        return
    used = status["measured_value_used"]
    label = DP_UCMGP_STATUS.get(status["status"], status["status"])
    value = f" {status['value_pmol_l']:.0f} pmol/L on {status['measured_on']}" if status.get("value_pmol_l") is not None else ""
    (st.success if used else st.info)(f"Measured dp-ucMGP value {label}.{value}")
    st.caption(status.get("note", "") + (f" Assay: {status['assay']}." if status.get("assay") else ""))


def render_trajectory(res: dict | None, form: dict) -> None:
    if not res:
        st.info("Compute the risk at every echo from time zero to the prediction time.")
        return
    if res.get("error") or res.get("status") != 200:
        st.error(res.get("error") or f"The predict service answered {res['status']}: {res['detail']}")
        return
    rows = []
    for item in res["body"]["items"]:
        p = item.get("prediction")
        if p:
            rows.append({"landmark": item["prediction_time"], "SVD before death, 12 months": 100 * p["p_svd_12m"],
                         "SVD before death, 5 years": 100 * p["p_svd_before_death"][2],
                         "Death before SVD, 5 years": 100 * p["p_death_before_svd"][2],
                         "Alive, valve in place, SVD-free, 5 years": 100 * p["p_alive_intact"][2],
                         "finding": "yes" if p["messages"]["current_abnormality"] else "",
                         "earlier assessment": "yes" if p["messages"]["earlier_assessment"] else "", "status": "predicted"})
        else:
            rows.append({"landmark": item["prediction_time"], "status": f"no prediction ({item['status']}): {item['error']}"})
    table = pd.DataFrame(rows)
    predicted = table[table["status"] == "predicted"].copy()
    risk_cols = ["SVD before death, 12 months", "SVD before death, 5 years", "Death before SVD, 5 years",
                 "Alive, valve in place, SVD-free, 5 years"]
    if len(predicted):
        predicted["landmark"] = pd.to_datetime(predicted["landmark"])
        st.markdown("**Probabilities at each echo (percent)**")
        st.line_chart(predicted.set_index("landmark")[risk_cols], color=["#f1948a", "#c0392b", "#7f8c8d", "#1e8449"], height=300)
    echoes = pd.DataFrame(form["echoes"])
    if len(echoes):
        echoes["date"] = pd.to_datetime(echoes["date"], errors="coerce")
        echoes = echoes.dropna(subset=["date"]).sort_values("date").set_index("date")
        left, right = st.columns(2)
        left.markdown("**Mean gradient (mmHg)**")
        left.line_chart(echoes[["mean_gradient_mmhg"]], height=200)
        right.markdown("**EOA (cm²) and DVI**")
        right.line_chart(echoes[["eoa_cm2", "dvi"]], height=200)
    shown = table.copy()
    for col in risk_cols:
        if col in shown:
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.1f}%")
    st.dataframe(shown, hide_index=True, width="stretch")


def render_whatif(res: dict | None) -> None:
    if not res:
        st.info("Choose the values of a hypothetical next echo, then compare.")
        return
    if res.get("form_error"):
        st.error(res["form_error"])
        return
    stage = res.get("new_stage")
    if stage:
        st.write(f"Hypothetical echo on {stage['date']}: VARC-3 stage **{stage['VARC-3 stage']}** against the reference "
                 f"study ({stage['note']}).")
    now = res["now"]
    base = now.get("body") if now.get("status") == 200 else None
    for column, title, outcome, when in zip(st.columns(2, gap="large"), ["Now, at the latest echo", "After the hypothetical echo"],
                                            [res["now"], res["then"]], res["dates"]):
        with column:
            st.markdown(f"**{title}** ({when})")
            if outcome.get("error"):
                st.error(outcome["error"])
                continue
            if outcome["status"] == 409:
                st.warning(f"No prediction: {outcome['detail']}")
                continue
            if outcome["status"] != 200:
                st.error(f"The predict service answered {outcome['status']}: {outcome['detail']}")
                continue
            p = outcome["body"]
            compare = base is not None and title.startswith("After")

            def delta(value: float, base_value: float, compare: bool = compare):
                return f"{100 * (value - base_value):+.1f} points" if compare else None

            st.metric("SVD before death, 12 months", pct(p["p_svd_12m"]), border=True, delta_color="inverse",
                      delta=delta(p["p_svd_12m"], base["p_svd_12m"]) if compare else None)
            st.metric("SVD before death, 5 years", pct(p["p_svd_before_death"][2]), border=True, delta_color="inverse",
                      delta=delta(p["p_svd_before_death"][2], base["p_svd_before_death"][2]) if compare else None)
            st.metric("Death before SVD, 5 years", pct(p["p_death_before_svd"][2]), border=True, delta_color="off",
                      delta=delta(p["p_death_before_svd"][2], base["p_death_before_svd"][2]) if compare else None)
            flags = [label for label, key in (("current abnormality", "current_abnormality"),
                                              ("earlier assessment", "earlier_assessment"),
                                              ("overdue surveillance", "overdue_surveillance")) if p["messages"][key]]
            st.caption("Messages: " + (", ".join(flags) if flags else "none"))


def findings_markdown(draft) -> str:
    sections = [("Findings", draft.findings), ("Interpretation", draft.interpretation),
                ("Recommendations for clinical review", draft.recommendations), ("Limitations", draft.limitations)]
    lines = []
    for title, items in sections:
        lines.extend([f"**{title}**", ""])
        for item in items:
            text = getattr(item, "text", None) or getattr(item, "rationale", "")
            evidence = ", ".join(f"`{eid}`" for eid in item.evidence_ids)
            lines.append(f"- {text} ({evidence})")
        lines.append("")
    return "\n".join(lines)


def render_comparator(comparator: dict) -> None:
    status = comparator.get("status", "indeterminate")
    alert = st.success if status == "positive" else st.info if status == "negative" else st.warning
    displayed = "Criteria not met" if status == "negative" else status.capitalize()
    alert(f"Independent comparator assessment: {displayed}")
    st.caption("VARC-3-derived HVD criteria using the KAIROS reference window. A negative result means that the "
               "moderate/severe change criteria were not met; it does not mean that the valve is normal.")
    branches = comparator.get("branches") or {}
    branch_rows = [{"branch": key.replace("_", " "),
                    "result": "met" if value is True else "not met" if value is False else "unknown"}
                   for key, value in branches.items()]
    if branch_rows:
        st.dataframe(pd.DataFrame(branch_rows), hide_index=True, width="stretch")
    if comparator.get("missing_inputs") or comparator.get("reason_codes"):
        st.caption("Why the comparator triggered or is incomplete: "
                   + "; ".join([*(comparator.get("reason_codes") or []),
                                *(f"missing {x}" for x in comparator.get("missing_inputs") or [])]))


def render_patient_summary(request, payload: dict | None, payload_key: str | None, selected_family: str | None,
                           source_kind: str, comparison: dict | None) -> None:
    st.header("Patient Summary")
    st.caption("Consolidated consultation view. The selected model estimates future risk; the independent echo rule "
               "asks whether current change criteria are met. Neither output changes the other.")
    if request is None or payload is None:
        st.info("Complete the Patient inputs before updating the summary.")
        return
    binding = ss.get("summary_binding")
    current = bool(binding and binding.get("snapshot_key") == payload_key
                   and binding.get("selected_family") == selected_family
                   and binding.get("namespace") == inventory.get("namespace")
                   and (selected_family is None or binding.get("model_version")
                        == (models_info.get(selected_family) or {}).get("model_version")))
    if binding and not current:
        st.warning("Out of date — update summary. The previous result and findings are excluded from this view and download.")
    update = st.button("Update summary", key="update_summary", type="primary")
    if update:
        response = None
        if selected_family:
            existing = ss.get("res_predict")
            if (existing and existing.get("snapshot_key") == payload_key and existing.get("status") == 200
                    and existing.get("requested_family") == selected_family):
                response = existing
            else:
                response = run_or_reuse(payload, selected_family)
                ss["res_predict"] = response
        comparator = comparator_for_request(request, source_kind=source_kind)
        family_compare = None
        include_compare = bool(ss.get("include_family_comparison"))
        if include_compare and comparison and comparison.get("key") == compare_key:
            valid = {fam: res["body"] for fam, res in comparison.get("results", {}).items()
                     if res.get("status") == 200 and res.get("body")}
            if len(valid) >= 2:
                family_compare = {fam: {"model_version": p.get("model_version"), "p_svd_12m": p.get("p_svd_12m"),
                                        "earlier_assessment": (p.get("messages") or {}).get("earlier_assessment")}
                                  for fam, p in valid.items()}
        model_body = response.get("body") if response and response.get("status") == 200 else None
        facts = assemble_patient_summary(request, model_body, comparator, selected_family=selected_family,
                                         source_provenance=source_kind, family_comparison=family_compare)
        ss["summary_binding"] = binding = {
            "snapshot_key": payload_key, "selected_family": selected_family,
            "returned_family": facts.returned_family, "model_version": facts.model_version,
            "namespace": inventory.get("namespace"), "threshold": 0.05,
            "request": json.loads(json.dumps(payload, default=str)), "response": model_body,
            "comparator": comparator.to_dict(), "facts": facts.model_dump(mode="json"),
            "include_family_comparison": bool(family_compare),
        }
        ss.pop("summary_draft", None)
        current = True
    if not current:
        st.info("Select Update summary to bind the current inputs, model identity and independent comparator. "
                "No hosted-model request is made by opening this tab.")
        return

    facts_data = binding["facts"]
    from kairos.summary.schema import PatientSummaryFacts

    facts = PatientSummaryFacts.model_validate(facts_data)
    response, comparator = binding.get("response"), binding["comparator"]
    fact_map = {f.id: f for f in facts.facts}
    st.subheader("Patient and valve")
    header_cols = st.columns(5)
    for col, fact_id, label in zip(header_cols,
                                    ["patient.case_id", "patient.source", "patient.assessment_date", "valve.route", "valve.age_years"],
                                    ["Case", "Source", "Assessment", "Route", "Valve age"]):
        fact = fact_map.get(fact_id)
        col.metric(label, f"{fact.value} {fact.unit or ''}".strip() if fact else "Not available")

    st.subheader("Key results")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Selected model", f"{family_label(facts.returned_family or facts.selected_family)} / {facts.model_version or 'Unavailable'}")
    if response:
        k2.metric("Twelve-month SVD risk", pct(response["p_svd_12m"]))
        flag = (response.get("messages") or {}).get("earlier_assessment")
        k3.metric("Earlier-assessment flag", "Yes" if flag else "No", help="Unchanged assumed threshold: 5%")
    else:
        k2.metric("Twelve-month SVD risk", "Unavailable")
        k3.metric("Earlier-assessment flag", "Unavailable")
    k4.metric("Comparator", comparator.get("status", "indeterminate").capitalize())
    st.caption(BANNER)
    if response:
        render_reason_list(response)

    st.subheader("Reference versus current echo")
    echo_rows = []
    for field, label, unit in (("mean_gradient_mmhg", "Mean gradient", "mmHg"), ("eoa_cm2", "EOA", "cm²"),
                               ("dvi", "DVI", ""), ("intraprosthetic_ar_grade", "Intraprosthetic AR", "grade"),
                               ("lvef_pct", "LVEF", "%"), ("svi_ml_m2", "Stroke-volume index", "mL/m²")):
        rv = (comparator.get("reference_values") or {}).get(field)
        cv = (comparator.get("current_values") or {}).get(field)
        echo_rows.append({"measurement": label, "reference": "Not available" if rv is None else f"{rv:g} {unit}".strip(),
                          "current": "Not available" if cv is None else f"{cv:g} {unit}".strip()})
    st.caption(f"Reference: {comparator.get('reference_study_date') or 'Not available'}; current: "
               f"{comparator.get('current_study_date') or 'Not available'}.")
    st.dataframe(pd.DataFrame(echo_rows), hide_index=True, width="stretch")

    st.subheader("Key clinical inputs")
    clinical = [f for f in facts.facts if f.category in ("patient", "laboratory", "exposure")
                and f.id not in ("patient.case_id", "patient.source", "patient.assessment_date")]
    st.dataframe(pd.DataFrame([{"input": f.label, "value": f.value, "unit": f.unit or "",
                               "date": f.measured_on or "", "source": f.provenance} for f in clinical]),
                 hide_index=True, width="stretch")

    st.subheader("Model outputs")
    if response:
        st.dataframe(outcome_frame(response).map(lambda value: f"{value:.1f}%"), width="stretch")
        st.markdown("**What influenced this estimate**")
        render_drivers(response)
        render_messages(response)
    else:
        st.warning("Prediction unavailable. Patient facts and the comparator remain available.")

    st.subheader("Model versus comparator")
    st.caption("This compares a future twelve-month risk forecast with a fixed current-echo criterion; it is distinct "
               "from the Compare families tab.")
    comparison_fact = fact_map.get("comparison.interpretation")
    st.info(str(comparison_fact.value if comparison_fact else "Comparison incomplete."))
    render_comparator(comparator)

    st.subheader("Findings and recommendations for clinical review")
    valid_family_comparison = bool(comparison and comparison.get("key") == compare_key
                                   and len([r for r in comparison.get("results", {}).values() if r.get("status") == 200]) >= 2)
    include = st.checkbox("Include family comparison in findings", key="include_family_comparison",
                          disabled=not valid_family_comparison,
                          help="Off by default; enabled only for a current, verified two-family comparison.")
    if include != binding.get("include_family_comparison"):
        st.warning("The findings binding changed — update summary before generating or downloading findings.")
        ss.pop("summary_draft", None)
        return
    draft_state = ss.get("summary_draft")
    draft_current = bool(draft_state and draft_state.get("snapshot_id") == facts.snapshot_id
                         and draft_state.get("family") == facts.returned_family
                         and draft_state.get("model_version") == facts.model_version
                         and draft_state.get("include_family_comparison") == include)
    label = "Regenerate findings" if draft_current else "Generate findings"
    if st.button(label, key="generate_findings"):
        try:
            with st.spinner("Generating a bounded clinician draft…"):
                draft, metadata = PatientSummaryWriter(settings).write(facts)
            ss["summary_draft"] = draft_state = {
                "snapshot_id": facts.snapshot_id, "family": facts.returned_family,
                "model_version": facts.model_version, "include_family_comparison": include,
                "kind": "AI-generated draft for clinician review", "markdown": findings_markdown(draft),
                "metadata": metadata.__dict__,
            }
        except Exception as exc:  # noqa: BLE001 - safe class-only failure shown with deterministic fallback
            ss["summary_draft"] = draft_state = {
                "snapshot_id": facts.snapshot_id, "family": facts.returned_family,
                "model_version": facts.model_version, "include_family_comparison": include,
                "kind": "Template summary", "markdown": deterministic_summary(facts),
                "error": PatientSummaryWriter.safe_error(exc),
            }
        draft_current = True
    if draft_current:
        if draft_state.get("error"):
            st.warning(f"Narrative unavailable ({draft_state['error']}). A deterministic template is shown instead.")
        else:
            st.success("AI-generated draft for clinician review")
        st.markdown(draft_state["markdown"])
        export = export_summary(facts, draft_state["markdown"], draft_label=draft_state["kind"])
        st.download_button("Download summary", export, file_name=f"kairos-summary-{facts.snapshot_id[:12]}.md",
                           mime="text/markdown", key="download_summary")
    else:
        template = deterministic_summary(facts)
        st.markdown(template)
        st.download_button("Download summary", export_summary(facts, template),
                           file_name=f"kairos-summary-{facts.snapshot_id[:12]}.md", mime="text/markdown",
                           key="download_summary")
    with st.expander("Full input audit", expanded=False):
        st.dataframe(pd.DataFrame([f.model_dump() for f in facts.facts]), hide_index=True, width="stretch")


# --- page -------------------------------------------------------------------------------------------
st.title("KAIROS: patient model")
st.warning(BANNER, icon="⚠️")
try:
    principal = st.context.headers.get("X-MS-CLIENT-PRINCIPAL-NAME")
except Exception:  # noqa: BLE001
    principal = None
notice = ss.pop("notice", None)
if notice:
    getattr(st, notice[0])(notice[1])

with st.sidebar:
    st.header("Example patient")
    st.selectbox("Synthetic example", list(dp.PRESETS) + [PUBLISHED], key="preset_choice")
    st.button("Load into the form", key="load_preset", on_click=cb_load_example, type="primary")
    st.caption("All examples are synthetic. Every input stays editable after loading.")
    st.divider()
    st.header("Prediction")
    st.radio("Guideline jurisdiction", dp.JURISDICTIONS, key="jurisdiction",
             format_func=lambda j: "ESC/EACTS" if j == "ESC_EACTS" else "ACC/AHA")
    st.caption("Selected model family: " + family_label(ss.get("model_family")))
    st.caption(f"Backend: {BACKEND}. Version {__version__}. "
               + (f"Signed in as {principal}." if principal else "Not behind Entra ID sign-in (local run)."))

inventory = model_inventory()
models_info = loaded_models()
available_families = list(models_info)
if "model_family" not in ss and available_families:
    reported_default = inventory.get("default_family")
    ss["model_family"] = reported_default if reported_default in available_families else available_families[0]
selected_missing = bool(ss.get("model_family") and ss.get("model_family") not in available_families)
family_options = list(available_families)
if selected_missing:
    family_options.append(ss["model_family"])

st.markdown("#### Patient workflow")
tool_family, tool_version, tool_default, tool_date, tool_refresh = st.columns([2, 1.4, 1.2, 1.5, 1.2])
if family_options:
    tool_family.selectbox("Model family", family_options, key="model_family", format_func=family_label,
                          help="The selected family is used by Prediction, Risk over time, What-if and Patient Summary. "
                               "Each family keeps the action flags returned by the service.")
else:
    tool_family.caption("Model family")
    tool_family.error("No compatible model family is available.")
selected_info = models_info.get(ss.get("model_family"), {})
tool_version.metric("Model version", selected_info.get("model_version") or "Unavailable")
default_family = inventory.get("default_family")
tool_default.metric("Service default", family_label(default_family) if default_family else "Unverified")
tool_date.date_input("Assessment date", key="prediction_time", min_value=MIN_DATE, max_value=MAX_DATE,
                     help="Only information dated on or before this day is used.")
if tool_refresh.button("Refresh available models", key="refresh_models"):
    model_inventory.clear()
    model_card.clear()
    st.rerun()
if not inventory.get("verified"):
    st.warning("Family availability could not be verified. Refresh available models; no unverified family option is invented.")
elif selected_missing:
    st.error("The selected model family is no longer available. Its bound results are out of date; select an available family.")
elif len(available_families) == 1:
    st.caption(f"Only {family_label(available_families[0])} is available; a two-family comparison has not been run.")
if inventory.get("verified") and inventory.get("models"):
    unavailable = {f: i for f, i in inventory["models"].items() if f not in models_info}
    with st.expander("Model-family availability", expanded=False):
        st.dataframe(pd.DataFrame([
            {"family": family_label(f), "version": i.get("model_version"),
             "compatible": i.get("compatible_with_default"),
             "status": "available" if f in models_info else "incompatible"}
            for f, i in inventory["models"].items()
        ]), hide_index=True, width="stretch")

card = model_card(ss.get("model_family")) if ss.get("model_family") in available_families else None
tab_summary, tab_inputs, tab_pred, tab_compare, tab_traj, tab_what, tab_note, tab_about = st.tabs(
    ["Patient Summary", "Patient inputs", "Prediction", "Compare families", "Risk over time", "What-if: next echo", "Fill from a note", "About"])

with tab_inputs:
    valve_col, patient_col = st.columns(2, gap="large")
    with valve_col:
        st.subheader("Valve")
        st.radio("Route", dp.ROUTES, key="v_route", horizontal=True,
                 format_func=lambda r: "Surgical (SAVR)" if r == "SAVR" else "Transcatheter (TAVR)")
        classes = dp.design_classes(ss["v_route"])
        ensure_option("v_class", classes)
        st.selectbox("Design class", classes, key="v_class")
        models = dp.models_for(ss["v_route"], ss["v_class"])
        ensure_option("v_model", models)
        st.selectbox("Valve model", models, key="v_model")
        sizes = dp.sizes_for(ss["v_model"])
        ensure_option("v_size", sizes)
        st.selectbox("Label size (mm)", sizes, key="v_size")
        st.date_input("Implant date", key="v_implant", min_value=MIN_DATE, max_value=MAX_DATE)
        device = dp.device_summary(ss["v_model"])
        badge_color = {"withdrawn": "red", "discontinued": "orange", "active": "green"}.get(device["market_status"], "gray")
        st.badge(f"Market status: {device['market_status']}", color=badge_color)
        st.caption(" ".join(x for x in (device["generation"], f"Tissue treatment: {device['tissue_treatment']}."
                                        if device["tissue_treatment"] and device["tissue_treatment"] not in ("none", "unknown") else "") if x))
    with patient_col:
        st.subheader("Patient at implant")
        left, right = st.columns(2)
        left.number_input("Age at implant (years)", min_value=18.0, max_value=100.0, step=1.0, format="%.0f", key="p_age")
        left.selectbox("Sex", ["F", "M"], key="p_sex", format_func=lambda s: "Female" if s == "F" else "Male")
        left.number_input("Body surface area (m²)", min_value=1.0, max_value=3.0, step=0.05, format="%.2f", key="p_bsa")
        left.number_input("Body mass index (kg/m²)", min_value=12.0, max_value=70.0, step=0.5, format="%.1f", key="p_bmi")
        left.number_input("eGFR at implant (mL/min/1.73 m²)", min_value=3.0, max_value=150.0, step=1.0, format="%.0f", key="p_egfr")
        right.toggle("Diabetes", key="p_diabetes")
        right.number_input("Diabetes duration (years)", min_value=0.0, max_value=60.0, step=1.0, format="%.0f",
                           key="p_diab_years", disabled=not ss["p_diabetes"])
        right.toggle("On dialysis", key="p_dialysis")
        right.toggle("Atrial fibrillation", key="p_af")
        right.toggle("Bicuspid native valve", key="p_bicuspid")
        right.toggle("Lipid-lowering treatment", key="p_lipid")

    version = ss["editor_v"]
    st.subheader("Echo series (prosthetic valve)")
    st.caption("Time zero is the first adequate echo, meaning a mean gradient plus EOA or DVI, 30 to 180 days after implant. "
               "Only echoes dated on or before the prediction time are used. Add, edit or delete rows.")
    echo_col, stage_col = st.columns([3, 2], gap="large")
    with echo_col:
        ss["echo_now"] = st.data_editor(
            ss["echo_df"], key=f"echo_editor_{version}", num_rows="dynamic", hide_index=True, width="stretch",
            column_config={
                "date": st.column_config.DateColumn("Date", required=True, format="YYYY-MM-DD"),
                "mean_gradient_mmhg": st.column_config.NumberColumn("Mean gradient (mmHg)", min_value=0.0, max_value=150.0, step=0.5, format="%.1f"),
                "eoa_cm2": st.column_config.NumberColumn("EOA (cm²)", min_value=0.1, max_value=6.0, step=0.05, format="%.2f"),
                "dvi": st.column_config.NumberColumn("DVI", min_value=0.05, max_value=1.5, step=0.01, format="%.2f"),
                "ar_grade": st.column_config.SelectboxColumn("Regurgitation", options=dp.AR_GRADES, required=True),
                "lvef_pct": st.column_config.NumberColumn("LVEF (%)", min_value=5.0, max_value=90.0, step=1.0, format="%.0f"),
                "svi_ml_m2": st.column_config.NumberColumn("SVi (mL/m²)", min_value=5.0, max_value=120.0, step=1.0, format="%.0f"),
            })
    form = form_from_state()
    with stage_col:
        stage_rows, reference_date = dp.staging_rows(form)
        implant_date = dp.to_date(form["valve"]["implant_date"])
        if reference_date:
            st.success(f"Time zero: reference study on {reference_date} ({(reference_date - implant_date).days} days after implant).")
        else:
            st.error("No reference study yet: add an echo with a mean gradient plus EOA or DVI, 30 to 180 days after implant.")
        if stage_rows:
            st.dataframe(pd.DataFrame(stage_rows), hide_index=True, width="stretch")

    labs_col, ac_col = st.columns(2, gap="large")
    with labs_col:
        st.subheader("Laboratory biomarkers")
        st.caption("The latest value on or before the prediction time enters the model; eGFR values also give a slope. "
                   "dp-ucMGP is a research assay: it is used for at most 12 months after the draw, only through the "
                   "substudy model, and never imputed.")
        ss["labs_now"] = st.data_editor(
            ss["labs_df"], key=f"labs_editor_{version}", num_rows="dynamic", hide_index=True, width="stretch",
            column_config={
                "date": st.column_config.DateColumn("Date", required=True, format="YYYY-MM-DD"),
                "marker": st.column_config.SelectboxColumn("Marker", options=[label for label, _unit in dp.MODEL_ANALYTES.values()], required=True),
                "value": st.column_config.NumberColumn("Value", min_value=0.0, step=0.1),
            })
    with ac_col:
        st.subheader("Antithrombotic exposure history")
        st.caption("VKA vitamin K antagonist, FXa factor Xa inhibitor, DTI direct thrombin inhibitor, SAPT or DAPT single or dual "
                   "antiplatelet therapy. Leave the stop date empty for ongoing treatment.")
        ss["ac_now"] = st.data_editor(
            ss["ac_df"], key=f"ac_editor_{version}", num_rows="dynamic", hide_index=True, width="stretch",
            column_config={
                "class": st.column_config.SelectboxColumn("Class", options=dp.EXPOSURE_CLASSES, required=True),
                "agent": st.column_config.TextColumn("Agent"),
                "indication": st.column_config.SelectboxColumn("Indication", options=dp.INDICATIONS),
                "start": st.column_config.DateColumn("Start", required=True, format="YYYY-MM-DD"),
                "stop": st.column_config.DateColumn("Stop", format="YYYY-MM-DD"),
                "post_suspicion": st.column_config.CheckboxColumn("Started within 90 days of a suspicious echo"),
            })
    form = form_from_state()
    st.subheader("Biomarker modules for this patient")
    st.dataframe(pd.DataFrame(dp.module_coverage(form, (card or {}).get("dropped_modules", []),
                                                 ((card or {}).get("dp_ucmgp_substudy") or {}).get("fitted"))),
                 hide_index=True, width="stretch")
    try:
        request = dp.request_from_form(form)
        form_errors: list[str] = []
    except dp.FormError as exc:
        request, form_errors = None, exc.messages
    if form_errors:
        st.error("The form needs attention before a prediction:\n\n" + "\n".join(f"- {m}" for m in form_errors))
    else:
        st.success("The inputs are complete. Open the Prediction tab.")

payload = request.model_dump(mode="json", by_alias=True) if request is not None else None
source_provenance = ss.get("source_provenance") or "unknown"
payload_key = snapshot_hash(payload, source_provenance) if payload is not None else None
family = ss.get("model_family") if ss.get("model_family") in available_families else None


def predict_for(body: dict, fam: str, path: str = "/predict") -> dict:
    response = {**result(*call("predict", "POST", path, json=body, params={"family": fam})),
                "family": fam, "requested_family": fam, "namespace": inventory.get("namespace")}
    if response.get("status") == 200 and path == "/predict":
        body_out = response.get("body") or {}
        actual = body_out.get("model_family") or (body_out.get("reliability") or {}).get("model_family")
        if actual != fam:
            return {"status": 502, "body": None, "family": fam, "requested_family": fam,
                    "namespace": inventory.get("namespace"),
                    "detail": ("The prediction response did not verify the requested model family "
                               f"({family_label(fam)} requested; {actual or 'identity missing'} returned).")}
        response["returned_family"] = actual
    return response


def cache_prediction(body: dict, fam: str, response: dict) -> dict:
    key = snapshot_hash(body, source_provenance)
    if response.get("status") == 200:
        version = (response.get("body") or {}).get("model_version")
        cache = ss.setdefault("prediction_cache", {})
        cache[f"{key}|{fam}|{version}|{inventory.get('namespace')}"] = response
    response["snapshot_key"] = key
    return response


def cached_prediction(body: dict, fam: str) -> dict | None:
    key = snapshot_hash(body, source_provenance)
    expected = (models_info.get(fam) or {}).get("model_version")
    if not expected:
        return None
    return ss.get("prediction_cache", {}).get(f"{key}|{fam}|{expected}|{inventory.get('namespace')}")


def run_or_reuse(body: dict, fam: str) -> dict:
    return cached_prediction(body, fam) or cache_prediction(body, fam, predict_for(body, fam))


def cb_use_family(fam: str) -> None:
    ss["model_family"] = fam
    ss.pop("summary_draft", None)


with tab_pred:
    if request is None:
        st.info("Fix the inputs listed on the Patient inputs tab first.")
    if st.button("Run prediction", key="run_predict", type="primary", disabled=request is None or family is None):
        ss["res_predict"] = run_or_reuse(payload, family)
    current_prediction = ss.get("res_predict")
    if current_prediction and current_prediction.get("snapshot_key") != payload_key:
        st.warning("Out of date — run prediction for the current inputs.")
    else:
        render_prediction(current_prediction)

with tab_compare:
    st.caption("This compares compatible estimator families on one fixed input snapshot; Patient Summary separately compares "
               "the selected model with the fixed current-echo rule. Sensitivities are not treatment effects.")
    compare_families = available_families
    compare_key = json.dumps({"snapshot": payload_key, "families": compare_families,
                              "versions": {f: models_info[f].get("model_version") for f in compare_families},
                              "namespace": inventory.get("namespace")}, sort_keys=True, default=str)
    cached = ss.get("res_compare")
    comparison_current = bool(cached and cached.get("key") == compare_key)
    if payload is None:
        st.info("Fix the inputs listed on the Patient inputs tab first.")
    elif len(compare_families) < 2:
        only = family_label(compare_families[0]) if compare_families else "No family"
        st.info(f"{only} is available; a two-family comparison has not been run.")
        st.button("Run family comparison", key="run_family_compare", disabled=True)
    else:
        if cached and not comparison_current:
            st.warning("Comparison out of date — update comparison for the current inputs and bundle versions.")
        label = "Update comparison" if cached else "Run family comparison"
        if st.button(label, key="run_family_compare", type="primary"):
            ss["res_compare"] = cached = {"key": compare_key, "snapshot_key": payload_key,
                                          "results": {fam: run_or_reuse(payload, fam) for fam in compare_families}}
            comparison_current = True
            ss.pop("summary_draft", None)
        if comparison_current:
            render_compare(cached["results"])
            good = {fam: res for fam, res in cached["results"].items() if res.get("status") == 200}
            if good:
                cols = st.columns(len(good))
                for col, fam in zip(cols, good):
                    col.button(f"Use {family_label(fam)} in summary", key=f"use_{fam}_summary",
                               on_click=cb_use_family, args=(fam,))
            st.markdown("**Shared independent comparator**")
            shared_comparator = comparator_for_request(request, source_kind=source_provenance)
            st.write(f"{shared_comparator.status.capitalize()} — "
                     f"stage {shared_comparator.highest_demonstrated_stage or 'not demonstrated'}; "
                     f"reasons: {', '.join(shared_comparator.reason_codes) or 'none'}.")

with tab_summary:
    render_patient_summary(request, payload, payload_key, family, source_provenance, ss.get("res_compare"))

with tab_traj:
    st.caption("One prediction at every echo from time zero to the prediction time: how the risk updates as the series grows.")
    if st.button("Compute risk over time", key="run_traj", type="primary", disabled=request is None or family is None):
        ss["res_traj"] = predict_for(payload, family, "/predict/trajectory")
    render_trajectory(ss.get("res_traj"), form)

with tab_what:
    last = dp.latest_echo(form)
    if request is None or last is None:
        st.info("Complete the patient inputs first.")
    else:
        st.write(f"Latest echo on {dp.to_date(last['date'])}: mean gradient {dp.to_num(last['mean_gradient_mmhg'])} mmHg, "
                 f"EOA {dp.to_num(last['eoa_cm2'])} cm², DVI {dp.to_num(last['dvi'])}, regurgitation {last.get('ar_grade') or 'none'}.")
        c1, c2, c3 = st.columns(3, gap="large")
        months = c1.slider("Months after the latest echo", 3, 36, 12, key=f"w_months_{version}")
        gradient = c1.slider("Mean gradient (mmHg)", 2.0, 80.0, min(80.0, max(2.0, (dp.to_num(last["mean_gradient_mmhg"]) or 12.0) + 4.0)),
                             step=0.5, key=f"w_grad_{version}")
        eoa = c2.slider("EOA (cm²)", 0.3, 3.0, min(3.0, max(0.3, round((dp.to_num(last["eoa_cm2"]) or 1.6) - 0.1, 2))),
                        step=0.05, key=f"w_eoa_{version}")
        dvi = c2.slider("DVI", 0.10, 1.00, min(1.0, max(0.1, round((dp.to_num(last["dvi"]) or 0.45) - 0.02, 2))),
                        step=0.01, key=f"w_dvi_{version}")
        last_ar = last.get("ar_grade") if last.get("ar_grade") in dp.AR_GRADES else "none"
        ar_grade = c3.selectbox("Regurgitation", dp.AR_GRADES, index=dp.AR_GRADES.index(last_ar), key=f"w_ar_{version}")
        ac_change = c3.selectbox("Antithrombotic change from the latest echo", dp.AC_CHANGES, key=f"w_ac_{version}")
        if st.button("Compare with the prediction now", key="run_whatif", type="primary", disabled=family is None):
            try:
                base_request = dp.request_from_form(dp.at_latest_echo(form))
                hypothetical = dp.hypothetical_form(form, months, gradient, eoa, dvi, ar_grade, ac_change)
                hypothetical_request = dp.request_from_form(hypothetical)
                rows, _ref = dp.staging_rows(hypothetical)
                ss["res_whatif"] = {
                    "now": predict_for(base_request.model_dump(mode="json", by_alias=True), family),
                    "then": predict_for(hypothetical_request.model_dump(mode="json", by_alias=True), family),
                    "dates": (base_request.prediction_time, hypothetical_request.prediction_time),
                    "new_stage": next((r for r in rows if r["date"] == dp.to_date(hypothetical["prediction_time"])), None)}
            except dp.FormError as exc:
                ss["res_whatif"] = {"form_error": "; ".join(exc.messages)}
        render_whatif(ss.get("res_whatif"))

with tab_note:
    st.caption("Paste an operative, procedure or progress note. The extract service reads it with rules first and with the "
               "hosted model second when one is configured. The note text is never stored; apply the result to the form.")
    st.text_area("Note text", key="note_text", height=220)
    n1, n2, n3, n4 = st.columns(4)
    source_kind = n1.radio("Source", ["synthetic", "real"], key="note_kind", horizontal=True)
    note_type = n2.selectbox("Note type", ["progress", "operative", "procedure", "discharge"], key="note_type")
    note_date = n3.date_input("Note date", key="note_date", min_value=MIN_DATE, max_value=MAX_DATE)
    method = n4.selectbox("Method", ["rules", "auto", "llm"], key="note_method",
                          format_func={"rules": "Rules only", "auto": "Rules, then hosted model", "llm": "Hosted model required"}.get)
    if source_kind == "real":
        st.info("Real de-identified text goes to the hosted model only if the server's ALLOW_REAL_NOTES_TO_LLM switch is on. "
                "Extracted real values shown beside a model trained on synthetic scenarios are illustrative, unvalidated.")
    if st.button("Extract", key="run_extract", type="primary", disabled=not (ss.get("note_text") or "").strip()):
        response, error = call("extract", "POST", "/extract", params={"store": "false"},
                               json={"text": ss["note_text"], "source_kind": source_kind, "note_ref": "demo-note",
                                     "note_type": note_type, "date": note_date.isoformat(), "method": method})
        ss["res_extract"] = {**result(response, error), "text": ss["note_text"]}
    extracted = ss.get("res_extract")
    if extracted:
        if extracted.get("error") or extracted.get("status") != 200:
            st.error(extracted.get("error") or f"The extract service answered {extracted['status']}: {extracted['detail']}")
        else:
            body = extracted["body"]
            passport = body["passport"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Route", passport["route"])
            m2.metric("Valve model", passport.get("canonical_model") or "not found")
            m3.metric("Size (mm)", passport.get("size_mm") or "not found")
            m4.metric("Implant date", passport.get("implant_date") or "not found")
            st.caption(f"Methods: {' + '.join(body['methods_used'])}. Hosted model used: {'yes' if body['llm_used'] else 'no'}. "
                       + (f"Warnings: {'; '.join(body['warnings'])}." if body["warnings"] else ""))
            if body["echo_observations"]:
                st.dataframe(pd.DataFrame(body["echo_observations"]).drop(columns=["passport_id", "evidence"], errors="ignore"),
                             hide_index=True, width="stretch")
            st.markdown(highlight(extracted["text"], passport.get("evidence", [])), unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            b1.button("Apply the valve and implant date to the form", key="apply_valve", on_click=cb_apply_extracted_valve,
                      disabled=not (passport.get("canonical_model") or passport.get("implant_date")))
            b2.button("Add the prosthetic echo values to the series", key="apply_echo", on_click=cb_add_extracted_echoes,
                      disabled=not any(o["native_vs_prosthetic"] == "prosthetic" for o in body["echo_observations"]))

with tab_about:
    st.markdown(f"""
**KAIROS** predicts structural valve deterioration (SVD) of a bioprosthetic aortic valve before death, with death and
replacement of the index valve for another cause as competing events. It updates at every echo.

- **Inputs.** Every input on the first tab is a feature of the prototype model: valve route, design class, model and size,
  patient characteristics at implant, the echo series against the patient's own reference study, biomarker values by
  module, and antithrombotic exposure history. Only information dated on or before the prediction time is used.
- **Outputs.** The probabilities of SVD before death, of death before SVD, of replacement for another cause and of being
  alive with the valve in place and SVD-free, at 1, 3 and 5 years, plus the 12-month SVD risk.
- **Messages.** A finding, a prediction and a reminder are kept apart. The guideline schedule is shown unchanged, and the
  model never recommends reintervention or a treatment change.
- **Evidence.** The model is trained on synthetic scenarios. *{ILLUSTRATIVE_LABEL}.*
""")
    if card:
        availability = card.get("module_availability", {})
        st.markdown("**Model card**")
        st.dataframe(pd.DataFrame([{"module": name, "in the model": "yes" if info.get("available") else "switched off",
                                    "measured in training rows": f"{info.get('fraction', 0):.0%}"} for name, info in availability.items()]),
                     hide_index=True, width="stretch")
        st.caption(f"{card['n_features']} features; training events: {card.get('event_counts')}; trained {card['trained_at']}.")
        vk = card.get("dp_ucmgp_substudy") or {}
        st.markdown("**dp-ucMGP substudy model**")
        if vk.get("fitted"):
            st.write(f"Fitted on {vk['n_rows_measured']} measured landmark rows from {vk['n_patients_measured']} patients "
                     f"({vk['n_svd_events_measured']} SVD events), with the full-cohort SVD linear predictor as an offset. "
                     f"Test of the dp-ucMGP terms: p = {vk['p_value']:.3g} ({vk.get('p_value_method', '')}).")
            st.dataframe(pd.DataFrame([{"term": k, "log hazard ratio (standardised)": v, "SE": vk.get("se", {}).get(k)}
                                       for k, v in vk.get("coefficients", {}).items()]), hide_index=True, width="stretch")
        else:
            st.caption(f"Not fitted: {vk.get('reason', 'unknown')}")

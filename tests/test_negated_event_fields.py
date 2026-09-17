"""A structured field that denies an event must not be counted as the event.

Transcatheter operative reports carry the key-value field ``Valve in Valve: No``.
Phrase matching counts those patients as valve-in-valve cases, which inflated the
reported count from 4 to 10 in the supplied corpus. These tests pin the fix.
"""
from kairos.passport import EVENT_PATS, in_negated_field, negated_field_spans


def _fires(key: str, text: str) -> bool:
    spans = negated_field_spans(text)
    return any(not in_negated_field(m.start(), spans) for m in EVENT_PATS[key].finditer(text))


def test_valve_in_valve_field_set_to_no_is_not_an_event():
    text = "PROCEDURE: TAVR\nValve in Valve: No\nAccess: transfemoral"
    assert EVENT_PATS["ViV"].search(text), "phrase is present, so the guard is what must reject it"
    assert not _fires("ViV", text)


def test_reoperation_field_denial_is_not_an_event():
    text = "Operative Approach: Full sternotomy\nReoperation: No previous surgeries"
    assert not _fires("redo", text)


def test_genuine_prose_valve_in_valve_still_detected():
    text = "Underwent successful transfemoral TAVR (valve-in-valve) with a 26 mm Evolut FX."
    assert _fires("ViV", text)


def test_positive_field_value_is_left_in_place():
    text = "PROCEDURE: TAVR\nValve in Valve: Yes\nPrior surgical bioprosthesis."
    assert _fires("ViV", text)


def test_denial_in_one_note_does_not_mask_prose_elsewhere_in_it():
    text = ("Valve in Valve: No\n"
            "History: the patient had a valve-in-valve procedure at another centre in 2019.")
    assert _fires("ViV", text)


def test_unrelated_events_are_untouched_by_the_guard():
    text = "Valve in Valve: No\nComplicated by prosthetic valve endocarditis."
    assert _fires("endocarditis", text)

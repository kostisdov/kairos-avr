# -*- coding: utf-8 -*-
"""
Build data/reference/published_rates.csv (Task 4, long format) by transcribing the numeric
durability/mortality figures already extracted (and cited) into
data/raw/papers/{notion_10yr,trifecta_vs_magnaease,trifecta_vs_perceval}_*_fulltext.txt.
This is a structured transcription step, not fresh PDF parsing -- the source numbers were
captured verbatim from the rendered articles (see docs/data_build_log.md for how, and the
PDF-access note in each *_fulltext.txt for why a browser render was used instead of a
direct download). No patient data; these are all published, aggregate, journal-reported
statistics.

Column contract: study, arm, device, outcome, definition (quoted from the source), time_years,
estimate_pct, ci_low, ci_high, n_at_risk, source_location, confidence.
ci_low/ci_high are left BLANK when the source only reports a Kaplan-Meier standard error
(e.g. "89.4 +/- 4.3%") rather than a genuine 95% CI -- turning an SE into a 95% CI would
require assuming normality and multiplying by ~1.96, which is a transformation this script
does not silently perform (that risks being read as the source's own number). Where the
source itself gives a 95% CI (e.g. NOTION's hazard ratios), it is carried over as-is.
"""
import csv
from datetime import datetime, timezone
from kairos.paths import repo_root

ROOT = str(repo_root())
OUT_CSV = f"{ROOT}/data/reference/published_rates.csv"
RETRIEVED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

NOTION_CITE = ("Thyregod HGH, Jorgensen TH, Ihlemann N, et al. Transcatheter or surgical "
               "aortic valve implantation: 10-year outcomes of the NOTION trial. "
               "Eur Heart J. 2024;45(13):1116-1124. PMCID PMC10984572.")
SUZUKI_CITE = ("Suzuki R, Ito T, Suzuki M, et al. Trifecta versus Perimount Magna Ease "
               "aortic valves: Failure mechanisms. Asian Cardiovasc Thorac Ann. "
               "2022;30(7):797-806. PMCID PMC9373186.")
NARDI_CITE = ("Nardi P, Altieri C, Salvati AC, et al. Medium-term results and the impact of "
              "structural valve deterioration of Trifecta versus Perceval bioprostheses: "
              "analysis from the Perfecta study. Kardiochir Torakochirurgia Pol. "
              "2026;23(1):39-47. PMCID PMC13122456.")

VARC3_HEMO_ONLY = ('VARC-3 ("haemodynamic" variant -- authors explicitly state they used '
                    'gradient+regurgitation only, without the EOA/DVI component of the full '
                    'VARC-3 SVD definition, because DVI was not systematically calculated)')

rows = []


def add(study, arm, device, outcome, definition, time_years, estimate_pct, ci_low, ci_high,
        n_at_risk, source_location, confidence):
    rows.append(dict(study=study, arm=arm, device=device, outcome=outcome, definition=definition,
                      time_years=time_years, estimate_pct=estimate_pct, ci_low=ci_low,
                      ci_high=ci_high, n_at_risk=n_at_risk, source_location=source_location,
                      confidence=confidence))


# ---------------------------------------------------------------------------
# NOTION 10-year (baseline N and every 10-year endpoint reported as text/Table 1)
# ---------------------------------------------------------------------------
add("NOTION", "TAVI", "CoreValve (1st/2nd generation, Medtronic)", "enrolled (baseline)",
    "intention-to-treat enrollment", 0, "", "", "", 145, "Methods / Results para 1", "high")
add("NOTION", "SAVR", "mixed bioprosthesis (5 surgical valve types, mfr not itemised)",
    "enrolled (baseline)", "intention-to-treat enrollment", 0, "", "", "", 135,
    "Methods / Results para 1", "high")
add("NOTION", "TAVI", "CoreValve", "all-cause mortality", "VARC-2, intention-to-treat", 10,
    62.7, "", "", "", "Table 1 (HR 1.0, 95%CI 0.7-1.3 vs SAVR, P=.8)", "high")
add("NOTION", "SAVR", "mixed bioprosthesis", "all-cause mortality", "VARC-2, intention-to-treat",
    10, 64.0, "", "", "", "Table 1", "high")
add("NOTION", "TAVI", "CoreValve", "composite: all-cause mortality/stroke/MI", "VARC-2", 10,
    65.5, "", "", "", "Results (HR 1.0, 95%CI 0.7-1.3, P=.9)", "high")
add("NOTION", "SAVR", "mixed bioprosthesis", "composite: all-cause mortality/stroke/MI",
    "VARC-2", 10, 65.5, "", "", "", "Results", "high")
add("NOTION", "TAVI", "CoreValve", "moderate-or-severe SVD", VARC3_HEMO_ONLY, 10, 15.4, "", "",
    "", "Results / Figure 3 (HR 0.7, 95%CI 0.4-1.3, P=.3; as-implanted population)", "high")
add("NOTION", "SAVR", "mixed bioprosthesis", "moderate-or-severe SVD", VARC3_HEMO_ONLY, 10,
    20.8, "", "", "", "Results / Figure 3; as-implanted population", "high")
add("NOTION", "TAVI", "CoreValve", "severe SVD", VARC3_HEMO_ONLY, 10, 1.5, "0.04", "0.7", "",
    "Abstract + Results (value shown here is the point estimate; ci_low/ci_high columns "
    "hold the reported HAZARD RATIO's 95%CI [TAVI vs SAVR], not a CI on the % itself -- "
    "the source does not give a CI on the raw percentage) / Figure 3", "high")
add("NOTION", "SAVR", "mixed bioprosthesis", "severe SVD", VARC3_HEMO_ONLY, 10, 10.0, "", "",
    "", "Abstract + Results / Figure 3", "high")
add("NOTION", "TAVI", "CoreValve", "severe BVD (bioprosthetic valve dysfunction)", "VARC-3", 10,
    20.5, "", "", "", "Results 'Bioprosthesis durability' paragraph / Figure 4 -- NOTE: the "
    "Abstract separately reports \"severe NSVD\" as 20.5%/43.0% (identical numbers to this "
    "row's severe-BVD figures), while the Results body separately reports \"severe NSVD\" as "
    "10.2%/31.9% -- an apparent label inconsistency between Abstract and body text in the "
    "source article itself. Both are transcribed faithfully in this table (see the "
    "'severe NSVD' rows below); flagged here rather than silently resolved.", "medium")
add("NOTION", "SAVR", "mixed bioprosthesis", "severe BVD (bioprosthetic valve dysfunction)",
    "VARC-3", 10, 43.0, "", "", "", "Results / Figure 4 -- see same label-inconsistency note "
    "as the TAVI row above", "medium")
add("NOTION", "TAVI", "CoreValve", "severe NSVD (as labelled in Results body text)", "VARC-3",
    10, 10.2, "", "", "", "Results 'Bioprosthesis durability' paragraph", "medium")
add("NOTION", "SAVR", "mixed bioprosthesis", "severe NSVD (as labelled in Results body text)",
    "VARC-3", 10, 31.9, "", "", "", "Results 'Bioprosthesis durability' paragraph "
    "(mainly driven by severe PPM)", "medium")
add("NOTION", "TAVI", "CoreValve", "severe NSVD (as labelled in Abstract)", "VARC-3", 10, 20.5,
    "", "", "", "Abstract -- identical value to this table's severe-BVD TAVI row; see "
    "inconsistency note above", "low")
add("NOTION", "SAVR", "mixed bioprosthesis", "severe NSVD (as labelled in Abstract)", "VARC-3",
    10, 43.0, "", "", "", "Abstract -- identical value to this table's severe-BVD SAVR row",
    "low")
add("NOTION", "TAVI", "CoreValve", "endocarditis", "modified Duke criteria (VARC-3 BVD "
    "component)", 10, 7.2, "", "", "", "Results / Table (P=1.0 vs SAVR)", "high")
add("NOTION", "SAVR", "mixed bioprosthesis", "endocarditis", "modified Duke criteria", 10,
    7.4, "", "", "", "Results / Table", "high")
add("NOTION", "TAVI", "CoreValve", "clinical valve thrombosis", "VARC-3 BVD component", 10,
    0.0, "", "", "", "Results ('No patients had clinical valve thrombosis')", "high")
add("NOTION", "SAVR", "mixed bioprosthesis", "clinical valve thrombosis", "VARC-3 BVD "
    "component", 10, 0.0, "", "", "", "Results", "high")
add("NOTION", "TAVI", "CoreValve", "bioprosthetic valve failure (BVF)", "VARC-3", 10, 9.7,
    "", "", "", "Abstract + Results (HR 0.7, 95%CI 0.4-1.5, P=.4)", "high")
add("NOTION", "SAVR", "mixed bioprosthesis", "bioprosthetic valve failure (BVF)", "VARC-3",
    10, 13.8, "", "", "", "Abstract + Results", "high")
add("NOTION", "TAVI", "CoreValve", "prosthesis re-intervention", "VARC-3 (any cause)", 10,
    4.3, "", "", "", "Results (reasons: 5 restenosis, 1 central regurg; TAVI used for ALL "
    "re-interventions incl. those after SAVR)", "high")
add("NOTION", "SAVR", "mixed bioprosthesis", "prosthesis re-intervention", "VARC-3 (any "
    "cause)", 10, 2.2, "", "", "", "Results (reasons: 2 restenosis, 1 central regurg)",
    "high")
# secondary/comparator figures NOTION's own Discussion quotes from OTHER trials -- keep
# clearly labelled as such, lower confidence (not this trial's own primary data)
add("PARTNER 2A (as cited by NOTION Discussion)", "TAVR (SAPIEN XT)", "SAPIEN XT",
    "SVD", "trial-specific, cited in Pibarot 2020 JACC per NOTION ref 28", 5, 1.61, "", "",
    "", "NOTION Discussion paragraph 'Bioprosthesis haemodynamics and durability' "
    "(1.61+/-0.24% vs SAVR 0.63+/-0.16%, P<=.01) -- secondary citation, verify against "
    "Pibarot et al 2020 primary source before using in the model", "low")
add("PARTNER 2A (as cited by NOTION Discussion)", "SAVR (80% pericardial)", "mixed",
    "SVD", "trial-specific, cited in Pibarot 2020 JACC per NOTION ref 28", 5, 0.63, "", "",
    "", "NOTION Discussion -- secondary citation", "low")
add("Self-expanding TAVR meta-comparison (as cited by NOTION Discussion, ref 25 O'Hair "
    "2023 JAMA Cardiol)", "TAVR (self-expanding)", "CoreValve/Evolut family", "SVD",
    "VARC-3 haemodynamic variant, per NOTION ref 25", 5, 2.20, "", "", "",
    "NOTION Discussion (HR 0.46, 95%CI 0.27-0.78, P=.004 vs SAVR) -- secondary citation",
    "low")
add("Self-expanding TAVR meta-comparison (as cited by NOTION Discussion, ref 25)", "SAVR",
    "mixed", "SVD", "VARC-3 haemodynamic variant, per NOTION ref 25", 5, 4.38, "", "", "",
    "NOTION Discussion -- secondary citation", "low")

# ---------------------------------------------------------------------------
# Suzuki 2022 (Trifecta vs Perimount Magna Ease), single center, Japan
# ---------------------------------------------------------------------------
add("Suzuki 2022 (Trifecta vs Magna Ease)", "Trifecta", "Trifecta (1st-gen + some GT)",
    "freedom from redo AVR (any cause)", "Kaplan-Meier, trial-specific (not VARC)", 5, 87.4,
    "", "", "", "Results 'Freedom from redo AVR', value is mean +/- SE (4.4), not a 95%CI "
    "-- ci_low/ci_high left blank rather than back-computed; N=137 at baseline", "medium")
add("Suzuki 2022 (Trifecta vs Magna Ease)", "Perimount Magna Ease", "Magna Ease",
    "freedom from redo AVR (any cause)", "Kaplan-Meier, trial-specific", 5, 99.2, "", "", "",
    "Results, mean +/- SE (0.8); N=133 at baseline", "medium")
add("Suzuki 2022 (Trifecta vs Magna Ease)", "Trifecta", "Trifecta (1st-gen + some GT)",
    "freedom from redo AVR due to SVD", "Rodriguez-Gabella 2017-modified, trial-specific "
    "(excludes prosthetic-valve-AR-driven cusp-tear failures from the SVD definition, see "
    "the paper's own methods -- NOT the same denominator as VARC-3 SVD)", 5, 89.4, "", "",
    "", "Results (log-rank P=.003 vs Magna Ease), mean +/- SE (4.3)", "medium")
add("Suzuki 2022 (Trifecta vs Magna Ease)", "Perimount Magna Ease", "Magna Ease",
    "freedom from redo AVR due to SVD", "Rodriguez-Gabella 2017-modified, trial-specific",
    5, 100.0, "", "", "", "Results (0 events in this arm)", "high")

# ---------------------------------------------------------------------------
# Nardi 2026 / Perfecta study (Trifecta vs Perceval), single center, Italy
# ---------------------------------------------------------------------------
add("Nardi 2026 / Perfecta study", "Trifecta", "Trifecta (St Jude/Abbott)",
    "enrolled (baseline)", "cohort enrollment", 0, "", "", "", 220,
    "Methods (274 survivors analysed: 214 Trifecta, 60 Perceval)", "high")
add("Nardi 2026 / Perfecta study", "Perceval", "Perceval (Corcym)", "enrolled (baseline)",
    "cohort enrollment", 0, "", "", "", 60, "Methods", "high")
add("Nardi 2026 / Perfecta study", "Trifecta", "Trifecta", "overall survival",
    "Kaplan-Meier, trial-specific", 7, 68.0, "", "", "", "Results, mean +/- SE (4.1) "
    "(P=.19 vs Perceval)", "medium")
add("Nardi 2026 / Perfecta study", "Perceval", "Perceval", "overall survival",
    "Kaplan-Meier, trial-specific", 7, 89.0, "", "", "", "Results, mean +/- SE (4.8)",
    "medium")
add("Nardi 2026 / Perfecta study", "Trifecta", "Trifecta", "freedom from cardiac death",
    "Kaplan-Meier, trial-specific", 7, 90.0, "", "", "", "Results, mean +/- SE (2.3) "
    "(P=.38 vs Perceval)", "medium")
add("Nardi 2026 / Perfecta study", "Perceval", "Perceval", "freedom from cardiac death",
    "Kaplan-Meier, trial-specific", 7, 94.0, "", "", "", "Results, mean +/- SE (1.7)",
    "medium")
add("Nardi 2026 / Perfecta study", "Trifecta", "Trifecta", "freedom from endocarditis",
    "Kaplan-Meier, trial-specific", 7, 96.0, "", "", "", "Results, mean +/- SE (4.0); "
    "source text states this pair two ways in one sentence (94+/-3.0% vs 96+/-4.0%) "
    "without unambiguous arm labelling in the extracted text -- this row's arm assignment "
    "(Trifecta=96) should be verified against the source PDF/HTML if precision matters",
    "low")
add("Nardi 2026 / Perfecta study", "Perceval", "Perceval", "freedom from endocarditis",
    "Kaplan-Meier, trial-specific", 7, 94.0, "", "", "", "Results, mean +/- SE (3.0); "
    "same arm-labelling caveat as the Trifecta row above", "low")
add("Nardi 2026 / Perfecta study", "Trifecta", "Trifecta", "freedom from thromboembolism",
    "Kaplan-Meier, trial-specific", 7, 94.0, "", "", "", "Results, mean +/- SE (2.3)",
    "medium")
add("Nardi 2026 / Perfecta study", "Perceval", "Perceval", "freedom from thromboembolism",
    "Kaplan-Meier, trial-specific", 7, 98.0, "", "", "", "Results, mean +/- SE (1.7)",
    "medium")
add("Nardi 2026 / Perfecta study", "Trifecta", "Trifecta", "freedom from redo operation "
    "(SVD-related)", "Kaplan-Meier, trial-specific", 7, 98.0, "", "", "", "Results, mean "
    "+/- SE (1.4) (P=.13 vs Perceval; 8 Trifecta reoperations = 2 surgical + 6 "
    "valve-in-valve TAVR)", "medium")
add("Nardi 2026 / Perfecta study", "Perceval", "Perceval", "freedom from redo operation "
    "(SVD-related)", "Kaplan-Meier, trial-specific", 7, 100.0, "", "", "", "Results "
    "(0 Perceval reoperations)", "high")
add("Nardi 2026 / Perfecta study", "Trifecta", "Trifecta", "freedom from overall SVD (all "
    "stages)", "VARC-3-style discrete staging (source text: 'initial SVD = stages 1 and "
    "2S; advanced/moderate-to-severe SVD = stages 2SR and 3')", 7, 74.0, "", "", "",
    "Results, mean +/- SE (4.2) (P=.09 vs Perceval)", "medium")
add("Nardi 2026 / Perfecta study", "Perceval", "Perceval", "freedom from overall SVD (all "
    "stages)", "VARC-3-style discrete staging", 7, 93.0, "", "", "", "Results, mean +/- SE "
    "(6.4)", "medium")
# secondary citations quoted in Nardi Discussion
add("Yokoyama meta-analysis (as cited by Nardi 2026 Discussion, ref 28)",
    "externally mounted (Trifecta+Mitroflow)", "Trifecta;Mitroflow", "reoperation for SVD",
    "trial-specific, per Nardi ref 28", "", "", "", "", "", "Nardi Discussion "
    "(15 studies, n=23,539; Trifecta n=6146, Mitroflow n=3192 vs internally-mounted "
    "Perimount n=14,201; P<.001, no single time point or % given in the extracted text) "
    "-- secondary citation, verify against Yokoyama primary source", "low")
add("Fukuhara (as cited by Nardi 2026 Discussion, ref 29)", "Trifecta", "Trifecta", "SVD",
    "trial-specific, per Nardi ref 29", "", 13.3, "", "", "", "Nardi Discussion (n=508 "
    "Trifecta vs n=550 non-Trifecta 4.4%, P=.01; age<65y subgroup 27.9% vs 6.9%, P=.004; "
    "no explicit follow-up duration given in the extracted text) -- secondary citation",
    "low")
add("Fukuhara (as cited by Nardi 2026 Discussion, ref 29)", "non-Trifecta", "mixed", "SVD",
    "trial-specific, per Nardi ref 29", "", 4.4, "", "", "", "Nardi Discussion -- "
    "secondary citation", "low")
add("Werner (as cited by Nardi 2026 Discussion, ref 22)", "Trifecta", "Trifecta", "SVD",
    "trial-specific, per Nardi ref 22", 5, 4.49, "", "", "", "Nardi Discussion (rising to "
    "23.78% at 7y) -- secondary citation, verify against Werner primary source", "low")
add("Werner (as cited by Nardi 2026 Discussion, ref 22)", "Trifecta", "Trifecta", "SVD",
    "trial-specific, per Nardi ref 22", 7, 23.78, "", "", "", "Nardi Discussion -- "
    "secondary citation", "low")
add("Werner (as cited by Nardi 2026 Discussion, ref 22)", "INTUITY", "INTUITY Elite", "SVD",
    "trial-specific, per Nardi ref 22", 5, 1.04, "", "", "", "Nardi Discussion -- "
    "secondary citation", "low")


def main():
    cols = ["study", "arm", "device", "outcome", "definition", "time_years", "estimate_pct",
            "ci_low", "ci_high", "n_at_risk", "source_location", "confidence", "source",
            "retrieved_at"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            cite = {"NOTION": NOTION_CITE, "Suzuki": SUZUKI_CITE, "Nardi": NARDI_CITE,
                    "PARTNER": NOTION_CITE, "Self-expanding": NOTION_CITE,
                    "Yokoyama": NARDI_CITE, "Fukuhara": NARDI_CITE, "Werner": NARDI_CITE}
            src = next((v for k, v in cite.items() if r["study"].startswith(k)), "")
            r["source"] = src
            r["retrieved_at"] = RETRIEVED_AT
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()

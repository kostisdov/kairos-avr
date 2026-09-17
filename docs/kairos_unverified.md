# KAIROS revision: details not verified

Items stated in the revised document on the strength of secondary sources or earlier team notes, which should be checked against the primary source or the repository before the submission is finalised. Each is tagged "Unverified" in the document where it appears.

## Clinical evidence

1. 2025 ESC/EACTS valvular heart disease guideline: the exact follow-up imaging table (intervals and class of recommendation) after surgical bioprostheses and TAVI. Only the 2021 ESC/EACTS position (yearly clinical review with imaging for bioprostheses) and the 2020 ACC/AHA position were confirmed from secondary summaries.
2. The 123-patient surgical series reporting an adjusted hazard ratio near 2 for SVD after three or more months of vitamin K antagonist exposure: citation not located.
3. The ANTIPRO substudy odds ratio of about 4.3 for higher bioprosthetic fluoride uptake on warfarin: PubMed record located earlier (39067525) but the figure was not read from the paper.
4. Quebec cohort sub-hazard ratio 0.50 for anticoagulation and structural deterioration: source not located in this revision. FRANCE-TAVI adjusted hazard ratio 0.54 is cited to Overtchouk et al., JACC 2019, but the specific value was not read from the paper.
5. The 97-patient imaging study finding no association between Lp(a) and degeneration over two years, and the 2026 389-patient reintervention study finding no difference by Lp(a): citations not located.
6. The haemodialysis series linking pre-operative phosphate to SVD with six events, and the dialysis cohort on accelerated SVD: citations not located.
7. The 1,193-patient observational study finding no slower deterioration with lipid-lowering therapy: citation not located.
8. Senage et al., Nat Med 2022 and Briand et al., Circulation 2006: cited from memory of the bibliographic details; pages and volume should be confirmed.
9. O'Hair et al., JAMA Cardiol 2023 volume and pages: cited via an ACC summary page; confirm against the journal.
10. Metabolic syndrome as a predictor of faster bioprosthetic degeneration: relies on Briand 2006; no more recent confirmation checked.

## Data and implementation

11. Years extending to 2027 in the laboratory and medication files: explained as de-identification date shifting because the medication export carries an empty date-shift column; not confirmed by the data provider.
12. Resolved on 16 September with authorisation: LOTUS Edge is now classed as "mechanically expanded intra-annular TAVR" in the curated table inside `src/kairos/build_device_table.py`, and `device_table.csv` and both FDA summary tables were regenerated. The free-text alias forms are now also held in the build script, so regeneration keeps them; 39 tests pass.
13. Reference-value table: 92 of 256 rows mapped to canonical models; the unmapped rows and the absence of Magna and Magna Ease from the ASE 2024 appendix were reported by the build process and spot-checked on ten cells, not exhaustively.
14. VARC-3 module: the stage 0 versus stage 1 boundary is the team's interpretation and awaits physician confirmation.
15. The ClinicalTrials.gov permissive durability flag (714 of 755 rows) was produced by a rule applied to titles and descriptions; individual rows were not reviewed.
16. The 20-note extraction audit is a pilot and its per-field agreement figures are not yet recorded here; extraction accuracy on real notes is therefore unknown and is presented only as a pending pilot.
17. The primary model, evaluation harness, patient demonstration and slides are built; this line is retained only to mark that they are demonstrated on synthetic scenarios, never validated on patients; their status in the document is "to build".
18. MAUDE counts: the openFDA API appears to drop the plus sign in "Evolut FX+" and "Evolut PRO+", so those rows duplicate their base models; not independently confirmed.

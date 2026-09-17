# Digitising a published Kaplan-Meier curve for KM reconstruction

## Why this document exists

Task 4 asked us to attempt pulling vector coordinates directly out of a Kaplan-Meier figure
in a PMC PDF using pdfplumber (`page.curves` / `page.rects` / `page.lines`), and to fall
back to manual digitisation instructions if the figure turns out to be a raster image.

That is exactly what happened. Checked this session (2026-09-16):

- The PMC journal articles (NOTION 10-year; Trifecta vs Perimount Magna Ease; Trifecta vs
  Perceval) could not be downloaded as PDFs at all -- direct HTTP GET to PMC is blocked by a
  Google reCAPTCHA Enterprise challenge page, and the "Download PDF" button resolves to a
  same-origin blob not exposed to a non-browser client or to network-request introspection.
  Full article **text** was captured via an interactive browser render instead (see
  `data/raw/papers/*_fulltext.txt` and `data/raw/MANIFEST.csv`), but their KM **figures**
  (NOTION Figures 1-4; Trifecta-vs-Magna-Ease Figure 2; Trifecta-vs-Perceval Figures 1-12)
  were never obtained as files at all, vector or raster -- there is nothing to digitise from
  this session's downloads for those three papers.
- The FDA SSED PDFs (which we do have) also contain KM-style curves (e.g. SAPIEN 3 SSED
  `P140031B.pdf` Figure 15 "All Stroke at 1 Year", page 37; LOTUS Edge SSED
  Figures 9-11, pages 29-34). These were inspected directly with pdfplumber:
  `page.rects`/`page.lines`/`page.curves` on the actual figure pages return either zero
  vector path elements or only the surrounding table's cell borders, while `page.images`
  consistently shows exactly one embedded raster image per figure (e.g. SAPIEN 3 page 37:
  one 362x228px image, zero curves; confirmed for several pages). **These are raster
  images, not vector-drawn charts** -- there is no coordinate data to pull out
  programmatically.

So: no curve in any PDF obtained this session is machine-readable. This is the documented,
confirmed reason `data/reference/published_curves/` contains no
`*_reconstructed_not_real.csv` files yet. `src/kairos/km_reconstruct.py` implements and
self-tests the reconstruction algorithm (Guyot 2012 "iKM", since R/`IPDfromKM` is not
available in this environment -- `Rscript` was confirmed not on PATH) against a synthetic
example, so it is ready to run the moment real digitised coordinates exist.

## What the modelling engineer / a team member needs to do

1. **Get an image of the figure.** The most useful curves for KAIROS scenario
   parameterisation, in priority order:
   - NOTION Figure 3 (SVD, moderate-or-severe and severe, TAVI vs SAVR) and Figure 4
     (BVD/BVF) -- `https://pmc.ncbi.nlm.nih.gov/articles/PMC10984572/` (view in a browser;
     right-click the figure -> "Open image in new tab" -> save, or use the journal's own
     figure-download link if present; the PMC page itself renders fine in a browser even
     though scripted downloads are blocked, per this session's finding above).
   - Suzuki 2022 Figure 2 (Trifecta vs Perimount Magna Ease freedom-from-reoperation) --
     `https://pmc.ncbi.nlm.nih.gov/articles/PMC9373186/`.
   - Nardi 2026 Figures 7/8/11/12 (freedom from SVD, Trifecta vs Perceval, and stratified by
     PPM) -- `https://pmc.ncbi.nlm.nih.gov/articles/PMC13122456/`.
   - SAPIEN 3 SSED Figure 15 (stroke) and LOTUS Edge SSED Figures 9-11 (gradient/EOA vs
     CoreValve) if finer granularity than the headline SSED numbers already in
     `data/reference/fda_ssed_events.csv` is needed -- both PDFs are already in
     `data/raw/fda_ssed/`.

2. **Digitise it with [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/)** (free,
   browser-based, no install; this is the same class of tool the IPDfromKM paper itself
   recommends -- see `data/raw/papers/ipdfromkm_method_PMC8168323_fulltext.txt`).
   - Upload the figure image.
   - Choose "2D (X-Y) Plot", calibrate the X axis (time) and Y axis (survival probability or
     cumulative incidence -- note which one the figure actually plots, and if it is
     cumulative incidence, remember `S(t) = 1 - CumInc(t)` before handing it to
     `km_reconstruct.py`, which expects a survival probability).
   - Digitise points **at both the top and bottom of every vertical drop** (per the
     IPDfromKM paper's own guidance), and as many points as practical along each flat
     segment. Extract one series per arm/curve (do them separately if curves overlap or
     cross -- WebPlotDigitizer's "Mask" tool or manual point placement handles this).
   - Export as CSV: two columns, `time, survival_probability`.

3. **Transcribe the numbers-at-risk table** printed under the figure (every published KM
   figure in these papers has one) into a second small CSV: `time, n_at_risk`, one row per
   column of the printed table, first row `0, <starting N for that arm>`.

4. **Run the reconstruction:**
   ```python
   import csv
   from kairos.km_reconstruct import reconstruct_km

   def load_pairs(path):
       with open(path) as f:
           return [(float(a), float(b)) for a, b in csv.reader(f)]  # skip header row first

   points = load_pairs("digitised_survival_points.csv")
   risk_table = load_pairs("digitised_risk_table.csv")
   result = reconstruct_km(points, risk_table)
   result.to_csv(
       "data/reference/published_curves/notion_savr_svd_reconstructed_not_real.csv",
       extra_cols={"study": "NOTION", "arm": "SAVR", "outcome": "moderate-or-severe SVD"},
   )
   ```

5. **Name the output file with `reconstructed_not_real` in it, always**, per the
   task's non-negotiable instructions -- this is pseudo-patient data statistically
   consistent with a published curve, not real patient data, and must never be confused
   with the genuine (and much more sensitive) `data/derived/private/passport.*` extraction
   from the real de-identified notes/labs/medications files.

6. **Sanity-check** the reconstruction with
   `km_reconstruct.kaplan_meier_from_pseudo_ipd(result.times, result.events)` and overlay it
   against the digitised points -- they should track closely (see the worked example in
   `src/kairos/km_reconstruct.py`'s `__main__` block, which does exactly this on synthetic
   data and prints both curves side by side).

## CSV templates

`digitised_survival_points.csv` (per arm):
```
time,survival_probability
0,1.00
1,0.94
...
```

`digitised_risk_table.csv` (per arm; first row must be `0,<starting N>`):
```
time,n_at_risk
0,145
1,138
...
```

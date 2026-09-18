# Notebooks

`01_kairos_proof_of_concept.ipynb` runs the pipeline end to end and is committed with its outputs.

1. **Extract** a valve passport from the real de-identified notes (skipped when the spreadsheets are
   absent, as they are in this repository; committed aggregates are shown instead).
2. **Stage** an echo against the patient's own reference study under VARC-3.
3. **Build** a landmark dataset from the synthetic `gradual_stenotic` scenario, with the leakage guard.
4. **Fit** the cause-specific model and show the four probabilities (SVD, death, replacement for
   another reason, alive with an intact valve) for two contrasting patients.
5. **Compare** with current practice: guideline surveillance with and without KAIROS, read from
   `docs/comparison/surveillance/`.

Steps 3 to 5 are synthetic and illustrative. To run it:

```bash
pip install -e ".[dev]" jupyter
jupyter nbconvert --to notebook --execute --inplace notebooks/01_kairos_proof_of_concept.ipynb
```

The heavier analyses have their own scripts: `scripts/build_all.py` (training and the model
ladder), `scripts/evaluate_surveillance.py` (the comparison with current practice) and
`scripts/score_real_extract.py` (the implant-time model on the real notes, where they are available).

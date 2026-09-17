# -*- coding: utf-8 -*-
"""
Task 5 (openFDA half). For each canonical valve model in data/reference/device_table.csv,
count openFDA MAUDE device-event reports overall and reports whose CODED product_problem
mentions structural-valve-deterioration-like terms, into
data/reference/maude_counts_by_brand.csv.

No API key used (openFDA allows ~40 req/min unauthenticated); a 1.6s pause between calls
keeps this run to ~37 req/min, safely under that limit, confirmed by wall-clock timing in
docs/data_build_log.md.

Query-syntax note (found the hard way, see docs/data_build_log.md): building the `search`
string with a literal "+AND+" and passing it through `requests.get(params=...)` double-
encodes the "+" to "%2B", which openFDA does NOT parse as AND/space -- every such query
returned a false "404 No matches found". The fix is to write the query with real space
characters (" AND ", "(term1 term2 term3)") and let `requests` do the URL-encoding; boolean
AND and same-field OR both then work as openFDA's documentation describes.

The SVD-like keyword list is NOT invented -- it was derived by pulling the actual MAUDE
`product_problems` coded-term frequency distribution for Trifecta (a brand with a large,
well-documented report volume) via `count=product_problems.exact`, then keeping every code
that is plausibly SVD/structural/thrombotic in nature (see docs/data_build_log.md for the
full observed frequency table). This is deliberately broader than the literal string
"structural valve deterioration" because MAUDE's own coded vocabulary never uses that exact
phrase -- it uses granular terms like "Material Split, Cut or Torn", "Device Stenosis",
"Calcified", "Intravalvular regurgitation", "Backflow", "Gradient Increase", etc.
"""
import csv
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from kairos.paths import repo_root

ROOT = str(repo_root())
RETRIEVED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

SVD_LIKE_TERMS = [
    "calcified", "calcification", "stenosis", "regurgitation", "regurgitant", "backflow",
    "leak", "tear", "torn", "split", "deteriorated", "deterioration", "degraded",
    "degradation", "fracture", "deformation", "obstruction", "coaptation", "thrombosis",
    "thrombus", "thrombotic", "gradient",
]
CAVEAT = "reported counts, not incidence; passive surveillance, no denominator"


def query_count(search_str, tries=3):
    for i in range(tries):
        try:
            r = requests.get("https://api.fda.gov/device/event.json",
                              params={"search": search_str, "limit": 1}, timeout=30)
            if r.status_code == 200:
                return r.json()["meta"]["results"]["total"]
            if r.status_code == 404:
                return 0  # openFDA's way of saying "zero matches", not an error
            print(f"  [{r.status_code}] {search_str[:80]}")
            return None
        except Exception as ex:
            print(f"  retry {i}: {ex}")
            time.sleep(2)
    return None


def brand_query_term(canonical_model, brand_name_gudid):
    """Pick a reasonably specific MAUDE brand_name search term for this canonical model.
    Prefers the canonical_model name itself (usually specific enough -- e.g. 'Trifecta GT',
    'SAPIEN 3 Ultra'); for single common-word models that risk collision with unrelated
    devices (Epic, Lotus, Mosaic -- the same generic-word collision problem found in GUDID,
    see build_device_table.py), qualifies with the manufacturer/family word too."""
    single_word_risk = {"Epic", "Epic Supra", "Lotus", "Mosaic", "Mosaic Ultra", "Freestyle",
                         "Prima Plus", "Crown PRT"}
    if canonical_model in single_word_risk:
        # use "<brand> valve" to bias toward cardiac devices
        return f"{canonical_model} valve"
    return canonical_model


def main():
    device_table = pd.read_csv(f"{ROOT}/data/reference/device_table.csv")
    models = device_table[["canonical_model", "brand_name_gudid"]].drop_duplicates()

    rows = []
    n_req = 0
    t0 = time.time()
    for _, row in models.iterrows():
        canon = row["canonical_model"]
        query_term = brand_query_term(canon, row.get("brand_name_gudid", ""))
        search_total = f'device.brand_name:"{query_term}"'
        total = query_count(search_total)
        n_req += 1
        time.sleep(1.6)

        terms_clause = "(" + " ".join(SVD_LIKE_TERMS) + ")"
        search_svd = f'device.brand_name:"{query_term}" AND product_problems:{terms_clause}'
        svd_like = query_count(search_svd)
        n_req += 1
        time.sleep(1.6)

        rows.append(dict(
            brand_query=query_term, canonical_model=canon,
            reports_total=total if total is not None else "",
            reports_svd_like=svd_like if svd_like is not None else "",
            years_covered="MAUDE coverage varies by device/report type (device reports "
                           "since 1996 mfr / 1991 user-facility / 1993 voluntary; not "
                           "queried per-year this session, see caveat)",
            query_string=search_svd, retrieved_at=RETRIEVED_AT, caveat=CAVEAT,
        ))
        print(f"{canon:35s} total={total!s:>8s} svd_like={svd_like!s:>8s}  "
              f"({n_req} requests, {time.time()-t0:.0f}s elapsed)")

    cols = ["brand_query", "canonical_model", "reports_total", "reports_svd_like",
            "years_covered", "query_string", "retrieved_at", "caveat"]
    with open(f"{ROOT}/data/reference/maude_counts_by_brand.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> data/reference/maude_counts_by_brand.csv "
          f"({n_req} total openFDA requests in {time.time()-t0:.0f}s "
          f"= {n_req/((time.time()-t0)/60):.1f} req/min)")


if __name__ == "__main__":
    main()

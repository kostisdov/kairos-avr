# -*- coding: utf-8 -*-
"""Stream-parse the 209 GUDID FULLDownload XML parts inside the full-release zip,
keep only device records that are replacement heart valves (FDA product code
DYE or NPT, verified via openFDA /device/classification.json) or whose brand
name / device description matches a curated bioprosthetic/TAVR valve keyword
list (covers edge cases like sutureless valves that may be coded differently).
Writes a flat JSON-lines intermediate to data/raw/gudid/ for the device-table
build step. No patient data involved -- this is public device data only.
"""
import json
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from kairos.paths import repo_root

ZIP_PATH = str(repo_root() / "data/raw/gudid/gudid_full_release_20260901.zip")
OUT_PATH = str(repo_root() / "data/raw/gudid/gudid_valve_records.jsonl")
NS = "{http://www.fda.gov/cdrh/gudid}"

PRODUCT_CODES = {"DYE", "NPT"}

BRAND_KEYWORDS = re.compile(
    r"perimount|magna|inspiris|resilia|avalus|trifecta|mitroflow|crown\s*prt|"
    r"\bepic\b|mosaic|hancock|freestyle|toronto\s*spv|perceval|intuity|sapien|"
    r"evolut|corevalve|corevalue|acurate|myval|portico|navitor|\blotus\b|"
    r"jenavalve|konect|carpentier",
    re.I,
)

TARGET_COMPANIES = re.compile(
    r"edwards\s*lifesciences|medtronic|abbott|st\.?\s*jude|livanova|corcym|"
    r"sorin|boston\s*scientific|symetis|meril|jenavalve",
    re.I,
)


def text(el, tag):
    child = el.find(NS + tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_device(dev):
    brand = text(dev, "brandName") or ""
    desc = text(dev, "deviceDescription") or ""
    company = text(dev, "companyName") or ""

    product_codes = []
    pc_parent = dev.find(NS + "productCodes")
    if pc_parent is not None:
        for pc in pc_parent.findall(NS + "fdaProductCode"):
            code = text(pc, "productCode")
            name = text(pc, "productCodeName")
            if code:
                product_codes.append({"code": code, "name": name})

    code_hit = any(pc["code"] in PRODUCT_CODES for pc in product_codes)
    brand_hit = bool(BRAND_KEYWORDS.search(brand)) or bool(BRAND_KEYWORDS.search(desc))
    company_hit = bool(TARGET_COMPANIES.search(company))

    if not (code_hit or brand_hit):
        return None
    # avoid pulling unrelated accessories from big device companies: require
    # code_hit OR brand_hit (already enforced); company_hit is informational only

    identifiers = []
    id_parent = dev.find(NS + "identifiers")
    if id_parent is not None:
        for ident in id_parent.findall(NS + "identifier"):
            identifiers.append({
                "deviceId": text(ident, "deviceId"),
                "deviceIdType": text(ident, "deviceIdType"),
                "deviceIdIssuingAgency": text(ident, "deviceIdIssuingAgency"),
            })

    gmdn = []
    gmdn_parent = dev.find(NS + "gmdnTerms")
    if gmdn_parent is not None:
        for g in gmdn_parent.findall(NS + "gmdn"):
            gmdn.append(text(g, "gmdnPTName"))

    sizes = []
    sz_parent = dev.find(NS + "deviceSizes")
    if sz_parent is not None:
        for sz in sz_parent.findall(NS + "deviceSize"):
            size_el = sz.find(NS + "size")
            sizes.append({
                "sizeType": text(sz, "sizeType"),
                "value": size_el.get("value") if size_el is not None else None,
                "unit": size_el.get("unit") if size_el is not None else None,
                "sizeText": text(sz, "sizeText"),
            })

    return {
        "publicDeviceRecordKey": text(dev, "publicDeviceRecordKey"),
        "brandName": brand,
        "versionModelNumber": text(dev, "versionModelNumber"),
        "catalogNumber": text(dev, "catalogNumber"),
        "companyName": company,
        "dunsNumber": text(dev, "dunsNumber"),
        "deviceDescription": desc,
        "commDistributionStatus": text(dev, "deviceCommDistributionStatus"),
        "commDistributionEndDate": text(dev, "deviceCommDistributionEndDate"),
        "devicePublishDate": text(dev, "devicePublishDate"),
        "identifiers": identifiers,
        "gmdnTerms": [g for g in gmdn if g],
        "productCodes": product_codes,
        "deviceSizes": sizes,
        "companyKeywordHit": company_hit,
    }


def main():
    z = zipfile.ZipFile(ZIP_PATH)
    names = sorted(
        [n for n in z.namelist() if n.startswith("FULLDownload_Part")],
        key=lambda n: int(re.search(r"Part(\d+)_Of", n).group(1)),
    )
    print(f"{len(names)} part files to scan")
    t0 = time.time()
    n_devices_seen = 0
    n_matched = 0
    out = open(OUT_PATH, "w", encoding="utf-8")
    for i, name in enumerate(names, 1):
        with z.open(name) as fh:
            for event, elem in ET.iterparse(fh, events=("end",)):
                if elem.tag == NS + "device":
                    n_devices_seen += 1
                    rec = parse_device(elem)
                    if rec is not None:
                        n_matched += 1
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    elem.clear()
                elif elem.tag == NS + "header":
                    elem.clear()
        if i % 20 == 0 or i == len(names):
            dt = time.time() - t0
            print(f"  [{i}/{len(names)}] parts scanned, {n_devices_seen} devices seen, "
                  f"{n_matched} matched, {dt:.1f}s elapsed")
    out.close()
    print(f"DONE. devices_seen={n_devices_seen} matched={n_matched} "
          f"elapsed={time.time()-t0:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Export the JSON schemas of every KAIROS payload to docs/schemas and check for drift.

    python scripts/export_schemas.py          # write docs/schemas/*.json
    python scripts/export_schemas.py --check  # exit 1 if a committed schema differs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kairos.extraction.schema import PAYLOAD_MODELS  # noqa: E402

OUT = ROOT / "docs" / "schemas"


def render() -> dict[str, str]:
    out = {}
    for name, model in PAYLOAD_MODELS.items():
        schema = model.model_json_schema(by_alias=True)
        out[name] = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    rendered = render()
    OUT.mkdir(parents=True, exist_ok=True)
    drift = []
    for name, text in rendered.items():
        path = OUT / f"{name}.json"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                drift.append(name)
        else:
            path.write_text(text, encoding="utf-8")
    if args.check:
        if drift:
            print("schema drift:", ", ".join(drift), "(run scripts/export_schemas.py and commit)")
            return 1
        print(f"schemas up to date ({len(rendered)} payloads)")
        return 0
    print(f"wrote {len(rendered)} schemas to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

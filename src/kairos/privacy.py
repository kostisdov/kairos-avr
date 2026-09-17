"""Privacy scan: no patient-level material in committable folders or build contexts.

Findings are raised for: the de-identified patient identifier pattern ``Patient_\\d{3}``;
spreadsheet files; anything under ``data/raw``, ``data/derived/private`` or ``tmp``; the
column names of the source spreadsheets appearing in data files; and long free-text
cells in data files that read like clinical narrative. Documentation and source code are
scanned for the identifier pattern only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PATIENT_ID_RE = re.compile(r"Patient_\d{3}")
NOTE_REF_RE = re.compile(r"\bPatient_\d{3}_\d+\b")
SOURCE_COLUMNS = ("Profile Key", "Authoring Provider", "Signed Status", "Service Date")
NARRATIVE_WORDS = re.compile(r"\b(patient|valve|gradient|echocardiogra|sternotomy|bioprosth)", re.I)
FORBIDDEN_PARTS = {"raw", "private", "tmp"}
FORBIDDEN_SUFFIXES = {".xlsx", ".xls", ".xlsm"}
DATA_SUFFIXES = {".csv", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".parquet"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
             "artifacts", ".azure", "tmp", "build", "dist"}
SKIP_DIR_PATHS = {("data", "raw"), ("data", "derived", "private")}
MAX_CELL_CHARS = 600


@dataclass
class Finding:
    path: str
    rule: str
    detail: str

    def as_dict(self) -> dict:
        return {"path": self.path, "rule": self.rule, "detail": self.detail}


def _is_skipped(rel: Path) -> bool:
    parts = rel.parts
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return True
    for skip in SKIP_DIR_PATHS:
        if parts[:len(skip)] == skip:
            return True
    return False


def _is_forbidden_location(rel: Path) -> bool:
    parts = rel.parts
    if parts[:2] == ("data", "raw") or parts[:3] == ("data", "derived", "private"):
        return True
    return parts[0] == "tmp" if parts else False


def iter_files(root: Path, include_forbidden: bool) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        # repository mode: the de-identified spreadsheets sit git-ignored at the root
        if not include_forbidden and len(rel.parts) == 1 and p.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        if _is_forbidden_location(rel):
            if include_forbidden:
                yield p
            continue
        if _is_skipped(rel):
            continue
        yield p


def scan_file(path: Path, root: Path, self_path: Path | None = None) -> list[Finding]:
    rel = path.relative_to(root)
    rels = rel.as_posix()
    out: list[Finding] = []
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        out.append(Finding(rels, "spreadsheet", "spreadsheet files never enter committable folders or build contexts"))
        return out
    if _is_forbidden_location(rel):
        out.append(Finding(rels, "forbidden-path", "data/raw, data/derived/private and tmp must not be present"))
        return out
    if self_path is not None and path.resolve() == self_path.resolve():
        return out
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            for col in df.columns:
                if col in SOURCE_COLUMNS:
                    out.append(Finding(rels, "source-column", f"column {col!r} of the source spreadsheets"))
                if df[col].dtype == object:
                    s = df[col].dropna().astype(str)
                    if (s.str.contains(PATIENT_ID_RE)).any():
                        out.append(Finding(rels, "patient-id", f"column {col!r} carries Patient_### identifiers"))
                    if (s.str.len() > MAX_CELL_CHARS).any():
                        out.append(Finding(rels, "narrative", f"column {col!r} has cells longer than {MAX_CELL_CHARS} characters"))
        except Exception as ex:  # noqa: BLE001
            out.append(Finding(rels, "unreadable", f"parquet could not be read: {type(ex).__name__}"))
        return out
    if path.suffix.lower() not in DATA_SUFFIXES and path.suffix.lower() not in {".py", ".toml", ".cfg", ".ini", ".bicep", ".bicepparam", ".ps1", ".sh", ".html", ".sql"}:
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return out
    is_source_or_doc = rel.parts[0] in ("src", "scripts", "services", "tests", "docs", "infra") or path.suffix.lower() in {".py", ".md", ".html", ".bicep", ".bicepparam", ".ps1", ".sh", ".toml"}
    for m in PATIENT_ID_RE.finditer(text):
        # documentation may name the pattern itself; a concrete identifier is a hit
        line = text[max(0, m.start() - 80): m.end() + 80]
        if "Patient_\\d" in line or "Patient_###" in line:
            continue
        out.append(Finding(rels, "patient-id", f"identifier {m.group(0)}"))
        break
    if is_source_or_doc:
        return out
    if path.suffix.lower() in {".csv", ".json", ".jsonl", ".yaml", ".yml"}:
        for col in SOURCE_COLUMNS:
            if col in text:
                out.append(Finding(rels, "source-column", f"source spreadsheet column name {col!r}"))
        if path.suffix.lower() in {".csv", ".json", ".jsonl"}:
            for chunk in re.split(r"[\n,\"]", text):
                if len(chunk) > MAX_CELL_CHARS and NARRATIVE_WORDS.search(chunk):
                    out.append(Finding(rels, "narrative", f"free-text cell of {len(chunk)} characters with clinical wording"))
                    break
    return out


def scan(root: Path, include_forbidden: bool = True, self_path: Path | None = None) -> list[Finding]:
    root = Path(root)
    findings: list[Finding] = []
    for p in iter_files(root, include_forbidden):
        findings.extend(scan_file(p, root, self_path))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KAIROS privacy scan")
    ap.add_argument("--root", default=".", help="directory to scan (repository root or a build staging directory)")
    ap.add_argument("--json", action="store_true", help="print findings as JSON")
    ap.add_argument("--allow-private-dirs", action="store_true",
                    help="do not fail on data/raw, data/derived/private and tmp (repository scan); build contexts must not use this")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    findings = scan(root, include_forbidden=not args.allow_private_dirs, self_path=Path(__file__))
    if args.json:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
    else:
        for f in findings:
            print(f"{f.rule:14s} {f.path}: {f.detail}")
        print(f"privacy scan: {len(findings)} finding(s) under {root}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

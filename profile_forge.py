#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VFP_DIR = ROOT / "flatten" / "vfp-vuln"
OUTPUT_FILE = ROOT / "forge_profile_output.txt"
CWE_RE = re.compile(r"^CWE-\d+$")
SEVERITIES = ("Critical", "High", "Medium", "Low")


def extract_cwes(value: object) -> list[str]:
    """Recursively collect CWE identifiers from dict/list/string category values."""
    cwes: list[str] = []

    if isinstance(value, dict):
        for nested in value.values():
            cwes.extend(extract_cwes(nested))
    elif isinstance(value, list):
        for nested in value:
            cwes.extend(extract_cwes(nested))
    elif isinstance(value, str) and CWE_RE.match(value.strip()):
        cwes.append(value.strip())

    return cwes


def has_source_code(vfp: dict[str, object]) -> bool:
    source_code = vfp.get("source_code")
    if source_code is None:
        return False
    if isinstance(source_code, str):
        return bool(source_code.strip())
    if isinstance(source_code, (list, dict, tuple, set)):
        return bool(source_code)
    return bool(source_code)


def main() -> None:
    if not VFP_DIR.is_dir():
        raise SystemExit(f"VFP directory not found: {VFP_DIR}")

    cwe_totals: Counter[str] = Counter()
    severity_by_cwe: dict[str, Counter[str]] = defaultdict(Counter)
    total_vfps = 0
    vfps_with_source = 0

    for json_path in sorted(VFP_DIR.glob("*.json")):
        total_vfps += 1
        with json_path.open("r", encoding="utf-8") as handle:
            vfp = json.load(handle)

        if isinstance(vfp, dict) and has_source_code(vfp):
            vfps_with_source += 1

        findings = vfp.get("findings", []) if isinstance(vfp, dict) else []
        if not isinstance(findings, list):
            continue

        for finding in findings:
            if not isinstance(finding, dict):
                continue

            severity = str(finding.get("severity", "Unknown")).strip() or "Unknown"
            cwes = extract_cwes(finding.get("category", {}))

            for cwe in cwes:
                cwe_totals[cwe] += 1
                severity_by_cwe[cwe][severity] += 1

    lines: list[str] = []
    lines.append(f"Total VFP file count: {total_vfps}")
    lines.append(f"VFPs with source code: {vfps_with_source}")
    lines.append(f"VFPs without source code: {total_vfps - vfps_with_source}")
    lines.append("")
    lines.append("Top 60 CWE categories by total count")
    lines.append(f"{'CWE ID':<12} {'Total':>8} {'High+Critical count':>20}")
    lines.append(f"{'-' * 12} {'-' * 8:>8} {'-' * 20:>20}")

    for cwe, total in cwe_totals.most_common(60):
        high_critical = severity_by_cwe[cwe]["High"] + severity_by_cwe[cwe]["Critical"]
        lines.append(f"{cwe:<12} {total:>8} {high_critical:>20}")

    lines.append("")
    lines.append(f"Total unique CWEs found: {len(cwe_totals)}")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    OUTPUT_FILE.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

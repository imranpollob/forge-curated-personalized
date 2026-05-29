#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
VFP_VULN_DIR = ROOT / "flatten" / "vfp-vuln"
VFP_ALL_DIR = ROOT / "flatten" / "vfp"
CONTRACTS_DIR = ROOT / "dataset-curated" / "contracts"
CONTRACTS_RAW_DIR = ROOT / "dataset-curated" / "contracts-raw"
OUTPUT_FILE = ROOT / "forge_profile_v2_output.txt"

CWE_RE = re.compile(r"^CWE-\d+$")

CWE_TO_CLASS = {
    "CWE-682": "arithmetic_oracle",
    "CWE-191": "arithmetic_oracle",
    "CWE-190": "arithmetic_oracle",
    "CWE-691": "reentrancy",
    "CWE-841": "reentrancy",
    "CWE-693": "defi_protection",
    "CWE-284": "access_control",
    "CWE-285": "access_control",
    "CWE-862": "access_control",
    "CWE-863": "access_control",
    "CWE-664": "state_management",
    "CWE-665": "state_management",
    "CWE-666": "state_management",
    "CWE-672": "state_management",
    "CWE-703": "unchecked_call",
    "CWE-754": "unchecked_call",
    "CWE-755": "unchecked_call",
    "CWE-252": "unchecked_call",
    "CWE-362": "race_condition",
    "CWE-20": "input_validation",
    "CWE-707": "input_validation",
    "CWE-1284": "input_validation",
}

CLASS_ORDER = (
    "arithmetic_oracle",
    "reentrancy",
    "defi_protection",
    "access_control",
    "state_management",
    "unchecked_call",
    "race_condition",
    "input_validation",
)


def extract_cwes(value: object) -> set[str]:
    cwes: set[str] = set()

    if isinstance(value, dict):
        for nested in value.values():
            cwes.update(extract_cwes(nested))
    elif isinstance(value, list):
        for nested in value:
            cwes.update(extract_cwes(nested))
    elif isinstance(value, str) and CWE_RE.match(value.strip()):
        cwes.add(value.strip())

    return cwes


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def affected_sol_count(vfp: dict[str, object]) -> int:
    affected_files = vfp.get("affected_files")
    if not isinstance(affected_files, dict) or not affected_files:
        return 0
    return sum(1 for filename in affected_files if str(filename).endswith(".sol"))


def collect_vfp_cwes(vfp: dict[str, object]) -> set[str]:
    cwes: set[str] = set()
    findings = vfp.get("findings", [])
    if not isinstance(findings, list):
        return cwes

    for finding in findings:
        if isinstance(finding, dict):
            cwes.update(extract_cwes(finding.get("category", {})))

    return cwes


def profile_vfp_dir(directory: Path) -> dict[str, object]:
    if not directory.is_dir():
        raise SystemExit(f"VFP directory not found: {directory}")

    json_paths = sorted(directory.glob("*.json"))
    class_counts: Counter[str] = Counter()
    all_cwes: set[str] = set()
    source_sol_counts: list[int] = []
    no_mapped_class_count = 0

    for json_path in json_paths:
        vfp = load_json(json_path)

        sol_count = affected_sol_count(vfp)
        if sol_count > 0:
            source_sol_counts.append(sol_count)

        cwes = collect_vfp_cwes(vfp)
        all_cwes.update(cwes)

        mapped_classes = {CWE_TO_CLASS[cwe] for cwe in cwes if cwe in CWE_TO_CLASS}
        if mapped_classes:
            for class_name in mapped_classes:
                class_counts[class_name] += 1
        else:
            no_mapped_class_count += 1

    total_vfps = len(json_paths)
    with_source = len(source_sol_counts)

    return {
        "total_vfps": total_vfps,
        "with_source": with_source,
        "without_source": total_vfps - with_source,
        "total_sol_files": sum(source_sol_counts),
        "avg_sol_files": mean(source_sol_counts) if source_sol_counts else 0.0,
        "min_sol_files": min(source_sol_counts) if source_sol_counts else 0,
        "max_sol_files": max(source_sol_counts) if source_sol_counts else 0,
        "unique_cwes": len(all_cwes),
        "class_counts": class_counts,
        "no_mapped_class_count": no_mapped_class_count,
    }


def append_class_table(lines: list[str], class_counts: Counter[str]) -> None:
    lines.append(f"{'Class':<22} {'VFP count (positive samples)':>28}")
    lines.append(f"{'-' * 22} {'-' * 28:>28}")
    for class_name in CLASS_ORDER:
        lines.append(f"{class_name:<22} {class_counts[class_name]:>28}")


def count_sol_files(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.rglob("*.sol") if path.is_file())


def main() -> None:
    vuln_profile = profile_vfp_dir(VFP_VULN_DIR)
    all_profile = profile_vfp_dir(VFP_ALL_DIR)

    lines: list[str] = []
    lines.append("Part A - Source Code Availability in flatten/vfp-vuln")
    lines.append(f"Total VFPs: {vuln_profile['total_vfps']}")
    lines.append(f"VFPs with source code: {vuln_profile['with_source']}")
    lines.append(f"VFPs without source code: {vuln_profile['without_source']}")
    lines.append(f"Total .sol files across VFPs with source code: {vuln_profile['total_sol_files']}")
    lines.append(f"Average .sol files per VFP with source code: {vuln_profile['avg_sol_files']:.2f}")
    lines.append(f"Min .sol files per VFP with source code: {vuln_profile['min_sol_files']}")
    lines.append(f"Max .sol files per VFP with source code: {vuln_profile['max_sol_files']}")
    lines.append("")

    lines.append("Part B - Per-VFP Multi-Label Profile in flatten/vfp-vuln")
    append_class_table(lines, vuln_profile["class_counts"])
    lines.append(f"VFPs with no mapped class: {vuln_profile['no_mapped_class_count']}")
    lines.append("")

    lines.append("Part C - Unfiltered Dataset Profile in flatten/vfp")
    lines.append(f"Total VFPs: {all_profile['total_vfps']}")
    lines.append(f"VFPs with source code: {all_profile['with_source']}")
    lines.append(f"VFPs without source code: {all_profile['without_source']}")
    lines.append(f"Total unique CWEs: {all_profile['unique_cwes']}")
    lines.append("")
    append_class_table(lines, all_profile["class_counts"])
    lines.append(f"VFPs with no mapped class: {all_profile['no_mapped_class_count']}")
    lines.append("")

    lines.append("Part D - dataset-curated Contracts")
    lines.append(f".sol files in dataset-curated/contracts/: {count_sol_files(CONTRACTS_DIR)}")
    lines.append(f".sol files in dataset-curated/contracts-raw/: {count_sol_files(CONTRACTS_RAW_DIR)}")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    OUTPUT_FILE.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

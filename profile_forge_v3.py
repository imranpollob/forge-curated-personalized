#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FINDINGS_DIR = ROOT / "dataset-curated" / "findings"
CONTRACTS_DIR = ROOT / "dataset-curated" / "contracts"
VFP_ALL_DIR = ROOT / "flatten" / "vfp"
OUTPUT_FILE = ROOT / "forge_profile_v3_output.txt"
RANDOM_SEED = 42
MAX_STRING_LENGTH = 200
SOL_RE = re.compile(r"[^\s,;:'\"()\[\]{}<>]+\.sol\b")


def truncated(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            return value[:MAX_STRING_LENGTH] + f"... [truncated {len(value) - MAX_STRING_LENGTH} chars]"
        return value
    if isinstance(value, list):
        return [truncated(item) for item in value]
    if isinstance(value, dict):
        return {key: truncated(item) for key, item in value.items()}
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def strings_from(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(strings_from(item))
        return strings
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(strings_from(item))
        return strings
    return []


def sol_refs_from(value: Any) -> set[str]:
    refs: set[str] = set()
    for text in strings_from(value):
        refs.update(match.group(0).strip() for match in SOL_RE.finditer(text))
    return refs


def candidate_paths_for_file(report: dict[str, Any], findings_path: Path, file_ref: str) -> list[Path]:
    cleaned = file_ref.strip().lstrip("./")
    candidates: list[Path] = []

    project_info = report.get("project_info")
    project_paths = project_info.get("project_path") if isinstance(project_info, dict) else None
    if isinstance(project_paths, dict):
        for prefix, source_root in project_paths.items():
            source_path = ROOT / str(source_root)
            prefix_text = str(prefix).strip("/")
            if cleaned == prefix_text:
                candidates.append(source_path)
            elif cleaned.startswith(prefix_text + "/"):
                candidates.append(source_path / cleaned[len(prefix_text) + 1 :])
            candidates.append(source_path / cleaned)

    report_path = report.get("path")
    if isinstance(report_path, str) and report_path:
        candidates.append(CONTRACTS_DIR / f"{Path(report_path).name}-source" / cleaned)

    candidates.append(CONTRACTS_DIR / f"{findings_path.name}-source" / cleaned)
    candidates.append(CONTRACTS_DIR / f"{findings_path.stem}-source" / cleaned)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def tree_lines(root: Path, max_depth: int = 2) -> list[str]:
    lines: list[str] = []
    if not root.exists():
        return [f"{root.name} [missing]"]

    def walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        except OSError:
            return
        for child in children:
            marker = "/" if child.is_dir() else ""
            lines.append(f"{prefix}{child.name}{marker}")
            if child.is_dir():
                walk(child, prefix + "  ", depth + 1)

    lines.append(f"{root.name}/")
    walk(root, "  ", 1)
    return lines


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    lines: list[str] = []

    finding_paths = sorted(FINDINGS_DIR.glob("*.json"))
    sample_finding_path = rng.choice(finding_paths) if finding_paths else None

    total_findings = 0
    findings_with_sol_location_or_files = 0
    findings_with_files: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    referenced_basenames: set[str] = set()

    for finding_path in finding_paths:
        report = load_json(finding_path)
        findings = as_list(report.get("findings"))
        total_findings += len(findings)
        for finding in findings:
            if not isinstance(finding, dict):
                continue

            location_refs = sol_refs_from(finding.get("location"))
            file_refs = sol_refs_from(finding.get("files"))
            if location_refs or file_refs:
                findings_with_sol_location_or_files += 1
            if file_refs:
                findings_with_files.append((finding_path, report, finding))
                referenced_basenames.update(Path(ref).name for ref in file_refs)

    lines.append("Part A - dataset-curated/findings Structure")
    lines.append(f"Total JSON files in dataset-curated/findings/: {len(finding_paths)}")
    if sample_finding_path is not None:
        lines.append(f"Sample JSON: {sample_finding_path.relative_to(ROOT)}")
        lines.append(json.dumps(truncated(load_json(sample_finding_path)), indent=2, sort_keys=True))
    lines.append(f"Total findings across JSON files: {total_findings}")
    lines.append(
        "Findings with location or files referencing .sol filenames: "
        f"{findings_with_sol_location_or_files}"
    )
    lines.append("")

    contract_subdirs = sorted(path for path in CONTRACTS_DIR.iterdir() if path.is_dir())
    sampled_contract_dir = rng.choice(contract_subdirs) if contract_subdirs else None
    sol_files = sorted(path for path in CONTRACTS_DIR.rglob("*.sol") if path.is_file())
    contract_subdir_names = {path.name for path in contract_subdirs}

    vfp_paths = sorted(VFP_ALL_DIR.glob("*.json"))
    sampled_vfp_paths = rng.sample(vfp_paths, min(10, len(vfp_paths)))

    lines.append("Part B - dataset-curated/contracts Structure")
    lines.append(f"Top-level subdirectory count: {len(contract_subdirs)}")
    lines.append("Top-level subdirectory names:")
    for subdir in contract_subdirs:
        lines.append(f"- {subdir.name}")
    if sampled_contract_dir is not None:
        lines.append("")
        lines.append(f"Random subdirectory sample: {sampled_contract_dir.name}")
        lines.extend(tree_lines(sampled_contract_dir, max_depth=2))
    lines.append(f"Total .sol files recursively: {len(sol_files)}")
    lines.append("")
    lines.append("10 random VFP project_name folder-match checks:")
    for vfp_path in sampled_vfp_paths:
        vfp = load_json(vfp_path)
        project_name = str(vfp.get("project_name", ""))
        exact_exists = project_name in contract_subdir_names
        source_suffix_exists = f"{project_name}-source" in contract_subdir_names
        lines.append(
            f"- {project_name} | exact folder: {'yes' if exact_exists else 'no'} | "
            f"project_name-source folder: {'yes' if source_suffix_exists else 'no'}"
        )
    lines.append("")

    sampled_findings = rng.sample(findings_with_files, min(5, len(findings_with_files)))
    lines.append("Part C - Finding File References vs Contracts")
    for index, (finding_path, report, finding) in enumerate(sampled_findings, start=1):
        title = str(finding.get("title", f"finding #{finding.get('id', '?')}"))
        lines.append(f"Sample {index}: {finding_path.name} | {title}")
        for file_ref in sorted(sol_refs_from(finding.get("files"))):
            candidates = candidate_paths_for_file(report, finding_path, file_ref)
            existing_path = next((path for path in candidates if path.is_file()), None)
            attempted_path = existing_path if existing_path is not None else candidates[0]
            exists = existing_path is not None
            lines.append(f"  {'yes' if exists else 'no '} {attempted_path}")
        if not sol_refs_from(finding.get("files")):
            lines.append("  no .sol file refs found after sampling")
    lines.append("")

    referenced_sol_file_count = sum(1 for path in sol_files if path.name in referenced_basenames)
    unreferenced_sol_file_count = len(sol_files) - referenced_sol_file_count

    lines.append("Part D - Potential Negative Samples")
    lines.append(f"Total .sol files in dataset-curated/contracts/: {len(sol_files)}")
    lines.append(f"Unique referenced .sol basenames from findings/files: {len(referenced_basenames)}")
    lines.append(f"Referenced .sol files by basename match: {referenced_sol_file_count}")
    lines.append(f"Unreferenced .sol files by basename (potential negatives): {unreferenced_sol_file_count}")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    OUTPUT_FILE.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

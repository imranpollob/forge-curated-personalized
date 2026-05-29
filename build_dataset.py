#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
VFP_DIR = ROOT / "flatten" / "vfp"
FINDINGS_DIR = ROOT / "dataset-curated" / "findings"
CONTRACTS_DIR = ROOT / "dataset-curated" / "contracts"
OUT_DIR = ROOT / "hypervul_dataset"
SAMPLES_DIR = OUT_DIR / "samples"
LABELS_CSV = OUT_DIR / "labels.csv"
SPLITS_JSON = OUT_DIR / "splits.json"
SUMMARY_TXT = OUT_DIR / "dataset_summary.txt"
ERROR_LOG = OUT_DIR / "build_errors.log"

SEED = 42
NEGATIVE_TARGET = 700
SOL_RE = re.compile(r"[^\s,;:'\"()\[\]{}<>]+\.sol\b")
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
    "CWE-20": "input_validation",
    "CWE-707": "input_validation",
    "CWE-1284": "input_validation",
}

CLASSES = [
    "reentrancy",
    "arithmetic_oracle",
    "defi_protection",
    "access_control",
    "state_management",
    "unchecked_call",
    "input_validation",
]


def reset_output_dir() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    if SAMPLES_DIR.exists():
        shutil.rmtree(SAMPLES_DIR)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for path in (LABELS_CSV, SPLITS_JSON, SUMMARY_TXT, ERROR_LOG):
        if path.exists():
            path.unlink()
    ERROR_LOG.touch()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def log_error(path: Path, exc: BaseException) -> None:
    with ERROR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{path}: {type(exc).__name__}: {exc}\n")


def extract_cwes(value: Any) -> set[str]:
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


def safe_filename(filename: str) -> str:
    safe = filename.strip().replace("\\", "_").replace("/", "_")
    safe = safe.replace("..", "_")
    return safe or "source.sol"


def unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def label_vector_for_cwes(cwes: set[str]) -> dict[str, int]:
    mapped = {CWE_TO_CLASS[cwe] for cwe in cwes if cwe in CWE_TO_CLASS}
    return {class_name: int(class_name in mapped) for class_name in CLASSES}


def label_values(row: dict[str, Any]) -> list[int]:
    return [int(row[class_name]) for class_name in CLASSES]


def build_positive_samples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vfp_paths = sorted(VFP_DIR.glob("*.json"))

    for processed, vfp_path in enumerate(vfp_paths, start=1):
        if processed % 100 == 0:
            print(f"Processed {processed} VFP JSON files")

        try:
            vfp = load_json(vfp_path)
            affected_files = vfp.get("affected_files")
            if not isinstance(affected_files, dict) or not affected_files:
                continue

            findings = vfp.get("findings")
            if not isinstance(findings, list):
                findings = []

            cwes: set[str] = set()
            severities: set[str] = set()
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                cwes.update(extract_cwes(finding.get("category", {})))
                severity = finding.get("severity")
                if isinstance(severity, str) and severity.strip():
                    severities.add(severity.strip())

            label_vector = label_vector_for_cwes(cwes)
            if sum(label_vector.values()) == 0:
                continue

            sample_id = str(vfp.get("vfp_id") or vfp_path.stem)
            sample_dir = SAMPLES_DIR / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            for filename, source_code in affected_files.items():
                if not isinstance(source_code, str):
                    continue
                target = unique_path(sample_dir, safe_filename(str(filename)))
                target.write_text(source_code, encoding="utf-8")

            label = {
                "vfp_id": sample_id,
                "label_vector": label_vector,
                "source": "forge_positive",
                "severity": sorted(severities),
                "cwe_list": sorted(cwes),
            }
            (sample_dir / "label.json").write_text(json.dumps(label, indent=2), encoding="utf-8")

            row = {"sample_id": sample_id, "source": "forge_positive", **label_vector}
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - build should continue after bad files.
            log_error(vfp_path, exc)

    return rows


def referenced_sol_basenames() -> set[str]:
    referenced: set[str] = set()
    finding_paths = sorted(FINDINGS_DIR.glob("*.json"))
    for processed, finding_path in enumerate(finding_paths, start=1):
        if processed % 100 == 0:
            print(f"Processed {processed} findings JSON files")
        try:
            report = load_json(finding_path)
            findings = report.get("findings")
            if not isinstance(findings, list):
                continue
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                for ref in sol_refs_from(finding.get("files")):
                    referenced.add(Path(ref).name)
        except Exception as exc:  # noqa: BLE001
            log_error(finding_path, exc)
    return referenced


def top_level_project(path: Path) -> str | None:
    try:
        relative = path.relative_to(CONTRACTS_DIR)
    except ValueError:
        return None
    return relative.parts[0] if relative.parts else None


def build_negative_samples() -> list[dict[str, Any]]:
    referenced = referenced_sol_basenames()
    grouped: dict[str, list[Path]] = defaultdict(list)

    sol_paths = sorted(path for path in CONTRACTS_DIR.rglob("*.sol") if path.is_file())
    for processed, sol_path in enumerate(sol_paths, start=1):
        if processed % 100 == 0:
            print(f"Scanned {processed} contract .sol files")
        if sol_path.name in referenced:
            continue
        project = top_level_project(sol_path)
        if project is None:
            continue
        grouped[project].append(sol_path)

    available_projects = [project for project, files in grouped.items() if files]
    if not available_projects:
        raise RuntimeError("No unreferenced Solidity files available for negative samples")

    rows: list[dict[str, Any]] = []
    zero_vector = {class_name: 0 for class_name in CLASSES}

    for index in range(NEGATIVE_TARGET):
        if (index + 1) % 100 == 0:
            print(f"Created {index + 1} negative samples")

        sample_id = f"neg_{index:05d}"
        project = random.choice(available_projects)
        files = grouped[project]
        count = random.randint(1, min(3, len(files)))
        selected = random.sample(files, count)

        sample_dir = SAMPLES_DIR / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for source_path in selected:
            relative_name = safe_filename(str(source_path.relative_to(CONTRACTS_DIR / project)))
            target = unique_path(sample_dir, relative_name)
            shutil.copy2(source_path, target)

        label = {
            "vfp_id": sample_id,
            "label_vector": zero_vector,
            "source": "forge_negative",
        }
        (sample_dir / "label.json").write_text(json.dumps(label, indent=2), encoding="utf-8")
        rows.append({"sample_id": sample_id, "source": "forge_negative", **zero_vector})

    return rows


def write_labels_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = ["sample_id", "source", *CLASSES]
    with LABELS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_iterative_split(rows: list[dict[str, Any]]) -> dict[str, list[str]] | None:
    try:
        from skmultilearn.model_selection import iterative_train_test_split
    except ImportError:
        print("scikit-multilearn not installed; running pip install scikit-multilearn")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-multilearn"])
        try:
            from skmultilearn.model_selection import iterative_train_test_split
        except Exception as exc:  # noqa: BLE001
            print(f"Could not import scikit-multilearn after install: {exc}")
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"Could not import scikit-multilearn: {exc}")
        return None

    ids = np.array([row["sample_id"] for row in rows], dtype=object).reshape(-1, 1)
    y = np.array([label_values(row) for row in rows], dtype=int)
    try:
        x_train, y_train, x_temp, y_temp = iterative_train_test_split(ids, y, test_size=0.30)
        x_val, _y_val, x_test, _y_test = iterative_train_test_split(x_temp, y_temp, test_size=0.50)
    except Exception as exc:  # noqa: BLE001
        print(f"Iterative split failed; using manual splitter: {exc}")
        return None

    return {
        "train": [str(item) for item in x_train.ravel()],
        "val": [str(item) for item in x_val.ravel()],
        "test": [str(item) for item in x_test.ravel()],
    }


def manual_split(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    target_sizes = {
        "train": round(len(rows) * 0.70),
        "val": round(len(rows) * 0.15),
    }
    target_sizes["test"] = len(rows) - target_sizes["train"] - target_sizes["val"]

    total_class_counts = Counter()
    for row in rows:
        for class_name in CLASSES:
            total_class_counts[class_name] += int(row[class_name])

    target_class_counts = {
        split: {
            class_name: total_class_counts[class_name] * target_sizes[split] / len(rows)
            for class_name in CLASSES
        }
        for split in target_sizes
    }

    shuffled = rows[:]
    random.shuffle(shuffled)
    shuffled.sort(key=lambda row: (sum(label_values(row)), row["source"] == "forge_negative"), reverse=True)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_class_counts: dict[str, Counter[str]] = {split: Counter() for split in splits}

    for row in shuffled:
        labels = {class_name for class_name in CLASSES if int(row[class_name]) == 1}
        candidates = [split for split in ("train", "val", "test") if len(splits[split]) < target_sizes[split]]
        if not candidates:
            candidates = ["train", "val", "test"]

        def score(split: str) -> tuple[float, float]:
            size_pressure = len(splits[split]) / target_sizes[split] if target_sizes[split] else 1.0
            if labels:
                deficits = [
                    target_class_counts[split][class_name] - split_class_counts[split][class_name]
                    for class_name in labels
                ]
                label_score = -sum(deficits)
            else:
                label_score = size_pressure
            return (label_score, size_pressure)

        chosen = min(candidates, key=score)
        splits[chosen].append(row)
        for class_name in labels:
            split_class_counts[chosen][class_name] += 1

    return {split: [row["sample_id"] for row in split_rows] for split, split_rows in splits.items()}


def build_splits(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    split = try_iterative_split(rows)
    if split is not None:
        return split
    return manual_split(rows)


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {class_name: sum(int(row[class_name]) for row in rows) for class_name in CLASSES}


def write_summary(rows: list[dict[str, Any]], splits: dict[str, list[str]]) -> None:
    rows_by_id = {row["sample_id"]: row for row in rows}
    positives = [row for row in rows if row["source"] == "forge_positive"]
    negatives = [row for row in rows if row["source"] == "forge_negative"]

    cardinalities = Counter(sum(label_values(row)) for row in rows)
    multi_label_count = sum(1 for row in rows if sum(label_values(row)) > 1)

    lines: list[str] = []
    lines.append(f"Total positive samples: {len(positives)}")
    lines.append(f"Total negative samples: {len(negatives)}")
    lines.append(f"Total samples: {len(rows)}")
    lines.append("")
    lines.append("Per-class positive count across full dataset:")
    for class_name, count in class_counts(rows).items():
        lines.append(f"{class_name}: {count}")
    lines.append("")
    lines.append("Per-class count in train / val / test splits:")
    for split_name in ("train", "val", "test"):
        split_rows = [rows_by_id[sample_id] for sample_id in splits[split_name]]
        lines.append(f"{split_name} samples: {len(split_rows)}")
        for class_name, count in class_counts(split_rows).items():
            lines.append(f"  {class_name}: {count}")
    lines.append("")
    lines.append(f"Number of multi-label samples: {multi_label_count}")
    lines.append("Distribution of label vector length:")
    lines.append(f"0 classes: {cardinalities[0]}")
    lines.append(f"1 class: {cardinalities[1]}")
    lines.append(f"2 classes: {cardinalities[2]}")
    lines.append(f"3+ classes: {sum(count for length, count in cardinalities.items() if length >= 3)}")

    summary = "\n".join(lines) + "\n"
    print(summary, end="")
    SUMMARY_TXT.write_text(summary, encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    reset_output_dir()

    print("Building positive samples")
    positive_rows = build_positive_samples()

    print("Building negative samples")
    negative_rows = build_negative_samples()

    rows = positive_rows + negative_rows
    write_labels_csv(rows)

    print("Building train/val/test splits")
    splits = build_splits(rows)
    SPLITS_JSON.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    write_summary(rows, splits)
    print(f"Wrote dataset to {OUT_DIR}")


if __name__ == "__main__":
    main()

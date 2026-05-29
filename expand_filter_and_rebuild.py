#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.iterative_stratification import IterativeStratification

import build_hypergraph as hg_builder
from extract_ast_features import setup_parsers


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "hypervul_dataset"
SAMPLES_DIR = DATASET_DIR / "samples"
LABELS_CSV = DATASET_DIR / "labels.csv"
HYPERGRAPHS_OLD_DIR = DATASET_DIR / "hypergraphs"
HYPERGRAPHS_V2_DIR = DATASET_DIR / "hypergraphs_v2"
CV_FOLDS_JSON = DATASET_DIR / "cv_folds.json"
MANIFEST_JSON = DATASET_DIR / "dataset_manifest.json"
OUTPUT_TXT = DATASET_DIR / "expand_filter_output.txt"
ERROR_LOG = DATASET_DIR / "build_errors_hg_v2.log"
SEED = 42

CLASSES = [
    "reentrancy",
    "arithmetic_oracle",
    "defi_protection",
    "access_control",
    "state_management",
    "unchecked_call",
    "input_validation",
]
EDGE_TYPES = ["CALL", "DELEGATECALL", "STATICCALL", "TRANSFER", "SEND", "INTERFACE_CALL"]

ADDITIONAL_EXTERNAL_PATTERNS = [
    r"\{value\s*:",
    r"\{value:",
    r"\.rawCall\s*\(",
    r"\.externalCall\s*\(",
    r"EfficientCall\.",
    r"SystemContractsCaller\.",
    r"\.staticcall\s*\(",
    r"\.delegatecall\s*\(",
    r"\.safeTransfer\s*\(",
    r"\.safeTransferFrom\s*\(",
    r"\.safeApprove\s*\(",
    r"IERC20\s*\(",
    r"IERC721\s*\(",
    r"IERC1155\s*\(",
    r"payable\s*\(.*\)\s*\.",
    r"assembly\s*\{",
]


def is_meaningful_external_call(call: dict, state_vars: list) -> bool:
    if call["call_type"] in {"CALL", "DELEGATECALL", "STATICCALL", "TRANSFER", "SEND"}:
        return True
    if call["call_type"] == "INTERFACE_CALL":
        state_var_names = {sv["name"] for sv in state_vars}
        raw = call.get("raw_text", "")
        for name in state_var_names:
            if re.search(rf"\b{re.escape(name)}\s*\.", raw):
                return True
        for pattern in ADDITIONAL_EXTERNAL_PATTERNS:
            if re.search(pattern, raw):
                return True
    return False


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def load_labels() -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with LABELS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels[row["sample_id"]] = {
                "sample_id": row["sample_id"],
                "source": row["source"],
                **{class_name: int(row[class_name]) for class_name in CLASSES},
            }
    return labels


def label_vector(row: dict[str, Any]) -> list[int]:
    return [int(row[class_name]) for class_name in CLASSES]


def build_v2_hypergraphs(sample_ids: list[str]) -> list[dict[str, Any]]:
    hg_builder.is_meaningful_external_call = is_meaningful_external_call
    if HYPERGRAPHS_V2_DIR.exists():
        shutil.rmtree(HYPERGRAPHS_V2_DIR)
    HYPERGRAPHS_V2_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.write_text("", encoding="utf-8")

    results: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        if index % 100 == 0:
            print(f"Processed {index} samples")
        try:
            hypergraph = hg_builder.build_hypergraph(str(SAMPLES_DIR / sample_id))
            (HYPERGRAPHS_V2_DIR / f"{sample_id}.json").write_text(
                json.dumps(hypergraph, indent=2),
                encoding="utf-8",
            )
            results.append(hypergraph)
        except Exception as exc:  # noqa: BLE001
            with ERROR_LOG.open("a", encoding="utf-8") as handle:
                handle.write(f"{sample_id}: {type(exc).__name__}: {exc}\n")
    return results


def old_usable_ids() -> set[str]:
    usable = set()
    for path in HYPERGRAPHS_OLD_DIR.glob("*.json"):
        hg = load_json(path)
        if int(hg.get("stats", {}).get("n_hyperedges", 0)) >= 1:
            usable.add(str(hg.get("sample_id") or path.stem))
    return usable


def make_cv_test_folds(ids: list[str], labels: dict[str, dict[str, Any]]) -> list[list[str]]:
    x = np.array(ids)
    y = np.array([label_vector(labels[sample_id]) for sample_id in ids], dtype=int)
    splitter = IterativeStratification(n_splits=5, order=1, sample_distribution_per_fold=[0.2] * 5)
    folds: list[list[str]] = []
    for _train_idx, test_idx in splitter.split(x, y):
        folds.append([str(x[index]) for index in test_idx])
    return folds


def stratified_val_split(trainval_ids: list[str], labels: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    x = np.array(trainval_ids, dtype=object).reshape(-1, 1)
    y = np.array([label_vector(labels[sample_id]) for sample_id in trainval_ids], dtype=int)
    try:
        x_train, _y_train, x_val, _y_val = iterative_train_test_split(x, y, test_size=0.125)
        return [str(item) for item in x_train.ravel()], [str(item) for item in x_val.ravel()]
    except Exception:
        shuffled = trainval_ids[:]
        random.shuffle(shuffled)
        val_n = round(len(shuffled) * 0.125)
        return shuffled[val_n:], shuffled[:val_n]


def build_cv_folds(usable_ids: list[str], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    test_folds = make_cv_test_folds(usable_ids, labels)
    all_ids = set(usable_ids)
    folds = []
    for test_ids in test_folds:
        test_set = set(test_ids)
        trainval_ids = sorted(all_ids - test_set)
        train_ids, val_ids = stratified_val_split(trainval_ids, labels)
        folds.append({"train": train_ids, "val": val_ids, "test": test_ids})
    return {"n_folds": 5, "folds": folds}


def class_counts(ids: list[str], labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        class_name: sum(int(labels[sample_id][class_name]) for sample_id in ids)
        for class_name in CLASSES
    }


def manifest_stats(hypergraphs: list[dict[str, Any]], labels: dict[str, dict[str, Any]], cv: dict[str, Any]) -> dict[str, Any]:
    usable = [hg for hg in hypergraphs if int(hg.get("stats", {}).get("n_hyperedges", 0)) >= 1]
    usable_ids = [str(hg["sample_id"]) for hg in usable]
    tau_counts = Counter()
    edge_sizes = []
    for hg in usable:
        for edge in hg.get("hyperedges", []):
            tau_counts[str(edge.get("tau", ""))] += 1
            edge_sizes.append(len(edge.get("node_ids", [])))

    positive = [sample_id for sample_id in usable_ids if labels[sample_id]["source"] == "forge_positive"]
    negative = [sample_id for sample_id in usable_ids if labels[sample_id]["source"] == "forge_negative"]

    return {
        "canonical_hypergraphs_dir": "hypergraphs_v2",
        "canonical_cv_folds": "cv_folds.json",
        "total_samples": len(usable_ids),
        "total_positive": len(positive),
        "total_negative": len(negative),
        "cv_folds": cv,
        "class_counts": {"usable": class_counts(usable_ids, labels)},
        "hyperedge_type_dist": {edge_type: tau_counts[edge_type] for edge_type in EDGE_TYPES},
        "avg_nodes_per_sample": sum(hg["stats"]["n_nodes"] for hg in usable) / len(usable) if usable else 0.0,
        "avg_hyperedges_per_sample": sum(hg["stats"]["n_hyperedges"] for hg in usable) / len(usable) if usable else 0.0,
        "avg_hyperedge_size": sum(edge_sizes) / len(edge_sizes) if edge_sizes else 0.0,
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    setup_parsers()

    labels = load_labels()
    sample_ids = sorted(labels)
    previous_usable = old_usable_ids()

    hypergraphs = build_v2_hypergraphs(sample_ids)
    usable_hypergraphs = [hg for hg in hypergraphs if int(hg.get("stats", {}).get("n_hyperedges", 0)) >= 1]
    usable_ids = sorted(str(hg["sample_id"]) for hg in usable_hypergraphs)
    recovered = sorted(set(usable_ids) - previous_usable)

    cv = build_cv_folds(usable_ids, labels)
    CV_FOLDS_JSON.write_text(json.dumps(cv, indent=2), encoding="utf-8")

    manifest = manifest_stats(hypergraphs, labels, cv)
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    usable_positive = [sample_id for sample_id in usable_ids if labels[sample_id]["source"] == "forge_positive"]
    usable_negative = [sample_id for sample_id in usable_ids if labels[sample_id]["source"] == "forge_negative"]
    usable_class_counts = class_counts(usable_positive, labels)

    train_counts = [len(fold["train"]) for fold in cv["folds"]]
    val_counts = [len(fold["val"]) for fold in cv["folds"]]
    test_counts = [len(fold["test"]) for fold in cv["folds"]]
    test_class_counts = [class_counts(fold["test"], labels) for fold in cv["folds"]]

    lines: list[str] = []
    lines.append("Part B - Expanded Filter Rebuild")
    lines.append(f"Previous usable: {len(previous_usable)}")
    lines.append(f"New usable: {len(usable_ids)}")
    lines.append(f"Recovered samples: {len(recovered)}")
    lines.append(f"New usable positive: {len(usable_positive)}")
    lines.append(f"New usable negative: {len(usable_negative)}")
    lines.append("")
    lines.append("Part D - Expanded Dataset Summary")
    lines.append("New per-class usable positive counts:")
    for class_name in CLASSES:
        lines.append(f"  {class_name}: {usable_class_counts[class_name]}")
    lines.append("")
    lines.append("5-fold CV stats (averaged across folds):")
    lines.append(f"  Avg train samples per fold: {sum(train_counts) / len(train_counts):.1f}")
    lines.append(f"  Avg val samples per fold:   {sum(val_counts) / len(val_counts):.1f}")
    lines.append(f"  Avg test samples per fold:  {sum(test_counts) / len(test_counts):.1f}")
    for class_name in ("reentrancy", "defi_protection", "access_control"):
        avg = sum(counts[class_name] for counts in test_class_counts) / len(test_class_counts)
        lines.append(f"  Avg {class_name} test per fold: {avg:.1f}")
    lines.append("")
    lines.append("Canonical dataset updated:")
    lines.append(f"  hypergraphs: {HYPERGRAPHS_V2_DIR}")
    lines.append(f"  cv_folds: {CV_FOLDS_JSON}")
    lines.append(f"  manifest: {MANIFEST_JSON}")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    OUTPUT_TXT.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

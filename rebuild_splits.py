#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "hypervul_dataset"
HYPERGRAPHS_DIR = DATASET_DIR / "hypergraphs"
LABELS_CSV = DATASET_DIR / "labels.csv"
SPLITS_USABLE_JSON = DATASET_DIR / "splits_usable.json"
MANIFEST_JSON = DATASET_DIR / "dataset_manifest.json"
DROPPED_TXT = DATASET_DIR / "dropped_samples.txt"
OUTPUT_TXT = DATASET_DIR / "rebuild_splits_output.txt"
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


def label_classes(row: dict[str, Any]) -> list[str]:
    return [class_name for class_name in CLASSES if int(row[class_name]) == 1]


def split_random(ids: list[str]) -> dict[str, list[str]]:
    shuffled = ids[:]
    random.shuffle(shuffled)
    train_n = round(len(shuffled) * 0.70)
    val_n = round(len(shuffled) * 0.15)
    return {
        "train": shuffled[:train_n],
        "val": shuffled[train_n : train_n + val_n],
        "test": shuffled[train_n + val_n :],
    }


def split_positive_iterative(ids: list[str], labels: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    x = np.array(ids, dtype=object).reshape(-1, 1)
    y = np.array([label_vector(labels[sample_id]) for sample_id in ids], dtype=int)
    x_train, y_train, x_temp, y_temp = iterative_train_test_split(x, y, test_size=0.30)
    x_val, _y_val, x_test, _y_test = iterative_train_test_split(x_temp, y_temp, test_size=0.50)
    return {
        "train": [str(item) for item in x_train.ravel()],
        "val": [str(item) for item in x_val.ravel()],
        "test": [str(item) for item in x_test.ravel()],
    }


def split_counts(ids: list[str], labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    rows = [labels[sample_id] for sample_id in ids]
    return {class_name: sum(int(row[class_name]) for row in rows) for class_name in CLASSES}


def split_summary_row(split: str, ids: list[str], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [labels[sample_id] for sample_id in ids]
    positives = sum(1 for row in rows if row["source"] == "forge_positive")
    negatives = sum(1 for row in rows if row["source"] == "forge_negative")
    return {
        "split": split,
        "total": len(ids),
        "positive": positives,
        "negative": negatives,
        **split_counts(ids, labels),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    labels = load_labels()

    hypergraphs: dict[str, dict[str, Any]] = {}
    usable_ids: list[str] = []
    dropped_ids: list[str] = []
    for hg_path in sorted(HYPERGRAPHS_DIR.glob("*.json")):
        hypergraph = load_json(hg_path)
        sample_id = str(hypergraph.get("sample_id") or hg_path.stem)
        hypergraphs[sample_id] = hypergraph
        n_hyperedges = int(hypergraph.get("stats", {}).get("n_hyperedges", 0))
        if n_hyperedges >= 1:
            usable_ids.append(sample_id)
        else:
            dropped_ids.append(sample_id)

    usable_positive = [sample_id for sample_id in usable_ids if labels[sample_id]["source"] == "forge_positive"]
    usable_negative = [sample_id for sample_id in usable_ids if labels[sample_id]["source"] == "forge_negative"]
    dropped_positive = [sample_id for sample_id in dropped_ids if labels[sample_id]["source"] == "forge_positive"]
    dropped_negative = [sample_id for sample_id in dropped_ids if labels[sample_id]["source"] == "forge_negative"]

    pos_splits = split_positive_iterative(usable_positive, labels)
    neg_splits = split_random(usable_negative)
    splits = {
        split: pos_splits[split] + neg_splits[split]
        for split in ("train", "val", "test")
    }
    for split in splits:
        random.shuffle(splits[split])

    SPLITS_USABLE_JSON.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    class_counts = {split: split_counts(ids, labels) for split, ids in splits.items()}
    tau_counts = Counter()
    edge_sizes: list[int] = []
    for sample_id in usable_ids:
        for edge in hypergraphs[sample_id].get("hyperedges", []):
            tau_counts[str(edge.get("tau", ""))] += 1
            edge_sizes.append(len(edge.get("node_ids", [])))

    avg_nodes = sum(hypergraphs[sample_id]["stats"]["n_nodes"] for sample_id in usable_ids) / len(usable_ids)
    avg_hyperedges = (
        sum(hypergraphs[sample_id]["stats"]["n_hyperedges"] for sample_id in usable_ids) / len(usable_ids)
    )
    avg_edge_size = sum(edge_sizes) / len(edge_sizes) if edge_sizes else 0.0

    manifest = {
        "total_samples": len(usable_ids),
        "total_positive": len(usable_positive),
        "total_negative": len(usable_negative),
        "splits": splits,
        "class_counts": class_counts,
        "hyperedge_type_dist": {edge_type: tau_counts[edge_type] for edge_type in EDGE_TYPES},
        "avg_nodes_per_sample": avg_nodes,
        "avg_hyperedges_per_sample": avg_hyperedges,
        "avg_hyperedge_size": avg_edge_size,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dropped_lines = []
    for sample_id in dropped_ids:
        row = labels[sample_id]
        classes = ",".join(label_classes(row)) if label_classes(row) else "none"
        dropped_lines.append(f"{sample_id} | {row['source']} | {classes}")
    DROPPED_TXT.write_text("\n".join(dropped_lines) + "\n", encoding="utf-8")

    dropped_class_counts = Counter()
    for sample_id in dropped_positive:
        dropped_class_counts.update(label_classes(labels[sample_id]))

    usable_class_counts = split_counts(usable_ids, labels)
    rows = [split_summary_row(split, splits[split], labels) for split in ("train", "val", "test")]

    lines: list[str] = []
    lines.append("Part A - Usable Sample Report")
    lines.append(f"Total usable: {len(usable_ids)}")
    lines.append(f"Usable positive: {len(usable_positive)}")
    lines.append(f"Usable negative: {len(usable_negative)}")
    lines.append("Per-class positive count in usable set:")
    for class_name in CLASSES:
        lines.append(f"  {class_name}: {usable_class_counts[class_name]}")
    lines.append("")

    lines.append("Part C - New Usable Split Summary")
    header = (
        f"{'Split':<8} | {'Total':>5} | {'Positive':>8} | {'Negative':>8} | "
        f"{'reentrancy':>10} | {'arithmetic_oracle':>17} | {'defi_protection':>15} | "
        f"{'access_control':>14} | {'state_management':>16} | {'unchecked_call':>14} | {'input_validation':>16}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row['split']:<8} | {row['total']:>5} | {row['positive']:>8} | {row['negative']:>8} | "
            f"{row['reentrancy']:>10} | {row['arithmetic_oracle']:>17} | {row['defi_protection']:>15} | "
            f"{row['access_control']:>14} | {row['state_management']:>16} | {row['unchecked_call']:>14} | "
            f"{row['input_validation']:>16}"
        )
    lines.append("")

    lines.append("Part D - Manifest Stats")
    lines.append(f"Manifest saved: {MANIFEST_JSON}")
    lines.append(f"Hyperedge type distribution: {manifest['hyperedge_type_dist']}")
    lines.append(f"Average nodes per usable sample: {avg_nodes:.2f}")
    lines.append(f"Average hyperedges per usable sample: {avg_hyperedges:.2f}")
    lines.append(f"Average hyperedge size: {avg_edge_size:.2f}")
    lines.append("")

    lines.append("Part E - Dropped Samples")
    lines.append(f"Dropped samples: {len(dropped_ids)}")
    lines.append(f"Dropped positive: {len(dropped_positive)}")
    lines.append(f"Dropped negative: {len(dropped_negative)}")
    lines.append("Positive classes lost the most samples:")
    for class_name, count in dropped_class_counts.most_common():
        lines.append(f"  {class_name}: {count}")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    OUTPUT_TXT.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

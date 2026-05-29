#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_hypergraph import is_meaningful_external_call, is_test_file
from extract_ast_features import (
    extract_call_graph,
    extract_external_calls,
    extract_functions,
    extract_state_var_access,
    extract_state_variables,
    setup_parsers,
)


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "hypervul_dataset"
SAMPLES_DIR = DATASET_DIR / "samples"
HYPERGRAPHS_DIR = DATASET_DIR / "hypergraphs"
OUTPUT_FILE = DATASET_DIR / "diagnosis_output.txt"
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def load_dataset() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for sample_dir in sorted(path for path in SAMPLES_DIR.iterdir() if path.is_dir()):
        sample_id = sample_dir.name
        label = load_json(sample_dir / "label.json")
        hg_path = HYPERGRAPHS_DIR / f"{sample_id}.json"
        hypergraph = load_json(hg_path) if hg_path.exists() else {}
        n_hyperedges = int(hypergraph.get("stats", {}).get("n_hyperedges", 0))
        data[sample_id] = {
            "sample_id": sample_id,
            "label": label,
            "hypergraph": hypergraph,
            "usable": n_hyperedges > 0,
        }
    return data


def pct(numerator: int, denominator: int) -> str:
    return f"{(numerator / denominator * 100):.1f}%" if denominator else "0.0%"


def label_classes(label: dict[str, Any]) -> list[str]:
    vector = label.get("label_vector", {})
    if not isinstance(vector, dict):
        return []
    return [class_name for class_name in CLASSES if int(vector.get(class_name, 0)) == 1]


def extract_sample_raw(sample_id: str) -> dict[str, Any]:
    state_vars: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    external_calls: list[dict[str, Any]] = []
    call_graph: list[dict[str, Any]] = []
    state_access: list[dict[str, Any]] = []
    sol_files: list[str] = []

    for sol_file in sorted((SAMPLES_DIR / sample_id).glob("*.sol")):
        source = sol_file.read_text(encoding="utf-8", errors="replace")
        if is_test_file(str(sol_file), source):
            continue
        sol_files.append(sol_file.name)
        file_state_vars = extract_state_variables(source)
        file_functions = extract_functions(source)
        file_external_calls = extract_external_calls(source)
        file_call_graph = extract_call_graph(source)
        file_state_access = extract_state_var_access(source, file_state_vars, file_functions)

        for item in file_state_vars:
            enriched = dict(item)
            enriched["file"] = sol_file.name
            state_vars.append(enriched)
        for item in file_functions:
            enriched = dict(item)
            enriched["file"] = sol_file.name
            functions.append(enriched)
        for item in file_external_calls:
            enriched = dict(item)
            enriched["file"] = sol_file.name
            external_calls.append(enriched)
        call_graph.extend(file_call_graph)
        state_access.extend(file_state_access)

    meaningful = [call for call in external_calls if is_meaningful_external_call(call, state_vars)]
    return {
        "sol_files": sol_files,
        "state_vars": state_vars,
        "functions": functions,
        "external_calls": external_calls,
        "meaningful_external_calls": meaningful,
        "call_graph": call_graph,
        "state_access": state_access,
    }


def interface_call_examples(records: dict[str, dict[str, Any]], count: int = 10) -> list[dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for sample_id, record in records.items():
        hypergraph = record["hypergraph"]
        nodes = {node["node_id"]: node for node in hypergraph.get("nodes", [])}
        for edge in hypergraph.get("hyperedges", []):
            if edge.get("tau") == "INTERFACE_CALL":
                anchor = nodes.get(edge.get("anchor_call_site_id"), {})
                candidates.append((sample_id, edge, anchor))

    random.shuffle(candidates)
    examples: list[dict[str, Any]] = []
    for sample_id, edge, anchor in candidates[:count]:
        nodes = {
            node["node_id"]: node
            for node in records[sample_id]["hypergraph"].get("nodes", [])
        }
        state_names = [
            nodes[node_id].get("name", "")
            for node_id in edge.get("node_ids", [])
            if node_id in nodes and nodes[node_id].get("node_type") == "state_var"
        ]
        examples.append(
            {
                "sample_id": sample_id,
                "raw_text": anchor.get("raw_text", ""),
                "enclosing_function": anchor.get("enclosing_function", ""),
                "state_var_names": state_names,
            }
        )
    return examples


def main() -> None:
    random.seed(SEED)
    setup_parsers()
    records = load_dataset()

    lines: list[str] = []
    lines.append("Part A - Empty vs Usable by Sample Source")
    lines.append(f"{'Source':<18} | {'Total':>5} | {'Usable':>6} | {'Empty':>5} | {'Usable%':>8}")
    lines.append(f"{'-' * 18}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 8}")
    for source in ("forge_positive", "forge_negative"):
        subset = [record for record in records.values() if record["label"].get("source") == source]
        usable = sum(1 for record in subset if record["usable"])
        total = len(subset)
        lines.append(f"{source:<18} | {total:>5} | {usable:>6} | {total - usable:>5} | {pct(usable, total):>8}")
    lines.append("")

    lines.append("Part B - Empty vs Usable by Vulnerability Class")
    lines.append(
        f"{'Class':<18} | {'Positive Total':>14} | {'Usable':>6} | {'Empty':>5} | {'Usable%':>8}"
    )
    lines.append(f"{'-' * 18}-+-{'-' * 14}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 8}")
    positives = [record for record in records.values() if record["label"].get("source") == "forge_positive"]
    for class_name in CLASSES:
        subset = [
            record
            for record in positives
            if int(record["label"].get("label_vector", {}).get(class_name, 0)) == 1
        ]
        usable = sum(1 for record in subset if record["usable"])
        total = len(subset)
        lines.append(f"{class_name:<18} | {total:>14} | {usable:>6} | {total - usable:>5} | {pct(usable, total):>8}")
    lines.append("")

    lines.append("Part C - Empty Positive Sample Investigation")
    empty_positives = [record for record in positives if not record["usable"]]
    sampled_empty = random.sample(empty_positives, min(5, len(empty_positives)))
    for record in sampled_empty:
        sample_id = record["sample_id"]
        raw = extract_sample_raw(sample_id)
        lines.append(f"Sample: {sample_id}")
        lines.append(f"  label_classes: {label_classes(record['label'])}")
        lines.append(f"  sol_files: {raw['sol_files']}")
        lines.append(f"  external_calls_raw_count: {len(raw['external_calls'])}")
        lines.append(f"  meaningful_external_calls_before_filter: {len(raw['external_calls'])}")
        lines.append(f"  meaningful_external_calls_after_filter: {len(raw['meaningful_external_calls'])}")
        lines.append(f"  state_variables_count: {len(raw['state_vars'])}")
        lines.append(f"  functions_count: {len(raw['functions'])}")
        lines.append("  raw_external_calls:")
        for call in raw["external_calls"][:25]:
            lines.append(
                "    "
                f"{call.get('file', '')}:{call.get('line', '')} "
                f"{call.get('call_type', '')} "
                f"{call.get('enclosing_function', '')} :: {call.get('raw_text', '')}"
            )
        if len(raw["external_calls"]) > 25:
            lines.append(f"    ... {len(raw['external_calls']) - 25} more")
    lines.append("")

    lines.append("Part D - INTERFACE_CALL Filter Quality")
    for index, example in enumerate(interface_call_examples(records), start=1):
        lines.append(f"Example {index}: {example['sample_id']}")
        lines.append(f"  enclosing_function: {example['enclosing_function']}")
        lines.append(f"  state_var_names_in_hyperedge: {example['state_var_names']}")
        lines.append(f"  raw_text: {example['raw_text']}")
    lines.append("")

    lines.append("Part E - Hyperedge Size Distribution")
    edge_sizes = [
        len(edge.get("node_ids", []))
        for record in records.values()
        if record["usable"]
        for edge in record["hypergraph"].get("hyperedges", [])
    ]
    if edge_sizes:
        lines.append(f"Min hyperedge size: {min(edge_sizes)}")
        lines.append(f"Max hyperedge size: {max(edge_sizes)}")
        lines.append(f"Mean hyperedge size: {statistics.mean(edge_sizes):.2f}")
        lines.append(f"Median hyperedge size: {statistics.median(edge_sizes):.2f}")
        lines.append(f"Size 1 degenerate count: {sum(1 for size in edge_sizes if size == 1)}")
        lines.append(f"Size 2-4 small count: {sum(1 for size in edge_sizes if 2 <= size <= 4)}")
        lines.append(f"Size 5-9 medium count: {sum(1 for size in edge_sizes if 5 <= size <= 9)}")
        lines.append(f"Size 10+ large count: {sum(1 for size in edge_sizes if size >= 10)}")
    else:
        lines.append("No hyperedges found.")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    OUTPUT_FILE.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

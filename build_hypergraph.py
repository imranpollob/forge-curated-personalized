#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

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
SPLITS_JSON = DATASET_DIR / "splits.json"
HYPERGRAPHS_DIR = DATASET_DIR / "hypergraphs"
EMPTY_LOG = DATASET_DIR / "empty_hypergraphs.log"
ERROR_LOG = DATASET_DIR / "build_errors_hg.log"
SUMMARY_TXT = DATASET_DIR / "hypergraph_summary.txt"
SEED = 42
EXTERNAL_CALL_TYPES = {"CALL", "DELEGATECALL", "STATICCALL", "TRANSFER", "SEND"}


def is_meaningful_external_call(call: dict, state_vars: list) -> bool:
    """
    Returns True if this call should anchor a hyperedge.
    Rules:
    - CALL, DELEGATECALL, STATICCALL, TRANSFER, SEND: always True
    - INTERFACE_CALL: True only if the call receiver identifier matches
      a state variable name (meaning we call an external contract stored as a field)
    """
    if call["call_type"] in {"CALL", "DELEGATECALL", "STATICCALL", "TRANSFER", "SEND"}:
        return True
    if call["call_type"] == "INTERFACE_CALL":
        state_var_names = {sv["name"] for sv in state_vars}
        raw = call.get("raw_text", "")
        for name in state_var_names:
            if re.search(rf"\b{re.escape(name)}\s*\.", raw):
                return True
    return False


def is_test_file(filepath: str, source: str) -> bool:
    """Skip Foundry/Hardhat test files - they inflate call counts."""
    if filepath.endswith(".t.sol"):
        return True
    if "forge-std" in source or 'import "hardhat' in source:
        return True
    if re.search(r"contract\s+\w*Test\b", source) and "function test" in source:
        return True
    return False


def get_ancestors(function_name: str, call_graph: list, depth: int) -> set:
    """
    Return all functions that transitively call function_name,
    up to given depth. BFS over reversed call graph edges.
    """
    reversed_edges: dict[str, set[str]] = defaultdict(set)
    for edge in call_graph:
        caller = edge.get("caller_function")
        callee = edge.get("callee_function")
        if caller and callee:
            reversed_edges[str(callee)].add(str(caller))

    ancestors: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(function_name, 0)])
    while queue:
        current, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        for caller in reversed_edges.get(current, set()):
            if caller in ancestors:
                continue
            ancestors.add(caller)
            queue.append((caller, current_depth + 1))
    return ancestors


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def log_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def unique_name_key(item: dict[str, Any]) -> str:
    return str(item.get("name", ""))


def merge_extractions(sample_dir: Path) -> dict[str, list[dict[str, Any]]]:
    state_vars: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    external_calls: list[dict[str, Any]] = []
    call_graph: list[dict[str, Any]] = []
    state_access: list[dict[str, Any]] = []

    for sol_file in sorted(sample_dir.glob("*.sol")):
        source = sol_file.read_text(encoding="utf-8", errors="replace")
        if is_test_file(str(sol_file), source):
            continue

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
        for item in file_call_graph:
            enriched = dict(item)
            enriched["file"] = sol_file.name
            call_graph.append(enriched)
        for item in file_state_access:
            enriched = dict(item)
            enriched["file"] = sol_file.name
            state_access.append(enriched)

    return {
        "state_vars": dedupe_by(state_vars, ("name", "file")),
        "functions": dedupe_by(functions, ("name", "file", "start_line")),
        "external_calls": dedupe_by(external_calls, ("call_type", "line", "raw_text", "file")),
        "call_graph": dedupe_by(call_graph, ("caller_function", "callee_function", "file")),
        "state_access": dedupe_by(state_access, ("function_name", "var_name", "access_type", "file")),
    }


def dedupe_by(items: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = tuple(item.get(field) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def build_hypergraph(sample_dir: str, delta: int = 3) -> dict:
    """
    Algorithm 1: Transaction-Centric Hyperedge Construction
    """
    sample_path = Path(sample_dir)
    sample_id = sample_path.name
    label = load_json(sample_path / "label.json") if (sample_path / "label.json").exists() else {}
    extracted = merge_extractions(sample_path)

    state_vars = extracted["state_vars"]
    functions = extracted["functions"]
    external_calls = extracted["external_calls"]
    call_graph = extracted["call_graph"]
    state_access = extracted["state_access"]

    meaningful_calls = [
        call for call in external_calls if is_meaningful_external_call(call, state_vars)
    ]

    nodes: list[dict[str, Any]] = []
    function_node_by_name: dict[str, int] = {}
    state_node_by_name: dict[str, int] = {}

    for function in functions:
        node_id = len(nodes)
        node = {
            "node_id": node_id,
            "node_type": "function",
            "name": function.get("name", ""),
            "file": function.get("file", ""),
            "start_line": function.get("start_line"),
            "end_line": function.get("end_line"),
            "visibility": function.get("visibility", ""),
            "is_external": bool(function.get("is_external", False)),
            "is_payable": bool(function.get("is_payable", False)),
        }
        nodes.append(node)
        function_node_by_name.setdefault(str(node["name"]), node_id)

    for state_var in state_vars:
        node_id = len(nodes)
        node = {
            "node_id": node_id,
            "node_type": "state_var",
            "name": state_var.get("name", ""),
            "var_type": state_var.get("type", ""),
            "file": state_var.get("file", ""),
            "line": state_var.get("line"),
        }
        nodes.append(node)
        state_node_by_name.setdefault(str(node["name"]), node_id)

    call_site_nodes: list[dict[str, Any]] = []
    for call in meaningful_calls:
        node_id = len(nodes)
        node = {
            "node_id": node_id,
            "node_type": "call_site",
            "name": f"{call.get('call_type', '')}:{call.get('file', '')}:{call.get('line', '')}",
            "call_type": call.get("call_type", ""),
            "line": call.get("line"),
            "enclosing_function": call.get("enclosing_function", ""),
            "file": call.get("file", ""),
            "raw_text": call.get("raw_text", ""),
        }
        nodes.append(node)
        call_site_nodes.append(node)

    access_by_function: dict[str, set[str]] = defaultdict(set)
    for access in state_access:
        function_name = access.get("function_name")
        var_name = access.get("var_name")
        if function_name and var_name:
            access_by_function[str(function_name)].add(str(var_name))

    hyperedges: list[dict[str, Any]] = []
    for call_node in call_site_nodes:
        enclosing = str(call_node.get("enclosing_function", ""))
        if not enclosing:
            continue
        function_names = {enclosing}
        function_names.update(get_ancestors(enclosing, call_graph, delta))

        node_ids = {int(call_node["node_id"])}
        for function_name in function_names:
            node_id = function_node_by_name.get(function_name)
            if node_id is not None:
                node_ids.add(node_id)

        state_names: set[str] = set()
        for function_name in function_names:
            state_names.update(access_by_function.get(function_name, set()))
        for state_name in state_names:
            node_id = state_node_by_name.get(state_name)
            if node_id is not None:
                node_ids.add(node_id)

        if len(node_ids) < 2:
            continue

        hyperedges.append(
            {
                "edge_id": len(hyperedges),
                "node_ids": sorted(node_ids),
                "tau": str(call_node.get("call_type", "")),
                "anchor_call_site_id": int(call_node["node_id"]),
            }
        )

    avg_hyperedge_size = (
        sum(len(edge["node_ids"]) for edge in hyperedges) / len(hyperedges)
        if hyperedges
        else 0.0
    )
    node_features_raw = [
        {
            "node_id": node["node_id"],
            "node_type": node["node_type"],
            "name": node.get("name", ""),
            "file": node.get("file", ""),
        }
        for node in nodes
    ]

    return {
        "sample_id": sample_id,
        "nodes": nodes,
        "hyperedges": hyperedges,
        "node_features_raw": node_features_raw,
        "stats": {
            "n_nodes": len(nodes),
            "n_hyperedges": len(hyperedges),
            "n_functions": len(functions),
            "n_state_vars": len(state_vars),
            "n_call_sites": len(call_site_nodes),
            "avg_hyperedge_size": avg_hyperedge_size,
        },
        "label": label,
    }


def split_sample_ids() -> tuple[list[str], dict[str, list[str]]]:
    splits = load_json(SPLITS_JSON)
    split_map = {
        split: [str(sample_id) for sample_id in splits.get(split, [])]
        for split in ("train", "val", "test")
    }
    ordered = split_map["train"] + split_map["val"] + split_map["test"]
    seen = set()
    unique_ordered = []
    for sample_id in ordered:
        if sample_id in seen:
            continue
        seen.add(sample_id)
        unique_ordered.append(sample_id)
    return unique_ordered, split_map


def write_summary(results: list[dict[str, Any]], split_map: dict[str, list[str]]) -> str:
    total = len(results)
    usable = [hg for hg in results if hg["stats"]["n_hyperedges"] > 0]
    empty = total - len(usable)
    avg_nodes = sum(hg["stats"]["n_nodes"] for hg in results) / total if total else 0.0
    avg_edges = sum(hg["stats"]["n_hyperedges"] for hg in results) / total if total else 0.0
    all_edge_sizes = [
        len(edge["node_ids"])
        for hg in results
        for edge in hg["hyperedges"]
    ]
    avg_edge_size = sum(all_edge_sizes) / len(all_edge_sizes) if all_edge_sizes else 0.0
    tau_counts = Counter(edge["tau"] for hg in results for edge in hg["hyperedges"])
    hg_by_id = {hg["sample_id"]: hg for hg in results}

    lines = []
    lines.append(f"Total samples processed: {total}")
    lines.append(f"Samples with at least 1 hyperedge (usable): {len(usable)}")
    lines.append(f"Samples with 0 hyperedges (empty): {empty}")
    lines.append(f"Average nodes per sample: {avg_nodes:.2f}")
    lines.append(f"Average hyperedges per sample: {avg_edges:.2f}")
    lines.append(f"Average hyperedge size: {avg_edge_size:.2f}")
    lines.append("Hyperedge type distribution:")
    for tau in ["CALL", "DELEGATECALL", "STATICCALL", "TRANSFER", "SEND", "INTERFACE_CALL"]:
        lines.append(f"  {tau}: {tau_counts[tau]}")
    lines.append("Per-split usable sample count:")
    for split in ("train", "val", "test"):
        usable_count = sum(
            1
            for sample_id in split_map[split]
            if sample_id in hg_by_id and hg_by_id[sample_id]["stats"]["n_hyperedges"] > 0
        )
        lines.append(f"  {split}: {usable_count}/{len(split_map[split])}")
    return "\n".join(lines) + "\n"


def main() -> None:
    random.seed(SEED)
    setup_parsers()

    if HYPERGRAPHS_DIR.exists():
        shutil.rmtree(HYPERGRAPHS_DIR)
    HYPERGRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    EMPTY_LOG.write_text("", encoding="utf-8")
    ERROR_LOG.write_text("", encoding="utf-8")

    sample_ids, split_map = split_sample_ids()
    results: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        if index % 100 == 0:
            print(f"Processed {index} samples")
        sample_dir = SAMPLES_DIR / sample_id
        try:
            hypergraph = build_hypergraph(str(sample_dir))
            (HYPERGRAPHS_DIR / f"{sample_id}.json").write_text(
                json.dumps(hypergraph, indent=2),
                encoding="utf-8",
            )
            results.append(hypergraph)
            if hypergraph["stats"]["n_hyperedges"] == 0:
                log_line(EMPTY_LOG, sample_id)
        except Exception as exc:  # noqa: BLE001
            log_line(ERROR_LOG, f"{sample_id}: {type(exc).__name__}: {exc}")

    summary = write_summary(results, split_map)
    print(summary, end="")
    SUMMARY_TXT.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()

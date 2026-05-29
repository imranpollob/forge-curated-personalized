#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "hypervul_dataset" / "samples"
SLITHER_FIX_OUTPUT = ROOT / "slither_fix_output.txt"
OUTPUT_FILE = ROOT / "ast_extraction_output.txt"

FUNCTION_RE = re.compile(r"\bfunction\s+(\w+)\s*\((.*?)\)([^{;]*)\{", re.DOTALL)
CONTRACT_RE = re.compile(r"\b(contract|interface|library)\s+\w+[^{};]*\{", re.MULTILINE)
STATE_RE = re.compile(
    r"^\s*(?:"
    r"(?:uint|int|address|bool|bytes\d*|bytes|string)\b[^;=]*?\s+(\w+)"
    r"|mapping\s*\([^;]+?\)\s+(?:\w+\s+)*(?P<mapping_name>\w+)"
    r"|struct\s+(?P<struct_name>\w+)"
    r")\s*(?:[=;{])",
    re.MULTILINE,
)
CALL_PATTERNS = [
    ("DELEGATECALL", re.compile(r"\.delegatecall\s*\(")),
    ("STATICCALL", re.compile(r"\.staticcall\s*\(")),
    ("TRANSFER", re.compile(r"\.transfer\s*\(")),
    ("SEND", re.compile(r"\.send\s*\(")),
    ("CALL", re.compile(r"\.call\s*(?:\{|\()")),
]
GENERIC_MEMBER_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*(?:\{[^}]*\})?\s*\(")
COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.DOTALL | re.MULTILINE)

TREE_SITTER_AVAILABLE = False
SOLIDITY_PARSER_AVAILABLE = False
ts_parser: Any = None
sol_parser: Any = None


def ensure_package(import_name: str, package_name: str) -> Any | None:
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"{import_name} is not installed; running pip install {package_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return importlib.import_module(import_name)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not install/import {package_name}: {exc}")
            return None


def setup_parsers() -> None:
    global TREE_SITTER_AVAILABLE, SOLIDITY_PARSER_AVAILABLE, ts_parser, sol_parser

    tree_sitter = ensure_package("tree_sitter", "tree-sitter")
    tree_sitter_solidity = ensure_package("tree_sitter_solidity", "tree-sitter-solidity")

    if tree_sitter is not None and tree_sitter_solidity is not None:
        try:
            language = tree_sitter.Language(tree_sitter_solidity.language())
            try:
                ts_parser = tree_sitter.Parser(language)
            except TypeError:
                ts_parser = tree_sitter.Parser()
                ts_parser.set_language(language)
            TREE_SITTER_AVAILABLE = True
        except Exception as exc:  # noqa: BLE001
            print(f"tree-sitter Solidity setup failed: {exc}")

    if not TREE_SITTER_AVAILABLE:
        solidity_parser = ensure_package("solidity_parser", "solidity-parser")
        if solidity_parser is not None:
            try:
                from solidity_parser import parser as fallback_parser  # type: ignore

                sol_parser = fallback_parser
                SOLIDITY_PARSER_AVAILABLE = True
            except Exception as exc:  # noqa: BLE001
                print(f"solidity-parser fallback setup failed: {exc}")


def strip_comments(source: str) -> str:
    return COMMENT_RE.sub("", source)


def line_for_offset(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def find_matching_brace(source: str, open_index: int) -> int:
    depth = 0
    in_string: str | None = None
    escaped = False
    for index in range(open_index, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {'"', "'"}:
            in_string = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(source) - 1


def contract_ranges(source: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in CONTRACT_RE.finditer(source):
        open_index = source.find("{", match.start())
        if open_index != -1:
            ranges.append((open_index, find_matching_brace(source, open_index)))
    return ranges


def function_ranges(source: str) -> list[dict[str, Any]]:
    functions: list[dict[str, Any]] = []
    for match in FUNCTION_RE.finditer(source):
        open_index = source.find("{", match.end() - 1)
        if open_index == -1:
            continue
        close_index = find_matching_brace(source, open_index)
        signature = match.group(3)
        visibility = ""
        for candidate in ("public", "external", "internal", "private"):
            if re.search(rf"\b{candidate}\b", signature):
                visibility = candidate
                break
        functions.append(
            {
                "name": match.group(1),
                "start": match.start(),
                "body_start": open_index,
                "end": close_index,
                "start_line": line_for_offset(source, match.start()),
                "end_line": line_for_offset(source, close_index),
                "visibility": visibility,
                "is_external": bool(re.search(r"\bexternal\b", signature)),
                "is_payable": bool(re.search(r"\bpayable\b", signature)),
            }
        )
    return functions


def node_text(source_bytes: bytes, node: Any) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def walk_nodes(node: Any):
    yield node
    for child in getattr(node, "children", []):
        yield from walk_nodes(child)


def tree_root(source: str) -> Any | None:
    if not TREE_SITTER_AVAILABLE or ts_parser is None:
        return None
    return ts_parser.parse(source.encode("utf-8")).root_node


def regex_state_variables(source: str) -> list[dict[str, Any]]:
    clean = strip_comments(source)
    functions = function_ranges(clean)
    ranges = contract_ranges(clean)
    state_vars: list[dict[str, Any]] = []

    for match in STATE_RE.finditer(clean):
        start = match.start()
        if any(func["start"] <= start <= func["end"] for func in functions):
            continue
        if ranges and not any(contract_start <= start <= contract_end for contract_start, contract_end in ranges):
            continue
        name = match.group(1) or match.group("mapping_name") or match.group("struct_name")
        if not name:
            continue
        raw = match.group(0).strip().rstrip("{;=")
        type_text = raw[: raw.rfind(name)].strip() if name in raw else raw
        state_vars.append({"name": name, "type": type_text, "line": line_for_offset(clean, start)})
    return dedupe_dicts(state_vars, ("name", "line"))


def extract_state_variables(source: str) -> list[dict]:
    """
    Returns list of dicts: {name, type, line}
    State variables are variable declarations at contract level (not inside functions)
    """
    try:
        root = tree_root(source)
        if root is None:
            return regex_state_variables(source)

        source_bytes = source.encode("utf-8")
        variables: list[dict[str, Any]] = []
        funcs = function_ranges(source)
        for node in walk_nodes(root):
            node_type = getattr(node, "type", "")
            if node_type not in {"state_variable_declaration", "variable_declaration"}:
                continue
            start = node.start_byte
            if any(func["start"] <= start <= func["end"] for func in funcs):
                continue
            text = node_text(source_bytes, node)
            found = regex_state_variables(text)
            if found:
                for item in found:
                    item["line"] = node.start_point[0] + item["line"]
                    variables.append(item)
            else:
                name_match = re.search(r"\b([A-Za-z_]\w*)\s*(?:=|;)", text)
                if name_match:
                    name = name_match.group(1)
                    variables.append(
                        {
                            "name": name,
                            "type": text[: name_match.start(1)].strip(),
                            "line": node.start_point[0] + 1,
                        }
                    )
        return dedupe_dicts(variables, ("name", "line"))
    except Exception:
        return regex_state_variables(source)


def regex_functions(source: str) -> list[dict[str, Any]]:
    clean = strip_comments(source)
    return [
        {
            "name": func["name"],
            "start_line": func["start_line"],
            "end_line": func["end_line"],
            "visibility": func["visibility"],
            "is_external": func["is_external"],
            "is_payable": func["is_payable"],
        }
        for func in function_ranges(clean)
    ]


def extract_functions(source: str) -> list[dict]:
    """
    Returns list of dicts: {name, start_line, end_line, visibility, is_external, is_payable}
    """
    try:
        root = tree_root(source)
        if root is None:
            return regex_functions(source)
        source_bytes = source.encode("utf-8")
        functions: list[dict[str, Any]] = []
        for node in walk_nodes(root):
            if getattr(node, "type", "") not in {"function_definition", "function_declaration"}:
                continue
            text = node_text(source_bytes, node)
            match = re.search(r"\bfunction\s+(\w+)\s*\(", text)
            if not match:
                continue
            visibility = ""
            for candidate in ("public", "external", "internal", "private"):
                if re.search(rf"\b{candidate}\b", text):
                    visibility = candidate
                    break
            functions.append(
                {
                    "name": match.group(1),
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "visibility": visibility,
                    "is_external": bool(re.search(r"\bexternal\b", text)),
                    "is_payable": bool(re.search(r"\bpayable\b", text)),
                }
            )
        return functions or regex_functions(source)
    except Exception:
        return regex_functions(source)


def enclosing_function(line: int, functions: list[dict[str, Any]]) -> str:
    for func in functions:
        if func["start_line"] <= line <= func["end_line"]:
            return func["name"]
    return ""


def regex_external_calls(source: str) -> list[dict[str, Any]]:
    clean = strip_comments(source)
    functions = extract_functions(clean)
    calls: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()

    for call_type, pattern in CALL_PATTERNS:
        for match in pattern.finditer(clean):
            line = line_for_offset(clean, match.start())
            raw_line = clean.splitlines()[line - 1].strip() if line - 1 < len(clean.splitlines()) else ""
            key = (line, call_type, raw_line)
            if key in seen:
                continue
            seen.add(key)
            calls.append(
                {
                    "call_type": call_type,
                    "line": line,
                    "enclosing_function": enclosing_function(line, functions),
                    "raw_text": raw_line[:240],
                }
            )

    known_call_methods = {"call", "delegatecall", "staticcall", "transfer", "send"}
    local_functions = {func["name"] for func in functions}
    for match in GENERIC_MEMBER_CALL_RE.finditer(clean):
        method = match.group(2)
        if method in known_call_methods or method in local_functions:
            continue
        line = line_for_offset(clean, match.start())
        raw_line = clean.splitlines()[line - 1].strip() if line - 1 < len(clean.splitlines()) else ""
        key = (line, "INTERFACE_CALL", raw_line)
        if key in seen:
            continue
        seen.add(key)
        calls.append(
            {
                "call_type": "INTERFACE_CALL",
                "line": line,
                "enclosing_function": enclosing_function(line, functions),
                "raw_text": raw_line[:240],
            }
        )
    return sorted(calls, key=lambda item: item["line"])


def extract_external_calls(source: str) -> list[dict]:
    """
    Returns list of dicts: {call_type, line, enclosing_function, raw_text}
    """
    return regex_external_calls(source)


def extract_call_graph(source: str) -> list[dict]:
    """
    Returns list of dicts: {caller_function, callee_function}
    Intra-contract only: function A calls function B defined in the same contract
    """
    clean = strip_comments(source)
    funcs = function_ranges(clean)
    known = {func["name"] for func in funcs}
    edges: set[tuple[str, str]] = set()
    for func in funcs:
        body = clean[func["body_start"] : func["end"]]
        for callee in known:
            if callee == func["name"]:
                continue
            if re.search(rf"(?<!\.)\b{re.escape(callee)}\s*\(", body):
                edges.add((func["name"], callee))
    return [{"caller_function": caller, "callee_function": callee} for caller, callee in sorted(edges)]


def extract_state_var_access(source: str, state_vars: list, functions: list) -> list[dict]:
    """
    Returns list of dicts: {function_name, var_name, access_type}
    """
    clean = strip_comments(source)
    ranges = function_ranges(clean)
    state_names = [var["name"] for var in state_vars if var.get("name")]
    accesses: list[dict[str, str]] = []
    for func in ranges:
        body = clean[func["body_start"] : func["end"]]
        for var_name in state_names:
            if not re.search(rf"\b{re.escape(var_name)}\b", body):
                continue
            write_pattern = rf"(\b{re.escape(var_name)}\b\s*(?:[+\-*/%|&^]?=|\+\+|--)|(?:\+\+|--)\s*\b{re.escape(var_name)}\b)"
            access_type = "WRITE" if re.search(write_pattern, body) else "READ"
            accesses.append(
                {
                    "function_name": func["name"],
                    "var_name": var_name,
                    "access_type": access_type,
                }
            )
    return accesses


def dedupe_dicts(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = tuple(item.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def parse_sample_ids_from_slither_fix() -> list[str]:
    text = SLITHER_FIX_OUTPUT.read_text(encoding="utf-8") if SLITHER_FIX_OUTPUT.exists() else ""
    ids = re.findall(r"^(vfp_\d+|neg_\d+)\s+", text, flags=re.MULTILINE)
    if ids:
        return ids[:25]
    return [
        "vfp_00605",
        "vfp_00143",
        "vfp_00029",
        "vfp_00329",
        "vfp_00301",
        "vfp_00281",
        "vfp_00187",
        "vfp_00136",
        "vfp_00537",
        "vfp_00121",
        "vfp_00569",
        "vfp_00462",
        "vfp_00034",
        "vfp_00033",
        "vfp_00125",
        "vfp_00275",
        "vfp_00289",
        "vfp_00513",
        "vfp_00575",
        "vfp_00030",
        "neg_00307",
        "neg_00101",
        "neg_00453",
        "neg_00389",
        "neg_00439",
    ]


def extract_file(source: str) -> dict[str, Any]:
    state_vars = extract_state_variables(source)
    functions = extract_functions(source)
    external_calls = extract_external_calls(source)
    call_graph = extract_call_graph(source)
    state_access = extract_state_var_access(source, state_vars, functions)
    return {
        "state_vars": state_vars,
        "functions": functions,
        "external_calls": external_calls,
        "call_graph": call_graph,
        "state_access": state_access,
    }


def summarize_sample(sample_id: str) -> dict[str, Any]:
    sample_dir = SAMPLES_DIR / sample_id
    sol_files = sorted(sample_dir.glob("*.sol"))
    summary = {
        "sample_id": sample_id,
        "sol_files": len(sol_files),
        "functions_found": 0,
        "state_vars_found": 0,
        "external_calls_found": 0,
        "call_graph_edges": 0,
        "success": False,
    }
    for sol_file in sol_files:
        source = sol_file.read_text(encoding="utf-8", errors="replace")
        data = extract_file(source)
        summary["functions_found"] += len(data["functions"])
        summary["state_vars_found"] += len(data["state_vars"])
        summary["external_calls_found"] += len(data["external_calls"])
        summary["call_graph_edges"] += len(data["call_graph"])
    summary["success"] = summary["functions_found"] > 0
    return summary


def main() -> None:
    setup_parsers()
    lines: list[str] = []

    lines.append("Part A - Parser Setup")
    lines.append(f"tree-sitter Solidity available: {TREE_SITTER_AVAILABLE}")
    lines.append(f"solidity-parser fallback available: {SOLIDITY_PARSER_AVAILABLE}")
    test_files = sorted((SAMPLES_DIR / "vfp_00001").glob("*.sol"))
    if test_files and TREE_SITTER_AVAILABLE:
        source = test_files[0].read_text(encoding="utf-8", errors="replace")
        root = tree_root(source)
        lines.append(f"Test file: {test_files[0].relative_to(ROOT)}")
        lines.append(f"Root node type: {getattr(root, 'type', 'unavailable')}")
    elif test_files:
        lines.append(f"Test file: {test_files[0].relative_to(ROOT)}")
        lines.append("Root node type: unavailable; using fallback parser/regex")
    else:
        lines.append("No test file found in hypervul_dataset/samples/vfp_00001/")
    lines.append("")

    lines.append("Part C - Five Real Sample Details")
    for sample_id in ["vfp_00001", "vfp_00002", "vfp_00003", "vfp_00004", "vfp_00006"]:
        sample_dir = SAMPLES_DIR / sample_id
        lines.append(f"Sample {sample_id}")
        sample_totals = {"state_vars": 0, "functions": 0, "external_calls": 0, "call_graph": 0}
        for sol_file in sorted(sample_dir.glob("*.sol")):
            source = sol_file.read_text(encoding="utf-8", errors="replace")
            data = extract_file(source)
            sample_totals["state_vars"] += len(data["state_vars"])
            sample_totals["functions"] += len(data["functions"])
            sample_totals["external_calls"] += len(data["external_calls"])
            sample_totals["call_graph"] += len(data["call_graph"])
            lines.append(f"  File: {sol_file.name}")
            lines.append(f"    state_variables: {json.dumps(data['state_vars'][:10], ensure_ascii=False)}")
            lines.append(f"    functions: {json.dumps(data['functions'][:10], ensure_ascii=False)}")
            lines.append(f"    external_calls: {json.dumps(data['external_calls'][:10], ensure_ascii=False)}")
            lines.append(f"    call_graph: {json.dumps(data['call_graph'][:10], ensure_ascii=False)}")
            lines.append(f"    state_var_access: {json.dumps(data['state_access'][:10], ensure_ascii=False)}")
        lines.append(
            "  Totals: "
            f"state_vars={sample_totals['state_vars']}, functions={sample_totals['functions']}, "
            f"external_calls={sample_totals['external_calls']}, call_graph_edges={sample_totals['call_graph']}"
        )
    lines.append("")

    lines.append("Part D - 25-Sample Summary")
    sample_ids = parse_sample_ids_from_slither_fix()
    summaries = [summarize_sample(sample_id) for sample_id in sample_ids]
    lines.append(
        f"{'sample_id':<14} {'sol_files':>9} {'functions_found':>15} {'state_vars_found':>16} "
        f"{'external_calls_found':>22} {'call_graph_edges':>16} {'success':>10}"
    )
    lines.append(
        f"{'-' * 14} {'-' * 9:>9} {'-' * 15:>15} {'-' * 16:>16} "
        f"{'-' * 22:>22} {'-' * 16:>16} {'-' * 10:>10}"
    )
    for summary in summaries:
        lines.append(
            f"{summary['sample_id']:<14} {summary['sol_files']:>9} {summary['functions_found']:>15} "
            f"{summary['state_vars_found']:>16} {summary['external_calls_found']:>22} "
            f"{summary['call_graph_edges']:>16} {('yes' if summary['success'] else 'no'):>10}"
        )
    success_count = sum(1 for summary in summaries if summary["success"])
    success_rate = success_count / len(summaries) * 100 if summaries else 0.0
    successful = [summary for summary in summaries if summary["success"]]
    avg_external = (
        sum(summary["external_calls_found"] for summary in successful) / len(successful)
        if successful
        else 0.0
    )
    lines.append("")
    lines.append(f"Overall success rate: {success_count}/{len(summaries)} ({success_rate:.1f}%)")
    lines.append(f"Average external calls per successful sample: {avg_external:.2f}")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    OUTPUT_FILE.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

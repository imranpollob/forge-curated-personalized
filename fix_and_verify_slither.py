#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "hypervul_dataset"
SAMPLES_DIR = DATASET_DIR / "samples"
SPLITS_JSON = DATASET_DIR / "splits.json"
OUTPUT_FILE = ROOT / "slither_fix_output.txt"

SEED = 42
TIMEOUT_SECONDS = 60
PRAGMA_RE = re.compile(r"pragma\s+solidity\s+[\^>=<~]*\s*([\d.]+)")
VERSIONS_NEEDED = [
    "0.8.0",
    "0.8.16",
    "0.8.18",
    "0.8.20",
    "0.8.21",
    "0.8.24",
    "0.8.26",
    "0.8.27",
    "0.8.28",
    "0.8.29",
]
DEFAULT_SOLC = "0.8.18"


def ensure_solcx():
    try:
        import solcx  # type: ignore
    except ImportError:
        print("py-solc-x is not installed; running pip install py-solc-x")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "py-solc-x"])
        import solcx  # type: ignore
    return solcx


def solc_executable(solcx: Any, version: str) -> str:
    try:
        return str(solcx.get_executable(version))
    except AttributeError:
        from solcx.install import get_executable  # type: ignore

        return str(get_executable(version))


def install_required_solc_versions(solcx: Any) -> tuple[list[str], str]:
    installed_successfully: list[str] = []
    for version in VERSIONS_NEEDED:
        solcx.install_solc(version, show_progress=False)
        installed_versions = {str(item) for item in solcx.get_installed_solc_versions()}
        if version in installed_versions:
            installed_successfully.append(version)
    return installed_successfully, str(solcx.get_solcx_install_folder())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def detect_pragma_version(sol_file: Path) -> str | None:
    try:
        with sol_file.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(20):
                line = handle.readline()
                if not line:
                    break
                match = PRAGMA_RE.search(line)
                if match:
                    return match.group(1)
    except OSError:
        return None
    return None


def get_solc_path(solcx: Any, sol_file: Path) -> str:
    installed_versions = {str(item) for item in solcx.get_installed_solc_versions()}
    version = detect_pragma_version(sol_file)
    if version not in installed_versions:
        version = DEFAULT_SOLC
    return solc_executable(solcx, version)


def label_source(sample_id: str) -> str:
    label = load_json(SAMPLES_DIR / sample_id / "label.json")
    source = label.get("source")
    return source if isinstance(source, str) else ""


def select_samples() -> list[str]:
    splits = load_json(SPLITS_JSON)
    train = splits.get("train")
    if not isinstance(train, list):
        raise RuntimeError("hypervul_dataset/splits.json is missing train list")

    positives = [str(sample_id) for sample_id in train if label_source(str(sample_id)) == "forge_positive"]
    negatives = [str(sample_id) for sample_id in train if label_source(str(sample_id)) == "forge_negative"]
    if len(positives) < 20:
        raise RuntimeError(f"Need at least 20 positive train samples, found {len(positives)}")
    if len(negatives) < 5:
        raise RuntimeError(f"Need at least 5 negative train samples, found {len(negatives)}")

    random.seed(SEED)
    return random.sample(positives, 20) + random.sample(negatives, 5)


def classify_error(text: str, timed_out: bool = False) -> str:
    if timed_out:
        return "Timeout"
    lowered = text.lower()
    if "import" in lowered and (
        "not found" in lowered
        or "cannot find" in lowered
        or "file not found" in lowered
        or "source" in lowered
    ):
        return "ImportError"
    if "parsererror" in lowered or "syntax error" in lowered or "syntaxerror" in lowered:
        return "SyntaxError"
    if "compilererror" in lowered or "compiler error" in lowered or "invalid solc compilation" in lowered:
        return "CompilerError"
    return "Other"


def run_command(command: list[str]) -> tuple[int, str, float, bool]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        elapsed = time.perf_counter() - start
        return result.returncode, result.stdout or "", elapsed, False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        output = exc.stdout or exc.stderr or "Timed out"
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return 124, str(output), elapsed, True


def parse_json_output(output: str) -> dict[str, Any] | None:
    stripped = output.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None


def output_success(output: str) -> bool:
    data = parse_json_output(output)
    return bool(isinstance(data, dict) and data.get("success") is True)


def walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from walk_json(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_json(nested)


def slither_stats(output: str) -> dict[str, Any]:
    data = parse_json_output(output)
    contracts: set[str] = set()
    functions: set[str] = set()
    external_calls = 0
    has_state_variables = False

    if data is None:
        return {
            "contracts": 0,
            "functions": 0,
            "external_calls": 0,
            "has_state_variables": False,
        }

    for item in walk_json(data):
        item_type = str(item.get("type", "")).lower()
        name = item.get("name")
        canonical_name = item.get("canonical_name")
        item_id = str(canonical_name or name or item.get("source_mapping", {}).get("filename_absolute", ""))

        if item_type == "contract" and item_id:
            contracts.add(item_id)
        elif item_type == "function" and item_id:
            functions.add(item_id)
        elif item_type in {"variable", "statevariable", "state_variable"}:
            has_state_variables = True

        for key in ("contracts", "contract_names"):
            values = item.get(key)
            if isinstance(values, list):
                contracts.update(str(value) for value in values if value)
        for key in ("functions", "function_names"):
            values = item.get(key)
            if isinstance(values, list):
                functions.update(str(value) for value in values if value)
        for key in ("state_variables", "stateVariables", "variables"):
            values = item.get(key)
            if isinstance(values, list) and values:
                has_state_variables = True

        for key in ("high_level_calls", "low_level_calls"):
            values = item.get(key)
            if isinstance(values, list):
                external_calls += len(values)
            elif values:
                external_calls += 1

    return {
        "contracts": len(contracts),
        "functions": len(functions),
        "external_calls": external_calls,
        "has_state_variables": has_state_variables,
    }


def first_error_line(message: str) -> str:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return ""
    markers = (
        "Error:",
        "ParserError",
        "CompilerError",
        "InvalidCompilation",
        "ImportError",
        "File not found",
        "not found",
    )
    for line in lines:
        if any(marker in line for marker in markers):
            return line[:300]
    return lines[0][:300]


def run_diagnostic_command(sample_dir: Path, solc_path: str, allow_paths: bool = False) -> str:
    command = ["slither", str(sample_dir), "--solc", solc_path, "--disable-color"]
    if allow_paths:
        command.extend(["--solc-args", "--allow-paths ."])
    _returncode, output, _elapsed, _timed_out = run_command(command)
    return output


def verify_sample(solcx: Any, sample_id: str) -> dict[str, Any]:
    sample_dir = SAMPLES_DIR / sample_id
    label = load_json(sample_dir / "label.json")
    vfp_id = str(label.get("vfp_id") or sample_id)
    sol_files = sorted(sample_dir.glob("*.sol"))
    if not sol_files:
        return {
            "sample_id": sample_id,
            "vfp_id": vfp_id,
            "pragma": "no pragma",
            "solc_path": "",
            "success": False,
            "error_type": "Other",
            "error_message": "No .sol files found",
            "seconds": 0.0,
            "stats": slither_stats(""),
        }

    first_sol = sol_files[0]
    pragma = detect_pragma_version(first_sol) or "no pragma"
    solc_path = get_solc_path(solcx, first_sol)

    command = ["slither", str(sample_dir), "--solc", solc_path, "--json", "-"]
    returncode, output, elapsed, timed_out = run_command(command)
    if returncode != 0 and not output.strip() and not timed_out:
        output = run_diagnostic_command(sample_dir, solc_path)
    error_type = classify_error(output, timed_out)

    success = returncode == 0 or output_success(output)

    if not success and error_type == "ImportError" and not timed_out:
        retry_command = [
            "slither",
            str(sample_dir),
            "--solc",
            solc_path,
            "--json",
            "-",
            "--solc-args",
            "--allow-paths .",
        ]
        retry_code, retry_output, retry_elapsed, retry_timed_out = run_command(retry_command)
        if retry_code != 0 and not retry_output.strip() and not retry_timed_out:
            retry_output = run_diagnostic_command(sample_dir, solc_path, allow_paths=True)
        if retry_code == 0 or output_success(retry_output):
            returncode = retry_code
            output = retry_output
        else:
            output = retry_output or output
        elapsed += retry_elapsed
        timed_out = retry_timed_out
        error_type = classify_error(output, timed_out)
        success = returncode == 0 or output_success(output)

    return {
        "sample_id": sample_id,
        "vfp_id": vfp_id,
        "pragma": pragma,
        "solc_path": solc_path,
        "success": success,
        "error_type": "" if success else error_type,
        "error_message": "" if success else first_error_line(output),
        "seconds": elapsed,
        "stats": slither_stats(output) if success else slither_stats(""),
    }


def main() -> None:
    random.seed(SEED)
    solcx = ensure_solcx()

    installed, install_folder = install_required_solc_versions(solcx)
    print("Solc versions installed successfully via py-solc-x:")
    for version in installed:
        print(f"- {version}")
    print(f"py-solc-x install folder: {install_folder}")

    selected_samples = select_samples()
    results: list[dict[str, Any]] = []
    for index, sample_id in enumerate(selected_samples, start=1):
        print(f"Verifying sample {index}/25: {sample_id}")
        results.append(verify_sample(solcx, sample_id))

    total = len(results)
    successes = [result for result in results if result["success"]]
    failures = [result for result in results if not result["success"]]
    failure_counts = Counter(result["error_type"] for result in failures)
    avg_external_calls = (
        sum(result["stats"]["external_calls"] for result in successes) / len(successes)
        if successes
        else 0.0
    )
    avg_success_time = sum(result["seconds"] for result in successes) / len(successes) if successes else 0.0

    lines: list[str] = []
    lines.append("Part A - py-solc-x Compiler Setup")
    lines.append(f"Installed versions: {', '.join(installed)}")
    lines.append(f"py-solc-x install folder: {install_folder}")
    lines.append("")
    lines.append("Part C/D - Project-Level Slither Results")
    lines.append(
        f"{'sample_id':<14} {'pragma':<10} {'result':<8} {'seconds':>8} "
        f"{'error_type':<14} {'contracts':>9} {'functions':>9} {'ext_calls':>9} {'state_vars':>10} error"
    )
    lines.append(
        f"{'-' * 14} {'-' * 10} {'-' * 8} {'-' * 8:>8} "
        f"{'-' * 14} {'-' * 9:>9} {'-' * 9:>9} {'-' * 9:>9} {'-' * 10:>10} {'-' * 5}"
    )
    for result in results:
        stats = result["stats"]
        status = "success" if result["success"] else "fail"
        lines.append(
            f"{result['sample_id']:<14} {result['pragma']:<10} {status:<8} {result['seconds']:>8.2f} "
            f"{result['error_type']:<14} {stats['contracts']:>9} {stats['functions']:>9} "
            f"{stats['external_calls']:>9} {str(stats['has_state_variables']):>10} "
            f"{result['error_message']}"
        )
        if result["success"]:
            lines.append(
                f"  Slither found: contracts={stats['contracts']}, functions={stats['functions']}, "
                f"external_calls={stats['external_calls']}, state_variables_detected={stats['has_state_variables']}"
            )

    success_pct = len(successes) / total * 100 if total else 0.0
    failure_pct = len(failures) / total * 100 if total else 0.0
    lines.append("")
    lines.append("Part E - Summary")
    lines.append(f"Total files tested: {total}")
    lines.append(f"Success count: {len(successes)} ({success_pct:.1f}%)")
    lines.append(f"Failure count: {len(failures)} ({failure_pct:.1f}%)")
    lines.append("Failures by error type:")
    for error_type in ("CompilerError", "ImportError", "SyntaxError", "Timeout", "Other"):
        lines.append(f"  {error_type}: {failure_counts[error_type]}")
    lines.append(f"Average number of external calls detected in successful contracts: {avg_external_calls:.2f}")
    lines.append(f"Average processing time for successful runs: {avg_success_time:.2f}s")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    OUTPUT_FILE.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

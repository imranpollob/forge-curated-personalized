#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "hypervul_dataset"
SAMPLES_DIR = DATASET_DIR / "samples"
SPLITS_JSON = DATASET_DIR / "splits.json"
OUTPUT_FILE = ROOT / "slither_verification_output.txt"
SEED = 42
TIMEOUT_SECONDS = 30
PRAGMA_RE = re.compile(r"pragma\s+solidity\s+[\^>=<~]*\s*([\d.]+)")


def run_version(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)

    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output


def setup_check() -> tuple[str, str] | None:
    slither_ok, slither_version = run_version(["slither", "--version"])
    solc_ok, solc_version = run_version(["solc", "--version"])

    if slither_ok and solc_ok:
        print(f"Slither version: {slither_version}")
        print(f"solc version: {solc_version}")
        return slither_version, solc_version

    if not slither_ok:
        print("slither-analyzer is missing or failed to run.")
        print("Install command: pip install slither-analyzer")
        print(f"Observed error: {slither_version}")
    if not solc_ok:
        print("solc is missing or failed to run.")
        print("Install commands:")
        print("pip install py-solc-x")
        print("python -c \"from solcx import install_solc; install_solc('0.8.19')\"")
        print(f"Observed error: {solc_version}")
    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def source_for_sample(sample_id: str) -> str:
    label_path = SAMPLES_DIR / sample_id / "label.json"
    label = load_json(label_path)
    source = label.get("source")
    return source if isinstance(source, str) else ""


def select_samples() -> list[str]:
    splits = load_json(SPLITS_JSON)
    train_ids = splits.get("train", [])
    if not isinstance(train_ids, list):
        raise RuntimeError("hypervul_dataset/splits.json does not contain a train list")

    positives = [sample_id for sample_id in train_ids if source_for_sample(str(sample_id)) == "forge_positive"]
    negatives = [sample_id for sample_id in train_ids if source_for_sample(str(sample_id)) == "forge_negative"]

    if len(positives) < 20:
        raise RuntimeError(f"Need 20 positive train samples, found {len(positives)}")
    if len(negatives) < 5:
        raise RuntimeError(f"Need 5 negative train samples, found {len(negatives)}")

    random.seed(SEED)
    return random.sample(positives, 20) + random.sample(negatives, 5)


def detect_pragma(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(50):
                line = handle.readline()
                if not line:
                    break
                match = PRAGMA_RE.search(line)
                if match:
                    return match.group(1)
    except OSError:
        return "no pragma"
    return "no pragma"


def classify_error(text: str, timed_out: bool = False) -> str:
    if timed_out:
        return "Timeout"
    lowered = text.lower()
    if "syntaxerror" in lowered or "syntax error" in lowered or "parsererror" in lowered:
        return "SyntaxError"
    if "compilererror" in lowered or "compiler error" in lowered or "solc" in lowered:
        return "CompilerError"
    if "import" in lowered and (
        "not found" in lowered
        or "cannot find" in lowered
        or "file not found" in lowered
        or "source" in lowered
    ):
        return "ImportError"
    return "Other"


def run_slither_once(command: list[str]) -> tuple[bool, str, float, bool]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        elapsed = time.perf_counter() - start
        message = "\n".join(part for part in (result.stderr, result.stdout) if part).strip()
        return result.returncode == 0, message, elapsed, False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        raw_message = exc.stderr or exc.stdout or "Timed out"
        if isinstance(raw_message, bytes):
            message = raw_message.decode("utf-8", errors="replace")
        else:
            message = str(raw_message)
        return False, message.strip(), elapsed, True


def run_diagnostic_slither(path: Path) -> str:
    try:
        result = subprocess.run(
            ["slither", str(path), "--disable-color"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raw_message = exc.stderr or exc.stdout or "Timed out"
        if isinstance(raw_message, bytes):
            return raw_message.decode("utf-8", errors="replace").strip()
        return str(raw_message).strip()
    return "\n".join(part for part in (result.stderr, result.stdout) if part).strip()


def run_slither(path: Path) -> tuple[bool, str, float, str]:
    first_command = ["slither", str(path), "--json", "-"]
    success, error, elapsed, timed_out = run_slither_once(first_command)
    if success or timed_out:
        return success, error, elapsed, classify_error(error, timed_out)

    second_command = ["slither", str(path), "--json", "-", "--solc-remaps", "", "--disable-color"]
    retry_success, retry_error, retry_elapsed, retry_timed_out = run_slither_once(second_command)
    total_elapsed = elapsed + retry_elapsed
    if retry_success:
        return True, "", total_elapsed, ""
    final_error = retry_error or error
    if not final_error.strip() and not retry_timed_out:
        final_error = run_diagnostic_slither(path)
    return False, final_error, total_elapsed, classify_error(final_error, retry_timed_out)


def first_error_line(message: str) -> str:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return ""
    markers = ("Error:", "ParserError", "CompilerError", "SyntaxError", "InvalidCompilation", "ImportError")
    for line in lines:
        if any(marker in line for marker in markers):
            return line[:300]
    return lines[0][:300]


def main() -> None:
    versions = setup_check()
    if versions is None:
        return
    slither_version, solc_version = versions

    selected_samples = select_samples()
    results: list[dict[str, Any]] = []

    for index, sample_id in enumerate(selected_samples, start=1):
        print(f"Testing sample {index}/25: {sample_id}")
        sample_dir = SAMPLES_DIR / sample_id
        sol_files = sorted(path for path in sample_dir.glob("*.sol") if path.is_file())
        if not sol_files:
            results.append(
                {
                    "sample_id": sample_id,
                    "filename": "(no .sol files)",
                    "pragma": "no pragma",
                    "success": False,
                    "time": 0.0,
                    "error_type": "Other",
                    "error_message": "No .sol files found",
                }
            )
            continue

        for sol_file in sol_files:
            pragma = detect_pragma(sol_file)
            success, error_message, elapsed, error_type = run_slither(sol_file)
            results.append(
                {
                    "sample_id": sample_id,
                    "filename": sol_file.name,
                    "pragma": pragma,
                    "success": success,
                    "time": elapsed,
                    "error_type": "" if success else error_type,
                    "error_message": "" if success else first_error_line(error_message),
                }
            )

    total = len(results)
    successes = sum(1 for result in results if result["success"])
    failures = total - successes
    failure_types = Counter(result["error_type"] for result in results if not result["success"])
    pragma_counts = Counter(result["pragma"] for result in results)
    successful_times = [float(result["time"]) for result in results if result["success"]]
    average_success_time = sum(successful_times) / len(successful_times) if successful_times else 0.0

    lines: list[str] = []
    lines.append(f"Slither version: {slither_version}")
    lines.append(f"solc version: {solc_version}")
    lines.append("")
    lines.append("Per-file results")
    lines.append(
        f"{'sample_id':<14} {'filename':<48} {'pragma':<12} "
        f"{'result':<8} {'seconds':>8} {'error_type':<14} error_message"
    )
    lines.append(
        f"{'-' * 14} {'-' * 48} {'-' * 12} "
        f"{'-' * 8} {'-' * 8:>8} {'-' * 14} {'-' * 13}"
    )
    for result in results:
        status = "success" if result["success"] else "fail"
        lines.append(
            f"{result['sample_id']:<14} {result['filename'][:48]:<48} {result['pragma']:<12} "
            f"{status:<8} {result['time']:>8.2f} {result['error_type']:<14} {result['error_message']}"
        )

    lines.append("")
    lines.append("Summary")
    lines.append(f"Total files tested: {total}")
    success_pct = (successes / total * 100) if total else 0.0
    failure_pct = (failures / total * 100) if total else 0.0
    lines.append(f"Slither success count: {successes} ({success_pct:.1f}%)")
    lines.append(f"Slither failure count: {failures} ({failure_pct:.1f}%)")
    lines.append("Failures by error type:")
    for error_type in ("CompilerError", "ImportError", "SyntaxError", "Timeout", "Other"):
        lines.append(f"  {error_type}: {failure_types[error_type]}")
    lines.append("Most common pragma solidity versions found:")
    for pragma, count in pragma_counts.most_common(10):
        lines.append(f"  {pragma}: {count}")
    lines.append(f"Average processing time for successful runs: {average_success_time:.2f}s")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    OUTPUT_FILE.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

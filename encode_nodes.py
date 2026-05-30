#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "hypervul_dataset"
SAMPLES_DIR = DATASET_DIR / "samples"
HYPERGRAPHS_DIR = DATASET_DIR / "hypergraphs_v2"
MANIFEST_JSON = DATASET_DIR / "dataset_manifest.json"
FEATURES_DIR = DATASET_DIR / "node_features"
PROJECTIONS_PT = DATASET_DIR / "node_projections.pt"
ERROR_LOG = DATASET_DIR / "encoding_errors.log"
SUMMARY_TXT = DATASET_DIR / "encoding_summary.txt"

MODEL_NAME = "web3se/SmartBERT-v3"
FALLBACK_MODEL_NAME = "microsoft/unixcoder-base"
OPCODE_TYPES = ["CALL", "DELEGATECALL", "STATICCALL", "TRANSFER", "SEND", "INTERFACE_CALL"]
OPCODE_TO_IDX = {t: i for i, t in enumerate(OPCODE_TYPES)}
BATCH_SIZE = 32
SEED = 42

tokenizer: Any = None
encoder: Any = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_transformers() -> tuple[Any, Any]:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        print("transformers is not installed; running pip install transformers")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        transformers = importlib.import_module("transformers")
    return transformers.RobertaTokenizer, transformers.RobertaModel


def setup_encoder() -> tuple[str, int, bool]:
    global tokenizer, encoder
    RobertaTokenizer, RobertaModel = ensure_transformers()

    loaded_name = MODEL_NAME
    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        encoder = RobertaModel.from_pretrained(MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load {MODEL_NAME}: {exc}")
        print(f"Falling back to {FALLBACK_MODEL_NAME}")
        loaded_name = FALLBACK_MODEL_NAME
        tokenizer = RobertaTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
        encoder = RobertaModel.from_pretrained(FALLBACK_MODEL_NAME)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.to(device)
    param_count = sum(param.numel() for param in encoder.parameters())
    all_frozen = all(not param.requires_grad for param in encoder.parameters())
    print(f"Model name: {loaded_name}")
    print(f"Parameter count: {param_count}")
    print(f"All parameters frozen: {all_frozen}")
    print(f"Device: {device}")
    return loaded_name, param_count, all_frozen


def encode_text(text: str, max_length: int = 256) -> torch.Tensor:
    """
    Encode a text string using SmartBERT-v3.
    Returns: 768-d tensor (CLS token embedding)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    ).to(device)
    with torch.no_grad():
        outputs = encoder(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()


def encode_text_batch(texts: list[str], max_length: int = 256) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 768), dtype=torch.float32)
    encoded_batches = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        ).to(device)
        with torch.no_grad():
            outputs = encoder(**inputs)
        encoded_batches.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(encoded_batches, dim=0)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def usable_sample_ids() -> list[str]:
    manifest = load_json(MANIFEST_JSON)
    ids: set[str] = set()
    cv_folds = manifest.get("cv_folds", {}).get("folds", [])
    if isinstance(cv_folds, list):
        for fold in cv_folds:
            if not isinstance(fold, dict):
                continue
            for split in ("train", "val", "test"):
                ids.update(str(sample_id) for sample_id in fold.get(split, []))
    if ids:
        return sorted(ids)

    return sorted(
        path.stem
        for path in HYPERGRAPHS_DIR.glob("*.json")
        if int(load_json(path).get("stats", {}).get("n_hyperedges", 0)) >= 1
    )


def load_sample_source(sample_id: str) -> dict[str, str]:
    source: dict[str, str] = {}
    sample_dir = SAMPLES_DIR / sample_id
    for sol_file in sample_dir.glob("*.sol"):
        source[sol_file.name] = sol_file.read_text(encoding="utf-8", errors="replace")
    return source


def function_text(node: dict[str, Any], sample_source_code: dict[str, str]) -> str:
    source = sample_source_code.get(str(node.get("file", "")))
    if not source:
        return f"{node.get('name', '')} function"
    lines = source.splitlines()
    start_line = int(node.get("start_line") or 1)
    end_line = int(node.get("end_line") or start_line)
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    text = "\n".join(lines[start_idx:end_idx]).strip()
    return text or f"{node.get('name', '')} function"


def node_text_and_kind(node: dict[str, Any], sample_source_code: dict[str, str]) -> tuple[str, str, int]:
    node_type = str(node.get("node_type", ""))
    if node_type == "function":
        return function_text(node, sample_source_code), "code", 256
    if node_type == "state_var":
        return f"{node.get('var_type', '')} {node.get('name', '')}".strip(), "code", 256
    if node_type == "call_site":
        return str(node.get("raw_text") or node.get("name") or ""), "call", 64
    return str(node.get("name", "")), "code", 256


def encode_node(node: dict, sample_source_code: dict) -> torch.Tensor:
    """
    Encode a single node into a raw embedding.
    Function/state_var nodes return 768-d SmartBERT embeddings.
    Call-site nodes return 774-d: 768-d SmartBERT raw_text + 6-d opcode one-hot.
    """
    text, kind, max_length = node_text_and_kind(node, sample_source_code)
    base = encode_text(text, max_length=max_length)
    if kind != "call":
        return base
    opcode = torch.zeros(len(OPCODE_TYPES), dtype=torch.float32)
    idx = OPCODE_TO_IDX.get(str(node.get("call_type", "")))
    if idx is not None:
        opcode[idx] = 1.0
    return torch.cat([base, opcode], dim=0)


def build_node_raw_embeddings(nodes: list[dict[str, Any]], sample_source_code: dict[str, str]) -> tuple[torch.Tensor, list[int]]:
    code_indices: list[int] = []
    code_texts: list[str] = []
    call_indices: list[int] = []
    call_texts: list[str] = []

    for index, node in enumerate(nodes):
        text, kind, _max_length = node_text_and_kind(node, sample_source_code)
        if kind == "call":
            call_indices.append(index)
            call_texts.append(text)
        else:
            code_indices.append(index)
            code_texts.append(text)

    hidden_size = int(getattr(encoder.config, "hidden_size", 768))
    raw_embeddings = torch.zeros((len(nodes), hidden_size + len(OPCODE_TYPES)), dtype=torch.float32)
    embedding_dims = [hidden_size for _ in nodes]

    code_embeddings = encode_text_batch(code_texts, max_length=256)
    for row, node_index in enumerate(code_indices):
        raw_embeddings[node_index, :hidden_size] = code_embeddings[row]

    call_embeddings = encode_text_batch(call_texts, max_length=64)
    for row, node_index in enumerate(call_indices):
        node = nodes[node_index]
        raw_embeddings[node_index, :hidden_size] = call_embeddings[row]
        opcode = torch.zeros(len(OPCODE_TYPES), dtype=torch.float32)
        idx = OPCODE_TO_IDX.get(str(node.get("call_type", "")))
        if idx is not None:
            opcode[idx] = 1.0
        raw_embeddings[node_index, hidden_size:] = opcode
        embedding_dims[node_index] = hidden_size + len(OPCODE_TYPES)

    return raw_embeddings, embedding_dims


def save_projection_layers(hidden_size: int = 768) -> None:
    torch.manual_seed(SEED)
    proj_function = torch.nn.Linear(hidden_size, 128)
    proj_state_var = torch.nn.Linear(hidden_size, 128)
    proj_call_site = torch.nn.Linear(hidden_size + len(OPCODE_TYPES), 128)
    torch.save(
        {
            "proj_function": proj_function.state_dict(),
            "proj_state_var": proj_state_var.state_dict(),
            "proj_call_site": proj_call_site.state_dict(),
            "metadata": {
                "hidden_size": hidden_size,
                "opcode_types": OPCODE_TYPES,
                "note": "Projection layers are trainable during G-HAN training; node cache stores raw encoder outputs.",
            },
        },
        PROJECTIONS_PT,
    )


def directory_size_mb(path: Path) -> float:
    total_bytes = sum(file.stat().st_size for file in path.glob("*.pt") if file.is_file())
    return total_bytes / (1024 * 1024)


def main() -> None:
    start_time = time.perf_counter()
    torch.manual_seed(SEED)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.write_text("", encoding="utf-8")

    model_name, param_count, frozen = setup_encoder()
    hidden_size = int(getattr(encoder.config, "hidden_size", 768))
    save_projection_layers(hidden_size)

    sample_ids = usable_sample_ids()
    total_nodes = 0
    encoded_samples = 0
    node_type_counts = {"function": 0, "state_var": 0, "call_site": 0}

    for index, sample_id in enumerate(sample_ids, start=1):
        if index % 50 == 0:
            print(f"Encoded progress: {index}/{len(sample_ids)} samples")
        try:
            hypergraph = load_json(HYPERGRAPHS_DIR / f"{sample_id}.json")
            nodes = hypergraph.get("nodes", [])
            if not isinstance(nodes, list):
                raise ValueError("hypergraph nodes field is not a list")
            sample_source_code = load_sample_source(sample_id)
            raw_embeddings, embedding_dims = build_node_raw_embeddings(nodes, sample_source_code)
            node_ids = [int(node.get("node_id", idx)) for idx, node in enumerate(nodes)]
            node_types = [str(node.get("node_type", "")) for node in nodes]
            torch.save(
                {
                    "node_ids": node_ids,
                    "node_types": node_types,
                    "raw_embeddings": raw_embeddings,
                    "embedding_dims": embedding_dims,
                    "storage_note": (
                        "raw_embeddings is rectangular [n_nodes, hidden_size+6]; "
                        "function/state_var rows use zero opcode padding, call_site rows include opcode one-hot."
                    ),
                },
                FEATURES_DIR / f"{sample_id}.pt",
            )
            encoded_samples += 1
            total_nodes += len(nodes)
            for node_type in node_types:
                if node_type in node_type_counts:
                    node_type_counts[node_type] += 1
        except Exception as exc:  # noqa: BLE001
            with ERROR_LOG.open("a", encoding="utf-8") as handle:
                handle.write(f"{sample_id}: {type(exc).__name__}: {exc}\n")

    elapsed = time.perf_counter() - start_time
    cache_size = directory_size_mb(FEATURES_DIR)
    errors = ERROR_LOG.read_text(encoding="utf-8").strip().splitlines()
    avg_nodes = total_nodes / encoded_samples if encoded_samples else 0.0

    lines = []
    lines.append(f"Model name: {model_name}")
    lines.append(f"Parameter count: {param_count}")
    lines.append(f"All params frozen: {frozen}")
    lines.append(f"GPU used: {'yes' if device.type == 'cuda' else 'no'}")
    lines.append(f"Total samples encoded successfully: {encoded_samples}")
    lines.append(f"Total nodes encoded: {total_nodes}")
    lines.append(f"Average nodes per sample: {avg_nodes:.2f}")
    lines.append("Average nodes per sample breakdown:")
    for node_type in ("function", "state_var", "call_site"):
        avg = node_type_counts[node_type] / encoded_samples if encoded_samples else 0.0
        lines.append(f"  {node_type}: {avg:.2f}")
    lines.append(f"Time taken total: {elapsed:.2f}s")
    lines.append(f"Time taken per sample: {(elapsed / encoded_samples if encoded_samples else 0.0):.2f}s")
    lines.append(f"Cache size on disk: {cache_size:.2f} MB")
    lines.append(f"Encoding errors: {len(errors)}")
    if errors:
        lines.append("First errors:")
        lines.extend(f"  {line}" for line in errors[:10])
    lines.append(f"Projection weights saved: {PROJECTIONS_PT}")
    lines.append(f"Node feature cache: {FEATURES_DIR}")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    SUMMARY_TXT.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

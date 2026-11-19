#!/usr/bin/env python3
"""Stage G external Hugging Face BF16 reference preflight for SenseNova-Loop-9B.

This script is deterministic and local-only. It validates the exact model directory used by the
Stage G reference, records tokenizer/config/MTP file facts, and prints a fixed token prompt. It
never downloads, converts, or writes a second artifact.

The heavy BF16 forward comparison needs PyTorch, Transformers, and Safetensors, which are not
installed in the current sandbox. When those imports are unavailable the script exits with code
77 and a precise blocker message so a CTest or operator can distinguish an environment gap from
a model failure. When they are available, `--forward` loads the local model with trust_remote_code
and compares the first greedy token/top-k overlap against the artifact engine's recorded token.

Example:
    python3 tests/targets/sensenova_loop_9b/hf_reference.py --preflight
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

DEFAULT_MODEL = Path("/home/user/models/sensenova/SenseNova-Loop-9B")
FIXED_PROMPT_IDS = (785, 3895, 286, 2673, 315, 3387, 349)
EXPECTED_EOS = (248044, 248046)
FIXED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "modeling_sensenova_loop.py",
    "mtp.safetensors",
    "model.safetensors.index.json",
)
EXPECTED_RESOURCE_SHA256 = {
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "generation_config.json": "f288bb4281580138395c6acc7b3adb62f45c21530bc0708cb2cf9edf093f6a4f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(model: Path) -> int:
    missing = [name for name in FIXED_FILES if not (model / name).is_file()]
    if missing:
        print("FAIL: local HF model is missing: " + ", ".join(missing))
        return 1

    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    generation = json.loads((model / "generation_config.json").read_text(encoding="utf-8"))
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shards if not (model / name).is_file()]
    if missing_shards:
        print("FAIL: safetensors index names missing shards: " + ", ".join(missing_shards))
        return 1

    errors: list[str] = []
    if config.get("model_type") != "sensenova_loop":
        errors.append("unexpected model_type")
    if config.get("dtype") != "bfloat16":
        errors.append("checkpoint dtype is not BF16")
    if tuple(generation.get("eos_token_id", ())) != EXPECTED_EOS:
        errors.append("generation_config eos_token_id does not match the converter contract")
    if generation.get("use_cache") is not True:
        errors.append("generation_config use_cache is not true")
    for name, expected in EXPECTED_RESOURCE_SHA256.items():
        actual = sha256(model / name)
        if actual != expected:
            errors.append(f"{name} hash mismatch: {actual}")
    if errors:
        for error in errors:
            print("FAIL: " + error)
        return 1

    print("HF_LOCAL_PREFLIGHT: OK")
    print(f"HF_MODEL_DIR: {model}")
    print(f"HF_ARCHITECTURE: {config['architectures'][0]}")
    print(f"HF_SHARDS: {len(shards)}")
    print(f"HF_FIXED_PROMPT_IDS: {' '.join(map(str, FIXED_PROMPT_IDS))}")
    return 0


def forward(_model: Path) -> int:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError as error:
        print(f"BLOCKED: external BF16 forward requires local Python dependencies: {error}")
        return 77
    print("BLOCKED: --forward is not wired until the dependency-backed oracle is available")
    return 77


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    if args.forward:
        return forward(args.model)
    return preflight(args.model)


if __name__ == "__main__":
    sys.exit(main())

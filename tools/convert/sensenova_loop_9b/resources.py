"""Frontend resources and strict source-key checks for SenseNova-Loop-9B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from tools.convert.qwen3_6.common.conversion import ResourcePayload, load_resources

from . import inventory


RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    ),
    "frontend/tokenizer_config.json": (
        "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b"
    ),
    "frontend/chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
    "frontend/generation_config.json": (
        "f288bb4281580138395c6acc7b3adb62f45c21530bc0708cb2cf9edf093f6a4f"
    ),
    "frontend/preprocessor_config.json": (
        "3a159dfec9978a186a72ba085e0ad6a050f3968d8b364218d7bd13f5c89381f2"
    ),
    "frontend/video_preprocessor_config.json": (
        "bd96830cce1312679aa1f2ea43ed17da20e8ebbcea2398cfa64e2d275a96fdd7"
    ),
}


def validate_resources(resources: Sequence[ResourcePayload]) -> None:
    actual = {resource.name: hashlib.sha256(resource.data).hexdigest() for resource in resources}
    if tuple(actual) != tuple(RESOURCE_SHA256):
        raise ValueError("frontend resource set does not match the SenseNova-Loop-9B profile")
    for name, expected in RESOURCE_SHA256.items():
        if actual[name] != expected:
            raise ValueError(f"frontend resource hash mismatch: {name}")


def load_checked_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    resources = load_resources(model_dir, inventory.RESOURCE_SPECS)
    validate_resources(resources)
    return resources


def _is_ignored_vision_source(name: str) -> bool:
    return name.startswith("model.visual.") or name == "model.language_model.loop_gate.logits"


def validate_source_keys(
    source_names: Iterable[str],
    required_names: Iterable[str],
) -> None:
    """Reject unknown source tensors, except the explicit text-only ignore list."""

    names = set(source_names)
    required = set(required_names)
    missing = sorted(required - names)
    if missing:
        raise ValueError("source tensors are missing: " + ", ".join(missing))
    unknown = sorted(name for name in names - required if not _is_ignored_vision_source(name))
    if unknown:
        raise ValueError("unexpected source tensors: " + ", ".join(unknown))


def validate_mtp_source_keys(
    source_names: Iterable[str],
    required_names: Iterable[str],
) -> None:
    names = set(source_names)
    required = set(required_names)
    if names != required:
        missing = sorted(required - names)
        unknown = sorted(names - required)
        raise ValueError(
            "MTP source tensor set mismatch; missing="
            + repr(missing)
            + " unexpected="
            + repr(unknown)
        )


def validate_generation_config(config: Mapping[str, object]) -> None:
    expected_eos = (248044, 248046)
    eos = config.get("eos_token_id")
    if not isinstance(eos, list) or tuple(eos) != expected_eos:
        raise ValueError(f"generation_config eos_token_id must be {expected_eos!r}, got {eos!r}")
    if config.get("use_cache") is not True:
        raise ValueError("generation_config use_cache must be true")


def load_generation_config(model_dir: str | Path) -> dict[str, object]:
    path = Path(model_dir) / "generation_config.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("generation_config.json root must be an object")
    validate_generation_config(value)
    return value


__all__ = [
    "RESOURCE_SHA256",
    "load_checked_resources",
    "load_generation_config",
    "validate_generation_config",
    "validate_mtp_source_keys",
    "validate_resources",
    "validate_source_keys",
]

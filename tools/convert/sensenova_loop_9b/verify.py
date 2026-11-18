"""Structural verification for a SenseNova-Loop-9B .ninfer artifact."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    object_alignment,
)

from . import inventory


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class StructureSummary:
    objects: int
    tensors: int
    resources: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class LogicalSummary:
    physical_layers: int
    logical_visits: int
    full_attention_layers: int
    gdn_layers: int
    mhc_layers: int
    mtp_layers: int


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    identity: dict[str, str]
    structure: StructureSummary
    logical: LogicalSummary
    formats: dict[str, int]
    layouts: dict[str, int]


class VerificationError(ValueError):
    """The artifact does not match the registered SenseNova target contract."""


def verify(artifact: Artifact) -> VerificationSummary:
    expected_names = [spec.name for spec in inventory.OBJECT_SPECS]
    actual_names = [obj.name for obj in artifact.objects]
    if actual_names != expected_names:
        raise VerificationError("artifact object inventory does not match the target")
    if artifact.identity != ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID):
        raise VerificationError("artifact identity does not match the target")

    tensors = [obj for obj in artifact.objects if isinstance(obj, TensorObject)]
    resources = [obj for obj in artifact.objects if isinstance(obj, ResourceObject)]
    for obj in artifact.objects:
        alignment = object_alignment(obj)
        if obj.offset % alignment:
            raise VerificationError(f"object {obj.name} has unaligned offset")
        if obj.bytes <= 0:
            raise VerificationError(f"object {obj.name} has nonpositive size")

    expected_by_name = {
        spec.name: spec
        for spec in inventory.OBJECT_SPECS
        if isinstance(spec, inventory.StoredObjectSpec) and hasattr(spec, "shape")
    }
    for obj in tensors:
        spec = expected_by_name[obj.name]
        if obj.shape != spec.shape or obj.format != spec.format or obj.layout != spec.layout:
            raise VerificationError(f"tensor {obj.name} does not match its inventory spec")

    payload_bytes = artifact.objects[-1].offset + artifact.objects[-1].bytes
    return VerificationSummary(
        identity={
            "model_id": artifact.identity.model_id,
            "weights_id": artifact.identity.weights_id,
        },
        structure=StructureSummary(
            objects=len(artifact.objects),
            tensors=len(tensors),
            resources=len(resources),
            payload_bytes=payload_bytes,
        ),
        logical=LogicalSummary(
            physical_layers=len(inventory.TEXT_LAYERS),
            logical_visits=len(inventory.LOOP_PLAN),
            full_attention_layers=len(inventory.FULL_ATTENTION_LAYERS),
            gdn_layers=len(inventory.GDN_LAYERS),
            mhc_layers=len(inventory.MHC_LAYERS),
            mtp_layers=1,
        ),
        formats=dict(sorted(Counter(obj.format for obj in tensors).items())),
        layouts=dict(sorted(Counter(obj.layout for obj in tensors).items())),
    )


def verify_file(path: str | Path) -> VerificationSummary:
    with Artifact.open(path) as artifact:
        return verify(artifact)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    summary = verify_file(args.artifact)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

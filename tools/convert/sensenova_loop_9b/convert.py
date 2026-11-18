"""Convert SenseNova-Loop-9B into the text + MTP groupwise-int artifact.

Canonical invocation::

    python -m tools.convert.sensenova_loop_9b.convert \
      --model /home/user/models/sensenova/SenseNova-Loop-9B \
      --out out/sensenova-loop-9b.ninfer

The converter stores one copy of each physical layer. The fixed two-pass loop
is a runtime contract. MTP draft tensors are read from `mtp.safetensors`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import draft_head, inventory, recipe, resources


RECIPE_ID = "sensenova-loop-9b-text-mtp-v1"
ENCODER_PROFILE = "MAXABS_F16_RECIP_RNE_V1"

_ROOT_CONFIG = {
    "architectures": ["SenseNovaLoopForConditionalGeneration"],
    "model_type": "sensenova_loop",
    "hidden_size": 4096,
    "tie_word_embeddings": False,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
_LOOP_CONFIG = {"start": 4, "end": 28, "eval_count": 2, "counts": [2]}
_MHC_CONFIG = {
    "start": 4,
    "end": 28,
    "streams": 2,
    "iterations": 20,
    "fixed_identity": False,
    "pre_scale": 1.0,
}
_TEXT_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "attn_output_gate": True,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "vocab_size": 248320,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
}
_ROPE_CONFIG = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 262144,
}
_VISION_CONFIG = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 4096,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [],
}


ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: recipe.SourcePreflight
    mtp_source: recipe.SourcePreflight
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate every checkpoint fact that fixes storage or execution shape."""

    family_conversion.check_members("config", config, _ROOT_CONFIG)
    loop = config.get("loop_settings")
    mhc = config.get("mhc")
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(loop, Mapping) or not isinstance(mhc, Mapping):
        raise ValueError("config.json must contain loop_settings and mhc")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("config.json must contain text_config and vision_config")

    family_conversion.check_members("config.loop_settings", loop, _LOOP_CONFIG)
    family_conversion.check_members("config.mhc", mhc, _MHC_CONFIG)
    family_conversion.check_members("config.text_config", text, _TEXT_CONFIG)

    expected_layer_types = tuple(
        "full_attention" if layer in inventory.FULL_ATTENTION_LAYERS else "linear_attention"
        for layer in inventory.TEXT_LAYERS
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError("text_config.layer_types does not match the 32-layer schedule")

    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    family_conversion.check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    family_conversion.check_members("config.vision_config", vision, _VISION_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "loop": {name: loop[name] for name in _LOOP_CONFIG},
        "mhc": {name: mhc[name] for name in _MHC_CONFIG},
        "text": {name: text[name] for name in _TEXT_CONFIG},
        "layer_types": {
            "layers": len(layer_types),
            "full_attention_layers": list(inventory.FULL_ATTENTION_LAYERS),
            "linear_attention_layers": list(inventory.GDN_LAYERS),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": {name: vision[name] for name in _VISION_CONFIG},
    }


def preflight_inventory() -> None:
    counts = (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    )
    if counts != (6, 427, 2, 12, 441, 447):
        raise ValueError(f"target inventory is incomplete: {counts}")
    expected_formats = {
        inventory.BF16: 160,
        inventory.FP32: 144,
        inventory.I32: 1,
        inventory.Q4: 1,
        inventory.Q5: 0,
        inventory.Q6: 1,
        inventory.W8: 134,
    }
    if inventory.FORMAT_COUNTS != expected_formats:
        raise ValueError(f"target format counts drifted: {inventory.FORMAT_COUNTS}")
    recipe.validate_recipe_coverage()


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    return resources.load_checked_resources(model_dir)


def build_object_plan(resources_map: Mapping[str, bytes]) -> ObjectPlan:
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources_map)


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    model = Path(model_dir)
    config = family_conversion.load_json(model / "config.json")
    config_summary = validate_config(config)
    preflight_inventory()

    with ShardReader.from_index(model / "model.safetensors.index.json") as reader:
        base_source = recipe.preflight_base_sources(model)
        resources.validate_source_keys(reader.names, recipe.source_requirements(recipe.BASE_RECIPES))
    with ShardReader.from_file(model / recipe.MTP_FILE) as reader:
        mtp_source = recipe.preflight_mtp_sources(model)
        resources.validate_mtp_source_keys(reader.names, recipe.source_requirements(recipe.MTP_RECIPES))
    resources.load_generation_config(model)

    frontend = load_resources(model)
    resource_map = {item.name: item.data for item in frontend}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=base_source,
        mtp_source=mtp_source,
        resources=frontend,
        draft=draft,
        object_plan=object_plan,
    )


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    return family_conversion.encode_tensor_payload(tensor, spec, device)


def build_conversion_report(
    *,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    base_source_preflight: recipe.SourcePreflight,
    mtp_source_preflight: recipe.SourcePreflight,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path,
    revision: str | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    combined_source = recipe.SourcePreflight(
        recipe_count=base_source_preflight.recipe_count + mtp_source_preflight.recipe_count,
        source_tensor_count=(
            base_source_preflight.source_tensor_count + mtp_source_preflight.source_tensor_count
        ),
        source_shard_count=(
            base_source_preflight.source_shard_count + mtp_source_preflight.source_shard_count
        ),
        source_dtype_counts={
            "BF16": (
                base_source_preflight.source_dtype_counts.get("BF16", 0)
                + mtp_source_preflight.source_dtype_counts.get("BF16", 0)
            )
        },
    )
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary=dict(config_summary),
        source_preflight=combined_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking_path,
        revision=revision,
        environment_summary=environment,
    )
    report["source_preflight"] = {
        "base": {
            "recipes": base_source_preflight.recipe_count,
            "tensors": base_source_preflight.source_tensor_count,
            "shards": base_source_preflight.source_shard_count,
            "dtypes": dict(base_source_preflight.source_dtype_counts),
        },
        "mtp": {
            "recipes": mtp_source_preflight.recipe_count,
            "tensors": mtp_source_preflight.source_tensor_count,
            "files": mtp_source_preflight.source_shard_count,
            "dtypes": dict(mtp_source_preflight.source_dtype_counts),
        },
        "combined": {
            "recipes": combined_source.recipe_count,
            "tensors": combined_source.source_tensor_count,
            "files": combined_source.source_shard_count,
            "dtypes": dict(combined_source.source_dtype_counts),
        },
    }
    report["draft_head"] = {
        "rows": inventory.DRAFT_HEAD_ROWS,
        "tokenizer_vocab_size": inventory.TOKENIZER_DOMAIN,
        "ranking_source_target": "qwen3_6_27b",
        "shared_semantic_vocabulary": True,
    }
    report["quantization"] = {
        "encoder_profile": ENCODER_PROFILE,
        "physical_layers": len(inventory.TEXT_LAYERS),
        "logical_visits": len(inventory.LOOP_PLAN),
    }
    return report


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the complete conversion and return its report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model)
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"base={preflight.source.source_tensor_count} source tensors, "
        f"mtp={preflight.mtp_source.source_tensor_count} source tensors, "
        f"device={resolved_device}",
        flush=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    resource_map = {item.name: item.data for item in preflight.resources}
    with ArtifactWriter(
        output,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        preflight.object_plan.specs,
    ) as writer:
        index = 0

        def write_payload(spec, payload: bytes) -> None:
            nonlocal index
            writer.write(spec.name, payload)
            index += 1
            print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)

        for spec in inventory.RESOURCE_SPECS:
            write_payload(spec, resource_map[spec.name])

        with ShardReader.from_index(model / "model.safetensors.index.json") as base_reader, \
             ShardReader.from_file(model / recipe.MTP_FILE) as mtp_reader:
            for spec in inventory.TENSOR_SPECS:
                tensor = recipe.materialize(spec, base_reader, mtp_reader, preflight.draft)
                payload = encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
                write_payload(spec, payload)
                del payload

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = build_conversion_report(
        model_dir=model,
        out_path=output,
        arguments={
            "model": str(model_dir),
            "out": str(out_path),
            "device": requested_device,
        },
        config_summary=preflight.config_summary,
        base_source_preflight=preflight.source,
        mtp_source_preflight=preflight.mtp_source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()

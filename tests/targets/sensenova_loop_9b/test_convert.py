"""Fixture tests for the SenseNova-Loop-9B converter.

These tests use generated safetensors fixtures and do not touch the real
19 GB checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


import tools.convert.sensenova_loop_9b.convert as MODULE  # noqa: E402
from tools.convert.sensenova_loop_9b import inventory, recipe, resources  # noqa: E402


CONFIG = {
    "architectures": ["SenseNovaLoopForConditionalGeneration"],
    "model_type": "sensenova_loop",
    "hidden_size": 4096,
    "tie_word_embeddings": False,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
    "loop_settings": {"start": 4, "end": 28, "eval_count": 2, "counts": [2]},
    "mhc": {
        "start": 4,
        "end": 28,
        "streams": 2,
        "iterations": 20,
        "fixed_identity": False,
        "pre_scale": 1.0,
    },
    "text_config": {
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
        "layer_types": [
            "full_attention" if layer in inventory.FULL_ATTENTION_LAYERS else "linear_attention"
            for layer in range(32)
        ],
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 262144,
        },
    },
    "vision_config": {
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
    },
}


class TensorReader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def test_inventory_and_loop_plan_are_fixed() -> None:
    assert inventory.MODEL_ID == "sensenova-loop-9b"
    assert inventory.WEIGHTS_ID == "groupwise-int"
    assert inventory.LOOP_PLAN == tuple(range(4)) + tuple(range(4, 28)) * 2 + tuple(range(28, 32))
    assert len(inventory.LOOP_PLAN) == 56
    assert inventory.FULL_ATTENTION_LAYERS == (3, 7, 11, 15, 19, 23, 27, 31)
    assert inventory.MHC_LAYERS == tuple(range(4, 28))
    assert "text/layers/4/mhc/attn/weight" in {spec.name for spec in inventory.TENSOR_SPECS}
    assert "text/layers/3/mhc/attn/weight" not in {spec.name for spec in inventory.TENSOR_SPECS}


def test_config_validation_accepts_expected_and_rejects_dynamic_loop() -> None:
    summary = MODULE.validate_config(CONFIG)
    assert summary["loop"]["counts"] == [2]
    bad = json.loads(json.dumps(CONFIG))
    bad["loop_settings"] = {
        "start": 4,
        "end": 28,
        "eval_count": 2,
        "counts": [1, 2],
    }
    with pytest.raises(ValueError, match="loop"):
        MODULE.validate_config(bad)


def test_source_key_validation_ignores_vision_but_rejects_unknown_text_key() -> None:
    required = {"model.language_model.embed_tokens.weight", "lm_head.weight"}
    accepted = set(required) | {"model.visual.pos_embed.weight", "model.language_model.loop_gate.logits"}
    resources.validate_source_keys(accepted, required)
    with pytest.raises(ValueError, match="unexpected source tensors"):
        resources.validate_source_keys(accepted | {"model.language_model.unexpected"}, required)


def test_attention_recipe_extracts_query_key_gate_value_rows() -> None:
    prefix = "model.language_model.layers.3.self_attn."
    q = torch.arange(8192, dtype=torch.int32).view(-1, 1).expand(-1, 4096)
    k = torch.full((1024, 1), -1, dtype=torch.int32).expand(-1, 4096)
    v = torch.full((1024, 1), -2, dtype=torch.int32).expand(-1, 4096)
    fused = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/3/attention/query_key_gate_value"],
        TensorReader(
            {
                prefix + "q_proj.weight": q,
                prefix + "k_proj.weight": k,
                prefix + "v_proj.weight": v,
            }
        ),
    )
    query_rows = torch.cat(
        [torch.arange(head * 512, head * 512 + 256) for head in range(16)]
    ).to(torch.int32)
    gate_rows = torch.cat(
        [torch.arange(head * 512 + 256, head * 512 + 512) for head in range(16)]
    ).to(torch.int32)
    assert fused.shape == (10240, 4096)
    assert torch.equal(fused[:4096, 0], query_rows)
    assert torch.all(fused[4096:5120, 0] == -1)
    assert torch.equal(fused[5120:9216, 0], gate_rows)
    assert torch.all(fused[9216:, 0] == -2)


def test_gdn_recipe_uses_four_kilo_output_width() -> None:
    prefix = "model.language_model.layers.0.linear_attn."
    a = torch.full((32, 1), 1, dtype=torch.float32).expand(-1, 4096)
    b = torch.full((32, 1), 2, dtype=torch.float32).expand(-1, 4096)
    ab = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/gdn/a_b_projection"],
        TensorReader({prefix + "in_proj_a.weight": a, prefix + "in_proj_b.weight": b}),
    )
    assert ab.shape == (64, 4096)
    assert torch.all(ab[:32] == 1)
    assert torch.all(ab[32:] == 2)


def test_mhc_recipe_keeps_fp32_sources() -> None:
    prefix = "model.language_model.layers.4."
    weight = torch.arange(8 * 8192, dtype=torch.float32).reshape(8, 8192)
    bias = torch.arange(8, dtype=torch.float32)
    materialized = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/4/mhc/attn/weight"],
        TensorReader({prefix + "mhc_attn.weight": weight}),
    )
    assert materialized.dtype == torch.float32
    assert torch.equal(materialized, weight)


def test_resource_loader_uses_sensenova_hashes(tmp_path: Path) -> None:
    for name in inventory.RESOURCE_SPECS:
        (tmp_path / name.name.removeprefix("frontend/")).write_bytes(b"fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        resources.load_checked_resources(tmp_path)

"""Hugging Face source recipe for SenseNova-Loop-9B text + MTP."""

from __future__ import annotations

from pathlib import Path

import torch

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common.draft_head import (
    DraftHeadContext,
    materialize_draft_head,
    materialize_draft_head_token_ids,
)
from tools.convert.qwen3_6.common.recipe import (
    SOURCE_DTYPE,
    Cast,
    Concat,
    DraftHeadTokenIds,
    Expression,
    GatherRows,
    Reshape,
    Slice,
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    Transpose,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    preflight_source_reader,
    source,
    source_requirements as _source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import draft_head, inventory


DRAFT_HEAD_OBJECT = "text/draft_head"
DRAFT_HEAD_TOKEN_IDS_OBJECT = "text/draft_head_token_ids"
MTP_FILE = "mtp.safetensors"


def _attention_part(source_name: str, *, gate: bool) -> Expression:
    """Extract 16 Q-head query or output-gate rows from head-interleaved Q."""

    per_head = Reshape(source(source_name, (inventory.ATTENTION_Q_ROWS, inventory.HIDDEN)), (16, 512, inventory.HIDDEN))
    begin = 256 if gate else 0
    return Reshape(
        Slice(per_head, 1, begin, begin + 256),
        (16 * 256, inventory.HIDDEN),
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            source("model.language_model.embed_tokens.weight", (inventory.VOCAB_ROWS, inventory.HIDDEN)),
        )
    ]

    for layer in inventory.TEXT_LAYERS:
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                source(source_prefix + "input_layernorm.weight", (inventory.HIDDEN,)),
            )
        )

        if layer in inventory.FULL_ATTENTION_LAYERS:
            q_proj = source_prefix + "self_attn.q_proj.weight"
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        Concat(
                            (
                                _attention_part(q_proj, gate=False),
                                source(
                                    source_prefix + "self_attn.k_proj.weight",
                                    (inventory.ATTENTION_KV_ROWS, inventory.HIDDEN),
                                ),
                                _attention_part(q_proj, gate=True),
                                source(
                                    source_prefix + "self_attn.v_proj.weight",
                                    (inventory.ATTENTION_KV_ROWS, inventory.HIDDEN),
                                ),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/query_norm",
                        source(source_prefix + "self_attn.q_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/key_norm",
                        source(source_prefix + "self_attn.k_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/output",
                        source(
                            source_prefix + "self_attn.o_proj.weight",
                            (inventory.HIDDEN, inventory.ATTENTION_OUTPUT_K),
                        ),
                    ),
                )
            )
        else:
            convolution = source(
                source_prefix + "linear_attn.conv1d.weight",
                (inventory.GDN_KVQ_ROWS, 1, 4),
            )
            qkv = source(
                source_prefix + "linear_attn.in_proj_qkv.weight",
                (inventory.GDN_KVQ_ROWS, inventory.HIDDEN),
            )
            z = source(
                source_prefix + "linear_attn.in_proj_z.weight",
                (inventory.GDN_Z_ROWS, inventory.HIDDEN),
            )
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(SourceTensor(source_prefix + "linear_attn.A_log", (32,), dtype="F32"), inventory.FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(SourceTensor(source_prefix + "linear_attn.dt_bias", (32,), dtype="F32"), inventory.FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(Reshape(Slice(convolution, 1, 0, 1), (inventory.GDN_KVQ_ROWS, 4)), (1, 0)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_b_projection",
                        Concat(
                            (
                                source(source_prefix + "linear_attn.in_proj_a.weight", (32, inventory.HIDDEN)),
                                source(source_prefix + "linear_attn.in_proj_b.weight", (32, inventory.HIDDEN)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/query_key_value_z",
                        Concat((qkv, z), 0),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/norm",
                        source(source_prefix + "linear_attn.norm.weight", (128,)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        source(
                            source_prefix + "linear_attn.out_proj.weight",
                            (inventory.HIDDEN, inventory.GDN_OUTPUT_K),
                        ),
                    ),
                )
            )

        if layer in inventory.MHC_LAYERS:
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "mhc/attn/weight",
                        SourceTensor(source_prefix + "mhc_attn.weight", (inventory.MHC_LOGITS, 2 * inventory.HIDDEN), dtype="F32"),
                    ),
                    TensorRecipe(
                        object_prefix + "mhc/attn/bias",
                        SourceTensor(source_prefix + "mhc_attn.bias", (inventory.MHC_LOGITS,), dtype="F32"),
                    ),
                    TensorRecipe(
                        object_prefix + "mhc/ffn/weight",
                        SourceTensor(source_prefix + "mhc_ffn.weight", (inventory.MHC_LOGITS, 2 * inventory.HIDDEN), dtype="F32"),
                    ),
                    TensorRecipe(
                        object_prefix + "mhc/ffn/bias",
                        SourceTensor(source_prefix + "mhc_ffn.bias", (inventory.MHC_LOGITS,), dtype="F32"),
                    ),
                )
            )

        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    source(source_prefix + "post_attention_layernorm.weight", (inventory.HIDDEN,)),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            source(source_prefix + "mlp.gate_proj.weight", (inventory.INTERMEDIATE, inventory.HIDDEN)),
                            source(source_prefix + "mlp.up_proj.weight", (inventory.INTERMEDIATE, inventory.HIDDEN)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    source(
                        source_prefix + "mlp.down_proj.weight",
                        (inventory.HIDDEN, inventory.INTERMEDIATE),
                    ),
                ),
            )
        )

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                source("model.language_model.norm.weight", (inventory.HIDDEN,)),
            ),
            TensorRecipe(
                "text/output_head",
                source("lm_head.weight", (inventory.VOCAB_ROWS, inventory.HIDDEN)),
            ),
        )
    )
    return tuple(recipes)


def _build_draft_head_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(
            DRAFT_HEAD_OBJECT,
            GatherRows(
                source("lm_head.weight", (inventory.VOCAB_ROWS, inventory.HIDDEN)),
                token_ids_object=DRAFT_HEAD_TOKEN_IDS_OBJECT,
                rows=inventory.DRAFT_HEAD_ROWS,
            ),
        ),
        TensorRecipe(
            DRAFT_HEAD_TOKEN_IDS_OBJECT,
            DraftHeadTokenIds(
                ranking_path=draft_head.DEFAULT_RANKING,
                tokenizer_resource="frontend/tokenizer_config.json",
                vocab_rows=inventory.VOCAB_ROWS,
                tokenizer_id_count=inventory.TOKENIZER_DOMAIN,
                rows=inventory.DRAFT_HEAD_ROWS,
            ),
        ),
    )


def _build_mtp_recipes() -> tuple[TensorRecipe, ...]:
    source_prefix = "mtp.layers.0."
    object_prefix = "mtp/layer/"
    q_proj = source_prefix + "self_attn.q_proj.weight"
    return (
        TensorRecipe(
            "mtp/input_projection",
            source("mtp.fc.weight", (inventory.HIDDEN, 2 * inventory.HIDDEN)),
        ),
        TensorRecipe(
            "mtp/embedding_norm",
            source("mtp.pre_fc_norm_embedding.weight", (inventory.HIDDEN,)),
        ),
        TensorRecipe(
            "mtp/hidden_norm",
            source("mtp.pre_fc_norm_hidden.weight", (inventory.HIDDEN,)),
        ),
        TensorRecipe(
            object_prefix + "input_norm",
            source(source_prefix + "input_layernorm.weight", (inventory.HIDDEN,)),
        ),
        TensorRecipe(
            object_prefix + "attention/query_key_gate_value",
            Concat(
                (
                    _attention_part(q_proj, gate=False),
                    source(source_prefix + "self_attn.k_proj.weight", (inventory.ATTENTION_KV_ROWS, inventory.HIDDEN)),
                    _attention_part(q_proj, gate=True),
                    source(source_prefix + "self_attn.v_proj.weight", (inventory.ATTENTION_KV_ROWS, inventory.HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "attention/query_norm",
            source(source_prefix + "self_attn.q_norm.weight", (256,)),
        ),
        TensorRecipe(
            object_prefix + "attention/key_norm",
            source(source_prefix + "self_attn.k_norm.weight", (256,)),
        ),
        TensorRecipe(
            object_prefix + "attention/output",
            source(source_prefix + "self_attn.o_proj.weight", (inventory.HIDDEN, inventory.ATTENTION_OUTPUT_K)),
        ),
        TensorRecipe(
            object_prefix + "post_attention_norm",
            source(source_prefix + "post_attention_layernorm.weight", (inventory.HIDDEN,)),
        ),
        TensorRecipe(
            object_prefix + "mlp/gate_up",
            Concat(
                (
                    source(source_prefix + "mlp.gate_proj.weight", (inventory.INTERMEDIATE, inventory.HIDDEN)),
                    source(source_prefix + "mlp.up_proj.weight", (inventory.INTERMEDIATE, inventory.HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "mlp/down",
            source(source_prefix + "mlp.down_proj.weight", (inventory.HIDDEN, inventory.INTERMEDIATE)),
        ),
        TensorRecipe("mtp/final_norm", source("mtp.norm.weight", (inventory.HIDDEN,))),
    )


BASE_RECIPES = _build_text_recipes() + _build_draft_head_recipes()
MTP_RECIPES = _build_mtp_recipes()
ALL_RECIPES = BASE_RECIPES + MTP_RECIPES
BASE_RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in BASE_RECIPES}
MTP_RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in MTP_RECIPES}
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in ALL_RECIPES}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(ALL_RECIPES, inventory.TENSOR_SPECS)


def source_requirements(
    recipes: tuple[TensorRecipe, ...] = ALL_RECIPES,
) -> dict[str, SourceTensor]:
    return _source_requirements(recipes)


def preflight_base_sources(model_dir: str | Path) -> SourcePreflight:
    with ShardReader.from_index(Path(model_dir) / "model.safetensors.index.json") as reader:
        return preflight_source_reader(reader, BASE_RECIPES)


def preflight_mtp_sources(model_dir: str | Path) -> SourcePreflight:
    with ShardReader.from_file(Path(model_dir) / MTP_FILE) as reader:
        return preflight_source_reader(reader, MTP_RECIPES)


def materialize(
    spec,
    base_reader: ShardReader,
    mtp_reader: ShardReader,
    draft: DraftHeadContext | None = None,
):
    if spec.name == DRAFT_HEAD_OBJECT or spec.name == DRAFT_HEAD_TOKEN_IDS_OBJECT:
        if draft is None:
            raise ValueError("draft-head context is required for derived draft objects")
        derived = {
            DRAFT_HEAD_TOKEN_IDS_OBJECT: materialize_draft_head_token_ids(draft),
        }
    else:
        derived = None
    reader = mtp_reader if spec.name in MTP_RECIPES_BY_NAME and spec.name not in BASE_RECIPES_BY_NAME else base_reader
    recipe = RECIPES_BY_NAME[spec.name]
    if spec.name == DRAFT_HEAD_OBJECT and draft is not None:
        lm_head = materialize_expression(recipe.expression.source, base_reader, derived)
        return materialize_draft_head(lm_head, draft)
    return materialize_recipe(recipe, reader, derived)


__all__ = [
    "ALL_RECIPES",
    "BASE_RECIPES",
    "BASE_RECIPES_BY_NAME",
    "DRAFT_HEAD_OBJECT",
    "DRAFT_HEAD_TOKEN_IDS_OBJECT",
    "MTP_FILE",
    "MTP_RECIPES",
    "MTP_RECIPES_BY_NAME",
    "SourcePreflight",
    "materialize",
    "preflight_base_sources",
    "preflight_mtp_sources",
    "source_requirements",
    "validate_recipe_coverage",
]

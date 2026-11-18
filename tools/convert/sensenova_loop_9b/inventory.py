"""Persistent-object inventory for SenseNova-Loop-9B.

The artifact stores 32 physical layers. The fixed two-pass execution plan is a
runtime contract, not a reason to duplicate weights. Each physical layer has
its own mHC routers; the runtime gives each logical visit its own state slot.
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FORMAT_NAMES,
    FP32,
    I32,
    LAYOUT_NAMES,
    Q4,
    Q5,
    Q6,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    W8,
    tensor_spec,
)


MODEL_ID = "sensenova-loop-9b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "sensenova_loop_9b"

TEXT_LAYERS = tuple(range(32))
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
GDN_LAYERS = tuple(layer for layer in TEXT_LAYERS if layer not in FULL_ATTENTION_LAYERS)
MHC_LAYERS = tuple(range(4, 28))
LOOP_PLAN = tuple(range(4)) + tuple(range(4, 28)) * 2 + tuple(range(28, 32))

DRAFT_HEAD_ROWS = 131072
VOCAB_ROWS = 248320
TOKENIZER_DOMAIN = 248077
HIDDEN = 4096
INTERMEDIATE = 12288
MHC_LOGITS = 8
MHC_INPUT = 2 * HIDDEN
ATTENTION_Q_ROWS = 8192
ATTENTION_KV_ROWS = 1024
ATTENTION_OUTPUT_K = HIDDEN
GDN_KVQ_ROWS = 8192
GDN_Z_ROWS = HIDDEN
GDN_A_B_ROWS = 64
GDN_OUTPUT_K = HIDDEN

def _tensor(name: str, shape: tuple[int, ...], numeric_format: str):
    return tensor_spec(name, shape, numeric_format)


def _build_text_core_specs() -> tuple:
    specs: list = [
        _tensor("text/token_embedding", (VOCAB_ROWS, HIDDEN), W8),
    ]

    for layer in TEXT_LAYERS:
        prefix = f"text/layers/{layer}/"
        specs.append(_tensor(prefix + "input_norm", (HIDDEN,), BF16))

        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    _tensor(
                        prefix + "attention/query_key_gate_value",
                        (10240, HIDDEN),
                        W8,
                    ),
                    _tensor(prefix + "attention/query_norm", (256,), BF16),
                    _tensor(prefix + "attention/key_norm", (256,), BF16),
                    _tensor(prefix + "attention/output", (HIDDEN, ATTENTION_OUTPUT_K), W8),
                )
            )
        else:
            specs.extend(
                (
                    _tensor(prefix + "gdn/a_log", (32,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (32,), FP32),
                    _tensor(prefix + "gdn/convolution", (4, 8192), BF16),
                    _tensor(prefix + "gdn/a_b_projection", (GDN_A_B_ROWS, HIDDEN), BF16),
                    _tensor(
                        prefix + "gdn/query_key_value_z",
                        (GDN_KVQ_ROWS + GDN_Z_ROWS, HIDDEN),
                        W8,
                    ),
                    _tensor(prefix + "gdn/norm", (128,), BF16),
                    _tensor(prefix + "gdn/output", (HIDDEN, GDN_OUTPUT_K), W8),
                )
            )

        if layer in MHC_LAYERS:
            specs.extend(
                (
                    _tensor(prefix + "mhc/attn/weight", (MHC_LOGITS, MHC_INPUT), FP32),
                    _tensor(prefix + "mhc/attn/bias", (MHC_LOGITS,), FP32),
                    _tensor(prefix + "mhc/ffn/weight", (MHC_LOGITS, MHC_INPUT), FP32),
                    _tensor(prefix + "mhc/ffn/bias", (MHC_LOGITS,), FP32),
                )
            )

        specs.extend(
            (
                _tensor(prefix + "post_attention_norm", (HIDDEN,), BF16),
                _tensor(prefix + "mlp/gate_up", (2 * INTERMEDIATE, HIDDEN), W8),
                _tensor(prefix + "mlp/down", (HIDDEN, INTERMEDIATE), W8),
            )
        )

    specs.extend(
        (
            _tensor("text/final_norm", (HIDDEN,), BF16),
            _tensor("text/output_head", (VOCAB_ROWS, HIDDEN), Q6),
        )
    )
    return tuple(specs)


def _build_draft_head_specs() -> tuple:
    return (
        _tensor("text/draft_head", (DRAFT_HEAD_ROWS, HIDDEN), Q4),
        _tensor("text/draft_head_token_ids", (DRAFT_HEAD_ROWS,), I32),
    )


def _build_mtp_specs() -> tuple:
    return (
        _tensor("mtp/input_projection", (HIDDEN, 2 * HIDDEN), W8),
        _tensor("mtp/embedding_norm", (HIDDEN,), BF16),
        _tensor("mtp/hidden_norm", (HIDDEN,), BF16),
        _tensor("mtp/layer/input_norm", (HIDDEN,), BF16),
        _tensor(
            "mtp/layer/attention/query_key_gate_value",
            (10240, HIDDEN),
            W8,
        ),
        _tensor("mtp/layer/attention/query_norm", (256,), BF16),
        _tensor("mtp/layer/attention/key_norm", (256,), BF16),
        _tensor("mtp/layer/attention/output", (HIDDEN, ATTENTION_OUTPUT_K), W8),
        _tensor("mtp/layer/post_attention_norm", (HIDDEN,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (2 * INTERMEDIATE, HIDDEN), W8),
        _tensor("mtp/layer/mlp/down", (HIDDEN, INTERMEDIATE), W8),
        _tensor("mtp/final_norm", (HIDDEN,), BF16),
    )


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
TENSOR_SPECS = TEXT_CORE_TENSOR_SPECS + DRAFT_HEAD_TENSOR_SPECS + MTP_TENSOR_SPECS
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

__all__ = [
    "ATTENTION_KV_ROWS",
    "ATTENTION_Q_ROWS",
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DIRECT_FORMATS",
    "DRAFT_HEAD_ROWS",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "HIDDEN",
    "I32",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "LOOP_PLAN",
    "MHC_LAYERS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "TARGET_KEY",
    "TEXT_CORE_TENSOR_SPECS",
    "TEXT_LAYERS",
    "TENSOR_SPECS",
    "TOKENIZER_DOMAIN",
    "VOCAB_ROWS",
    "W8",
]

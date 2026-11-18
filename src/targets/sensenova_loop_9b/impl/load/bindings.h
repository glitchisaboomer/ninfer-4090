#pragma once

#include <ninfer/targets/sensenova_loop_9b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::sensenova_loop_9b::detail {

inline constexpr std::size_t kPhysicalLayers = 32;
inline constexpr std::size_t kFullAttentionLayers = 8;
inline constexpr std::size_t kGdnLayers = 24;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

struct TensorPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

struct AttentionPlan {
    WeightPlan query_key_gate_value;
    TensorPlan query_norm;
    TensorPlan key_norm;
    WeightPlan output;
};

struct MhcPlan {
    TensorPlan weight;
    TensorPlan bias;
};

struct GdnPlan {
    TensorPlan a_log;
    TensorPlan dt_bias;
    TensorPlan convolution;
    WeightPlan a_b_projection;
    WeightPlan query_key_value_z;
    TensorPlan norm;
    WeightPlan output;
};

struct TextLayerPlan {
    TensorPlan input_norm;
    AttentionPlan attention;
    GdnPlan gdn;
    MhcPlan mhc_attn;
    MhcPlan mhc_ffn;
    bool is_full_attention = false;
    bool has_mhc           = false;
    TensorPlan post_attention_norm;
    WeightPlan mlp_gate_up;
    WeightPlan mlp_down;
};

struct MtpPlan {
    WeightPlan input_projection;
    TensorPlan embedding_norm;
    TensorPlan hidden_norm;
    TensorPlan input_norm;
    AttentionPlan attention;
    TensorPlan post_attention_norm;
    WeightPlan mlp_gate_up;
    WeightPlan mlp_down;
    TensorPlan final_norm;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    WeightPlan token_embedding;
    std::array<TextLayerPlan, kPhysicalLayers> text_layers;
    TensorPlan final_norm;
    WeightPlan output_head;
    WeightPlan draft_head;
    TensorPlan draft_head_token_ids;
    MtpPlan mtp;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

[[nodiscard]] ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile);

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::sensenova_loop_9b::detail

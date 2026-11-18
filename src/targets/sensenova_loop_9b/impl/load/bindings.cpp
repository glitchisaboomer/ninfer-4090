#include "targets/sensenova_loop_9b/impl/load/bindings.h"

#include "artifact/typed_binding.h"
#include "targets/sensenova_loop_9b/impl/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::sensenova_loop_9b::detail {
namespace {

using artifact::NumericFormat;

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

TensorPlan bind_tensor(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    return TensorPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

Tensor bind_tensor_view(const artifact::MaterializedArtifact& artifact, const TensorPlan& plan,
                        std::initializer_list<std::int32_t> shape) {
    return artifact::materialized_tensor(artifact, plan.object, plan.format, shape);
}

Weight bind_weight_view(const artifact::MaterializedArtifact& artifact, const WeightPlan& plan,
                        std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(artifact, plan.object, plan.format, rows, columns);
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    const auto bytes = binder.payload(handle).data;
    std::vector<bool> seen(TextConfig::token_domain, false);
    for (int i = 0; i < kDraftHeadRows; ++i) {
        const std::byte* value = bytes.data() + static_cast<std::size_t>(i) * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= static_cast<std::uint32_t>(TextConfig::token_domain) || seen[id]) {
            throw artifact::ArtifactError("invalid SenseNova optimized draft-head token ids");
        }
        seen[id] = true;
    }
}

AttentionPlan bind_attention(artifact::Binder& binder, std::string_view prefix) {
    return AttentionPlan{
        .query_key_gate_value =
            bind_weight(binder, std::string(prefix) + "attention/query_key_gate_value",
                        NumericFormat::W8G32_F16S,
                        {10240, static_cast<std::uint64_t>(TextConfig::hidden)}),
        .query_norm = bind_tensor(binder, std::string(prefix) + "attention/query_norm",
                                  NumericFormat::BF16, {256}),
        .key_norm = bind_tensor(binder, std::string(prefix) + "attention/key_norm",
                                NumericFormat::BF16, {256}),
        .output = bind_weight(binder, std::string(prefix) + "attention/output",
                              NumericFormat::W8G32_F16S,
                              {TextConfig::hidden, TextConfig::hidden}),
    };
}

FullAttentionWeights load_attention(const AttentionPlan& plan,
                                    const artifact::MaterializedArtifact& artifact) {
    return FullAttentionWeights{
        .query_key_gate_value =
            bind_weight_view(artifact, plan.query_key_gate_value, 10240, TextConfig::hidden),
        .query_norm        = bind_tensor_view(artifact, plan.query_norm, {256}),
        .key_norm          = bind_tensor_view(artifact, plan.key_norm, {256}),
        .output = bind_weight_view(artifact, plan.output, TextConfig::hidden, TextConfig::hidden),
    };
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile) {
    if (weights_profile != WeightsProfile::GroupwiseInt) {
        throw std::invalid_argument("sensenova_loop_9b: invalid weights profile");
    }

    ArtifactLoadPlan load_plan;
    BindingPlan& out   = load_plan.bindings;
    out.frontend       = qwen3_6::bind_frontend_resources(binder);
    out.token_embedding = bind_weight(binder, "text/token_embedding", NumericFormat::W8G32_F16S,
                                      {TextConfig::output_rows, TextConfig::hidden});

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < TextConfig::layers; ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        TextLayerPlan& target    = out.text_layers[layer];
        target.input_norm = bind_tensor(binder, prefix + "input_norm", NumericFormat::BF16,
                                        {static_cast<std::uint64_t>(TextConfig::hidden)});
        target.is_full_attention = TextConfig::is_full_attention(static_cast<int>(layer));
        if (target.is_full_attention) {
            target.attention = bind_attention(binder, prefix);
            ++full_index;
        } else {
            target.gdn = GdnPlan{
                .a_log = bind_tensor(binder, prefix + "gdn/a_log", NumericFormat::FP32, {32}),
                .dt_bias =
                    bind_tensor(binder, prefix + "gdn/dt_bias", NumericFormat::FP32, {32}),
                .convolution = bind_tensor(binder, prefix + "gdn/convolution", NumericFormat::BF16,
                                           {4, 8192}),
                .a_b_projection =
                    bind_weight(binder, prefix + "gdn/a_b_projection", NumericFormat::BF16,
                                {64, static_cast<std::uint64_t>(TextConfig::hidden)}),
                .query_key_value_z =
                    bind_weight(binder, prefix + "gdn/query_key_value_z", NumericFormat::W8G32_F16S,
                                {12288, static_cast<std::uint64_t>(TextConfig::hidden)}),
                .norm = bind_tensor(binder, prefix + "gdn/norm", NumericFormat::BF16, {128}),
                .output = bind_weight(binder, prefix + "gdn/output", NumericFormat::W8G32_F16S,
                                      {TextConfig::hidden, TextConfig::hidden}),
            };
            ++gdn_index;
        }

        target.has_mhc = TextConfig::is_mhc_layer(static_cast<int>(layer));
        if (target.has_mhc) {
            target.mhc_attn = MhcPlan{
                .weight = bind_tensor(binder, prefix + "mhc/attn/weight", NumericFormat::FP32,
                                      {8, 8192}),
                .bias = bind_tensor(binder, prefix + "mhc/attn/bias", NumericFormat::FP32, {8}),
            };
            target.mhc_ffn = MhcPlan{
                .weight = bind_tensor(binder, prefix + "mhc/ffn/weight", NumericFormat::FP32,
                                      {8, 8192}),
                .bias = bind_tensor(binder, prefix + "mhc/ffn/bias", NumericFormat::FP32, {8}),
            };
        }

        target.post_attention_norm =
            bind_tensor(binder, prefix + "post_attention_norm", NumericFormat::BF16,
                        {static_cast<std::uint64_t>(TextConfig::hidden)});
        target.mlp_gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::W8G32_F16S,
                        {2 * 12288, static_cast<std::uint64_t>(TextConfig::hidden)});
        target.mlp_down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::W8G32_F16S,
                        {TextConfig::hidden, 12288});
    }
    if (full_index != kFullAttentionLayers || gdn_index != kGdnLayers) {
        throw std::logic_error("sensenova_loop_9b: text topology binding is incomplete");
    }

    out.final_norm = bind_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                 {static_cast<std::uint64_t>(TextConfig::hidden)});
    out.output_head = bind_weight(binder, "text/output_head", NumericFormat::Q6G64_F16S,
                                  {TextConfig::output_rows, TextConfig::hidden});
    out.draft_head = bind_weight(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                 {kDraftHeadRows, TextConfig::hidden});
    out.draft_head_token_ids = bind_tensor(binder, "text/draft_head_token_ids", NumericFormat::I32,
                                           {kDraftHeadRows});
    validate_draft_ids(binder, out.draft_head_token_ids.object);

    out.mtp = MtpPlan{
        .input_projection =
            bind_weight(binder, "mtp/input_projection", NumericFormat::W8G32_F16S,
                        {TextConfig::hidden, 2 * TextConfig::hidden}),
        .embedding_norm = bind_tensor(binder, "mtp/embedding_norm", NumericFormat::BF16,
                                      {static_cast<std::uint64_t>(TextConfig::hidden)}),
        .hidden_norm = bind_tensor(binder, "mtp/hidden_norm", NumericFormat::BF16,
                                   {static_cast<std::uint64_t>(TextConfig::hidden)}),
        .input_norm = bind_tensor(binder, "mtp/layer/input_norm", NumericFormat::BF16,
                                  {static_cast<std::uint64_t>(TextConfig::hidden)}),
        .attention = bind_attention(binder, "mtp/layer/"),
        .post_attention_norm = bind_tensor(binder, "mtp/layer/post_attention_norm",
                                           NumericFormat::BF16,
                                           {static_cast<std::uint64_t>(TextConfig::hidden)}),
        .mlp_gate_up = bind_weight(binder, "mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S,
                                   {2 * 12288, static_cast<std::uint64_t>(TextConfig::hidden)}),
        .mlp_down = bind_weight(binder, "mtp/layer/mlp/down", NumericFormat::W8G32_F16S,
                                {TextConfig::hidden, 12288}),
        .final_norm = bind_tensor(binder, "mtp/final_norm", NumericFormat::BF16,
                                  {static_cast<std::uint64_t>(TextConfig::hidden)}),
    };

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena     = &backing.device_arena();
    runtime.token_embedding   = bind_weight_view(backing, plan.token_embedding,
                                                 TextConfig::output_rows, TextConfig::hidden);
    runtime.output_head       = bind_weight_view(backing, plan.output_head,
                                                 TextConfig::output_rows, TextConfig::hidden);
    runtime.draft_head        = bind_weight_view(backing, plan.draft_head, kDraftHeadRows,
                                                 TextConfig::hidden);
    runtime.draft_head_token_ids =
        bind_tensor_view(backing, plan.draft_head_token_ids, {kDraftHeadRows});
    runtime.final_norm = bind_tensor_view(backing, plan.final_norm, {TextConfig::hidden});

    for (std::size_t layer = 0; layer < TextConfig::layers; ++layer) {
        TextLayerWeights& target = runtime.layers[layer];
        const TextLayerPlan& source = plan.text_layers[layer];
        target.input_norm = bind_tensor_view(backing, source.input_norm, {TextConfig::hidden});
        target.is_full_attention = source.is_full_attention;
        target.has_mhc           = source.has_mhc;
        if (source.is_full_attention) {
            target.attention = load_attention(source.attention, backing);
        } else {
            target.gdn = GdnWeights{
                .a_log    = bind_tensor_view(backing, source.gdn.a_log, {32}),
                .dt_bias  = bind_tensor_view(backing, source.gdn.dt_bias, {32}),
                .convolution = bind_tensor_view(backing, source.gdn.convolution, {4, 8192}),
                .a_b_projection = bind_weight_view(backing, source.gdn.a_b_projection, 64,
                                                   TextConfig::hidden),
                .query_key_value_z = bind_weight_view(backing, source.gdn.query_key_value_z, 12288,
                                                      TextConfig::hidden),
                .norm   = bind_tensor_view(backing, source.gdn.norm, {128}),
                .output = bind_weight_view(backing, source.gdn.output, TextConfig::hidden,
                                           TextConfig::hidden),
            };
        }
        if (source.has_mhc) {
            target.mhc_attn = MhcRouterWeights{
                .weight = bind_tensor_view(backing, source.mhc_attn.weight, {8, 8192}),
                .bias   = bind_tensor_view(backing, source.mhc_attn.bias, {8}),
            };
            target.mhc_ffn = MhcRouterWeights{
                .weight = bind_tensor_view(backing, source.mhc_ffn.weight, {8, 8192}),
                .bias   = bind_tensor_view(backing, source.mhc_ffn.bias, {8}),
            };
        }
        target.post_attention_norm =
            bind_tensor_view(backing, source.post_attention_norm, {TextConfig::hidden});
        target.mlp_gate_up = bind_weight_view(backing, source.mlp_gate_up, 2 * 12288,
                                              TextConfig::hidden);
        target.mlp_down = bind_weight_view(backing, source.mlp_down, TextConfig::hidden, 12288);
    }

    runtime.mtp = MtpWeights{
        .input_projection = bind_weight_view(backing, plan.mtp.input_projection,
                                             TextConfig::hidden, 2 * TextConfig::hidden),
        .embedding_norm =
            bind_tensor_view(backing, plan.mtp.embedding_norm, {TextConfig::hidden}),
        .hidden_norm = bind_tensor_view(backing, plan.mtp.hidden_norm, {TextConfig::hidden}),
        .input_norm  = bind_tensor_view(backing, plan.mtp.input_norm, {TextConfig::hidden}),
        .attention   = load_attention(plan.mtp.attention, backing),
        .post_attention_norm =
            bind_tensor_view(backing, plan.mtp.post_attention_norm, {TextConfig::hidden}),
        .mlp_gate_up = bind_weight_view(backing, plan.mtp.mlp_gate_up, 2 * 12288,
                                        TextConfig::hidden),
        .mlp_down = bind_weight_view(backing, plan.mtp.mlp_down, TextConfig::hidden, 12288),
        .final_norm = bind_tensor_view(backing, plan.mtp.final_norm, {TextConfig::hidden}),
    };
}

} // namespace ninfer::targets::sensenova_loop_9b::detail

#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "runtime/contract/types.h"
#include "targets/sensenova_loop_9b/impl/runtime_stub_fwd.h"
#include <ninfer/targets/qwen3_6/frontend.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceArena;
struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::sensenova_loop_9b {

struct Package;

namespace detail {

enum class WeightsProfile : std::uint8_t { GroupwiseInt };

struct FullAttentionWeights {
    Weight query_key_gate_value;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
};

struct MhcRouterWeights {
    Tensor weight;
    Tensor bias;
};

struct GdnWeights {
    Tensor a_log;
    Tensor dt_bias;
    Tensor convolution;
    Weight a_b_projection;
    Weight query_key_value_z;
    Tensor norm;
    Weight output;
};

struct TextLayerWeights {
    Tensor input_norm;
    FullAttentionWeights attention;
    GdnWeights gdn;
    MhcRouterWeights mhc_attn;
    MhcRouterWeights mhc_ffn;
    bool is_full_attention       = false;
    bool has_mhc                 = false;
    Tensor post_attention_norm;
    Weight mlp_gate_up;
    Weight mlp_down;
};

struct MtpWeights {
    Weight input_projection;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Tensor input_norm;
    FullAttentionWeights attention;
    Tensor post_attention_norm;
    Weight mlp_gate_up;
    Weight mlp_down;
    Tensor final_norm;
};

struct RuntimeModelView {
    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<TextLayerWeights, 32> layers;
    Tensor final_norm;
    Weight output_head;
    Weight draft_head;
    Tensor draft_head_token_ids;
    MtpWeights mtp;
};

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct sensenova_loop_9b::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct sensenova_loop_9b::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "sensenova-loop-9b";
    static constexpr std::string_view target_key = "sensenova_loop_9b";
    static constexpr std::string_view weights_id = "groupwise-int";

    using WeightsProfile  = detail::WeightsProfile;
    using LoadPlan        = detail::LoadPlan;
    using LoadedModel     = detail::LoadedModel;
    using Frontend        = qwen3_6::Frontend;
    using PreparedPrompt  = qwen3_6::PreparedPrompt;
    using OutputSession   = qwen3_6::OutputSession;
    using SequencePlanner = detail::SequencePlanner;
    using SequencePlan    = detail::SequencePlan;
    using RequestBasePlan = detail::RequestBasePlan;
    using RequestPlan     = detail::RequestPlan;
    using Program         = detail::Program;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::sensenova_loop_9b
} // namespace ninfer

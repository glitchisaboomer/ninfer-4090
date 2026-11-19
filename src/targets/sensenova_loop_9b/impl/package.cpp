#include <ninfer/targets/sensenova_loop_9b/package.h>
#include <ninfer/targets/qwen3_6/frontend.h>

#include "artifact/reader.h"
#include "targets/sensenova_loop_9b/impl/load/bindings.h"
#include "targets/sensenova_loop_9b/impl/runtime_stub.h"
#include "targets/sensenova_loop_9b/impl/runtime_stub_internal.h"
#include "targets/sensenova_loop_9b/impl/config.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::sensenova_loop_9b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;
const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::sensenova_loop_9b::detail

namespace ninfer::targets::sensenova_loop_9b {
namespace {

using detail::make_curve;

constexpr ModelSamplingDefaults kSamplingDefaults{
    .thinking = {.temperature = 0.6F, .top_k = 20, .top_p = 0.95F},
    .non_thinking = {.temperature = 0.6F, .top_k = 20, .top_p = 0.95F},
};

detail::SequencePlanner make_planner(DeviceContext& device, const EngineOptions& options,
                             detail::WeightsProfile weights_profile) {
    (void)device;
    return detail::SequencePlanner::create(options, weights_profile);
}

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kSamplingDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == weights_id) {
        return WeightsProfile::GroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions&,
                                     WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, detail::bind_artifact(binder, weights_profile)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.impl_->data.frontend, false, 0);
}

detail::SequencePlanner Package::make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                                               WeightsProfile weights_profile) {
    return make_planner(device, options, weights_profile);
}

std::unique_ptr<detail::Program> Package::create_program(const LoadedModel& model, SequencePlan&& plan,
                                                 DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return detail::Program::create(model.impl_->data.runtime, device.stream, plan.max_concurrency(),
                                   plan.main_page_groups(), plan.capacity(),
                                   plan.device_reservation_bytes(), plan.use_cuda_graph(),
                                   plan.draft_window());
}

} // namespace ninfer::targets::sensenova_loop_9b

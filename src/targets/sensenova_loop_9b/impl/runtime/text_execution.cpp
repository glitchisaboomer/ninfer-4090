#include "targets/sensenova_loop_9b/impl/runtime/text_execution.h"

#include <stdexcept>

namespace ninfer::targets::sensenova_loop_9b::detail {

namespace {

// Start of the GDN region: immediately after all 15 KV domains of `kv_capacity`
// tokens (BF16, no scales — the only KV format this target stores).
constexpr std::size_t gdn_region_offset(std::uint32_t kv_capacity) noexcept {
    return StateImageLayout::total_kv_bytes(kv_capacity, false, false);
}

// Start of the mHC region: after the KV domains and the full GDN region
// (matching the Program's state image binding in runtime_stub).
constexpr std::size_t mhc_region_offset(std::uint32_t kv_capacity) noexcept {
    return gdn_region_offset(kv_capacity) +
           StateImageLayout::kGdnSlots * StateLayout::gdn_state_bytes_per_slot();
}

} // namespace

TextExecutionContext::TextExecutionContext(const RuntimeModelView& model, cudaStream_t stream,
                                           std::byte* state_image_base,
                                           std::uint32_t kv_capacity, WorkspaceArena& workspace)
    : model_(model), stream_(stream), state_image_base_(state_image_base),
      kv_capacity_(kv_capacity), workspace_(workspace) {
    if (stream == nullptr) {
        throw std::invalid_argument("TextExecutionContext requires a non-null CUDA stream");
    }
    if (state_image_base == nullptr) {
        throw std::invalid_argument(
            "TextExecutionContext requires a non-null state image base");
    }
    if (kv_capacity == 0) {
        throw std::invalid_argument("TextExecutionContext requires a positive KV capacity");
    }
    if (workspace.capacity() == 0) {
        throw std::invalid_argument(
            "TextExecutionContext requires a workspace with positive capacity");
    }
}

std::byte* TextExecutionContext::gdn_state_base(std::size_t visit) const {
    if (visit >= StateImageLayout::kGdnSlots) {
        throw std::out_of_range("SenseNova GDN visit slot is out of range");
    }
    return state_image_base_ + gdn_region_offset(kv_capacity_) +
           StateImageLayout::gdn_offset(visit);
}

std::byte* TextExecutionContext::mhc_state_base(std::size_t visit) const {
    if (visit >= StateImageLayout::kMhcSlots) {
        throw std::out_of_range("SenseNova mHC visit slot is out of range");
    }
    return state_image_base_ + mhc_region_offset(kv_capacity_) +
           StateImageLayout::mhc_offset(visit);
}

std::size_t TextExecutionContext::kv_domain(std::size_t visit) const noexcept {
    return StateImageLayout::kv_domain_for_visit(visit);
}

} // namespace ninfer::targets::sensenova_loop_9b::detail
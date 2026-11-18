#pragma once

#include "targets/sensenova_loop_9b/impl/state_layout.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::sensenova_loop_9b::detail {

// Stable, host-computable addresses for every logical visit state. This type deliberately does not
// own device memory: the Program allocates one backing span and binds these regions into it.
// Keeping the arithmetic here makes aliasing and loop-visit identity testable without CUDA.
class StateImageLayout {
public:
    static constexpr std::size_t kKvDomains =
        StateLayout::full_attention_state_slots + StateLayout::mtp_kv_domains;
    static constexpr std::size_t kGdnSlots = StateLayout::gdn_state_slots;
    static constexpr std::size_t kMhcSlots = StateLayout::mhc_state_slots;

    [[nodiscard]] static constexpr std::size_t kv_bytes_per_domain(std::size_t tokens,
                                                                   bool int8_kv,
                                                                   bool with_scales) noexcept {
        return tokens * StateLayout::kv_bytes_per_token(int8_kv, with_scales);
    }

    [[nodiscard]] static constexpr std::size_t total_kv_bytes(std::size_t tokens, bool int8_kv,
                                                              bool with_scales) noexcept {
        return kKvDomains * kv_bytes_per_domain(tokens, int8_kv, with_scales);
    }

    [[nodiscard]] static constexpr std::size_t kv_offset(std::size_t domain, std::size_t tokens,
                                                         bool int8_kv, bool with_scales) noexcept {
        return domain * kv_bytes_per_domain(tokens, int8_kv, with_scales);
    }

    [[nodiscard]] static constexpr std::size_t gdn_offset(std::size_t slot) noexcept {
        return slot * StateLayout::gdn_state_bytes_per_slot();
    }

    [[nodiscard]] static constexpr std::size_t mhc_offset(std::size_t slot) noexcept {
        return slot * StateLayout::mhc_state_bytes_per_slot();
    }

    [[nodiscard]] static constexpr std::size_t fixed_bytes() noexcept {
        return kGdnSlots * StateLayout::gdn_state_bytes_per_slot() +
               kMhcSlots * StateLayout::mhc_state_bytes_per_slot();
    }

    [[nodiscard]] static constexpr std::size_t total_bytes(std::size_t tokens, bool int8_kv,
                                                           bool with_scales) noexcept {
        return total_kv_bytes(tokens, int8_kv, with_scales) + fixed_bytes();
    }

    [[nodiscard]] static constexpr std::size_t kv_domain_for_visit(std::size_t visit) noexcept {
        // full-attention visits are assigned compactly in visit order; the single MTP domain is
        // deliberately last and is never aliased with a text visit.
        return visit;
    }

    [[nodiscard]] static constexpr std::size_t gdn_slot_for_visit(std::size_t visit) noexcept {
        return visit;
    }

    [[nodiscard]] static constexpr std::size_t mhc_slot_for_visit(std::size_t visit) noexcept {
        return visit;
    }
};

} // namespace ninfer::targets::sensenova_loop_9b::detail

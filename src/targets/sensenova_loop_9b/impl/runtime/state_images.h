#pragma once

#include "targets/sensenova_loop_9b/impl/state_layout.h"

#include "core/paged_kv_delta_layout.h"

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

    // Internal layout of one KV domain. Each domain holds two page-major planes -- all K pages,
    // then all V pages -- in the exact tensor shape the paged GQA views consume:
    // [head_dim, kKvPageTokens, kv_heads, pages]. kKvPageTokens must match the engine-wide GQA
    // page size: the kernels derive logical pages and page offsets from absolute positions with a
    // compile-time 64-token shift. Both halves of StateLayout's per-token KV bytes (K and V,
    // 4 heads x 256) fit exactly, so a domain is attended with no copy or repacking. Changing the
    // page size changes only the number of pages, never the total KV state bytes.
    static constexpr std::size_t kKvPageTokens = static_cast<std::size_t>(kPagedKVPageSize);

    [[nodiscard]] static constexpr std::size_t kv_pages(std::size_t tokens) noexcept {
        return tokens / kKvPageTokens;
    }

    // Bytes of one plane (K only, or V only) for a domain of `tokens` tokens. Requires `tokens`
    // to be a whole number of pages; the domain layout reserves exactly two of these.
    [[nodiscard]] static constexpr std::size_t kv_plane_bytes(std::size_t tokens) noexcept {
        return tokens * StateLayout::kv_heads * StateLayout::head_dim * sizeof(std::uint16_t);
    }

    static_assert(2 * kKvPageTokens * StateLayout::kv_heads * StateLayout::head_dim *
                          sizeof(std::uint16_t) ==
                      kKvPageTokens * StateLayout::kv_bytes_per_token(false, false),
                  "KV domain planes must tile the per-token KV bytes exactly");

    [[nodiscard]] static constexpr std::size_t kv_domain_for_visit(std::size_t visit) noexcept {
        // Full-attention visits are assigned compactly in visit order. The single MTP domain is
        // deliberately last and is never aliased with a text visit. Non-attention visits have no
        // KV domain; callers must check LoopPlan::is_full_attention(visit) before using this.
        std::size_t domain = 0;
        for (std::size_t v = 0; v < visit; ++v) {
            if (LoopPlan::is_full_attention(v)) { ++domain; }
        }
        return domain;
    }

    // GDN and mHC slots are compact: one per visit of that kind, assigned in visit order. The
    // slot count (42 and 48) is therefore smaller than the 56 visits, and indexing the region with
    // a raw visit number would run past its end for the final 14 visits. A full-attention visit
    // has no GDN slot and a non-mHC visit has no mHC slot; callers must gate on
    // LoopPlan::is_gdn / LoopPlan::is_mhc first.
    [[nodiscard]] static constexpr std::size_t gdn_slot_for_visit(std::size_t visit) noexcept {
        std::size_t slot = 0;
        for (std::size_t v = 0; v < visit; ++v) {
            if (LoopPlan::is_gdn(v)) { ++slot; }
        }
        return slot;
    }

    [[nodiscard]] static constexpr std::size_t mhc_slot_for_visit(std::size_t visit) noexcept {
        std::size_t slot = 0;
        for (std::size_t v = 0; v < visit; ++v) {
            if (LoopPlan::is_mhc(v)) { ++slot; }
        }
        return slot;
    }
};

} // namespace ninfer::targets::sensenova_loop_9b::detail

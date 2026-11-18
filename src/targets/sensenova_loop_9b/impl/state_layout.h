#pragma once

#include "targets/sensenova_loop_9b/impl/config.h"
#include "targets/sensenova_loop_9b/impl/loop_plan.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::sensenova_loop_9b::detail {

// Static state identities and byte accounting for the fixed 56-visit plan. This is deliberately
// independent of DeviceArena/PagedKV layout types so the mapping can be tested host-only.
struct StateLayout {
    static constexpr std::size_t full_attention_state_slots = LoopPlan::full_attention_visits();
    static constexpr std::size_t gdn_state_slots           = LoopPlan::gdn_visits();
    static constexpr std::size_t mhc_state_slots           = LoopPlan::mhc_visits();
    static constexpr std::size_t mtp_kv_domains            = 1;

    static constexpr std::size_t kv_heads       = TextConfig::kv_heads;
    static constexpr std::size_t head_dim       = TextConfig::head_dim;
    static constexpr std::size_t qk_dim         = kv_heads * head_dim;
    static constexpr std::size_t gdn_key_dim    = TextConfig::key_dim;
    static constexpr std::size_t gdn_value_dim  = TextConfig::value_dim;
    static constexpr std::size_t gdn_conv_width = TextConfig::convolution_dim;
    static constexpr std::size_t kv_quant_group       = 64;
    static constexpr std::size_t kv_scale_element_size = sizeof(float);

    [[nodiscard]] static constexpr std::size_t kv_bytes_per_token(bool int8_kv,
                                                                   bool with_scales) noexcept {
        const std::size_t element_bytes = int8_kv ? 1 : 2;
        const std::size_t scale_bytes =
            int8_kv && with_scales ? 2 * qk_dim / kv_quant_group * kv_scale_element_size : 0;
        return 2 * qk_dim * element_bytes + scale_bytes;
    }

    [[nodiscard]] static constexpr std::size_t gdn_state_bytes_per_slot() noexcept {
        constexpr std::size_t recurrent_f32 = 4 * gdn_value_dim * sizeof(float);
        constexpr std::size_t conv_bf16     = gdn_conv_width * sizeof(std::uint16_t);
        return recurrent_f32 + conv_bf16;
    }

    [[nodiscard]] static constexpr std::size_t mhc_state_bytes_per_slot() noexcept {
        return MhcConfig::input_width * sizeof(std::uint16_t);
    }

    [[nodiscard]] static constexpr std::size_t total_state_slots() noexcept {
        return full_attention_state_slots + gdn_state_slots + mhc_state_slots;
    }

    [[nodiscard]] static constexpr std::size_t total_fixed_state_bytes() noexcept {
        return gdn_state_slots * gdn_state_bytes_per_slot() +
               mhc_state_slots * mhc_state_bytes_per_slot();
    }

    [[nodiscard]] static constexpr std::size_t total_kv_bytes_per_token(bool int8_kv,
                                                                        bool with_scales,
                                                                        bool include_mtp) noexcept {
        const std::size_t text =
            full_attention_state_slots * kv_bytes_per_token(int8_kv, with_scales);
        return include_mtp ? text + mtp_kv_domains * kv_bytes_per_token(int8_kv, with_scales)
                           : text;
    }
};

static_assert(StateLayout::full_attention_state_slots == 14);
static_assert(StateLayout::gdn_state_slots == 42);
static_assert(StateLayout::mhc_state_slots == 48);
static_assert(StateLayout::total_state_slots() == 104);

} // namespace ninfer::targets::sensenova_loop_9b::detail

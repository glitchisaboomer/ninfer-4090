#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::sensenova_loop_9b::detail {

// Exact source-contract constants for SenseNova-Loop-9B. The converter validates the same values
// in `tools/convert/sensenova_loop_9b`; this header is the runtime-side copy of that contract.
struct TextConfig {
    static constexpr int hidden       = 4096;
    static constexpr int layers       = 32;
    static constexpr int intermediate = 12288;

    // The LM head is padded to 248320 rows. Tokenizer-addressable IDs stop at 248077.
    static constexpr int output_rows  = 248320;
    static constexpr int token_domain = 248077;

    // Generation-config end-of-sequence ids: <|endoftext|> and <|im_end|>. Both lie inside the
    // tokenizer-addressable domain, so the head can sample them; the target passes a sampled id
    // through unchanged and the request's output session decides to stop. Neither is ever masked
    // out of a round.
    static constexpr int end_of_text_token    = 248044;
    static constexpr int end_of_message_token = 248046;

    static constexpr int gdn_conv_kernel      = 4;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = 16;
    static constexpr int gdn_key_head_dim     = 128;
    static constexpr int gdn_value_heads      = 32;
    static constexpr int gdn_value_head_dim   = 128;

    static constexpr int query_heads = 16;
    static constexpr int kv_heads    = 4;
    static constexpr int head_dim    = 256;
    static constexpr int rotary_dim  = 64;

    static constexpr int full_attention_interval = 4;
    static constexpr float rms_epsilon           = 1.0e-6F;
    static constexpr float rope_theta            = 1.0e7F;

    static constexpr int key_dim               = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim             = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim       = 2 * key_dim + value_dim;
    static constexpr int query_size            = query_heads * head_dim;
    static constexpr int kv_size               = kv_heads * head_dim;
    static constexpr int query_projection_rows = query_heads * head_dim +
                                                 kv_heads * head_dim;

    static constexpr int mtp_layers               = 1;
    static constexpr int mtp_input_rows           = 2 * hidden;
    static constexpr int mtp_attention_input_rows = 2 * query_size + 2 * kv_size;
    static constexpr int mtp_mlp_gate_up_rows     = 2 * intermediate;

    static constexpr std::array<int, 8> full_attention_layer_ids{
        3, 7, 11, 15, 19, 23, 27, 31};

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        return (layer + 1) % full_attention_interval == 0;
    }

    [[nodiscard]] static constexpr bool is_mhc_layer(int layer) {
        return layer >= 4 && layer < 28;
    }

    [[nodiscard]] static constexpr int full_attention_layers() {
        return static_cast<int>(full_attention_layer_ids.size());
    }

    [[nodiscard]] static constexpr int gdn_layers() {
        return layers - full_attention_layers();
    }

    [[nodiscard]] static constexpr int mhc_layers() { return 24; }

    [[nodiscard]] static constexpr int full_attention_index(int layer) {
        return layer / full_attention_interval;
    }

    [[nodiscard]] static constexpr int gdn_index(int layer) {
        return layer - full_attention_index(layer);
    }
};

struct MhcConfig {
    static constexpr int streams            = 2;
    static constexpr int iterations         = 20;
    static constexpr int logits             = 8;
    static constexpr int hidden             = TextConfig::hidden;
    static constexpr int input_width        = streams * hidden;
    static constexpr float pre_scale        = 1.0F;
    static constexpr bool fixed_identity    = false;
    static constexpr int start_physical     = 4;
    static constexpr int end_physical       = 28;
};

inline constexpr float kAttentionScale = 1.0F / static_cast<float>(TextConfig::head_dim);
inline constexpr std::uint32_t kMaximumMtpDraftTokens = 15;
inline constexpr std::uint32_t kNativeContext         = 262144;
inline constexpr int kDraftHeadRows                   = 131072;

static_assert(TextConfig::end_of_text_token < TextConfig::token_domain);
static_assert(TextConfig::end_of_message_token < TextConfig::token_domain);
static_assert(TextConfig::full_attention_layers() == 8);
static_assert(TextConfig::gdn_layers() == 24);
static_assert(TextConfig::mhc_layers() == 24);
static_assert(TextConfig::output_rows > TextConfig::token_domain);
static_assert(MhcConfig::input_width == 8192);

} // namespace ninfer::targets::sensenova_loop_9b::detail

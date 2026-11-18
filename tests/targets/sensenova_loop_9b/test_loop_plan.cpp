#include "targets/sensenova_loop_9b/impl/config.h"
#include "targets/sensenova_loop_9b/impl/loop_plan.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

namespace sn = ninfer::targets::sensenova_loop_9b::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_config() {
    static_assert(sn::TextConfig::hidden == 4096);
    static_assert(sn::TextConfig::layers == 32);
    static_assert(sn::TextConfig::intermediate == 12288);
    static_assert(sn::TextConfig::query_heads == 16);
    static_assert(sn::TextConfig::kv_heads == 4);
    static_assert(sn::TextConfig::head_dim == 256);
    static_assert(sn::TextConfig::rotary_dim == 64);
    static_assert(sn::TextConfig::gdn_key_heads == 16);
    static_assert(sn::TextConfig::gdn_value_heads == 32);
    static_assert(sn::TextConfig::gdn_key_head_dim == 128);
    static_assert(sn::TextConfig::gdn_conv_kernel == 4);
    static_assert(sn::TextConfig::output_rows == 248320);
    static_assert(sn::TextConfig::token_domain == 248077);
    static_assert(sn::kNativeContext == 262144);
    static_assert(sn::kMaximumMtpDraftTokens == 15);
    static_assert(sn::kDraftHeadRows == 131072);
    static_assert(sn::MhcConfig::streams == 2);
    static_assert(sn::MhcConfig::iterations == 20);
    static_assert(sn::MhcConfig::logits == 8);
    static_assert(sn::MhcConfig::input_width == 8192);

    std::array<int, 8> expected_full{3, 7, 11, 15, 19, 23, 27, 31};
    expect(sn::TextConfig::full_attention_layer_ids == expected_full, "full-attention layer IDs");
    for (int layer = 0; layer < sn::TextConfig::layers; ++layer) {
        const bool expected = (layer + 1) % 4 == 0;
        expect(sn::TextConfig::is_full_attention(layer) == expected, "layer kind");
        expect(sn::TextConfig::is_mhc_layer(layer) == (layer >= 4 && layer < 28), "mHC layer range");
    }
}

void test_exact_plan() {
    static_assert(sn::LoopPlan::size() == 56);
    static_assert(sn::LoopPlan::pass_count() == 2);

    std::vector<int> expected;
    for (int layer = 0; layer < 4; ++layer) { expected.push_back(layer); }
    for (int layer = 4; layer < 28; ++layer) { expected.push_back(layer); }
    for (int layer = 4; layer < 28; ++layer) { expected.push_back(layer); }
    for (int layer = 28; layer < 32; ++layer) { expected.push_back(layer); }

    expect(expected.size() == sn::LoopPlan::size(), "expected plan length");
    for (std::size_t visit = 0; visit < sn::LoopPlan::size(); ++visit) {
        expect(sn::LoopPlan::physical_layer(visit) == expected[visit], "visit sequence");
        expect(sn::LoopPlan::state_slot(visit) == visit, "state slot identity");
    }

    expect(sn::LoopPlan::physical_layer(0) == 0, "first visit");
    expect(sn::LoopPlan::physical_layer(55) == 31, "last visit");
    expect(sn::LoopPlan::physical_layer(4) == 4 && sn::LoopPlan::physical_layer(27) == 27,
           "first loop pass");
    expect(sn::LoopPlan::physical_layer(28) == 4 && sn::LoopPlan::physical_layer(51) == 27,
           "second loop pass");
    expect(sn::LoopPlan::pass(3) == 0 && sn::LoopPlan::pass(4) == 0, "pass zero");
    expect(sn::LoopPlan::pass(27) == 0 && sn::LoopPlan::pass(28) == 1, "pass transition");
    expect(sn::LoopPlan::pass(51) == 1 && sn::LoopPlan::pass(52) == 0, "loop exit");
}

void test_counts() {
    std::size_t full = 0;
    std::size_t gdn  = 0;
    std::size_t mhc  = 0;
    std::array<int, 32> visits{};
    for (std::size_t visit = 0; visit < sn::LoopPlan::size(); ++visit) {
        full += sn::LoopPlan::is_full_attention(visit) ? 1U : 0U;
        gdn += sn::LoopPlan::is_gdn(visit) ? 1U : 0U;
        mhc += sn::LoopPlan::is_mhc(visit) ? 1U : 0U;
        ++visits[static_cast<std::size_t>(sn::LoopPlan::physical_layer(visit))];
    }
    expect(full == 14 && sn::LoopPlan::full_attention_visits() == 14, "full-attention visit count");
    expect(gdn == 42 && sn::LoopPlan::gdn_visits() == 42, "GDN visit count");
    expect(mhc == 48 && sn::LoopPlan::mhc_visits() == 48, "mHC visit count");
    expect(full + gdn == 56, "mixer visit partition");

    for (int layer : {3, 31}) {
        expect(visits[layer] == 1, "single-visit non-looped layer");
    }
    for (int layer = 4; layer < 28; ++layer) {
        expect(visits[layer] == 2, "looped physical layer visited twice");
    }
    for (int layer : {7, 11, 15, 19, 23, 27}) {
        expect(visits[layer] == 2, "looped full-attention layer visited twice");
    }
}

void test_state_layout() {
    static_assert(sn::StateLayout::full_attention_state_slots == 14);
    static_assert(sn::StateLayout::gdn_state_slots == 42);
    static_assert(sn::StateLayout::mhc_state_slots == 48);
    static_assert(sn::StateLayout::total_state_slots() == 104);

    constexpr std::size_t bf16_kv = sn::StateLayout::kv_bytes_per_token(false, false);
    constexpr std::size_t i8_kv   = sn::StateLayout::kv_bytes_per_token(true, true);
    expect(bf16_kv == 2 * 4 * 256 * 2, "BF16 KV bytes per token");
    expect(i8_kv == 2 * 4 * 256 * 1 + 2 * (4 * 256 / 64) * sizeof(float),
           "INT8 KV bytes per token");
    expect(sn::StateLayout::kv_bytes_per_token(true, false) ==
               2 * 4 * 256 * 1,
           "INT8 KV without scale planes");
    expect(sn::StateLayout::total_kv_bytes_per_token(false, false, false) == 14 * bf16_kv,
           "text KV uses 14 logical visits");
    expect(sn::StateLayout::total_kv_bytes_per_token(false, false, true) == 15 * bf16_kv,
           "MTP adds one KV domain");
    expect(sn::StateLayout::total_fixed_state_bytes() ==
               42 * sn::StateLayout::gdn_state_bytes_per_slot() +
                   48 * sn::StateLayout::mhc_state_bytes_per_slot(),
           "fixed GDN/mHC state accounting");
}

} // namespace

int main() {
    test_config();
    test_exact_plan();
    test_counts();
    test_state_layout();
    if (failures != 0) {
        std::cerr << "FAIL senseNova loop plan: " << failures << '\n';
        return 1;
    }
    std::cout << "OK senseNova loop plan\n";
    return 0;
}

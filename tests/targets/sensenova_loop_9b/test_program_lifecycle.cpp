#include "targets/sensenova_loop_9b/impl/runtime_stub.h"
#include "targets/sensenova_loop_9b/impl/runtime/state_images.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/sensenova_loop_9b/package.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace sn      = ninfer::targets::sensenova_loop_9b::detail;
namespace qwen3_6 = ninfer::targets::qwen3_6;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_lifecycle() {
    auto program = sn::Program::create(2, 32, 4096, 1U << 30U);
    const auto capacity = program->admission_capacity();
    expect(capacity.active_lanes == 2 && capacity.main_kv_pages == 32, "admission capacity");
    expect(program->config_signature_slug().find("sensenova_loop_9b") != std::string::npos,
           "config signature");

    auto plan = sn::RequestPlanForTest::make();

    expect(program->can_admit_lane(0, plan), "empty lane admits");
    expect(program->can_admit_lane(2, plan) == false, "out-of-range lane rejects");

    auto active = sn::Program::inspect_lane_for_test(*program, 0);
    expect(!active.occupied && !active.retained, "lane starts empty");
}

void test_state_images() {
    auto program = sn::Program::create(1, 16, 2048, 1U << 20U);
    const auto images = sn::Program::inspect_state_images_for_test(*program);
    expect(images.base != nullptr, "state images are device-backed");
    expect(images.bytes == images.kv_bytes + images.gdn_bytes + images.mhc_bytes,
           "state image regions cover the allocation");
    expect(images.kv_bytes == sn::StateImageLayout::total_kv_bytes(16 * 128, false, false),
           "KV region uses declared KV capacity");
    expect(images.gdn_bytes == 42 * sn::StateLayout::gdn_state_bytes_per_slot(),
           "GDN region covers 42 visit slots");
    expect(images.mhc_bytes == 48 * sn::StateLayout::mhc_state_bytes_per_slot(),
           "mHC region covers 48 visit slots");
    expect(images.gdn_region != images.mhc_region && images.gdn_region != images.mtp_kv_region,
           "state domains do not alias");

    sn::Program::write_state_image_markers_for_test(*program, 0xa5);
    expect(sn::Program::verify_state_image_markers_for_test(*program, 0xa5),
           "state image marker write/verify");
    expect(!sn::Program::verify_state_image_markers_for_test(*program, 0x5a),
           "state image marker mismatch is observed");
    sn::Program::write_state_image_markers_for_test(*program, 0);

    const auto again = sn::Program::inspect_state_images_for_test(*program);
    expect(again.base == images.base && again.gdn_region == images.gdn_region &&
               again.mhc_region == images.mhc_region,
           "state addresses are stable across access");
}

void test_prefill_requires_model() {
    auto program = sn::Program::create(1, 32, 4096, 1U << 20U);

    // The test factory has no loaded model. An empty prompt must not be the
    // reason this fails: the missing model is reported first, and no lane state
    // is mutated.
    qwen3_6::PreparedPrompt prompt;
    auto plan = sn::RequestPlanForTest::make();
    bool reported_missing_model = false;
    try {
        (void)program->start_prefill_lane(0, std::move(prompt), std::move(plan), {});
    } catch (const std::logic_error& error) {
        reported_missing_model =
            std::string_view(error.what()).find("loaded model") != std::string_view::npos;
    } catch (...) {
    }
    expect(reported_missing_model, "prefill without a loaded model reports the missing model");

    const auto lane = sn::Program::inspect_lane_for_test(*program, 0);
    expect(!lane.occupied && lane.frontier == 0 && lane.prompt_tokens == 0,
           "a rejected prefill leaves the lane untouched");
}

void test_decode_batch_validation() {
    auto program = sn::Program::create(2, 32, 4096, 1U << 20U);

    std::array<std::uint32_t, 1> lanes{0};
    std::array<ninfer::runtime::RoundBudget, 1> budgets{ninfer::runtime::RoundBudget{8}};
    bool rejected_not_pending = false;
    try {
        (void)program->decode_batch(lanes, budgets);
    } catch (const std::logic_error& error) {
        rejected_not_pending =
            std::string_view(error.what()).find("pending lane") != std::string_view::npos;
    } catch (...) {
    }
    expect(rejected_not_pending, "decode rejects a lane that is not pending");

    bool rejected_empty = false;
    try {
        (void)program->decode_batch({}, {});
    } catch (const std::invalid_argument&) { rejected_empty = true; } catch (...) {
    }
    expect(rejected_empty, "decode rejects an empty batch");
}

void test_output_contract() {
    // The output session stops on the generation-config EOS ids. The target's obligation is the
    // other half: sample inside a domain that contains them and pass them through unfiltered.
    expect(sn::TextConfig::end_of_text_token == 248044 &&
               sn::TextConfig::end_of_message_token == 248046,
           "generation-config EOS ids are the published values");
    expect(sn::TextConfig::end_of_text_token < sn::TextConfig::token_domain &&
               sn::TextConfig::end_of_message_token < sn::TextConfig::token_domain,
           "both EOS ids are inside the sampled token domain");
    expect(sn::TextConfig::token_domain == 248077,
           "the sampled domain is the tokenizer-addressable domain");
    expect(sn::TextConfig::output_rows > sn::TextConfig::token_domain,
           "the head's padded rows are outside the sampled domain");

    auto program = sn::Program::create(1, 32, 4096, 1U << 20U);
    const auto timings = program->generation_timings_lane(0);
    expect(timings.prefill_seconds == 0.0 && timings.decode_seconds == 0.0 &&
               timings.total_seconds == 0.0,
           "a lane that has not run reports zero timings");
}

void test_prefix_state_snapshot() {
    auto program = sn::Program::create(1, 8, 4096, 1U << 20U);
    sn::Program::seed_lane_for_test(*program, 0, 7, sn::TextConfig::end_of_text_token);
    sn::Program::write_state_image_markers_for_test(*program, 0x5A);

    const sn::PrefixStateSnapshot snapshot = program->capture_prefix_state(0);
    const auto image = sn::Program::inspect_state_images_for_test(*program);
    expect(!snapshot.empty(), "prefix snapshot owns a device copy");
    expect(snapshot.bytes() == image.bytes,
           "prefix snapshot covers the whole state image, including every looped visit");
    expect(snapshot.token_count() == 7, "prefix snapshot records the frontier");
    expect(snapshot.last_token() == sn::TextConfig::end_of_text_token,
           "prefix snapshot records the last committed token");

    sn::Program::write_state_image_markers_for_test(*program, 0x00);
    program->restore_prefix_state(0, snapshot);
    expect(sn::Program::verify_state_image_markers_for_test(*program, 0x5A),
           "prefix restore brings back all 15 KV, 42 GDN, and 48 mHC domains");
    expect(sn::Program::inspect_lane_for_test(*program, 0).frontier == 7,
           "prefix restore reinstates the host frontier");

    bool rejected = false;
    try {
        program->restore_prefix_state(0, sn::PrefixStateSnapshot{});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "an empty snapshot is rejected");

    program->abort_lane(0);
    bool aborted_rejected = false;
    try {
        program->restore_prefix_state(0, snapshot);
    } catch (const std::logic_error&) {
        aborted_rejected = true;
    }
    expect(aborted_rejected, "a snapshot cannot resurrect an aborted lane");
}

void test_decode_graph_reporting() {
    auto program = sn::Program::create(1, 8, 4096, 1U << 20U);
    const auto inspection = sn::Program::inspect_decode_graph_for_test(*program, 0);
    expect(!inspection.configured, "the test factory does not configure decode graphs");
    expect(!inspection.attempted && !inspection.ready,
           "a Program without a model has no captured decode step");
}

void test_memory() {
    auto program   = sn::Program::create(1, 16, 2048, 987654321ULL);
    const auto mem = program->memory_summary();
    expect(mem.kv_capacity == 16 * 128, "memory summary KV capacity");
    expect(mem.runtime_reservation_bytes == 987654321ULL, "memory summary reservation");
    program->reset_memory_peaks();
}

} // namespace

int main() {
    test_lifecycle();
    test_state_images();
    test_prefill_requires_model();
    test_decode_batch_validation();
    test_output_contract();
    test_prefix_state_snapshot();
    test_decode_graph_reporting();
    test_memory();
    if (failures != 0) {
        std::cerr << "FAIL senseNova program lifecycle: " << failures << '\n';
        return 1;
    }
    std::cout << "OK senseNova program lifecycle\n";
    return 0;
}

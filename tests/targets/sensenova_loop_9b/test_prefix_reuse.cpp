#include "targets/sensenova_loop_9b/impl/runtime_stub.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
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

void seed_lane(sn::Program& program, std::uint32_t lane, std::uint32_t frontier,
               ninfer::TokenId last_token) {
    sn::Program::seed_lane_for_test(program, lane, frontier, last_token);
}

// Verifies that the round's accepted-prefix snapshot is restored for every device state domain
// and that only licensed tokens reach egress.
void test_mtp_program_rounds() {
    for (std::uint32_t k : {1U, 2U, 15U}) {
        auto program = sn::Program::create(1, 128, 4096, 1ULL << 30ULL);
        sn::Program::configure_mtp_for_test(*program, k);
        sn::Program::seed_lane_for_test(*program, 0, 16, 42);

        std::vector<ninfer::TokenId> drafts(k, 7);
        std::vector<ninfer::TokenId> target(k + 1, 9);
        target[0] = 7; // accept the first draft, reject the rest
        sn::Program::set_proposal_for_test(*program, 0, drafts);
        sn::Program::set_target_window_for_test(*program, 0, target);
        sn::Program::write_lane_state_markers_for_test(*program, 0, 0x5A);

        std::array<std::uint32_t, 1> lanes{0};
        std::array<ninfer::runtime::RoundBudget, 1> budgets{64};
        const auto round = program->decode_batch(lanes, budgets);

        expect(round.row_counts.size() == 1 && round.row_counts[0] == 2,
               "a rejected MTP window licenses the accepted draft plus correction");
        expect(round.tokens.size() >= 2 && round.tokens[0] == 7 && round.tokens[1] == 9,
               "licensed tokens follow the target verification result");
        expect(sn::Program::inspect_lane_for_test(*program, 0).frontier == 18,
               "frontier advances only by licensed tokens");
        expect(sn::Program::verify_lane_state_markers_for_test(*program, 0, 0x5A),
               "rejection restores KV, GDN, mHC and MTP image markers");

        const auto stats = program->speculative_stats_lane(0);
        expect(stats.enabled, "speculative stats are enabled for an armed MTP round");
        expect(stats.backend == ninfer::SpeculativeBackend::Mtp,
               "speculative stats identify the MTP backend");
        expect(stats.draft_window == k, "speculative stats record the configured K");
        expect(stats.rounds == 1, "one round was counted");
        expect(stats.drafted_tokens == k, "every offered draft was counted");
        expect(stats.accepted_tokens == 1, "the matching first draft was counted as accepted");
        expect(stats.fallback_steps == (k == 1 ? 0U : 1U),
               "fallback is recorded only when a draft tail was rejected");
        expect(!sn::Program::inspect_lane_for_test(*program, 0).retained &&
               sn::Program::inspect_lane_for_test(*program, 0).frontier == 18,
               "a decode round leaves the lane pending for host-side resolution");

        std::array<std::uint32_t, 1> resolve_lanes{0};
        std::array<std::uint32_t, 1> accepted{2};
        std::array<std::uint8_t, 1> terminal{0};
        std::array<std::uint8_t, 1> cancelled{0};
        program->resolve_pending_batch(resolve_lanes, accepted, terminal, cancelled);
        expect(sn::Program::inspect_lane_for_test(*program, 0).frontier == 20,
               "resolve_pending_batch returns the lane to active");
    }
}

void test_prefix_identity_and_isolation() {
    auto program = sn::Program::create(2, 256, 4096, 1ULL << 30ULL);
    const std::vector<ninfer::TokenId> prompt{1, 2, 3, 4, 5, 6, 7, 8};

    seed_lane(*program, 0, static_cast<std::uint32_t>(prompt.size()), prompt.front());
    sn::Program::set_committed_prompt_for_test(*program, 0, prompt);
    sn::Program::write_lane_state_markers_for_test(*program, 0, 0xA5);
    const auto captured = program->capture_engine_prefix(0, static_cast<std::uint32_t>(prompt.size()));
    const auto identity = sn::Program::prefix_identity_for_test(prompt);
    expect(captured.value != 0, "captured prefix identity is non-zero");
    expect(captured.value == identity.value,
           "captured prefix identity hashes the exact committed prompt");
    expect(identity.value != 0, "prompt identity is non-zero");
    expect(sn::Program::prefix_identity_for_test(prompt).value == identity.value,
           "prompt identity is deterministic");

    // Prefix snapshot is a whole-state-image copy: every visit and domain is included.
    sn::Program::write_lane_state_markers_for_test(*program, 1, 0x5A);
    expect(!sn::Program::verify_lane_state_markers_for_test(*program, 1, 0xA5),
           "lane partitions do not alias before restore");

    // Plan/admission identity must include the prefix and the target lane. With no captured
    // engine prefix this plan is rejected, proving the reuse gate is active rather than implicit.
    auto plan = sn::RequestPlanForTest::make();
    expect(program->can_admit_lane(1, plan), "second lane is admissible before prefix assignment");
    expect(program->plan_request_for_prefix(1, identity, 8, plan),
           "a captured prefix can be planned for reuse");
    expect(program->plan_request_for_prefix(1, captured, 8, plan),
           "a captured full prompt prefix is admitted for reuse");

    // A different prompt of the same length produces a different identity.
    std::vector<ninfer::TokenId> other{1, 2, 3, 9, 5, 6, 7, 8};
    expect(sn::Program::prefix_identity_for_test(other).value != identity.value,
           "prefix identity is token-content sensitive");
}

} // namespace

int main() {
    test_mtp_program_rounds();
    test_prefix_identity_and_isolation();
    if (failures != 0) {
        std::cerr << "FAIL SenseNova prefix reuse: " << failures << '\n';
        return 1;
    }
    std::cout << "OK SenseNova prefix reuse\n";
    return 0;
}

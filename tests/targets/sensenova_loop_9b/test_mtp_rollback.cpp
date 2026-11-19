#include "targets/sensenova_loop_9b/impl/config.h"
#include "targets/sensenova_loop_9b/impl/runtime/mtp.h"
#include "targets/sensenova_loop_9b/impl/runtime/mtp_stem.h"

#include <cstdint>
#include <iostream>
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

void test_boundaries() {
    for (std::uint32_t k : {1U, 2U, 15U}) {
        expect(sn::valid_draft_window(k), "K boundary is admitted");
    }
    expect(!sn::valid_draft_window(0), "K=0 is rejected");
    expect(!sn::valid_draft_window(16), "K=16 is rejected");
}

void test_accept_and_rejection() {
    using TokenId = ninfer::TokenId;

    // K=1 forced acceptance. `target[i]` verifies draft i and `target[K]` is the bonus column,
    // matching speculative_accept_greedy_drafts' [K+1,B] target_tokens row layout.
    sn::MtpProposal one_draft;
    one_draft.drafted = {11};
    one_draft.valid   = 1;
    sn::MtpDecision one = sn::verify_greedy(one_draft, std::vector<TokenId>{11, 77});
    expect(one.accepted == 1 && one.licensed == 2 && one.next_token == 77,
           "K=1 acceptance commits the draft and uses the bonus target token");

    sn::MtpProposal proposal;
    proposal.drafted = {11, 12, 13};
    proposal.valid   = 3;

    // First-token rejection: the correction is target[1].
    sn::MtpDecision first = sn::verify_greedy(one_draft, std::vector<TokenId>{22, 77});
    expect(first.accepted == 0 && first.licensed == 1 && first.next_token == 22,
           "first-token rejection commits only the correction");

    // Mid-window rejection: keep the matching prefix and correct at the mismatch.
    sn::MtpDecision mid = sn::verify_greedy(proposal, std::vector<TokenId>{11, 33, 99, 77});
    expect(mid.accepted == 1 && mid.licensed == 2 && mid.next_token == 33,
           "mid-window rejection restores the accepted prefix");

    // Full acceptance: no correction, so the target's bonus token is committed.
    sn::MtpDecision full = sn::verify_greedy(proposal, std::vector<TokenId>{11, 12, 13, 77});
    expect(full.accepted == 3 && full.licensed == 4 && full.next_token == 77,
           "full acceptance commits the target bonus token");

    bool rejected = false;
    try {
        (void)sn::verify_greedy(proposal, std::vector<TokenId>{11, 99});
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "a short target verification window is rejected");
}

void test_round_state() {
    sn::MtpRoundState state;
    state.reset(7, 42);
    expect(state.frontier() == 7 && state.last_token() == 42, "round state initializes");

    sn::MtpProposal proposal;
    proposal.drafted = {5, 6, 7};
    proposal.valid   = 3;
    state.propose(3, proposal);
    expect(state.pending_drafts().size() == 3, "proposal is retained until commit");
    state.commit(sn::verify_greedy(proposal, std::vector<ninfer::TokenId>{5, 99, 99, 77}));
    expect(state.frontier() == 9 && state.last_token() == 99 && state.pending_drafts().empty(),
           "commit advances by the licensed count and clears stale drafts");

    state.propose(1, sn::MtpProposal{{8}, 1});
    sn::MtpProposal next;
    next.drafted = state.pending_drafts();
    next.valid = static_cast<std::uint32_t>(next.drafted.size());
    state.commit(sn::verify_greedy(next, std::vector<ninfer::TokenId>{99, 88}));
    expect(state.frontier() == 10 && state.last_token() == 99,
           "next round starts from the correction token");
}

void test_stem_validation() {
    // This is a host-only contract test. The fixture has no device pointers, so
    // the stem must fail before dispatching any CUDA op.
    sn::RuntimeModelView model{};
    sn::MtpStemBindings bindings{};
    std::vector<std::byte> state(1);
    std::vector<std::byte> ws(4096);
    ninfer::WorkspaceArena arena{ninfer::DeviceSpan{ws.data(), ws.size()}};
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(std::uintptr_t{1});
    std::int32_t id = 0;
    std::int32_t draft = 0;
    ninfer::Tensor ids{&id, ninfer::DType::I32, {1}};
    ninfer::Tensor drafts{&draft, ninfer::DType::I32, {1}};
    bool rejected = false;
    try {
        sn::mtp_propose_tokens(model, stream, state.data(), 128, arena, bindings, ids, ids, ids, 1,
                               drafts);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "proposal rejects unbound MTP weights before dispatch");

    expect(sn::mtp_stem_workspace_capacity_bytes(1, 1) > 0, "stem reports a nonzero workspace");
    expect(sn::mtp_stem_workspace_capacity_bytes(15, 16) >=
               sn::mtp_stem_workspace_capacity_bytes(1, 1),
           "a wider MTP window needs at least as much transient storage");
}

} // namespace

int main() {
    test_boundaries();
    test_accept_and_rejection();
    test_round_state();
    test_stem_validation();
    if (failures != 0) {
        std::cerr << "FAIL SenseNova MTP rollback: " << failures << '\n';
        return 1;
    }
    std::cout << "OK SenseNova MTP rollback\n";
    return 0;
}

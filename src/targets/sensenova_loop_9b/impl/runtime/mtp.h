#pragma once

// Stage F: pure host-side state machine for SenseNova MTP propose/verify/rollback.
//
// This module deliberately owns no device memory. It defines the authoritative
// accepted-prefix state and the deterministic greedy commit rule shared by the
// device-backed MTP proposal path and its tests. Device state images are restored
// by Program using the accepted count this module returns; no token is published
// until the target logits have verified it.

#include "targets/sensenova_loop_9b/impl/config.h"

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::sensenova_loop_9b::detail {

// One MTP draft window. `drafted` holds at most K token ids on a successful
// proposal; `valid` is the number actually produced (a proposal near the KV
// capacity may be shorter). `target` is the target's greedy output for the K+1
// verification columns: column 0 is the anchor, column i+1 verifies draft i.
struct MtpProposal {
    std::vector<TokenId> drafted;
    std::uint32_t valid = 0;
};

struct MtpDecision {
    // Number of leading drafts the target licensed.
    std::uint32_t accepted = 0;
    // Number of tokens to publish this round: accepted drafts plus one target
    // correction/bonus token. Always at least one for a valid round.
    std::uint32_t licensed = 1;
    // Token to carry into the next round. Equal to target[accepted] by the
    // greedy speculative contract.
    TokenId next_token = 0;
};

// Validates K against the frozen SenseNova domain [1,15].
[[nodiscard]] bool valid_draft_window(std::uint32_t k) noexcept;

// Verifies one greedy proposal. `target` must contain at least `proposal.valid + 1`
// entries. The longest draft prefix equal to target[i+1] is accepted; the first
// mismatch (or the bonus column when all drafts matched) supplies `next_token`.
// This is the same rule as the repo's speculative accept op, expressed on host
// values so every rollback boundary can be proven without a licensed artifact.
[[nodiscard]] MtpDecision verify_greedy(const MtpProposal& proposal,
                                        std::span<const TokenId> target);

// Host-side proposal bookkeeping across rounds. `frontier` is the authoritative
// committed token count and `last_token` is the committed token at that frontier.
// The device-backed MTP KV domain and shortlist state live in the Program; this
// object records only the identities needed to decide what a round may commit.
class MtpRoundState {
public:
    void reset(std::uint32_t frontier, TokenId last_token) noexcept;

    [[nodiscard]] std::uint32_t frontier() const noexcept { return frontier_; }
    [[nodiscard]] TokenId last_token() const noexcept { return last_token_; }
    [[nodiscard]] std::uint32_t draft_window() const noexcept { return draft_window_; }
    [[nodiscard]] const std::vector<TokenId>& pending_drafts() const noexcept {
        return pending_drafts_;
    }

    // Records a proposal for the current frontier. The proposal is pending until
    // commit_round uses target verification to accept or reject it.
    void propose(std::uint32_t k, MtpProposal proposal);

    // Commits `decision`, clears stale proposal state, advances the frontier by
    // the licensed count, and keeps `decision.next_token` as the next round's
    // anchor. The caller restores device state to the accepted prefix before
    // calling this function.
    void commit(const MtpDecision& decision);

private:
    std::uint32_t frontier_     = 0;
    TokenId last_token_         = 0;
    std::uint32_t draft_window_ = 0;
    std::vector<TokenId> pending_drafts_;
};

} // namespace ninfer::targets::sensenova_loop_9b::detail

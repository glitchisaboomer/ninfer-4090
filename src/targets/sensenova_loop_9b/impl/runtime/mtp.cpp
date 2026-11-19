#include "targets/sensenova_loop_9b/impl/runtime/mtp.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::targets::sensenova_loop_9b::detail {

bool valid_draft_window(std::uint32_t k) noexcept {
    return k >= 1 && k <= kMaximumMtpDraftTokens;
}

MtpDecision verify_greedy(const MtpProposal& proposal, std::span<const TokenId> target) {
    if (!valid_draft_window(proposal.valid) || proposal.drafted.size() < proposal.valid ||
        target.size() < static_cast<std::size_t>(proposal.valid) + 1U) {
        throw std::invalid_argument("sensenova_loop_9b MTP: invalid verification window");
    }

    MtpDecision decision;
    std::uint32_t accepted = 0;
    while (accepted < proposal.valid && proposal.drafted[accepted] == target[accepted]) {
        ++accepted;
    }
    decision.accepted  = accepted;
    decision.licensed  = accepted + 1;
    decision.next_token = target[accepted];
    return decision;
}

void MtpRoundState::reset(std::uint32_t frontier, TokenId last_token) noexcept {
    frontier_        = frontier;
    last_token_      = last_token;
    draft_window_    = 0;
    pending_drafts_.clear();
}

void MtpRoundState::propose(std::uint32_t k, MtpProposal proposal) {
    if (!valid_draft_window(k) || proposal.valid > k ||
        proposal.drafted.size() < proposal.valid) {
        throw std::invalid_argument("sensenova_loop_9b MTP: proposal exceeds the draft window");
    }
    draft_window_   = k;
    pending_drafts_ = std::move(proposal.drafted);
    pending_drafts_.resize(proposal.valid);
}

void MtpRoundState::commit(const MtpDecision& decision) {
    if (decision.licensed == 0 || decision.licensed > draft_window_ + 1U) {
        throw std::invalid_argument("sensenova_loop_9b MTP: decision exceeds the draft window");
    }
    frontier_ += decision.licensed;
    last_token_ = decision.next_token;
    draft_window_ = 0;
    pending_drafts_.clear();
}

} // namespace ninfer::targets::sensenova_loop_9b::detail

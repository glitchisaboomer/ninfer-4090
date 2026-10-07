#pragma once

// Phase 5.5 of docs/SCR_ROADMAP.md: decide whether Suffix Cache Reuse changed the model's output
// enough to be unacceptable.
//
// SCR reuses KV computed under the pre-edit context, so it is an approximation. The roadmap fixes two
// measurements against the full-prefill path:
//
//   * mean next-token logprob divergence, threshold < 0.05 nats;
//   * greedy-decode agreement over the first 64 tokens, threshold > 95%.
//
// This header only *scores* two runs of the same prompt. It performs no device work and holds no
// state, so it is host-buildable and unit-testable without a CUDA toolchain, and whatever runs the
// two paths feeds it.
//
// Divergence is the mean absolute difference between the two paths' next-token logprobs. It is a
// magnitude, so a run that is confidently wrong and one that is uncertain both count; a signed mean
// would let opposing errors cancel and read as agreement.
//
// Only the shared prefix is compared, and "shared" means the leading run of positions at which both
// paths emitted the *same* token -- not merely the shorter of the two output lengths. Once the paths
// pick different tokens, position i holds a different token on each side, so `scr_logprobs[i]` and
// `baseline_logprobs[i]` are logprobs of two unrelated tokens. Averaging those measures how far the
// outputs drifted after the first flip, which is not a property of Suffix Cache Reuse, and it swamps
// the quantity being measured: a run that perturbs the distribution by a hair can then diverge for
// 60 more tokens and report a mean in the tenths.
//
// Agreement is deliberately *not* truncated this way. It is index-aligned over its configured prefix
// on purpose, because the roadmap's question is whether greedy decoding flipped at all; `
// first_divergence_index` is reported alongside it so a collapse can be read for what it is.

#include <ninfer/types.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// What one prompt produced on each path. The two runs must use the same prompt, sampling, and seed;
// only the reuse path differs, which is what the caller guarantees.
struct ScrQualitySample {
    // Next-token logprob of the token each path produced, in generation order.
    std::vector<float> baseline_logprobs;
    std::vector<float> scr_logprobs;
    // Greedy tokens each path produced, in generation order.
    std::vector<TokenId> baseline_tokens;
    std::vector<TokenId> scr_tokens;
};

struct ScrQualityThresholds {
    // `< 0.05 nats` in the roadmap, expressed as an inclusive bound so the comparison is `<=`.
    double mean_logprob_divergence_nats = 0.05;
    // `> 95%` in the roadmap, expressed as an inclusive bound so the comparison is `>=`.
    double greedy_agreement = 0.95;
    // Agreement is measured over this many leading tokens; a shorter run compares what it has.
    std::uint32_t agreement_prefix_tokens = 64;
};

struct ScrQualityPromptResult {
    // Tokens whose logprobs were compared, and the mean absolute divergence over them. This is the
    // agreed prefix: leading positions at which both paths emitted the *same* token, so each side's
    // logprob is the logprob of that one token. Positions past the first disagreement are excluded --
    // see `first_divergence_index`.
    std::uint32_t compared_tokens   = 0;
    double mean_logprob_divergence  = 0.0;
    // The worst single-step divergence over the agreed prefix. A mean can hide one badly perturbed
    // step behind many unremarkable ones; this cannot.
    double max_logprob_divergence = 0.0;
    // |baseline - SCR| logprob at the very first generated position. This is the one scalar that needs
    // no caveat: position 0 is compared before any divergence can have compounded, so it bounds how
    // much the reused prefill moved the next-token distribution before decoding feedback muddies it.
    // NaN when nothing was comparable.
    double first_token_divergence = 0.0;
    // Where the two paths first emitted different tokens. Equal to `logprobs_available` when they
    // never did. Reported because the pass/fail metrics hide it: agreement collapses to whatever
    // fraction of the drift happened to re-synchronize, which is not monotone in how much of the
    // context was reused.
    std::uint32_t first_divergence_index = 0;
    // Positions both sides had a token *and* a logprob for, before any prefix truncation. The gap
    // between this and `compared_tokens` is how much of the run the comparison discarded, and it is
    // what makes a large mean legible.
    std::uint32_t logprobs_available = 0;
    // False when a compared step's logprob difference was NaN or infinite, i.e. the path that emitted
    // it did not report a usable logprob. `first_nonfinite_index` then names the position.
    bool logprobs_finite        = true;
    std::uint32_t first_nonfinite_index = 0;
    // Leading tokens the agreement was measured over, and how many matched. Unlike divergence this
    // stays index-aligned over the whole configured prefix: the roadmap's question is whether greedy
    // decoding flipped, and a flip is a disagreement wherever it lands.
    std::uint32_t agreement_compared = 0;
    std::uint32_t agreement_matched  = 0;
    double greedy_agreement          = 0.0;

    bool ok = false;
    // Why the prompt did not pass, empty when it did.
    std::string note;
};

struct ScrQualityReport {
    std::vector<ScrQualityPromptResult> prompts;
    std::uint32_t passed = 0;
    // False for an empty report: a suite that measured nothing has not passed.
    bool all_passed = false;
};

// Score one prompt. A prompt passes only when both measurements are inside their thresholds and both
// had something to measure, so an empty sample fails rather than passing vacuously.
[[nodiscard]] ScrQualityPromptResult evaluate_scr_prompt(const ScrQualitySample& sample,
                                                        const ScrQualityThresholds& thresholds);

// Score a suite in order, counting the prompts that passed.
[[nodiscard]] ScrQualityReport evaluate_scr_quality(std::span<const ScrQualitySample> samples,
                                                    const ScrQualityThresholds& thresholds);

// Whether a null-control report is clean: every prompt's two runs chose the same token at every
// position both sides had one.
//
// This is the gate on attributing a measurement. Two runs of an identical configuration are not
// guaranteed bit-identical, so a measurement that reports divergence alongside a control that also
// diverges has no floor to subtract -- the spread could be entirely run-to-run variation, and no
// amount of reading the SCR-versus-baseline numbers will separate the two. A clean control says the
// floor is zero, so whatever the measurement reports is attributable to the difference between the
// paths being compared.
//
// An empty report is *not* clean: it measured nothing, and nothing cannot establish a floor.
[[nodiscard]] bool scr_quality_control_is_clean(const ScrQualityReport& control);

// The indices of the prompts in a control report that did not agree with themselves, for naming them.
// Empty exactly when scr_quality_control_is_clean is true.
[[nodiscard]] std::vector<std::uint32_t> scr_quality_control_divergent_prompts(
    const ScrQualityReport& control);

// One line of JSON per prompt plus an aggregate line, in a fixed key order so two runs are
// diffable. Numbers are emitted with enough precision to compare against the thresholds by hand.
[[nodiscard]] std::string format_scr_quality_json(const ScrQualityReport& report,
                                                  const ScrQualityThresholds& thresholds);

} // namespace ninfer::targets::qwen3_6::detail

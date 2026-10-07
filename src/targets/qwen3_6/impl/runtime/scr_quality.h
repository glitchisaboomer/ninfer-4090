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
// Only the shared prefix is compared. The paths can diverge and then produce different-length output,
// and comparing beyond the first divergence would compare different tokens, which says nothing about
// SCR's approximation.

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
    // Tokens whose logprobs were compared, and the mean absolute divergence over them.
    std::uint32_t compared_tokens   = 0;
    double mean_logprob_divergence  = 0.0;
    // Leading tokens the agreement was measured over, and how many matched.
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

// One line of JSON per prompt plus an aggregate line, in a fixed key order so two runs are
// diffable. Numbers are emitted with enough precision to compare against the thresholds by hand.
[[nodiscard]] std::string format_scr_quality_json(const ScrQualityReport& report,
                                                  const ScrQualityThresholds& thresholds);

} // namespace ninfer::targets::qwen3_6::detail

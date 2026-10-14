#pragma once

// Phase 5.5 of docs/SCR.md: decide whether Suffix Cache Reuse changed the model's output
// enough to be unacceptable.
//
// SCR reuses KV computed under the pre-edit context, so it is an approximation. The roadmap fixed
// two measurements against the full-prefill path:
//
//   * mean next-token logprob divergence, threshold < 0.05 nats;
//   * greedy-decode agreement over the first 64 tokens, threshold > 95%.
//
// Both are still computed and reported. Only the first gates `ok`, and that asymmetry is the
// result of the 2026-10-10 sweep rather than a change of taste -- see the note in
// `evaluate_scr_prompt`. SCR perturbs a distribution; it does not reproduce one, so "the same word
// must come out" is a demand the approximation never made, and on this suite it measures where the
// prompts happen to sit on a tie. Divergence measures the thing that is actually being asked.
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
// first_divergence_index` is reported alongside it so a collapse can be read for what it is. It is a
// diagnostic, not a bound -- see `ScrQualityThresholds::greedy_agreement`.

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
    // Best-minus-second-best logit gap per position, same order, optional. Without them a logprob
    // difference cannot be read: the same difference means "it flipped a coin" when the leading
    // candidates were a hair apart and "it moved a decision" when they were far apart.
    std::vector<float> baseline_margins;
    std::vector<float> scr_margins;
};

struct ScrQualityThresholds {
    // `< 0.05 nats` in the roadmap, expressed as an inclusive bound so the comparison is `<=`.
    // This is the bound that decides pass/fail: Suffix Cache Reuse is an approximation, so what it
    // can be held to is `how far did the distribution move`, not `did the same word come out`.
    double mean_logprob_divergence_nats = 0.05;
    // Reporting only, NOT a bound. `greedy_agreement` and `agreement_prefix_tokens` are still
    // computed and shipped in every report -- they are how a reader sees where the two streams
    // parted, and how early -- but they do not gate `ok`. See the long note in
    // `evaluate_scr_prompt`: the 2026-10-10 sweep showed agreement is a function of where the
    // suite's near-ties sit (0.094 at 34/55/71% relocated share, 0.750 at 93%), not of how much
    // context was reused, so a threshold on it measures the test suite rather than the feature.
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
    // The decision margin at the first position, from whichever side reported it. Small means the greedy
    // token both paths agreed on was won by a hair, so a measured divergence there is the expected
    // consequence of any perturbation at all rather than evidence of a large one. Large means the token
    // was comfortably ahead and a flip would have required moving it a long way.
    //
    // Position 0 specifically, and not the minimum over the prefix: the quantity being explained is
    // `first_token_divergence`, and pairing it with a margin from some other position would be a
    // correlation rather than an explanation.
    double first_token_margin = 0.0;
    bool margins_available    = false;

    // Leading tokens the agreement was measured over, and how many matched. Unlike divergence this
    // stays index-aligned over the whole configured prefix on purpose: the roadmap's question is
    // whether greedy decoding flipped at all, and a flip is a disagreement wherever it lands. Where
    // the paths parted is reported separately, because agreement alone cannot distinguish an early
    // flip from a late one.
    std::uint32_t agreement_compared = 0;
    std::uint32_t agreement_matched  = 0;
    double greedy_agreement          = 0.0;

    // The null control's measurements on the same prompt: the same two-run comparison performed on
    // two runs of one identical configuration. See evaluate_scr_prompt_against_control.
    //
    // These are *not* subtracted from the thresholds. A control establishes the run-to-run floor, and
    // the floor is not zero -- two runs of one configuration pick the same tokens from logits that
    // differ by a few thousandths of a nat -- so a raw measurement has to be read against it rather
    // than against an implied zero.
    bool control_available            = false;
    double floor_first_token_divergence = 0.0;
    double floor_mean_logprob_divergence = 0.0;
    // `first_token_divergence` minus the floor: what the reuse path contributed rather than what the
    // run-to-run variation contributed. Negative means the measurement sat inside the floor.
    double signal_first_token_divergence = 0.0;
    // Whether the measurement exceeded the floor. A plain comparison, stated as a fact -- this is not
    // a significance test and carries no confidence claim, because one control run cannot support one.
    bool above_floor = false;

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

// Score one prompt against a null control: the same comparison run on two executions of one identical
// configuration, which is what measures the run-to-run floor.
//
// The floor exists because two runs of the same configuration are not bit-identical. Greedy token
// choice proved stable across every prompt measured so far, while the logprobs behind it moved by up
// to ~0.08 nats -- the argmax is insensitive to a perturbation that the logprob metric is not. So a
// control that agrees on tokens says the *tokens* are reproducible; it says nothing about the logprobs
// being so, and reading a raw measurement against an implied zero overstates whatever it reports.
//
// This fills the floor fields on the result and leaves `ok` exactly as evaluate_scr_prompt set it: the
// roadmap's thresholds are stated against the measurement, not against a noise-corrected quantity, and
// quietly moving the goalposts here would mean the score no longer means what the roadmap says.
// Deciding what a given signal-to-floor ratio is worth is a threshold question and belongs with the
// other thresholds, not in a comparison helper.
//
// `control` must describe the same prompt as `sample`; a control that measured nothing leaves the floor
// fields at their defaults with `control_available` false rather than reporting a floor of zero.
[[nodiscard]] ScrQualityPromptResult evaluate_scr_prompt_against_control(
    const ScrQualitySample& sample, const ScrQualitySample& control,
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

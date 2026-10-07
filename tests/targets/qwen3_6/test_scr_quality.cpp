// Phase 5.5 SCR checks: scoring two runs of the same prompt.
//
// The scorer is host-only and dependency-free, so this file also builds with a plain host compiler:
//   g++ -std=c++20 -I src -I include
//       src/targets/qwen3_6/impl/runtime/scr_quality.cpp
//       tests/targets/qwen3_6/test_scr_quality.cpp -o /tmp/test_scr_quality

#include "targets/qwen3_6/impl/runtime/scr_quality.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::targets::qwen3_6::detail::evaluate_scr_prompt;
using ninfer::targets::qwen3_6::detail::evaluate_scr_quality;
using ninfer::targets::qwen3_6::detail::format_scr_quality_json;
using ninfer::targets::qwen3_6::detail::ScrQualityReport;
using ninfer::targets::qwen3_6::detail::scr_quality_control_divergent_prompts;
using ninfer::targets::qwen3_6::detail::scr_quality_control_is_clean;
using ninfer::targets::qwen3_6::detail::ScrQualitySample;
using ninfer::targets::qwen3_6::detail::ScrQualityThresholds;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool contains(const std::string& haystack, std::string_view needle) {
    return haystack.find(needle) != std::string::npos;
}

// A sample with `tokens` greedy steps that agree, and a fixed logprob offset on every scr step.
ScrQualitySample agreeing(std::size_t tokens, double logprob_offset) {
    ScrQualitySample sample;
    for (std::size_t i = 0; i < tokens; ++i) {
        sample.baseline_tokens.push_back(static_cast<TokenId>(i));
        sample.scr_tokens.push_back(static_cast<TokenId>(i));
        sample.baseline_logprobs.push_back(0.0F);
        sample.scr_logprobs.push_back(static_cast<float>(logprob_offset));
    }
    return sample;
}

// A sample where the first `matched` of `tokens` greedy steps agree and the rest do not.
ScrQualitySample agreement(std::size_t tokens, std::size_t matched) {
    ScrQualitySample sample = agreeing(tokens, 0.0);
    for (std::size_t i = matched; i < tokens; ++i) {
        sample.scr_tokens[i] = static_cast<TokenId>(i) + 1000;
    }
    return sample;
}

void test_identical_runs_pass() {
    const ScrQualityThresholds thresholds;
    const auto result = evaluate_scr_prompt(agreeing(128, 0.0), thresholds);
    expect(result.ok, "identical runs pass");
    expect(result.compared_tokens == 128, "every logprob is compared");
    expect(result.mean_logprob_divergence == 0.0, "identical runs diverge by nothing");
    expect(result.agreement_compared == 64, "agreement covers the roadmap prefix of 64 tokens");
    expect(result.agreement_matched == 64, "identical runs agree on the whole prefix");
    expect(result.greedy_agreement == 1.0, "identical runs agree completely");
    expect(result.note.empty(), "a passing prompt carries no note");
}

void test_threshold_bounds_are_inclusive() {
    // Thresholds chosen so the boundary values are exactly representable, which a 0.05 nat bound
    // comparing a float-derived difference against a double bound is not.
    ScrQualityThresholds thresholds;
    thresholds.mean_logprob_divergence_nats = 0.5;
    thresholds.greedy_agreement            = 0.5;
    thresholds.agreement_prefix_tokens     = 64;

    // Both bounds are exercised exactly. 0.5 is exactly representable, so a per-step difference of
    // 0.5 gives a mean of exactly 0.5 rather than something just above or below it.
    const auto divergence_on_bound = evaluate_scr_prompt(agreeing(64, 0.5), thresholds);
    expect(divergence_on_bound.mean_logprob_divergence == 0.5,
           "the divergence fixture sits exactly on the bound");
    expect(divergence_on_bound.ok, "a divergence exactly on the bound passes");

    const auto at_bound = evaluate_scr_prompt(agreement(64, 32), thresholds);
    expect(at_bound.mean_logprob_divergence == 0.0, "the agreement fixture diverges by nothing");
    expect(at_bound.greedy_agreement == 0.5, "the agreement fixture sits exactly on the bound");
    expect(at_bound.ok, "an agreement exactly on the bound passes");

    const auto beyond = evaluate_scr_prompt(agreement(64, 31), thresholds);
    expect(beyond.greedy_agreement < 0.5, "the beyond fixture sits under the bound");
    expect(!beyond.ok, "a value under the bound fails");
    expect(contains(beyond.note, "greedy agreement"), "the note names the failing measurement");
}

void test_divergence_is_the_mean_absolute_difference() {
    ScrQualityThresholds thresholds;
    // 0.01 + 0.03 over two steps is a mean of 0.02, which is inside the default bound.
    ScrQualitySample sample;
    sample.baseline_tokens   = {1, 2};
    sample.scr_tokens        = {1, 2};
    sample.baseline_logprobs = {0.0F, 0.0F};
    sample.scr_logprobs      = {0.01F, 0.03F};

    const auto result = evaluate_scr_prompt(sample, thresholds);
    expect(result.compared_tokens == 2, "both steps are compared");
    expect(result.mean_logprob_divergence > 0.019 && result.mean_logprob_divergence < 0.021,
           "divergence is the mean of the per-step differences");
    expect(result.ok, "a small divergence passes the default bound");

    // Opposing errors must not cancel: -0.05 and +0.05 is a divergence of 0.05, not of zero.
    ScrQualitySample cancelling;
    cancelling.baseline_tokens   = {1, 2};
    cancelling.scr_tokens        = {1, 2};
    cancelling.baseline_logprobs = {0.0F, 0.0F};
    cancelling.scr_logprobs      = {-0.05F, 0.05F};
    const auto cancelling_result = evaluate_scr_prompt(cancelling, thresholds);
    expect(cancelling_result.mean_logprob_divergence > 0.049,
           "divergence is a magnitude, so opposing errors do not cancel");
}

void test_divergence_beyond_the_default_bound_fails() {
    const ScrQualityThresholds thresholds;
    const auto result = evaluate_scr_prompt(agreeing(128, 0.06), thresholds);
    expect(!result.ok, "divergence over 0.05 nats fails");
    expect(result.greedy_agreement == 1.0, "agreement is still measured independently");
    expect(contains(result.note, "divergence"), "the note names the failing measurement");
}

void test_agreement_prefix_is_bounded() {
    // The first 64 of 128 steps agree, so the default prefix measurement passes even though the run
    // diverges later.
    ScrQualityThresholds thresholds;
    const auto prefix_only = evaluate_scr_prompt(agreement(128, 64), thresholds);
    expect(prefix_only.agreement_compared == 64, "agreement stops at the configured prefix");
    expect(prefix_only.greedy_agreement == 1.0, "the agreeing prefix scores fully");
    expect(prefix_only.ok, "a run that diverges after the prefix still passes");

    // Widening the prefix to the whole run exposes the divergence.
    thresholds.agreement_prefix_tokens = 128;
    const auto whole_run = evaluate_scr_prompt(agreement(128, 64), thresholds);
    expect(whole_run.greedy_agreement == 0.5, "the whole run is half in agreement");
    expect(!whole_run.ok, "the wider prefix fails");
}

void test_short_runs_compare_what_they_have() {
    const ScrQualityThresholds thresholds;
    const auto result = evaluate_scr_prompt(agreeing(3, 0.0), thresholds);
    expect(result.agreement_compared == 3, "a run shorter than the prefix compares its own length");
    expect(result.greedy_agreement == 1.0, "the short run agrees completely");
    expect(result.ok, "a short agreeing run passes");
}

void test_only_the_shared_prefix_is_compared() {
    ScrQualitySample sample;
    sample.baseline_tokens   = {1, 2, 3, 4};
    sample.scr_tokens        = {1, 2};
    sample.baseline_logprobs = {0.0F, 0.0F, 9.0F, 9.0F};
    sample.scr_logprobs      = {0.0F, 0.0F};

    const auto result = evaluate_scr_prompt(sample, ScrQualityThresholds{});
    expect(result.compared_tokens == 2, "logprobs are compared over the shared prefix only");
    expect(result.mean_logprob_divergence == 0.0, "the divergent tail is not compared");
    expect(result.agreement_compared == 2, "agreement is compared over the shared prefix only");
    expect(result.ok, "a differing tail does not fail the shared prefix");
}

void test_unmeasured_inputs_fail() {
    static const ScrQualityThresholds thresholds;

    const auto empty = evaluate_scr_prompt(ScrQualitySample{}, thresholds);
    expect(!empty.ok, "a sample with nothing to compare does not pass");
    expect(contains(empty.note, "no overlapping logprobs"), "the note says what was missing");

    // Agreement is what failed here, so the note must say so rather than blaming logprobs.
    ScrQualitySample no_tokens;
    no_tokens.baseline_logprobs = {0.0F};
    no_tokens.scr_logprobs      = {0.0F};
    const auto no_tokens_result = evaluate_scr_prompt(no_tokens, thresholds);
    expect(!no_tokens_result.ok, "logprobs without tokens do not pass");
    expect(contains(no_tokens_result.note, "no overlapping tokens"),
           "the note distinguishes missing tokens from missing logprobs");
}

void test_report_aggregates() {
    const ScrQualityThresholds thresholds;
    std::vector<ScrQualitySample> samples{agreeing(64, 0.0), agreeing(64, 0.5)};

    const auto report = evaluate_scr_quality(samples, thresholds);
    expect(report.prompts.size() == 2, "every sample gets a result");
    expect(report.passed == 1, "only the passing sample is counted");
    expect(!report.all_passed, "a suite with a failure has not passed");

    std::vector<ScrQualitySample> all_good{agreeing(64, 0.0), agreeing(64, 0.01)};
    const auto passing = evaluate_scr_quality(all_good, thresholds);
    expect(passing.passed == 2, "both passing samples are counted");
    expect(passing.all_passed, "a suite of passing samples passes");

    const auto none = evaluate_scr_quality({}, thresholds);
    expect(none.prompts.empty() && none.passed == 0, "an empty suite reports nothing");
    expect(!none.all_passed, "an empty suite has not passed");
}

// The case the previous implementation got wrong, and the reason this file grew: two runs of the same
// length whose tokens diverge partway through. Past the flip, scr_logprobs[i] is the logprob of a
// different token than baseline_logprobs[i], so averaging those pairs measures how far the outputs
// drifted rather than how much relocation perturbed the distribution.
void test_divergence_stops_at_the_first_divergent_token() {
    const ScrQualityThresholds thresholds;
    ScrQualitySample sample;
    for (std::size_t i = 0; i < 64; ++i) {
        sample.baseline_tokens.push_back(static_cast<TokenId>(i));
        sample.scr_tokens.push_back(static_cast<TokenId>(i));
        sample.baseline_logprobs.push_back(0.0F);
        sample.scr_logprobs.push_back(0.01F);
    }
    // Flip at step 5, and make the post-flip logprobs wildly different so a contaminated mean cannot
    // be mistaken for a clean one.
    sample.scr_tokens[5] = static_cast<TokenId>(5) + 1000;
    for (std::size_t i = 5; i < 64; ++i) {
        // Every later step stays different, so the paths do not re-synchronize and agreement fails
        // on its own instead of being rescued by a chance re-match.
        sample.scr_tokens[i] = static_cast<TokenId>(i) + 1000;
        sample.scr_logprobs[i] = 4.0F;
        sample.baseline_logprobs[i] = 0.0F;
    }

    const auto result = evaluate_scr_prompt(sample, thresholds);
    expect(result.first_divergence_index == 5, "the first divergent token is located");
    expect(result.compared_tokens == 5, "divergence covers only the agreed prefix");
    expect(result.logprobs_available == 64, "the run length that was available is reported");
    expect(result.mean_logprob_divergence < 0.011 && result.mean_logprob_divergence > 0.009,
           "the mean is 0.01, not the 0.5 the post-flip pairs would produce");
    expect(result.max_logprob_divergence < 0.011 && result.max_logprob_divergence > 0.009,
           "the worst agreed step is also 0.01");
    expect(result.first_token_divergence < 0.011 && result.first_token_divergence > 0.009,
           "the first token's divergence is reported");
    // Agreement deliberately stays index-aligned: greedy agreement asks whether decoding flipped, and
    // where it flipped is now reported rather than hidden.
    // Index-aligned over the whole configured prefix: only the 5 agreed steps match. Where the paths
    // parted is reported separately, because agreement alone cannot distinguish an early flip from a
    // late one.
    expect(result.agreement_matched == 5, "agreement stays index-aligned over the whole prefix");
    expect(!result.ok, "a flipped run does not pass the agreement bound");
    expect(contains(result.note, "first divergence at token 5"),
           "the note says where the paths parted");
}

// The scalar that answers "did relocation move the distribution, before decoding feedback could
// amplify it" -- usable even when everything after diverges, which is exactly when the mean is not.
void test_first_token_divergence_survives_a_later_divergence() {
    const ScrQualityThresholds thresholds;
    ScrQualitySample sample;
    sample.baseline_tokens   = {7, 8, 9};
    sample.scr_tokens        = {7, 42, 43};
    sample.baseline_logprobs = {-0.25F, -1.0F, -2.0F};
    sample.scr_logprobs      = {-0.30F, -5.0F, -6.0F};

    const auto result = evaluate_scr_prompt(sample, thresholds);
    expect(result.compared_tokens == 1, "only the agreed first token is compared");
    expect(result.first_token_divergence > 0.049 && result.first_token_divergence < 0.051,
           "the first token moved by 0.05 nats, which is what is being measured");
    expect(result.first_divergence_index == 1, "the paths parted at token 1");
}

void test_differing_first_tokens_is_not_a_vacuous_pass() {
    const ScrQualityThresholds thresholds;
    ScrQualitySample sample;
    sample.baseline_tokens   = {11, 12};
    sample.scr_tokens        = {13, 14};
    sample.baseline_logprobs = {-0.1F, -0.2F};
    sample.scr_logprobs      = {-0.1F, -0.2F};

    const auto result = evaluate_scr_prompt(sample, thresholds);
    // Perfectly equal logprobs and zero divergence would otherwise read as the best possible run.
    expect(!result.ok, "disagreeing first tokens do not pass");
    expect(result.compared_tokens == 0, "there is no agreed prefix to measure");
    expect(result.greedy_agreement == 0.0, "agreement still reports the disagreement");
    expect(contains(result.note, "different first tokens"),
           "the note says the paths chose different first tokens");
}

void test_non_finite_logprobs_are_reported_not_averaged() {
    const ScrQualityThresholds thresholds;
    ScrQualitySample sample;
    for (std::size_t i = 0; i < 8; ++i) {
        sample.baseline_tokens.push_back(static_cast<TokenId>(i));
        sample.scr_tokens.push_back(static_cast<TokenId>(i));
        sample.baseline_logprobs.push_back(0.0F);
        sample.scr_logprobs.push_back(0.0F);
    }
    // A path that emitted no logprob at one step must not be silently scored as agreement there.
    sample.scr_logprobs[3] = std::numeric_limits<float>::quiet_NaN();

    const auto result = evaluate_scr_prompt(sample, thresholds);
    expect(!result.logprobs_finite, "the non-finite step is detected");
    expect(result.first_nonfinite_index == 3, "the non-finite step is located");
    expect(std::isnan(result.mean_logprob_divergence),
           "the mean is NaN rather than an average that hides the gap");
    expect(!result.ok, "a run with an unusable logprob does not pass");
    expect(contains(result.note, "not finite at position 3"), "the note names the position");
}

// The gate on attributing a measurement at all. Without it a run whose divergence is entirely
// run-to-run variation reads exactly like a run that perturbed the distribution.
void test_control_cleanliness_gates_attribution() {
    const ScrQualityThresholds thresholds;

    // Two runs of one configuration that agree throughout: the floor is zero, so anything the
    // SCR-versus-baseline comparison reports is attributable.
    std::vector<ScrQualitySample> identical{agreeing(64, 0.0), agreeing(32, 0.0)};
    const auto clean_report = evaluate_scr_quality(identical, thresholds);
    expect(scr_quality_control_is_clean(clean_report), "a control that agrees with itself is clean");
    expect(scr_quality_control_divergent_prompts(clean_report).empty(),
           "a clean control names no prompts");

    // One prompt parts at token 3. Nothing about the size of that gap matters -- a single flipped
    // token is enough to make the floor unknown for that prompt.
    std::vector<ScrQualitySample> noisy{agreeing(64, 0.0), agreement(64, 3)};
    const auto noisy_report = evaluate_scr_quality(noisy, thresholds);
    expect(!scr_quality_control_is_clean(noisy_report), "a control that disagrees with itself is not clean");
    const auto divergent = scr_quality_control_divergent_prompts(noisy_report);
    expect(divergent.size() == 1 && divergent[0] == 1, "the disagreeing prompt is named by index");

    // An empty control measured nothing. Nothing cannot establish a floor, so it is not clean --
    // otherwise a control that failed to run would silently authorize every conclusion.
    expect(!scr_quality_control_is_clean(evaluate_scr_quality({}, thresholds)),
           "an empty control is not clean");

    // A tiny logprob gap with an identical token is still agreement: cleanliness is about the token
    // choice, not about the size of the perturbation.
    std::vector<ScrQualitySample> nudged{agreeing(64, 0.04)};
    expect(scr_quality_control_is_clean(evaluate_scr_quality(nudged, thresholds)),
           "a control whose tokens all match is clean however large the logprob difference");
}

void test_json_report() {
    const ScrQualityThresholds thresholds;
    std::vector<ScrQualitySample> samples{agreeing(64, 0.0), agreeing(64, 0.5)};
    const std::string json =
        format_scr_quality_json(evaluate_scr_quality(samples, thresholds), thresholds);

    expect(contains(json, "\"thresholds\":"), "the report states the thresholds it applied");
    expect(contains(json, "\"agreement_prefix_tokens\":64"), "the prefix length is reported");
    expect(contains(json, "\"index\":0"), "the first prompt is reported");
    expect(contains(json, "\"index\":1"), "the second prompt is reported");
    expect(contains(json, "\"passed\":1"), "the pass count is reported");
    expect(contains(json, "\"total\":2"), "the suite size is reported");
    expect(contains(json, "\"all_passed\":false"), "the aggregate verdict is reported");
    expect(json.find("FAIL") == std::string::npos, "a note never contains unrelated text");
}

} // namespace

int main() {
    test_identical_runs_pass();
    test_threshold_bounds_are_inclusive();
    test_divergence_is_the_mean_absolute_difference();
    test_divergence_beyond_the_default_bound_fails();
    test_agreement_prefix_is_bounded();
    test_short_runs_compare_what_they_have();
    test_only_the_shared_prefix_is_compared();
    test_divergence_stops_at_the_first_divergent_token();
    test_first_token_divergence_survives_a_later_divergence();
    test_differing_first_tokens_is_not_a_vacuous_pass();
    test_non_finite_logprobs_are_reported_not_averaged();
    test_control_cleanliness_gates_attribution();
    test_unmeasured_inputs_fail();
    test_report_aggregates();
    test_json_report();
    if (failures != 0) {
        std::cerr << failures << " SCR quality checks failed\n";
        return 1;
    }
    std::cout << "SCR quality checks passed\n";
    return 0;
}

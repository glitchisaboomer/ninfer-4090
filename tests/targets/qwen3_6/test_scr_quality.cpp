// Phase 5.5 SCR checks: scoring two runs of the same prompt.
//
// The scorer is host-only and dependency-free, so this file also builds with a plain host compiler:
//   g++ -std=c++20 -I src -I include
//       src/targets/qwen3_6/impl/runtime/scr_quality.cpp
//       tests/targets/qwen3_6/test_scr_quality.cpp -o /tmp/test_scr_quality

#include "targets/qwen3_6/impl/runtime/scr_quality.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::targets::qwen3_6::detail::evaluate_scr_prompt;
using ninfer::targets::qwen3_6::detail::evaluate_scr_quality;
using ninfer::targets::qwen3_6::detail::format_scr_quality_json;
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

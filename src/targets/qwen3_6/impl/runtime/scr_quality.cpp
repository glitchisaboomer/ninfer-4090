#include "targets/qwen3_6/impl/runtime/scr_quality.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>

namespace ninfer::targets::qwen3_6::detail {
namespace {

// Fixed notation with enough digits to read a value against its threshold by hand. A NaN prints as
// "nan", which is honest.
std::string format_number(double value) {
    std::ostringstream out;
    out.precision(6);
    out << std::fixed << value;
    return out.str();
}

} // namespace

ScrQualityPromptResult evaluate_scr_prompt(const ScrQualitySample& sample,
                                           const ScrQualityThresholds& thresholds) {
    ScrQualityPromptResult result;

    const std::size_t compared =
        std::min(sample.baseline_logprobs.size(), sample.scr_logprobs.size());
    result.compared_tokens = static_cast<std::uint32_t>(compared);
    if (compared != 0) {
        double total = 0.0;
        for (std::size_t i = 0; i < compared; ++i) {
            total += std::abs(static_cast<double>(sample.baseline_logprobs[i]) -
                              static_cast<double>(sample.scr_logprobs[i]));
        }
        result.mean_logprob_divergence = total / static_cast<double>(compared);
    }

    const std::size_t agreement_limit =
        std::min<std::size_t>(thresholds.agreement_prefix_tokens,
                              std::min(sample.baseline_tokens.size(), sample.scr_tokens.size()));
    result.agreement_compared = static_cast<std::uint32_t>(agreement_limit);
    for (std::size_t i = 0; i < agreement_limit; ++i) {
        if (sample.baseline_tokens[i] == sample.scr_tokens[i]) { ++result.agreement_matched; }
    }
    if (agreement_limit != 0) {
        result.greedy_agreement =
            static_cast<double>(result.agreement_matched) / static_cast<double>(agreement_limit);
    }

    // Nothing measured is not a pass. A suite that silently scores absent data as agreement would
    // report success for exactly the runs it learned nothing about.
    if (compared == 0) {
        result.note = "no overlapping logprobs to compare";
        return result;
    }
    if (agreement_limit == 0) {
        result.note = "no overlapping tokens to compare";
        return result;
    }

    // Both bounds are inclusive. Compared in double, and a NaN fails both because every comparison
    // against NaN is false.
    const bool divergence_ok =
        result.mean_logprob_divergence <= thresholds.mean_logprob_divergence_nats;
    const bool agreement_ok = result.greedy_agreement >= thresholds.greedy_agreement;
    result.ok               = divergence_ok && agreement_ok;
    if (result.ok) { return result; }

    std::ostringstream note;
    if (!divergence_ok) {
        note << "mean logprob divergence " << format_number(result.mean_logprob_divergence)
             << " exceeds " << format_number(thresholds.mean_logprob_divergence_nats);
    }
    if (!agreement_ok) {
        if (!divergence_ok) { note << "; "; }
        note << "greedy agreement " << format_number(result.greedy_agreement) << " below "
             << format_number(thresholds.greedy_agreement);
    }
    result.note = note.str();
    return result;
}

ScrQualityReport evaluate_scr_quality(std::span<const ScrQualitySample> samples,
                                      const ScrQualityThresholds& thresholds) {
    ScrQualityReport report;
    report.prompts.reserve(samples.size());
    for (const ScrQualitySample& sample : samples) {
        report.prompts.push_back(evaluate_scr_prompt(sample, thresholds));
        if (report.prompts.back().ok) { ++report.passed; }
    }
    // An empty suite has measured nothing, so it has not passed.
    report.all_passed = !report.prompts.empty() && report.passed == report.prompts.size();
    return report;
}

std::string format_scr_quality_json(const ScrQualityReport& report,
                                    const ScrQualityThresholds& thresholds) {
    std::ostringstream out;
    out << "{\"thresholds\":{\"mean_logprob_divergence_nats\":"
        << format_number(thresholds.mean_logprob_divergence_nats)
        << ",\"greedy_agreement\":" << format_number(thresholds.greedy_agreement)
        << ",\"agreement_prefix_tokens\":" << thresholds.agreement_prefix_tokens << "}"
        << ",\"prompts\":[";
    for (std::size_t i = 0; i < report.prompts.size(); ++i) {
        const ScrQualityPromptResult& prompt = report.prompts[i];
        if (i != 0) { out << ','; }
        out << "{\"index\":" << i << ",\"ok\":" << (prompt.ok ? "true" : "false")
            << ",\"compared_tokens\":" << prompt.compared_tokens
            << ",\"mean_logprob_divergence\":" << format_number(prompt.mean_logprob_divergence)
            << ",\"agreement_compared\":" << prompt.agreement_compared
            << ",\"agreement_matched\":" << prompt.agreement_matched
            << ",\"greedy_agreement\":" << format_number(prompt.greedy_agreement) << ",\"note\":\""
            << prompt.note << "\"}";
    }
    out << "],\"passed\":" << report.passed << ",\"total\":" << report.prompts.size()
        << ",\"all_passed\":" << (report.all_passed ? "true" : "false") << "}\n";
    return out.str();
}

} // namespace ninfer::targets::qwen3_6::detail

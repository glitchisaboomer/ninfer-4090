#include "targets/qwen3_6/impl/runtime/scr_quality.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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

    const std::size_t tokens_overlap =
        std::min(sample.baseline_tokens.size(), sample.scr_tokens.size());
    const std::size_t logprobs_overlap =
        std::min(sample.baseline_logprobs.size(), sample.scr_logprobs.size());
    // A position is measurable only when both sides have a token *and* a logprob there: without both
    // there is no pair of logprobs to compare, and no way to tell whether the tokens even agree.
    const std::size_t comparable = std::min(tokens_overlap, logprobs_overlap);
    result.logprobs_available    = static_cast<std::uint32_t>(comparable);

    // The agreed prefix: leading positions at which both paths emitted the same token. Truncating
    // here rather than at `comparable` is the entire point -- see the header.
    std::size_t agreed = comparable;
    for (std::size_t i = 0; i < comparable; ++i) {
        if (sample.baseline_tokens[i] != sample.scr_tokens[i]) {
            agreed = i;
            break;
        }
    }
    result.first_divergence_index = static_cast<std::uint32_t>(agreed);
    result.compared_tokens       = static_cast<std::uint32_t>(agreed);

    // A non-finite step is not a small divergence, it is an unusable measurement. Reporting NaN is
    // honest, and it makes the bound below fail, where averaging it in would quietly pass.
    if (agreed != 0) {
        double total = 0.0;
        double worst = 0.0;
        bool finite  = true;
        for (std::size_t i = 0; i < agreed; ++i) {
            const double delta = std::abs(static_cast<double>(sample.baseline_logprobs[i]) -
                                          static_cast<double>(sample.scr_logprobs[i]));
            if (!std::isfinite(delta)) {
                finite                       = false;
                result.logprobs_finite       = false;
                result.first_nonfinite_index = static_cast<std::uint32_t>(i);
                break;
            }
            total += delta;
            worst  = std::max(worst, delta);
        }
        if (finite) {
            result.mean_logprob_divergence = total / static_cast<double>(agreed);
            result.max_logprob_divergence  = worst;
        } else {
            result.mean_logprob_divergence = std::numeric_limits<double>::quiet_NaN();
            result.max_logprob_divergence  = std::numeric_limits<double>::quiet_NaN();
        }
        result.first_token_divergence = std::abs(static_cast<double>(sample.baseline_logprobs[0]) -
                                                 static_cast<double>(sample.scr_logprobs[0]));
    } else {
        result.mean_logprob_divergence = std::numeric_limits<double>::quiet_NaN();
        result.max_logprob_divergence  = std::numeric_limits<double>::quiet_NaN();
        result.first_token_divergence  = std::numeric_limits<double>::quiet_NaN();
    }

    // The margin at position 0, read from whichever side reported one. Both sides describe the same
    // column there when they agreed on the token, so either is a fact about the decision being
    // measured; a run that measured neither leaves this NaN rather than inventing a margin.
    {
        const auto at_zero = [](const std::vector<float>& margins) -> double {
            if (margins.empty()) { return std::numeric_limits<double>::quiet_NaN(); }
            return static_cast<double>(margins[0]);
        };
        const double baseline_margin = at_zero(sample.baseline_margins);
        const double scr_margin      = at_zero(sample.scr_margins);
        // The smaller is the conservative reading: it is the tightest lead either run had to overcome.
        const double reported = std::min(baseline_margin, scr_margin);
        if (std::isfinite(reported)) {
            result.margins_available = true;
            // An infinite margin is the degenerate single-candidate column, which is a real answer but
            // not a finite one; it stays NaN so nothing downstream reads it as "infinitely confident".
            result.first_token_margin = reported;
        } else {
            result.first_token_margin = std::numeric_limits<double>::quiet_NaN();
        }
    }

    const std::size_t agreement_limit =
        std::min<std::size_t>(thresholds.agreement_prefix_tokens, tokens_overlap);
    result.agreement_compared = static_cast<std::uint32_t>(agreement_limit);
    for (std::size_t i = 0; i < agreement_limit; ++i) {
        if (sample.baseline_tokens[i] == sample.scr_tokens[i]) { ++result.agreement_matched; }
    }
    if (agreement_limit != 0) {
        result.greedy_agreement =
            static_cast<double>(result.agreement_matched) / static_cast<double>(agreement_limit);
    }

    // Nothing measured is not a pass. A suite that silently scored absent data as agreement would
    // report success for exactly the runs it learned nothing about. The two absent things are
    // reported separately: a run with logprobs but no tokens failed to log its output, while a run
    // with neither logged no logprobs, and only one of those is a harness-side surprise.
    if (logprobs_overlap == 0) {
        result.note = "no overlapping logprobs to compare";
        return result;
    }
    if (tokens_overlap == 0) {
        result.note = "no overlapping tokens to compare";
        return result;
    }
    if (!result.logprobs_finite) {
        result.note = "logprobs are not finite at position " +
                      std::to_string(result.first_nonfinite_index);
        return result;
    }
    // The paths disagreed on the very first token, so there is no agreed prefix to average over. The
    // thresholds cannot be evaluated; say that rather than reporting a mean over nothing as perfect.
    if (agreed == 0) {
        result.note = "the two paths chose different first tokens (baseline " +
                      std::to_string(sample.baseline_tokens[0]) + ", scr " +
                      std::to_string(sample.scr_tokens[0]) +
                      "), so there is no agreed prefix to measure divergence over";
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
             << " over the " << agreed << " agreed leading token(s) exceeds "
             << format_number(thresholds.mean_logprob_divergence_nats);
    }
    if (!agreement_ok) {
        if (!divergence_ok) { note << "; "; }
        note << "greedy agreement " << format_number(result.greedy_agreement) << " below "
             << format_number(thresholds.greedy_agreement) << " (first divergence at token "
             << result.first_divergence_index << ")";
    }
    result.note = note.str();
    return result;
}

ScrQualityPromptResult evaluate_scr_prompt_against_control(const ScrQualitySample& sample,
                                                           const ScrQualitySample& control,
                                                           const ScrQualityThresholds& thresholds) {
    ScrQualityPromptResult result = evaluate_scr_prompt(sample, thresholds);
    const ScrQualityPromptResult floor = evaluate_scr_prompt(control, thresholds);

    // A control with nothing to compare establishes no floor. Reporting zero here would be the exact
    // mistake this function exists to prevent, in the one case where it is easiest to miss.
    if (floor.compared_tokens == 0 || floor.logprobs_available == 0) { return result; }

    result.control_available            = true;
    result.floor_first_token_divergence = floor.first_token_divergence;
    result.floor_mean_logprob_divergence = floor.mean_logprob_divergence;
    if (std::isfinite(result.first_token_divergence) &&
        std::isfinite(floor.first_token_divergence)) {
        result.signal_first_token_divergence =
            result.first_token_divergence - floor.first_token_divergence;
        result.above_floor = result.first_token_divergence > floor.first_token_divergence;
    } else {
        result.signal_first_token_divergence = std::numeric_limits<double>::quiet_NaN();
    }
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

std::vector<std::uint32_t> scr_quality_control_divergent_prompts(const ScrQualityReport& control) {
    std::vector<std::uint32_t> divergent;
    for (std::size_t i = 0; i < control.prompts.size(); ++i) {
        // parted@ == logprobs_available means the two runs chose the same token at every position
        // both of them had. Anything earlier is a disagreement, however small the logprob gap was.
        if (control.prompts[i].first_divergence_index < control.prompts[i].logprobs_available) {
            divergent.push_back(static_cast<std::uint32_t>(i));
        }
    }
    return divergent;
}

bool scr_quality_control_is_clean(const ScrQualityReport& control) {
    return !control.prompts.empty() && scr_quality_control_divergent_prompts(control).empty();
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
            << ",\"logprobs_available\":" << prompt.logprobs_available
            << ",\"first_divergence_index\":" << prompt.first_divergence_index
            << ",\"compared_tokens\":" << prompt.compared_tokens
            << ",\"mean_logprob_divergence\":" << format_number(prompt.mean_logprob_divergence)
            << ",\"max_logprob_divergence\":" << format_number(prompt.max_logprob_divergence)
            << ",\"first_token_divergence\":" << format_number(prompt.first_token_divergence)
            << ",\"logprobs_finite\":" << (prompt.logprobs_finite ? "true" : "false")
            << ",\"margins_available\":" << (prompt.margins_available ? "true" : "false")
            << ",\"first_token_margin\":" << format_number(prompt.first_token_margin)
            << ",\"control_available\":" << (prompt.control_available ? "true" : "false")
            << ",\"floor_first_token_divergence\":"
            << format_number(prompt.floor_first_token_divergence)
            << ",\"floor_mean_logprob_divergence\":"
            << format_number(prompt.floor_mean_logprob_divergence)
            << ",\"signal_first_token_divergence\":"
            << format_number(prompt.signal_first_token_divergence)
            << ",\"above_floor\":" << (prompt.above_floor ? "true" : "false")
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

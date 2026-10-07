#include "serve/scr_quality_log.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

// The log's own schema version, written into server_start records. A `request_done` from a newer
// schema may have moved or renamed the fields read here, so it is refused rather than guessed at.
constexpr int kKnownRequestLogSchemaVersion = 9;

std::int32_t required_i32(const Json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_number_integer()) {
        throw std::invalid_argument(std::string("request log record is missing integer ") + key);
    }
    return it->get<std::int32_t>();
}

std::uint32_t required_u32(const Json& object, const char* key) {
    const auto value = required_i32(object, key);
    if (value < 0) {
        throw std::invalid_argument(std::string("request log record has negative ") + key);
    }
    return static_cast<std::uint32_t>(value);
}

std::string required_string(const Json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_string()) {
        throw std::invalid_argument(std::string("request log record is missing string ") + key);
    }
    return it->get<std::string>();
}

void reject_newer_schema(const Json& record) {
    const auto it = record.find("schema_version");
    if (it == record.end() || !it->is_number_integer()) { return; }
    if (it->get<int>() > kKnownRequestLogSchemaVersion) {
        throw std::invalid_argument("request log schema is newer than this reader understands");
    }
}

LoggedCompletion parse_record_checked(const Json& record);

LoggedCompletion parse_record(const Json& record) {
    // Every access below is guarded as a whole: a record missing a field this reader needs is
    // malformed for its purpose, whichever library error a particular access raises.
    try {
        return parse_record_checked(record);
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("request log record is malformed: ") +
                                    error.what());
    }
}

LoggedCompletion parse_record_checked(const Json& record) {
    reject_newer_schema(record);
    const Json& request = record.at("request");
    const Json& result  = record.at("result");
    LoggedCompletion completion;
    completion.request_id                = request.at("request_id").get<std::uint64_t>();
    completion.prompt_tokens             = required_i32(result, "prompt_tokens");
    completion.prefix_cache_hit_tokens   = required_u32(result, "prefix_cache_hit_tokens");
    completion.prefix_relocated_tokens   = required_u32(result, "prefix_relocated_tokens");
    completion.prefix_reuse_path         = required_string(result, "prefix_reuse_path");

    // `generated` was added with schema 9. An older record has no per-token data, which is not a
    // parse failure but does leave nothing to score for that request.
    const auto generated = result.find("generated");
    if (generated == result.end()) { return completion; }
    const auto tokens = generated->find("token_ids");
    const auto probs  = generated->find("logprobs");
    if (tokens == generated->end() || probs == generated->end() || !tokens->is_array() ||
        !probs->is_array()) {
        throw std::invalid_argument("request log result.generated is malformed");
    }
    if (tokens->size() != probs->size()) {
        throw std::invalid_argument(
            "request log carries a different number of tokens and logprobs");
    }
    completion.generated_token_ids.reserve(tokens->size());
    completion.generated_logprobs.reserve(probs->size());
    for (const Json& token : *tokens) {
        if (!token.is_number_integer()) {
            throw std::invalid_argument("request log token id is not an integer");
        }
        completion.generated_token_ids.push_back(token.get<ninfer::TokenId>());
    }
    for (const Json& prob : *probs) {
        // null is the log's spelling for "this path measured no logprob" (the MTP/DFlash licensed
        // tokens); NaN carries the same meaning into the scorer, which fails rather than passing.
        if (prob.is_null()) {
            completion.generated_logprobs.push_back(std::numeric_limits<float>::quiet_NaN());
            continue;
        }
        if (!prob.is_number()) {
            throw std::invalid_argument("request log logprob is neither a number nor null");
        }
        completion.generated_logprobs.push_back(prob.get<float>());
    }
    return completion;
}

} // namespace

std::vector<LoggedCompletion> parse_completion_log(std::string_view jsonl) {
    std::vector<LoggedCompletion> completions;
    std::size_t begin = 0;
    while (begin < jsonl.size()) {
        std::size_t end = jsonl.find('\n', begin);
        if (end == std::string_view::npos) { end = jsonl.size(); }
        const std::string_view line = jsonl.substr(begin, end - begin);
        begin                       = end + 1;
        if (line.find_first_not_of(" \t\r") == std::string_view::npos) { continue; }

        Json record;
        try {
            record = Json::parse(line);
        } catch (const Json::exception& error) {
            throw std::invalid_argument(std::string("request log line is not JSON: ") +
                                        error.what());
        }
        const auto event = record.find("event");
        if (event == record.end() || !event->is_string()) { continue; }
        if (event->get<std::string>() != "request_done") { continue; }
        completions.push_back(parse_record(record));
    }
    return completions;
}

std::vector<targets::qwen3_6::detail::ScrQualitySample>
pair_scr_quality_samples(std::span<const LoggedCompletion> scr,
                         std::span<const LoggedCompletion> baseline) {
    if (scr.size() != baseline.size()) {
        throw std::invalid_argument("the two runs logged a different number of completions");
    }
    std::vector<targets::qwen3_6::detail::ScrQualitySample> samples;
    samples.reserve(scr.size());
    for (std::size_t index = 0; index < scr.size(); ++index) {
        const LoggedCompletion& scr_completion      = scr[index];
        const LoggedCompletion& baseline_completion = baseline[index];
        if (scr_completion.prompt_tokens != baseline_completion.prompt_tokens) {
            throw std::invalid_argument(
                "completion " + std::to_string(index) +
                " was compared across two different prompts: " +
                std::to_string(scr_completion.prompt_tokens) + " vs " +
                std::to_string(baseline_completion.prompt_tokens) + " prompt tokens");
        }
        // Suffix Cache Reuse that relocated nothing is not a measurement of Suffix Cache Reuse.
        // Scoring it anyway would report near-zero divergence for a run in which the feature never
        // engaged, which is the one result that looks like success and means nothing.
        if (scr_completion.prefix_relocated_tokens == 0) {
            throw std::invalid_argument(
                "completion " + std::to_string(index) +
                " reused no relocated tokens (path=" + scr_completion.prefix_reuse_path +
                "); the prompt suite did not exercise Suffix Cache Reuse");
        }
        targets::qwen3_6::detail::ScrQualitySample sample;
        sample.baseline_logprobs = baseline_completion.generated_logprobs;
        sample.scr_logprobs      = scr_completion.generated_logprobs;
        sample.baseline_tokens   = baseline_completion.generated_token_ids;
        sample.scr_tokens        = scr_completion.generated_token_ids;
        samples.push_back(std::move(sample));
    }
    return samples;
}

std::string format_scr_quality_report(
    std::span<const LoggedCompletion> scr, std::span<const LoggedCompletion> baseline,
    const targets::qwen3_6::detail::ScrQualityThresholds& thresholds) {
    if (scr.empty() || baseline.empty()) { return {}; }
    const auto samples = pair_scr_quality_samples(scr, baseline);
    const auto report  = targets::qwen3_6::detail::evaluate_scr_quality(samples, thresholds);
    return targets::qwen3_6::detail::format_scr_quality_json(report, thresholds);
}

} // namespace ninfer::serve

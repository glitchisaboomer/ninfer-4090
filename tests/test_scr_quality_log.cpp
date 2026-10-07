// Phase 5.5 of docs/SCR_ROADMAP.md, served end: the request JSONL log is the only channel that
// carries per-token logprobs out of ninfer-serve, so the reader that turns it into scoring input is
// the place a quality run can silently go wrong. These checks are all host-side: no CUDA, no model,
// no server. The logprobs are fabricated, which is what makes the expected divergence exactly
// known -- the arithmetic itself is the target's own, tested separately.

#include "serve/scr_quality_log.h"
#include "serve/serve_options.h"
#include "targets/qwen3_6/impl/runtime/scr_quality.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace scr_quality = ninfer::targets::qwen3_6::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void expect_invalid(std::string_view message, const std::function<void()>& body) {
    try {
        body();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL: " << message << " threw the wrong type: " << error.what() << '\n';
        return;
    }
    ++failures;
    std::cerr << "FAIL: " << message << " did not throw\n";
}

// One request_done record with the schema the server writes. `logprobs` takes the JSON spelling
// directly so a null (an unmeasured logprob) can be expressed.
std::string record(std::uint64_t request_id, int prompt_tokens, std::uint32_t cache_hit_tokens,
                   std::uint32_t relocated_tokens, std::string_view reuse_path,
                   std::string_view token_ids, std::string_view logprobs) {
    return std::string{"{\"schema_version\":9,\"event\":\"request_done\",\"request\":{\"request_id\":"} +
           std::to_string(request_id) + "},\"result\":{\"prompt_tokens\":" +
           std::to_string(prompt_tokens) +
           ",\"prefix_cache_hit_tokens\":" + std::to_string(cache_hit_tokens) +
           ",\"prefix_relocated_tokens\":" + std::to_string(relocated_tokens) +
           ",\"prefix_reuse_path\":\"" + std::string(reuse_path) +
           "\",\"generated\":{\"token_ids\":" + std::string(token_ids) +
           ",\"logprobs\":" + std::string(logprobs) + "}}}";
}

std::string joined(const std::vector<std::string>& lines) {
    std::string out;
    for (const std::string& line : lines) { out += line + "\n"; }
    return out;
}

void test_parse_reads_only_completion_records() {
    const std::string log = joined({
        "{\"schema_version\":9,\"event\":\"server_start\"}",
        record(7, 512, 64, 64, "suffix_relocation", "[11,22,33]", "[0.5,0.25,0.125]"),
        "{\"schema_version\":9,\"event\":\"throughput\"}",
        record(8, 600, 32, 32, "append_at_frontier", "[44]", "[0.5]"),
    });
    const std::vector<ninfer::serve::LoggedCompletion> completions =
        ninfer::serve::parse_completion_log(log);

    expect(completions.size() == 2, "only request_done records become completions");
    expect(completions[0].request_id == 7 && completions[0].prompt_tokens == 512,
           "request id and prompt token count are read");
    expect(completions[0].prefix_relocated_tokens == 64 &&
               completions[0].prefix_reuse_path == "suffix_relocation",
           "reuse telemetry is read");
    expect(completions[0].generated_token_ids == std::vector<ninfer::TokenId>({11, 22, 33}),
           "generated token ids are read in order");
    expect(completions[0].generated_logprobs.size() == 3 &&
               completions[0].generated_logprobs[0] == 0.5f &&
               completions[0].generated_logprobs[2] == 0.125f,
           "generated logprobs are read in order");
    expect(completions[1].prefix_reuse_path == "append_at_frontier",
           "a non-relocating path is read as written");
}

void test_null_logprob_is_unmeasured() {
    const std::string log =
        record(1, 100, 64, 64, "suffix_relocation", "[1,2]", "[null,0.5]") + "\n";
    const auto completions = ninfer::serve::parse_completion_log(log);
    expect(completions.size() == 1 && completions[0].generated_logprobs.size() == 2,
           "a null logprob keeps the token it belongs to");
    expect(std::isnan(completions[0].generated_logprobs[0]),
           "a null logprob becomes NaN, which is how an unmeasured token is carried");
    expect(completions[0].generated_logprobs[1] == 0.5f, "the measured logprob is unaffected");
}

void test_malformed_records_are_refused() {
    expect_invalid("a length mismatch between tokens and logprobs", [] {
        (void)ninfer::serve::parse_completion_log(
            record(1, 100, 64, 64, "suffix_relocation", "[1,2]", "[0.5]") + "\n");
    });
    expect_invalid("a newer schema version", [] {
        (void)ninfer::serve::parse_completion_log(
            "{\"schema_version\":99,\"event\":\"request_done\",\"request\":{\"request_id\":1},"
            "\"result\":{\"prompt_tokens\":1,\"prefix_cache_hit_tokens\":0,"
            "\"prefix_relocated_tokens\":0,\"prefix_reuse_path\":\"full_reset\"}}\n");
    });
    expect_invalid("a line that is not JSON", [] {
        (void)ninfer::serve::parse_completion_log("{not json}\n");
    });
    expect_invalid("a record missing its reuse telemetry", [] {
        (void)ninfer::serve::parse_completion_log(
            "{\"schema_version\":9,\"event\":\"request_done\",\"request\":{\"request_id\":1},"
            "\"result\":{\"prompt_tokens\":1}}\n");
    });
    expect_invalid("a logprob that is neither a number nor null", [] {
        (void)ninfer::serve::parse_completion_log(
            record(1, 100, 64, 64, "suffix_relocation", "[1]", "[\"high\"]") + "\n");
    });
    expect_invalid("a token id that is not an integer", [] {
        (void)ninfer::serve::parse_completion_log(
            record(1, 100, 64, 64, "suffix_relocation", "[\"1\"]", "[0.5]") + "\n");
    });
}

void test_pairing_requires_comparable_runs() {
    const auto scr      = ninfer::serve::parse_completion_log(
        record(1, 800, 64, 64, "suffix_relocation", "[1,2,3]", "[0.5,0.5,0.5]") + "\n");
    const auto baseline = ninfer::serve::parse_completion_log(
        record(1, 800, 0, 0, "full_reset", "[1,2,3]", "[0.5,0.5,0.5]") + "\n");
    const auto samples = ninfer::serve::pair_scr_quality_samples(scr, baseline);
    expect(samples.size() == 1, "one prompt produces one comparison sample");
    expect(samples[0].baseline_logprobs.size() == 3 && samples[0].scr_logprobs.size() == 3,
           "the sample carries both paths' logprobs");
    expect(samples[0].baseline_tokens.size() == 3 && samples[0].scr_tokens.size() == 3,
           "the sample carries both paths' tokens");

    expect_invalid("a different number of completions", [&] {
        const auto short_log = ninfer::serve::parse_completion_log(
            record(2, 900, 32, 32, "suffix_relocation", "[9]", "[0.1]") + "\n");
        (void)ninfer::serve::pair_scr_quality_samples(scr, short_log);
    });

    expect_invalid("two different prompts", [&] {
        const auto other = ninfer::serve::parse_completion_log(
            record(1, 799, 0, 0, "full_reset", "[1,2,3]", "[0.5,0.5,0.5]") + "\n");
        (void)ninfer::serve::pair_scr_quality_samples(scr, other);
    });

    // The one failure that matters most: a run in which Suffix Cache Reuse never engaged must not
    // be scored, because two full prefills agree and would read as a pass.
    expect_invalid("a run that relocated nothing", [&] {
        const auto no_reuse = ninfer::serve::parse_completion_log(
            record(1, 800, 0, 0, "full_reset", "[1,2,3]", "[0.5,0.5,0.5]") + "\n");
        const auto base     = ninfer::serve::parse_completion_log(
            record(1, 800, 0, 0, "full_reset", "[1,2,3]", "[0.5,0.5,0.5]") + "\n");
        (void)ninfer::serve::pair_scr_quality_samples(no_reuse, base);
    });
}

void test_report_scores_and_fails_honestly() {
    const scr_quality::ScrQualityThresholds thresholds;

    // Identical paths: the divergence is exactly zero and every leading token agrees.
    const std::string tokens = "[1,2,3,4,5]";
    const std::string probs  = "[0.5,0.25,-0.125,-1.5,2.0]";
    const auto scr = ninfer::serve::parse_completion_log(
        record(1, 800, 64, 64, "suffix_relocation", tokens, probs) + "\n");
    const auto baseline = ninfer::serve::parse_completion_log(
        record(1, 800, 0, 0, "full_reset", tokens, probs) + "\n");
    const std::string passing =
        ninfer::serve::format_scr_quality_report(scr, baseline, thresholds);
    expect(passing.find("\"all_passed\":true") != std::string::npos,
           "identical runs pass with the target's own thresholds");
    expect(passing.find("\"mean_logprob_divergence\":0.000000") != std::string::npos,
           "identical runs report zero divergence");
    expect(passing.find("\"greedy_agreement\":1.000000") != std::string::npos,
           "identical runs report full agreement");

    // A baseline token swapped at the first position: agreement drops and the divergence is the
    // exact mean absolute difference the roadmap's threshold is written against.
    const auto flipped = ninfer::serve::parse_completion_log(
        record(1, 800, 0, 0, "full_reset", "[2,2,3,4,5]", probs) + "\n");
    const std::string failing =
        ninfer::serve::format_scr_quality_report(scr, flipped, thresholds);
    expect(failing.find("\"all_passed\":false") != std::string::npos,
           "a flipped token fails the comparison");
    expect(failing.find("\"greedy_agreement\":0.800000") != std::string::npos,
           "agreement is measured over the compared prefix");

    // An unmeasured logprob is NaN, so it can never read as agreement.
    const auto unmeasured = ninfer::serve::parse_completion_log(
        record(1, 800, 0, 0, "full_reset", tokens, "[null,0.25,-0.125,-1.5,2.0]") + "\n");
    const std::string nan_report =
        ninfer::serve::format_scr_quality_report(scr, unmeasured, thresholds);
    expect(nan_report.find("\"all_passed\":false") != std::string::npos,
           "a run carrying unmeasured logprobs fails rather than passing");

    expect(ninfer::serve::format_scr_quality_report({}, {}, thresholds).empty(),
           "an empty pair produces no report");
}

// End to end over the log schema, for the failure the in-memory scorer tests cannot see: two real
// records that diverge partway through. Every logprob past the flip is the logprob of a *different*
// token on each side, so a scorer that averaged the whole run would report a mean in the tenths for a
// run that agreed perfectly up to the flip. This is what the 2026-10-08 measurement actually did.
void test_divergence_after_a_flip_is_not_averaged_in() {
    const scr_quality::ScrQualityThresholds thresholds;
    // The paths agree on tokens 1..3 and part at token 4. After the flip each side logs a *different*
    // token, so each side's logprobs past position 3 are logprobs of its own tokens. They are made
    // deliberately far apart: a scorer that averaged all six pairs reports a mean near 1.9 nats for a
    // run whose first three steps are bit-identical, which is exactly the failure that produced the
    // 2026-10-08 report. A scorer that stops at the flip sees zero.
    const auto scr = ninfer::serve::parse_completion_log(
        record(1, 800, 64, 64, "suffix_relocation", "[1,2,3,9,9,9]",
               "[0.0,0.0,0.0,-0.5,-0.25,-0.125]") +
        "\n");
    const auto baseline = ninfer::serve::parse_completion_log(
        record(1, 800, 0, 0, "full_reset", "[1,2,3,4,5,6]",
               "[0.0,0.0,0.0,-3.0,-4.0,-5.0]") +
        "\n");

    const auto scored = ninfer::serve::evaluate_scr_quality_report(scr, baseline, thresholds);
    expect(scored.prompts.size() == 1, "one prompt is scored");
    const auto& prompt = scored.prompts.front();
    expect(prompt.first_divergence_index == 3, "the first divergent token is reported");
    expect(prompt.compared_tokens == 3, "only the agreed prefix is compared");
    expect(prompt.logprobs_available == 6, "the run length available is reported");
    expect(prompt.mean_logprob_divergence == 0.0,
           "the agreeing prefix diverges by nothing, not by the ~1.9 nats an all-pairs mean gives");
    expect(prompt.max_logprob_divergence == 0.0, "the worst agreed step is also zero");
    expect(prompt.first_token_divergence == 0.0, "the first token's divergence is reported");
    // Agreement still covers all six positions, because the roadmap asks whether greedy decoding
    // flipped, not only whether the prefix held.
    expect(prompt.agreement_compared == 6, "agreement measures the whole available prefix");
    expect(prompt.agreement_matched == 3, "the three divergent positions do not match");
    expect(prompt.greedy_agreement == 0.5, "agreement reports the flipped half");

    // The serialized form carries the new fields, since a report that omits them cannot be read.
    const std::string json = ninfer::serve::format_scr_quality_report(scr, baseline, thresholds);
    expect(json.find("\"first_divergence_index\":3") != std::string::npos,
           "the divergence point is serialized");
    expect(json.find("\"logprobs_available\":6") != std::string::npos,
           "the available length is serialized");
    expect(json.find("\"first_token_divergence\":") != std::string::npos,
           "the first-token divergence is serialized");
    expect(json.find("\"max_logprob_divergence\":") != std::string::npos,
           "the worst agreed step is serialized");
}


// The argv a measurement phase hands to ninfer-serve, checked against the server's own parser rather
// than against a reading of its usage text. The two disagree in exactly one way that matters:
// ninfer-serve takes the artifact as its first positional argument, so an invocation that passes
// --model leaves argv[1] unset and the server exits at argument-parse time. Spawned from a harness
// that failure surfaces as a health-check timeout minutes later, having loaded nothing, and the
// process output is the only evidence. Feeding the argv to parse_serve_options makes it a unit test.
void test_server_arguments_are_a_valid_invocation() {
    const std::string artifact = "models/qwen3_8_27b.ninfer";
    const std::vector<std::string> extra{"--kv-dtype", "bf16"};

    // The real parser, not a reading of the usage text. The two disagree in exactly one way that
    // matters: ninfer-serve reads the artifact from argv[1] and starts options at argv[2], so an
    // invocation passing --model leaves argv[1] unset and the server exits at argument-parse time.
    // Spawned from a harness, that failure surfaces as a health-check timeout minutes later with
    // nothing loaded and the server's usage text as the only evidence.
    const auto parse = [&artifact](const std::vector<std::string>& tail) {
        std::vector<std::string> argv{"ninfer-serve"};
        argv.insert(argv.end(), tail.begin(), tail.end());
        std::vector<char*> raw;
        raw.reserve(argv.size() + 1);
        for (std::string& argument : argv) { raw.push_back(argument.data()); }
        raw.push_back(nullptr);
        return ninfer::serve::parse_serve_options(static_cast<int>(raw.size()) - 1, raw.data());
    };

    for (const bool scr : {true, false}) {
        const std::string what = scr ? "scr" : "baseline";
        const std::vector<std::string> tail = ninfer::serve::build_scr_quality_server_arguments(
            artifact, 18080, 8192, "run/scr.jsonl", scr, extra);

        ninfer::serve::ServeOptions options;
        try {
            options = parse(tail);
        } catch (const std::exception& error) {
            expect(false, "the " + what + " phase argv is not a valid ninfer-serve invocation: " +
                 std::string(error.what()));
            continue;
        }

        expect(!tail.empty() && tail.front() == artifact,
               what + ": the artifact must be the first argument");
        expect(options.artifact_path == artifact,
               what + ": artifact_path is \"" + options.artifact_path + "\" rather than \"" +
                   artifact + "\"");
        expect(options.port == 18080, what + ": port did not survive the round trip");
        expect(options.max_context == 8192u, what + ": max_context did not survive the round trip");
        expect(options.request_log_jsonl == "run/scr.jsonl",
               what + ": the request log path did not survive the round trip");
        expect(options.max_concurrency == 1u,
               what + ": a phase must run one slot at a time or the log order is not send order");

        // The flags that make each phase mean what it claims, asserted as parsed state rather than
        // by scanning the argument list: the parser is the contract, and re-reading argv here would
        // only test this file against itself.
        expect(options.enable_suffix_cache_reuse == scr,
               what + ": enable_suffix_cache_reuse does not match the phase");
        expect(!options.scr_plan_segmented, what + ": the phase must plan a tail relocation");
    }

    // The baseline has to be a full prefill. With prefix reuse left on, the two phases share a cache
    // and would agree for reasons that have nothing to do with relocation -- which reads as a pass.
    const std::vector<std::string> with_baseline_flags{"--no-prefix-reuse"};
    const ninfer::serve::ServeOptions baseline = parse(
        ninfer::serve::build_scr_quality_server_arguments(artifact, 18081, 8192,
                                                          "run/baseline.jsonl", false,
                                                          with_baseline_flags));
    expect(!baseline.allow_prefix_reuse,
           "baseline: prefix reuse is still enabled, so the phases would agree for reasons "
           "unrelated to relocation");
    expect(!baseline.enable_suffix_cache_reuse, "baseline: Suffix Cache Reuse is still enabled");
    expect(baseline.port == 18081, "baseline: the second phase must not reuse the first phase's "
                                   "port");
}

// The control is the same configuration run twice, so it relocates nothing on either side -- which is
// exactly what pair_scr_quality_samples refuses. The control needs its own pairing entry point, and
// the point of these checks is that it is a *deliberate* bypass rather than a weakened guard: the
// measurement pairing still rejects a run that relocated nothing.
void test_control_pairs_two_full_prefills() {
    const std::string tokens = "[1,2,3,4]";
    const std::string probs  = "[0.5,0.25,-0.125,-1.5]";
    const auto first = ninfer::serve::parse_completion_log(
        record(1, 800, 0, 0, "full_reset", tokens, probs) + "\n");
    const auto second = ninfer::serve::parse_completion_log(
        record(1, 800, 0, 0, "full_reset", tokens, probs) + "\n");

    // The measurement pairing still refuses, so the guard has not been quietly relaxed.
    expect_invalid("the measurement pairing still requires relocation",
                   [&] { (void)ninfer::serve::pair_scr_quality_samples(first, second); });

    const auto samples = ninfer::serve::pair_scr_quality_samples_for_control(first, second);
    expect(samples.size() == 1, "the control pairs two full prefills");
    expect(samples[0].baseline_tokens.size() == 4, "both sides' tokens are carried through");

    // Two identical full prefills are a clean control: the floor is zero.
    const auto control = ninfer::serve::evaluate_scr_quality_control(
        first, second, scr_quality::ScrQualityThresholds{});
    expect(control.prompts.size() == 1, "the control is scored");
    expect(scr_quality::scr_quality_control_is_clean(control),
           "two identical full prefills are a clean control");
    expect(control.prompts.front().first_divergence_index == 4,
           "the control agreed at every position it compared");
    expect(control.prompts.front().logprobs_available == 4,
           "all four positions were available to disagree at");

    // A control that flips a token is not clean, even though the logprob gap that caused it is small.
    const auto flipped = ninfer::serve::parse_completion_log(
        record(1, 800, 0, 0, "full_reset", "[1,2,9,4]", probs) + "\n");
    const auto noisy = ninfer::serve::evaluate_scr_quality_control(
        first, flipped, scr_quality::ScrQualityThresholds{});
    expect(!scr_quality::scr_quality_control_is_clean(noisy),
           "a control that flipped a token cannot authorize a verdict");
    const auto divergent = scr_quality::scr_quality_control_divergent_prompts(noisy);
    expect(divergent.size() == 1 && divergent[0] == 0, "the flipped prompt is named");

    // The prompt-length check is not bypassed along with the relocation check.
    const auto other_length = ninfer::serve::parse_completion_log(
        record(1, 900, 0, 0, "full_reset", tokens, probs) + "\n");
    expect_invalid("the control still refuses mismatched prompt lengths", [&] {
        (void)ninfer::serve::pair_scr_quality_samples_for_control(first, other_length);
    });
}

} // namespace

int main() {
    test_parse_reads_only_completion_records();
    test_null_logprob_is_unmeasured();
    test_malformed_records_are_refused();
    test_pairing_requires_comparable_runs();
    test_report_scores_and_fails_honestly();
    test_divergence_after_a_flip_is_not_averaged_in();
    test_control_pairs_two_full_prefills();
    test_server_arguments_are_a_valid_invocation();
    if (failures != 0) {
        std::cerr << failures << " SCR quality log checks failed\n";
        return 1;
    }
    std::cout << "SCR quality log checks passed\n";
    return 0;
}

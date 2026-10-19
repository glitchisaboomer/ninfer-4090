#include "ninfer/engine.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

// Prefill must be reproducible. Position 0's logprob is produced by the last prefill chunk and
// nothing else -- no decode round, no sampler, no speculation -- so it isolates the forward pass.
//
// This test exists because four `ninfer-serve` processes given identical flags (the scr-quality
// harness's `baseline` and `control` phases) disagreed at position 0 by 3e-4..3e-3 nats, which is
// large enough to invalidate the Suffix Cache Reuse quality harness's noise floor. See
// docs/SCR.md.
//
// The tolerance is a deliberate tripwire, not a numerical criterion: it exists so that any future
// divergence fails loudly. `NINFER_PREFILL_DET_TOL` overrides it -- set it to 0 to demand
// bit-identity. `NINFER_PREFILL_DET_VERBOSE=1` prints the measured delta. Under CTest a passing
// test's stdout is suppressed, so the verbose line needs `-V`.
//
// Two things here report structure rather than a pass/fail:
//
//   * the first case prefills the same prompt three times and prints all three values, because the
//     shape of the trio -- not either delta -- is what says whether the divergence is a function of
//     call index; and
//   * `NINFER_PREFILL_TRACE=1` turns on the runtime's per-layer residual diff, which names the
//     layer whose output bytes first differ between two prefills and therefore the op that
//     diverged. It is opt-in there rather than forced here because it copies every layer's residual
//     tile to the host; the note this test prints says how to switch it on. Its window is a
//     thread_local and each Engine owns a worker thread, so naming a layer for the two-Engine leg
//     below also needs `NINFER_PREFILL_TRACE_CROSS=1`, which compares against the last chunk of the
//     same shape published by any Engine in the process.
constexpr float kLogprobTolerance = 1.0e-6F;

float logprob_tolerance() {
    const char* override_value = std::getenv("NINFER_PREFILL_DET_TOL");
    if (override_value == nullptr || *override_value == '\0') { return kLogprobTolerance; }
    try {
        return std::stof(override_value);
    } catch (const std::exception&) {
        std::cerr << "ignoring unparsable NINFER_PREFILL_DET_TOL=" << override_value << '\n';
        return kLogprobTolerance;
    }
}

// The configuration `ninfer-serve` actually runs with, copied from src/serve/serve_options.h and
// built the way ninfer::serve::build_scr_quality_server_arguments builds it. max_context is 8192
// here, not EngineOptions' 4096 default; an earlier version of this test used 4096 and reported
// delta=0, which does not describe the configuration the harness measures.
ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 8192;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.max_concurrency           = 1;
    options.prefill_chunk             = 1024;
    options.kv_cache                  = ninfer::KvCacheStorage::BFloat16;
    options.enable_vision             = false;
    // SpeculativeBackend::None and prefix reuse are the served defaults for this suite. A licensed
    // MTP/DFlash token reports NaN logprob by contract, which would make the comparison vacuous.
    return options;
}

// The k2 suite prompt, verbatim from apps/scr_quality built_in_prompt: same four shared exchanges,
// same deployment brief, same closing question, so the prompt runs through the artifact's real
// tokenizer and produces a real expert-routing distribution.
//
// Synthetic ids would be cheaper and were used by the first version of this test, but they are not
// equivalent: an id sequence chosen by hand does not reproduce the mixture of experts that this
// text activates, and this test exists to observe exactly that. Engine::prepare_tokens takes ids
// only and PreparedPrompt is opaque, so the text goes through Engine::prepare the way the served
// API reaches the same forward pass.
// Copied verbatim from apps/scr_quality, so the tail filler is byte-identical to the suite's and
// therefore tokenizes to the same ids. `blocks=2` is the k2 case under investigation.
constexpr const char* kTailSentences[] = {
    "The maintenance log for the deployment records every change in order.",
    "Each entry lists the operator, the timestamp, and a short justification.",
    "Rollbacks were performed twice, both times within the same maintenance window.",
    "The verification suite ran after every rollout without exception.",
    "Throughput stayed within the agreed band during the whole observation period.",
    "No customer-visible incidents were attributed to these changes.",
};
constexpr std::int32_t kTokensPerTailSentence = 8;
constexpr std::int32_t kBlockTokens           = 64;

std::string repeated_tail(int blocks) {
    const auto sentences = static_cast<std::size_t>(blocks) * kBlockTokens / kTokensPerTailSentence;
    constexpr std::size_t sentence_count =
        sizeof(kTailSentences) / sizeof(kTailSentences[0]);
    std::string tail;
    for (std::size_t index = 0; index < sentences; ++index) {
        if (!tail.empty()) { tail += ' '; }
        tail += kTailSentences[index % sentence_count];
    }
    return tail;
}

ninfer::PromptInput suite_prompt() {
    ninfer::ChatMessage system;
    system.role = "system";
    system.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text,
        .text = "You are an operations assistant. Answer from the deployment records you are given, "
                "state uncertainty when the records do not settle a question, and keep answers "
                "short.",
        .media = {}});

    ninfer::PromptInput input;
    // The harness posts plain messages and sets nothing else, so it takes the thinking-enabled
    // default. Left explicit because PromptOptions::enable_thinking defaults to true and a silent
    // change to it would move every token id in this fixture.
    input.options.enable_thinking = true;
    input.messages.push_back(std::move(system));
    for (int turn = 0; turn < 4; ++turn) {
        ninfer::ChatMessage ask;
        ask.role = "user";
        ask.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "SECTION ZERO." + std::to_string(turn) +
                    " Standing instructions for the maintenance window: confirm every figure "
                    "against the records, name the window it belongs to, and do not speculate "
                    "about counts that are not stated.",
            .media = {}});
        input.messages.push_back(std::move(ask));

        ninfer::ChatMessage answer;
        answer.role = "assistant";
        answer.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "Understood section zero " + std::to_string(turn) +
                    ". I will confirm each figure against the records and will say so when the "
                    "records do not settle a question.",
            .media = {}});
        input.messages.push_back(std::move(answer));
    }

    const std::string tail = repeated_tail(2);
    const ninfer::ChatMessage brief{
        .role = "user",
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                      .text = "Deployment brief. SECTION ONE. The migration moved "
                                              "the primary datastore to a three node cluster, and "
                                              "all writes were routed through the new leader. " +
                                                  tail,
                                      .media = {}}}};
    input.messages.push_back(brief);
    input.messages.push_back(
        ninfer::ChatMessage{.role = "assistant",
                            .parts = {ninfer::MessagePart{
                                .kind = ninfer::MessagePartKind::Text,
                                .text = "Understood. I have read the deployment records.",
                                .media = {}}}});
    input.messages.push_back(
        ninfer::ChatMessage{.role = "user",
                            .parts = {ninfer::MessagePart{
                                .kind = ninfer::MessagePartKind::Text,
                                .text = "SECTION TWO. Verification ran after every rollout.",
                                .media = {}}}});
    input.messages.push_back(ninfer::ChatMessage{.role = "assistant",
                                                 .parts = {ninfer::MessagePart{
                                                     .kind = ninfer::MessagePartKind::Text,
                                                     .text = "Noted.",
                                                     .media = {}}}});
    input.messages.push_back(
        ninfer::ChatMessage{.role = "user",
                            .parts = {ninfer::MessagePart{
                                .kind = ninfer::MessagePartKind::Text,
                                .text = "Given everything above, how many rollbacks were performed "
                                        "in the maintenance window?",
                                .media = {}}}});
    return input;
}

int compare_cold_prefills(const ninfer::GenerationResult& first,
                          const ninfer::GenerationResult& second, const char* subject) {
    if (first.generated_logprobs.size() != 1 || second.generated_logprobs.size() != 1) {
        std::cerr << subject << " did not produce exactly one measured token\n";
        return 1;
    }
    const float a = first.generated_logprobs[0];
    const float b = second.generated_logprobs[0];
    if (!std::isfinite(a) || !std::isfinite(b)) {
        std::cerr << subject << " produced a non-finite logprob: " << a << " " << b << '\n';
        return 1;
    }
    if (first.generated_token_ids != second.generated_token_ids) {
        std::cerr << subject << " chose different greedy tokens: " << first.generated_token_ids[0]
                  << " vs " << second.generated_token_ids[0] << '\n';
        return 1;
    }

    const float delta   = std::fabs(a - b);
    const float limit   = logprob_tolerance();
    const char* verbose = std::getenv("NINFER_PREFILL_DET_VERBOSE");
    if (verbose != nullptr && *verbose != '\0') {
        std::cout << subject << ": logprob=" << a << " vs " << b << " delta=" << delta
                  << " limit=" << limit << '\n';
    }
    if (delta > limit) {
        std::cerr << subject << " disagreed at position 0: logprob=" << a << " vs " << b
                  << " delta=" << delta << " (tolerance " << limit << ")\n"
                  << "Prefill is not bit-reproducible; see "
                     "docs/SCR.md\n";
        return 1;
    }
    return 0;
}

ninfer::RequestOptions single_token_options() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

// Case 1: one Engine, the same prompt three times.
//
// The forward pass is compared against itself with nothing else varying. This currently fails with
// delta in [1.7e-4, 5.1e-3] nats across runs, so it is the reproduction, not a control.
//
// Three, not two. A pair cannot separate "the second call read what the first one wrote" from "the
// first call is the odd one out", and those two send the search to different places -- buffer
// lifecycle versus first-call initialization. With three values the shape is readable: `p2` and `p3`
// agreeing while `p1` differs is a first-call effect, `p1` and `p3` differing while the middle two
// agree is a pairwise-coincidence warning, and three distinct values say the divergence is not a
// function of call index at all.
//
// The three values are printed whether or not NINFER_PREFILL_DET_VERBOSE is set. They are the datum
// this test exists to produce, and the bit-identity check below is expected to fail, so a run that
// stays silent about them has nothing to say. The token sweep stays verbose-gated: it is a table,
// not a fact.
int exercise_repeat_prefill(ninfer::Engine& engine, const ninfer::PromptInput& prompt) {
    const ninfer::RequestOptions options = single_token_options();
    const ninfer::GenerationResult first  = engine.generate(engine.prepare(prompt), options);
    const ninfer::GenerationResult second = engine.generate(engine.prepare(prompt), options);
    const ninfer::GenerationResult third  = engine.generate(engine.prepare(prompt), options);

    if (first.generated_logprobs.size() == 1 && second.generated_logprobs.size() == 1 &&
        third.generated_logprobs.size() == 1 && std::isfinite(first.generated_logprobs[0]) &&
        std::isfinite(second.generated_logprobs[0]) && std::isfinite(third.generated_logprobs[0])) {
        const float p1 = first.generated_logprobs[0];
        const float p2 = second.generated_logprobs[0];
        const float p3 = third.generated_logprobs[0];
        // Printed with full precision so a rerun can be compared digit for digit, which is how the
        // "is this a function of call index" question is answered.
        std::cout.precision(9);
        std::cout << "repeat: p1=" << p1 << " p2=" << p2 << " p3=" << p3
                  << " d12=" << std::fabs(p1 - p2) << " d23=" << std::fabs(p2 - p3)
                  << " d13=" << std::fabs(p1 - p3) << '\n';
        std::cout.precision(6);
    }

    int status = compare_cold_prefills(first, second, "same Engine, same prompt twice");
    status |= compare_cold_prefills(second, third, "same Engine, second and third prefill");
    return status;
}

// Case 3: two Engines in one process, one prefill each.
//
// Constructing a second Engine inside one process re-establishes everything a fresh process would
// re-establish, without the process boundary. It separates "something a fresh Engine computes
// differently" from "something that genuinely needs a fresh OS process". Kept as a control on
// case 1: it should not disagree *more* than case 1 does.
//
// std::optional because Engine has no default constructor -- only explicit EngineOptions.
int exercise_two_engines(const char* artifact, const ninfer::PromptInput& prompt) {
    std::optional<ninfer::Engine> first(engine_options(artifact));
    const ninfer::GenerationResult a = first->generate(first->prepare(prompt), single_token_options());
    // Released before the second Engine is built: two live 15.92 GiB weight allocations do not fit
    // in 24 GB, and that failure surfaces as a stack-smashing abort rather than a clean error.
    first.reset();

    std::optional<ninfer::Engine> second(engine_options(artifact));
    const ninfer::GenerationResult b =
        second->generate(second->prepare(prompt), single_token_options());
    return compare_cold_prefills(a, b, "two Engines in one process");
}

// Case 2: sweep the prompt length across the GEMM schedule boundaries.
//
// The dispatch tables select a different kernel per token count, so token count is a control on
// *which arithmetic runs*:
//
//   ops/linear/q5/q5_dispatch.cpp  k=5120,n=17408 (mlp/down):     t<=16 small_t, <=64 c32, else c128
//   ops/linear/w8/w8_dispatch.cpp  k=5120,n=34816 (mlp/gate_up): t<=40 small_t, else c128
//   ops/linear/q4/q4_dispatch.cpp  k=5120,n=34816 (mlp/gate_up): t<=4 simt_c4, <=16 simt_c8, else mma
//
// The sample points straddle those boundaries rather than being round numbers: 8 sits in the q5
// small_t band and the q4 simt band, 16 is the q5 small_t boundary, 40 the top of w8 small_t, 64 the
// q5 c32/c128 boundary, 128 the top of the c64 band. 661 is the harness's own length and lands in
// the bulk c128 path everywhere.
//
// A discriminator, not a criterion: it reports each band's delta and never fails on its own, because
// "which band diverges" is the answer rather than "did anything diverge".
// A single-user-turn prompt of roughly `target` tokens, built from real text so it tokenizes
// through the artifact's own tokenizer. Length is approximate on purpose: the dispatch boundaries
// are coarse (16, 40, 64), so landing within a token or two of the target is enough to select the
// intended schedule, and an exact count would add a tokenizer round-trip per sample for nothing.
std::string prompt_text_of_length(const ninfer::Engine& engine, std::size_t target,
                                  const std::string& seed) {
    std::string text = seed;
    // Grow geometrically rather than one token at a time: count_tokens is a full tokenizer pass and
    // this loop runs for every sweep point.
    while (true) {
        ninfer::PromptInput probe;
        ninfer::ChatMessage message;
        message.role = "user";
        message.parts.push_back(
            ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
        probe.messages.push_back(std::move(message));
        const std::uint32_t count = engine.count_tokens(probe);
        if (count >= target) { return text; }
        text += std::string((target - count) * 3, ' ') + "records of the deployment window ";
    }
}

// A sweep point cannot reach the MLP's small-T and SIMT bands from text at all. The chat template's
// floor -- one short user turn, rendered -- is already 67 tokens, and every boundary worth crossing
// is at or below 48 (q4 gate_up 16, w8 gate_up 40, q5 down 16). So targets 8, 16, 40 and 64 all
// measure the same 67-token prompt, which is said out loud as `same_prompt_as_previous=yes` rather
// than left for the reader to infer from four identical `tokens=` values. Reaching those bands needs
// `prepare_tokens` with a raw id sequence, which this test does not use: the earlier version that did
// reported delta=0, and the difference was believed to be the ids rather than the 4096-token context
// it also used.
//
// Four measurements of one configuration are still worth having -- they answer "is the delta ever
// zero at the shortest length" -- so the rows stay and only the reading is corrected.
int exercise_token_sweep(ninfer::Engine& engine, const ninfer::PromptInput& base) {
    const char* verbose = std::getenv("NINFER_PREFILL_DET_VERBOSE");
    if (verbose == nullptr || *verbose == '\0') { return 0; }

    const std::string seed = base.messages.back().parts.back().text;
    std::cout << "token-count sweep (prefill length selects the GEMM schedule):\n";
    std::string previous_text;
    for (const std::size_t point : {8u, 16u, 40u, 64u, 128u, 661u}) {
        ninfer::PromptInput prompt;
        prompt.options = base.options;
        ninfer::ChatMessage message;
        message.role = "user";
        const std::string text = prompt_text_of_length(engine, point, seed);
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
        prompt.messages.push_back(std::move(message));

        const ninfer::RequestOptions options = single_token_options();
        const ninfer::GenerationResult a = engine.generate(engine.prepare(prompt), options);
        const ninfer::GenerationResult b = engine.generate(engine.prepare(prompt), options);
        const float delta = a.generated_logprobs.size() == 1 && b.generated_logprobs.size() == 1
                                ? std::fabs(a.generated_logprobs[0] - b.generated_logprobs[0])
                                : std::numeric_limits<float>::quiet_NaN();
        std::cout << "  target=" << point << "  tokens=" << engine.count_tokens(prompt)
                  << "  same_prompt_as_previous=" << (text == previous_text ? "yes" : "no")
                  << "  delta=" << delta << '\n';
        previous_text = text;
    }
    return 0;
}

} // namespace

int main() {
    const char* weights = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (weights == nullptr || *weights == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }

    const char* trace = std::getenv("NINFER_PREFILL_TRACE");
    const bool tracing = trace != nullptr && *trace != '\0' && *trace != '0';
    if (!tracing) {
        std::cout << "note: NINFER_PREFILL_TRACE=1 adds a per-layer residual diff, which names the"
                     " first layer whose output bytes differ between two prefills (=2 prints every"
                     " layer, not only the ones that moved)\n";
    }
    // The trace's window is a thread_local and every Engine has its own worker thread, so the
    // two-Engine case below -- the leg whose two Engines never share a window -- prints nothing but
    // `baseline=yes` unless the cross-instance lookup is switched on too. Said out loud here, because a
    // reader who sets only NINFER_PREFILL_TRACE sees four declinations and no hint about the second
    // variable.
    const char* cross = std::getenv("NINFER_PREFILL_TRACE_CROSS");
    if (tracing && (cross == nullptr || *cross == '\0' || *cross == '0')) {
        std::cout << "note: add NINFER_PREFILL_TRACE_CROSS=1 so the per-layer diff also covers the"
                     " two-Engine case, whose Engines do not share a trace window\n";
    }

    const ninfer::PromptInput prompt = suite_prompt();

    // The sweep runs before the bit-identity assertion on purpose. The assertion is expected to
    // fail today, and returning early from it would skip the measurement that actually localizes the
    // divergence -- which is the only thing here a reader needs in order to act.
    int status = 0;
    {
        ninfer::Engine engine(engine_options(weights));
        status |= exercise_repeat_prefill(engine, prompt);
        status |= exercise_token_sweep(engine, prompt);
    }
    status |= exercise_two_engines(weights, prompt);

    if (status == 0) { std::cout << "ok\n"; }
    return status;
}
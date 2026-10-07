// Phase 5.5 of docs/SCR_ROADMAP.md: does Suffix Cache Reuse change the model's answer?
//
// This drives ninfer-serve rather than the Engine API directly, because the serve binary is where
// SCR is switchable (`--scr`) and it already writes the per-request telemetry the comparison needs
// (`--request-log-jsonl`). The scoring is not reimplemented here: the request logs are turned into
// the target's own input by src/serve/scr_quality_log.cpp and scored by
// qwen3_6::detail::evaluate_scr_quality, so the roadmap's thresholds exist in exactly one place.
//
// Three runs are needed and they cannot be concurrent -- a 27B model leaves no room for a second
// resident copy on one 4090 -- so the phases run one after the other on different ports:
//
//   phase SCR       --scr,                       request log A
//   phase baseline  --no-scr --no-prefix-reuse,  request log B
//   phase control   --no-scr --no-prefix-reuse,  request log C
//
// Each prompt is sent to every phase as [pre-edit, edited]; the edited reply is the one compared.
// The baseline deliberately disables prefix reuse as well, because the roadmap measures against the
// full-prefill path; two servers with different cache states would otherwise agree for reasons that
// have nothing to do with relocation.
//
// The control is the baseline run repeated under identical settings, and it is not optional. Two runs
// of the same configuration are not guaranteed to be bit-identical, and a comparison that reports
// divergence without a floor cannot say whether the reuse path caused it. The control measures that
// floor. When it is zero, the SCR-versus-baseline numbers are attributable; when it is not, the run
// is reported as unattributable and exits non-zero rather than reaching a verdict.
//
// Requests are sent one at a time, and an edited request is only sent after its own pre-edit reply
// has arrived. Concurrency would make the cache state the relocator sees depend on admission order,
// which is exactly the variable this harness exists to hold fixed.

#include "serve/scr_quality_log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "httplib.h"

#if defined(_WIN32)
#    include <process.h>
#    include <windows.h>
#else
#    include <signal.h>
#    include <spawn.h>
#    include <sys/wait.h>
#    include <unistd.h>
extern char** environ;
#endif

namespace {

using Json = nlohmann::json;

// The prose that makes up the shared tail. Ordinary sentences on purpose: the tail has to fill KV
// pages and survive the edit, and its content is irrelevant to the comparison.
constexpr const char* kTailSentences[] = {
    "The maintenance log for the deployment records every change in order.",
    "Each entry lists the operator, the timestamp, and a short justification.",
    "Rollbacks were performed twice, both times within the same maintenance window.",
    "The verification suite ran after every rollout without exception.",
    "Throughput stayed within the agreed band during the whole observation period.",
    "No customer-visible incidents were attributed to these changes.",
};

// Roughly how many tokens one tail sentence costs, which is all the K sweep needs: the report
// carries the token counts the server actually measured, so K is a request, not a claim.
constexpr std::int32_t kTokensPerTailSentence = 8;
constexpr std::int32_t kBlockTokens           = 64;

struct Options {
    std::string exe        = "build-ninja/apps/ninfer-serve.exe";
    std::string model;
    std::string work_dir   = "scr-quality-run";
    std::string prompts_file;
    // Empty means the default file; "-" means stdout. The default is a file, not stdout: a spawned
    // ninfer-serve inherits this process's stdout and writes its own console log there, so a report
    // on stdout would be interleaved with it.
    std::string json_out;
    std::string server_args;
    std::string baseline_server_args = "--no-prefix-reuse";
    std::string api_key;
    int port                 = 18080;
    int max_tokens           = 64;
    int max_context          = 8192;
    int startup_timeout_secs = 900;
    int request_timeout_secs = 900;
    int prompts_per_k        = 1;
    std::vector<int> k_blocks{1, 2, 3, 6, 12, 64};
    bool keep_logs = false;
};

struct SuitePrompt {
    std::string name;
    Json primed_messages; // the pre-edit conversation
    Json edited_messages; // the same conversation with one earlier message revised
};

[[noreturn]] void usage(const std::string& message) {
    std::cerr << "error: " << message << "\n"
              << "usage: ninfer-scr-quality --model <artifact.ninfer> [options]\n"
              << "  --exe <path>                  ninfer-serve binary (default: " << Options{}.exe
              << ")\n"
              << "  --work-dir <dir>              where the three request logs are written\n"
              << "  --prompts <file>              JSONL of {name,primed_messages,edited_messages}\n"
              << "  --k-blocks <a,b,c>            shared-tail lengths to sweep, in 64-token blocks\n"
              << "  --prompts-per-k <n>           built-in prompts generated per K (default 1)\n"
              << "  --max-tokens <n>              output tokens per request (default 64)\n"
              << "  --max-context <n>             server KV context budget (default 8192)\n"
              << "  --port <n>                    first phase port; the baseline uses n+1 and the\n"
                 "                                null control n+2\n"
              << "  --server-args \"<args>\"       extra arguments for both server runs\n"
              << "  --baseline-server-args \"<args>\"   extra arguments for the baseline run\n"
              << "  --api-key <key>               bearer token, when the server requires one\n"
              << "  --json <file|->               report file; "-" writes it to stdout instead\n"
              << "                                (default: <work-dir>/scr-quality.json)\n"
              << "  --keep-logs                   leave the request logs in place on success\n";
    std::exit(2);
}

std::vector<std::string> split_commas(const std::string& text) {
    std::vector<std::string> parts;
    std::istringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ',')) { parts.push_back(part); }
    return parts;
}

std::vector<std::string> split_whitespace(const std::string& text) {
    std::vector<std::string> parts;
    std::istringstream stream(text);
    std::string part;
    while (stream >> part) { parts.push_back(part); }
    return parts;
}

Options parse_options(int argc, char** argv) {
    Options options;
    const auto value_of = [&](int& index) -> std::string {
        if (index + 1 >= argc) { usage(std::string(argv[index]) + " needs a value"); }
        return argv[++index];
    };
    const auto positive_int = [](const std::string& text, const char* name) {
        int value = 0;
        try {
            value = std::stoi(text);
        } catch (const std::exception&) {
            usage(std::string(name) + " must be an integer");
        }
        if (value <= 0) { usage(std::string(name) + " must be positive"); }
        return value;
    };

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--exe") {
            options.exe = value_of(index);
        } else if (arg == "--model") {
            options.model = value_of(index);
        } else if (arg == "--work-dir") {
            options.work_dir = value_of(index);
        } else if (arg == "--prompts") {
            options.prompts_file = value_of(index);
        } else if (arg == "--json") {
            options.json_out = value_of(index);
        } else if (arg == "--server-args") {
            options.server_args = value_of(index);
        } else if (arg == "--baseline-server-args") {
            options.baseline_server_args = value_of(index);
        } else if (arg == "--api-key") {
            options.api_key = value_of(index);
        } else if (arg == "--port") {
            options.port = positive_int(value_of(index), "--port");
        } else if (arg == "--max-tokens") {
            options.max_tokens = positive_int(value_of(index), "--max-tokens");
        } else if (arg == "--max-context") {
            options.max_context = positive_int(value_of(index), "--max-context");
        } else if (arg == "--startup-timeout") {
            options.startup_timeout_secs = positive_int(value_of(index), "--startup-timeout");
        } else if (arg == "--request-timeout") {
            options.request_timeout_secs = positive_int(value_of(index), "--request-timeout");
        } else if (arg == "--prompts-per-k") {
            options.prompts_per_k = positive_int(value_of(index), "--prompts-per-k");
        } else if (arg == "--keep-logs") {
            options.keep_logs = true;
        } else if (arg == "--k-blocks") {
            options.k_blocks.clear();
            for (const std::string& part : split_commas(value_of(index))) {
                options.k_blocks.push_back(positive_int(part, "--k-blocks entry"));
            }
        } else {
            usage("unknown argument " + arg);
        }
    }
    if (options.model.empty()) { usage("--model is required"); }
    if (options.k_blocks.empty()) { usage("--k-blocks must not be empty"); }
    return options;
}

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

// A chat-shaped conversation, not filler: a real system prompt, a few turns of history, and an edit
// that revises an earlier user message, which is the actual CLM use case for reusing a suffix.
// Everything after the revised message survives the edit unchanged and is what SCR relocates.
void built_in_prompt(int blocks, int variant, Json& primed, Json& edited) {
    const std::string tail  = repeated_tail(blocks);
    const std::string system =
        "You are an operations assistant. Answer from the deployment records you are given, state "
        "uncertainty when the records do not settle a question, and keep answers short.";
    const std::string question =
        "Given everything above, how many rollbacks were performed in the maintenance window?" +
        (variant == 0 ? std::string{} : " Answer for survey " + std::to_string(variant) + ".");
    // Both revisions must tokenize to the same length. The surviving-span detector hashes whole
    // 64-token pages, so an edit that shifts the tail by even one token leaves nothing aligned to
    // match and Suffix Cache Reuse cannot engage at all. Same sentence shape, same word count, only
    // the facts differ.
    const std::string original =
        "SECTION ONE. The migration moved the primary datastore to a three node cluster, and all "
        "writes were routed through the new leader.";
    const std::string revised =
        "SECTION ONE. The migration moved the primary datastore to a four node cluster, and all "
        "writes were routed through the new leader.";

    const auto user_turn = [](const std::string& text) {
        return Json{{"role", "user"}, {"content", text}};
    };
    const auto assistant_turn = [](const std::string& text) {
        return Json{{"role", "assistant"}, {"content", text}};
    };
    const Json system_turn{{"role", "system"}, {"content", system}};

    // History that is identical in both conversations and comes *before* the edited turn. Without it
    // the only shared prefix is the system prompt and the chat template's own scaffolding, which is
    // a few dozen tokens: prefix reuse matches almost nothing, every request runs full_reset, and the
    // run reports a near-zero divergence for a feature that never engaged. The edit has to sit in the
    // middle of a conversation for there to be a prefix worth reusing and a suffix worth relocating.
    //
    // Four exchanges, deliberately longer than one KV page, so the reused prefix spans several pages
    // and the relocated run starts well inside the prompt.
    std::vector<Json> shared;
    for (int turn = 0; turn < 4; ++turn) {
        shared.push_back(user_turn("SECTION ZERO." + std::to_string(turn) +
                                   " Standing instructions for the maintenance window: confirm every "
                                   "figure against the records, name the window it belongs to, and "
                                   "do not speculate about counts that are not stated."));
        shared.push_back(assistant_turn("Understood section zero " + std::to_string(turn) +
                                        ". I will confirm each figure against the records and will "
                                        "say so when the records do not settle a question."));
    }

    const auto conversation = [&](const std::string& section) {
        Json messages = Json::array({system_turn});
        for (const Json& turn : shared) { messages.push_back(turn); }
        messages.push_back(user_turn("Deployment brief. " + section + " " + tail));
        messages.push_back(assistant_turn("Understood. I have read the deployment records."));
        messages.push_back(user_turn("SECTION TWO. Verification ran after every rollout."));
        messages.push_back(assistant_turn("Noted."));
        messages.push_back(user_turn(question));
        return messages;
    };
    primed = conversation(original);
    edited = conversation(revised);
}

std::vector<SuitePrompt> load_prompts_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot read prompts file " + path); }
    std::vector<SuitePrompt> prompts;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r") == std::string::npos) { continue; }
        Json record;
        try {
            record = Json::parse(line);
        } catch (const Json::exception& error) {
            throw std::runtime_error("prompts file line " + std::to_string(line_number) +
                                     " is not JSON: " + error.what());
        }
        SuitePrompt prompt;
        prompt.name            = record.at("name").get<std::string>();
        prompt.primed_messages = record.at("primed_messages");
        prompt.edited_messages = record.at("edited_messages");
        if (!prompt.primed_messages.is_array() || !prompt.edited_messages.is_array() ||
            prompt.primed_messages.empty() || prompt.edited_messages.empty()) {
            throw std::runtime_error("prompts file line " + std::to_string(line_number) +
                                     " needs non-empty message arrays");
        }
        prompts.push_back(std::move(prompt));
    }
    if (prompts.empty()) { throw std::runtime_error("prompts file has no prompts"); }
    return prompts;
}

std::vector<SuitePrompt> build_suite(const Options& options) {
    if (!options.prompts_file.empty()) { return load_prompts_file(options.prompts_file); }
    std::vector<SuitePrompt> prompts;
    for (const int blocks : options.k_blocks) {
        for (int variant = 0; variant < options.prompts_per_k; ++variant) {
            SuitePrompt prompt;
            prompt.name = "chat-k" + std::to_string(blocks) + "-p" + std::to_string(variant);
            built_in_prompt(blocks, variant, prompt.primed_messages, prompt.edited_messages);
            prompts.push_back(std::move(prompt));
        }
    }
    return prompts;
}

// The model field is deliberately omitted: the server fills in its own public id when the field is
// absent, and naming an id the artifact does not carry is the one way this request could be
// refused for a reason that has nothing to do with what is being measured.
std::string request_body(const Options& options, const Json& messages) {
    return Json{{"messages", messages},
                {"max_tokens", options.max_tokens},
                {"stream", false},
                {"temperature", 0}}
        .dump();
}

httplib::Headers request_headers(const Options& options) {
    httplib::Headers headers{{"Content-Type", "application/json"}};
    if (!options.api_key.empty()) { headers.emplace("Authorization", "Bearer " + options.api_key); }
    return headers;
}

// One child process, terminated on scope exit so a failing phase cannot leave a server holding the
// GPU. The argv strings are owned here because the spawn call keeps pointers into them.
class ServerProcess {
public:
    ServerProcess(const std::string& exe, const std::vector<std::string>& args) {
        arguments_.reserve(args.size() + 1);
        arguments_.push_back(exe);
        arguments_.insert(arguments_.end(), args.begin(), args.end());
        argv_.reserve(arguments_.size() + 1);
        for (std::string& argument : arguments_) { argv_.push_back(argument.data()); }
        argv_.push_back(nullptr);
    }

    ~ServerProcess() { terminate(); }

    ServerProcess(const ServerProcess&)            = delete;
    ServerProcess& operator=(const ServerProcess&) = delete;

    void start() {
#if defined(_WIN32)
        handle_ = ::_spawnvp(_P_NOWAIT, arguments_.front().c_str(), argv_.data());
        if (handle_ == -1) { throw std::runtime_error("failed to start " + arguments_.front()); }
#else
        if (::posix_spawn(&pid_, arguments_.front().c_str(), nullptr, nullptr, argv_.data(),
                          environ) != 0) {
            throw std::runtime_error("failed to start " + arguments_.front());
        }
#endif
        running_ = true;
    }

    // True once the child has exited. Without this a server that fails at startup -- a mistyped
    // artifact path, an unknown flag -- would leave the harness polling for health until the
    // whole startup timeout elapsed, reporting a timeout instead of the reason it never came up.
    [[nodiscard]] bool exited() {
        if (!running_) { return true; }
#if defined(_WIN32)
        return ::WaitForSingleObject(reinterpret_cast<HANDLE>(handle_), 0) == WAIT_OBJECT_0;
#else
        int status       = 0;
        const pid_t done = ::waitpid(pid_, &status, WNOHANG);
        if (done != pid_) { return false; }
        running_ = false;
        return true;
#endif
    }

    void terminate() noexcept {
        if (!running_) { return; }
        running_ = false;
#if defined(_WIN32)
        ::TerminateProcess(reinterpret_cast<HANDLE>(handle_), 0);
        ::WaitForSingleObject(reinterpret_cast<HANDLE>(handle_), 30000);
        ::CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
        ::kill(pid_, SIGTERM);
        int status = 0;
        (void)::waitpid(pid_, &status, 0);
#endif
    }

private:
    std::vector<std::string> arguments_;
    std::vector<char*> argv_;
#if defined(_WIN32)
    std::intptr_t handle_ = -1;
#else
    pid_t pid_ = -1;
#endif
    bool running_ = false;
};

void wait_for_health(const Options& options, int port, ServerProcess& server) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(5, 0);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(options.startup_timeout_secs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (server.exited()) {
            throw std::runtime_error("ninfer-serve exited before becoming healthy on port " +
                                     std::to_string(port) +
                                     "; see its console output for the reason");
        }
        const auto response = client.Get("/health");
        if (response && response->status >= 200 && response->status < 300) { return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    throw std::runtime_error("ninfer-serve did not become healthy on port " + std::to_string(port));
}

void post_chat(const Options& options, int port, const Json& messages, const std::string& label) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(options.request_timeout_secs, 0);
    const auto response = client.Post("/v1/chat/completions", request_headers(options),
                                      request_body(options, messages), "application/json");
    if (!response) { throw std::runtime_error("request failed for " + label); }
    if (response->status != 200) {
        throw std::runtime_error("request " + label + " returned HTTP " +
                                 std::to_string(response->status) + ": " + response->body);
    }
}

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

// One server run over the whole suite. Both phases send [pre-edit, edited] for every prompt, so the
// two logs describe the same requests in the same order and pairing stays positional. The baseline
// phase needs prefix reuse off as well, which is what makes its edited reply a full prefill.
void run_phase(const Options& options, const std::vector<SuitePrompt>& prompts, bool scr, int port,
               const std::string& log_path) {
    std::remove(log_path.c_str());
    const std::vector<std::string> extra = [&] {
        std::vector<std::string> args = split_whitespace(options.server_args);
        if (!scr) {
            const std::vector<std::string> baseline = split_whitespace(options.baseline_server_args);
            args.insert(args.end(), baseline.begin(), baseline.end());
        }
        return args;
    }();
    const std::vector<std::string> args = ninfer::serve::build_scr_quality_server_arguments(
        options.model, port, static_cast<std::uint32_t>(options.max_context), log_path, scr, extra);

    ServerProcess server(options.exe, args);
    server.start();
    wait_for_health(options, port, server);
    std::size_t sent = 0;
    for (const SuitePrompt& prompt : prompts) {
        post_chat(options, port, prompt.primed_messages, prompt.name + " prime");
        post_chat(options, port, prompt.edited_messages, prompt.name + " edited");
        sent += 2;
        std::cerr << "[" << (scr ? "scr" : "baseline") << "] " << prompt.name << " done (" << sent
                  << " of " << prompts.size() * 2 << " requests)\n";
    }
    server.terminate();
}

// The edited record of each prompt pair, in suite order, with the telemetry that says whether the
// regime the roadmap cares about (a long reused prefix, a long relocated suffix) was exercised.
std::vector<ninfer::serve::LoggedCompletion>
select_edited(const std::vector<ninfer::serve::LoggedCompletion>& completions, bool scr) {
    std::vector<ninfer::serve::LoggedCompletion> selected;
    selected.reserve(completions.size() / 2);
    std::vector<std::size_t> misaligned;
    for (std::size_t index = 0; index + 1 < completions.size(); index += 2) {
        const ninfer::serve::LoggedCompletion& primed = completions[index];
        const ninfer::serve::LoggedCompletion& edited = completions[index + 1];
        // Every surviving span is found by hashing whole KV pages, so an edit that changes the
        // prompt's token count by anything other than a multiple of the page size shifts every later
        // block off its boundary and leaves the detector nothing to match after the edit. The suite
        // then reports near-zero divergence for a run in which Suffix Cache Reuse never engaged --
        // the one result that looks like a pass and measures nothing.
        //
        // Checked rather than assumed. An earlier suite revised a message by a few words, which is
        // exactly this mistake, and the only symptom was that five of six prompts reported
        // full_reset with no relocation to explain it.
        constexpr std::int32_t kPageTokens = 64; // kPagedKVPageSize
        const std::int32_t offset = edited.prompt_tokens - primed.prompt_tokens;
        const bool aligned        = offset % kPageTokens == 0;
        std::cerr << "[" << (scr ? "scr" : "baseline") << "] prompt " << (index / 2)
                  << ": prompt_tokens=" << edited.prompt_tokens
                  << " primed=" << primed.prompt_tokens
                  << " edit_offset=" << offset << (aligned ? "" : " NOT_PAGE_ALIGNED")
                  << " reused=" << edited.prefix_cache_hit_tokens
                  << " relocated=" << edited.prefix_relocated_tokens
                  << " path=" << edited.prefix_reuse_path << "\n";
        if (!aligned && scr) { misaligned.push_back(index / 2); }
        selected.push_back(edited);
    }
    if (!misaligned.empty()) {
        std::ostringstream message;
        message << "the edit changes the prompt by a token count that is not a multiple of the "
                   "64-token KV page, so no surviving span survives it and Suffix Cache Reuse cannot "
                   "engage: prompt(s) ";
        for (std::size_t i = 0; i < misaligned.size(); ++i) {
            message << (i == 0 ? "" : ", ") << misaligned[i];
        }
        message << ". Pad the edited message so both versions tokenize to the same length.";
        throw std::runtime_error(message.str());
    }
    return selected;
}

// The K sweep is the point of this harness, and a single JSON blob does not show a curve. This is the
// roadmap's Figure 13 grid read directly: how much of the context was reused, where the two paths first
// parted, and how far the reused prefill had already moved the next-token distribution by then.
//
// first_token_divergence is the column that separates "relocation barely perturbs anything" from
// "relocation perturbs the first token and the rest is decoder feedback": it is measured before any
// divergence can compound, so unlike the mean it is not inflated by a run that flips at step 3 and
// then wanders for 60 more tokens.
void print_curve(const char* title, const std::vector<SuitePrompt>& prompts,
                 const std::vector<ninfer::serve::LoggedCompletion>& edited,
                 const ninfer::targets::qwen3_6::detail::ScrQualityReport& scored,
                 bool with_floor) {
    std::cerr << "\n" << title << "\n\n";
    if (with_floor) {
        std::cerr << "  K   prompt  relocated  parted@  d(first)   floor    signal   "
                     "margin    d(mean)  agree  verdict\n";
    } else {
        std::cerr << "  K   prompt  relocated  parted@  d(first)     margin    d(mean)  "
                     "agree  verdict\n";
    }
    std::cerr << "  ----------------------------------------------------------------------"
                 "---------------\n";
    for (std::size_t i = 0; i < scored.prompts.size() && i < prompts.size(); ++i) {
        const auto& result = scored.prompts[i];
        std::ostringstream row;
        row << "  " << prompts[i].name << "  " << edited[i].prompt_tokens << "  "
            << edited[i].prefix_relocated_tokens << "  " << result.first_divergence_index;
        if (with_floor) {
            // signal = d(first) - floor. Negative means the measurement sat inside the run-to-run
            // variation, which is the reading that matters: it is not a small perturbation, it is an
            // unmeasurable one.
            row << "  " << result.first_token_divergence << "  "
                << result.floor_first_token_divergence << "  " << result.signal_first_token_divergence;
        } else {
            row << "  " << result.first_token_divergence;
        }
        row << "  " << result.first_token_margin << "  " << result.mean_logprob_divergence << "  "
            << result.greedy_agreement << "  " << (result.ok ? "pass" : "FAIL") << "\n";
        std::cerr << row.str();
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::filesystem::create_directories(options.work_dir);
        const std::string scr_log      = options.work_dir + "/scr.jsonl";
        const std::string baseline_log = options.work_dir + "/baseline.jsonl";
        const std::string control_log  = options.work_dir + "/control.jsonl";
        const std::vector<SuitePrompt> prompts = build_suite(options);
        const std::string report_path = (options.json_out.empty() || options.json_out == "-")
                                            ? options.work_dir + "/scr-quality.json"
                                            : options.json_out;
        std::cerr << "suite: " << prompts.size() << " prompt(s); report to " << report_path
                  << "\n";

        run_phase(options, prompts, true, options.port, scr_log);
        // A separate port keeps each later phase off a socket still in TIME_WAIT.
        run_phase(options, prompts, false, options.port + 1, baseline_log);
        // The control is the baseline configuration run a second time. Without it the measurement has
        // no noise floor, and every divergence it reports is unattributable: the same spread is
        // produced by two runs that differ in nothing. The 2026-10-08 sweep is the case in point --
        // d(first) varied from 1.8e-4 to 7.5e-2 nats with no relationship to how much context was
        // reused, which is what a floor of comparable size looks like and not what a relocation error
       // that grows with the reused region looks like.
        run_phase(options, prompts, false, options.port + 2, control_log);

        const auto load = [&](const std::string& path, const char* phase) {
            const std::vector<ninfer::serve::LoggedCompletion> completions =
                ninfer::serve::parse_completion_log(read_file(path));
            if (completions.size() != prompts.size() * 2) {
                throw std::runtime_error(std::string("the ") + phase +
                                         " request log does not describe the prompts that were sent");
            }
            return completions;
        };
        const std::vector<ninfer::serve::LoggedCompletion> scr_edited =
            select_edited(load(scr_log, "SCR"), true);
        const std::vector<ninfer::serve::LoggedCompletion> baseline_edited =
            select_edited(load(baseline_log, "baseline"), false);
        const std::vector<ninfer::serve::LoggedCompletion> control_edited =
            select_edited(load(control_log, "control"), false);

        const ninfer::targets::qwen3_6::detail::ScrQualityThresholds thresholds;
        // Throws when a prompt relocated nothing: a suite that never engaged Suffix Cache Reuse
        // would otherwise compare two full prefills and read as a pass.
        // Scored twice over: once on its own, and once paired with the control so each prompt carries
        // the floor it has to be read against. Same thresholds, same function, no second arithmetic.
        auto scored = ninfer::serve::evaluate_scr_quality_report(scr_edited, baseline_edited,
                                                                 thresholds);
        // Scored through its own entry point because the control relocates nothing by construction,
        // which is precisely what pair_scr_quality_samples refuses.
        const auto control =
            ninfer::serve::evaluate_scr_quality_control(baseline_edited, control_edited, thresholds);
        {
            const auto measurement_samples =
                ninfer::serve::pair_scr_quality_samples(scr_edited, baseline_edited);
            const auto control_samples =
                ninfer::serve::pair_scr_quality_samples_for_control(baseline_edited, control_edited);
            for (std::size_t i = 0; i < scored.prompts.size() && i < control_samples.size(); ++i) {
                scored.prompts[i] = ninfer::targets::qwen3_6::detail::evaluate_scr_prompt_against_control(
                    measurement_samples[i], control_samples[i], thresholds);
            }
        }
        const std::vector<std::uint32_t> unattributable =
            ninfer::targets::qwen3_6::detail::scr_quality_control_divergent_prompts(control);
        const bool attributable =
            ninfer::targets::qwen3_6::detail::scr_quality_control_is_clean(control);

        print_curve("MEASUREMENT  scr vs baseline", prompts, scr_edited, scored, true);
        print_curve("CONTROL  baseline vs baseline (the noise floor)", prompts, baseline_edited,
                    control, false);

        std::cerr << "\n  parted@ is the first greedy token the two runs disagreed on, so the columns "
                     "to its right\n  describe a prefix both runs chose identically and are "
                     "comparable; the agreement column is\n  index-aligned and is not.\n";

        // The envelope carries the control beside the measurement, because a report that states a
        // verdict without the floor it was measured against is not a report. The inner objects are
        // the target's own format, so the thresholds still exist in exactly one place.
        std::ostringstream report;
        report << "{\"attributable\":" << (attributable ? "true" : "false") << ",\"control\":"
               << ninfer::targets::qwen3_6::detail::format_scr_quality_json(control, thresholds)
               << ",\"measurement\":"
               << ninfer::targets::qwen3_6::detail::format_scr_quality_json(scored, thresholds)
               << "}\n";

        if (options.json_out == "-") {
            std::cout << report.str();
            std::cout.flush();
        } else {
            std::ofstream output(report_path, std::ios::binary | std::ios::trunc);
            if (!output) { throw std::runtime_error("cannot write " + report_path); }
            output << report.str();
            std::cerr << "wrote " << report_path << "\n";
        }
        std::cerr << "\nThis measures argmax robustness and logprob divergence, not distribution "
                     "shift: high greedy agreement means SCR does not flip decisions, not that the "
                     "logits are unperturbed.\n";
        if (!options.keep_logs) {
            std::remove(scr_log.c_str());
            std::remove(baseline_log.c_str());
            std::remove(control_log.c_str());
        }
        if (!attributable) {
            std::cerr << "\nUNATTRIBUTABLE: the control chose a different token than itself on "
                         "prompt(s) ";
            for (std::size_t i = 0; i < unattributable.size(); ++i) {
                std::cerr << (i == 0 ? "" : ", ") << unattributable[i];
            }
            std::cerr << ".\n  Two runs of the same configuration disagreed there, so the token "
                         "stream itself is not\n  reproducible and the measurement's divergence for "
                         "those prompts has no floor to\n  subtract. Settle run-to-run determinism "
                         "before reading them.\n";
            return 3;
        }
        // The control agreeing on tokens is NOT a statement that the logprobs matched: greedy choice is
        // insensitive to a perturbation the logprob metric is not, and the measured floor is a few
        // thousandths of a nat with occasional excursions an order of magnitude higher. An earlier
        // version of this message claimed the floor was zero, which the control's own numbers
        // contradicted directly underneath it.
        std::cerr << "\nThe control chose the same token as itself at every position, so the greedy "
                     "token stream is\n  reproducible run to run. Its logprobs were not identical: "
                     "read d(first) against the floor column,\n  not against zero.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}

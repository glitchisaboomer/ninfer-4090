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
#include <cmath>
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
using ninfer::targets::qwen3_6::detail::ScrQualityPromptResult;
using ninfer::targets::qwen3_6::detail::ScrQualityReport;
using ninfer::targets::qwen3_6::detail::ScrQualitySample;

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
    // NINFER_SCR_DIAG=1 for the spawned servers: the runtime's own account of what relocation did,
    // which is the only channel that shows the applied plan, the page remap, and any refusal the
    // planner made. Without it the harness can only report what it asked for, never what happened.
    bool scr_diag = false;
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
              << "  --keep-logs                   leave the request logs in place on success\n"
              << "  --scr-diag                    set NINFER_SCR_DIAG=1 for the spawned servers, so each\n"
                 "                                prints its applied relocation plan, page remap, and any\n"
                 "                                planner refusal to this console\n";
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
        } else if (arg == "--scr-diag") {
            options.scr_diag = true;
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

Json post_chat(const Options& options, int port, const Json& messages, const std::string& label) {
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
    try {
        return Json::parse(response->body);
    } catch (const Json::exception& error) {
        throw std::runtime_error("request " + label + " returned a body that is not JSON: " +
                                 error.what());
    }
}

// One chat reply's prose and disposition. The scorer compares tokens and logprobs; this is what the
// model actually said -- the part a reader can judge directly, without decoding token ids. Two runs
// that flip a token at position 6 but still answer the question identically are a different finding
// from two runs whose answers disagree, and the scored report alone cannot tell them apart.
struct ChatReply {
    std::string content;
    std::string reasoning;
    std::string finish_reason;
};

// `content` is null whenever the model produced only reasoning, and null is not a string, so the
// field is read defensively rather than with `at` + `get`, which would throw on the null.
std::string string_field(const Json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || it->is_null()) { return {}; }
    if (!it->is_string()) {
        throw std::runtime_error(std::string("chat reply field ") + key +
                                 " is neither a string nor null");
    }
    return it->get<std::string>();
}

ChatReply extract_reply(const Json& body, const std::string& label) {
    try {
        const Json& choice = body.at("choices").at(0);
        ChatReply reply;
        reply.content       = string_field(choice.at("message"), "content");
        reply.reasoning     = string_field(choice.at("message"), "reasoning_content");
        reply.finish_reason = string_field(choice, "finish_reason");
        return reply;
    } catch (const Json::exception& error) {
        throw std::runtime_error("reply for " + label + " is missing fields the harness needs: " +
                                 error.what());
    }
}

// Primed and edited replies of one prompt, both kept: the primed reply is the request that seeds the
// resident KV, so a primed reply that differs between phases means the two sides did not start from
// the same state and nothing downstream of it is a clean A/B.
struct PhaseExchange {
    ChatReply primed;
    ChatReply edited;
};

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
//
// Returns both replies of every prompt: the scored comparison says where the token streams parted,
// and this is what the two runs actually answered -- the thing a reader can judge without decoding
// token ids. `label` is how the phase names itself in every line it prints; the baseline and the
// control run the same flags, and printing both as "baseline" made them read as one phase.
std::vector<PhaseExchange> run_phase(const Options& options, const std::vector<SuitePrompt>& prompts,
                                     bool scr, int port, const std::string& log_path,
                                     const char* label) {
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
    std::vector<PhaseExchange> exchanges;
    exchanges.reserve(prompts.size());
    for (const SuitePrompt& prompt : prompts) {
        const Json primed_body =
            post_chat(options, port, prompt.primed_messages, prompt.name + " prime");
        const Json edited_body =
            post_chat(options, port, prompt.edited_messages, prompt.name + " edited");
        exchanges.push_back(PhaseExchange{extract_reply(primed_body, prompt.name + " prime"),
                                          extract_reply(edited_body, prompt.name + " edited")});
        sent += 2;
        std::cerr << "[" << label << "] " << prompt.name << " done (" << sent << " of "
                  << prompts.size() * 2 << " requests)\n";
    }
    server.terminate();
    return exchanges;
}

// Reuse telemetry for one prompt pair, kept beside the scores: a report that says how far the runs
// drifted without saying how much was reused cannot be read as the K curve the roadmap asks for,
// and `reused=0` with `relocated>0` is a shape that has to be explained rather than guessed at.
struct ReuseInfo {
    std::string name;
    std::int32_t primed_prompt_tokens     = 0;
    std::int32_t prompt_tokens            = 0;
    std::int32_t edit_offset              = 0;
    std::uint32_t prefix_cache_hit_tokens = 0;
    std::uint32_t prefix_relocated_tokens = 0;
    std::string prefix_reuse_path;
};

// The edited record of each prompt pair, in suite order, with the telemetry that says whether the
// regime the roadmap cares about (a long reused prefix, a long relocated suffix) was exercised.
// `label` names the phase in every line printed; `reuse` receives the same telemetry in structured
// form so it can ride in the report instead of only in stderr.
std::vector<ninfer::serve::LoggedCompletion> select_edited(
    const std::vector<ninfer::serve::LoggedCompletion>& completions, const char* label, bool scr,
    const std::vector<SuitePrompt>& prompts, std::vector<ReuseInfo>& reuse) {
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
        const std::size_t prompt_index = index / 2;
        const std::string name = prompt_index < prompts.size()
                                     ? prompts[prompt_index].name
                                     : "prompt " + std::to_string(prompt_index);
        std::cerr << "[" << label << "] " << name
                  << ": prompt_tokens=" << edited.prompt_tokens
                  << " primed=" << primed.prompt_tokens
                  << " edit_offset=" << offset << (aligned ? "" : " NOT_PAGE_ALIGNED")
                  << " reused=" << edited.prefix_cache_hit_tokens
                  << " relocated=" << edited.prefix_relocated_tokens
                  << " path=" << edited.prefix_reuse_path << "\n";
        if (!aligned && scr) { misaligned.push_back(prompt_index); }
        reuse.push_back(ReuseInfo{name, primed.prompt_tokens, edited.prompt_tokens, offset,
                                  edited.prefix_cache_hit_tokens, edited.prefix_relocated_tokens,
                                  edited.prefix_reuse_path});
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

// --- diagnostics the scored report cannot carry ------------------------------------------------
//
// The report says *where* the token streams parted and how far the logprobs moved before it did. It
// does not say which token each side chose at the flip, how confident either side was there, how
// much was actually reused, or what the two runs answered -- and those are the four things a reader
// needs before the pass/fail column means anything.

// Index of the first character at which the two strings differ, or npos when they are equal.
std::size_t first_diff(const std::string& left, const std::string& right) {
    const std::size_t common = std::min(left.size(), right.size());
    std::size_t index        = 0;
    while (index < common && left[index] == right[index]) { ++index; }
    if (index == common && left.size() == right.size()) { return std::string::npos; }
    return index;
}

// Six decimals, and "n/a" for anything not finite. The scorer prints nan, which is honest in a
// numeric field, but here the value sits beside strings where nan would read as a measurement
// rather than as an absence.
std::string metric(double value) {
    if (!std::isfinite(value)) { return "n/a"; }
    std::ostringstream out;
    out.precision(6);
    out << std::fixed << value;
    return out.str();
}

std::string margin_at(const std::vector<float>& margins, std::size_t index) {
    if (index >= margins.size()) { return "n/a"; }
    return metric(static_cast<double>(margins[index]));
}

// One line of a reply, capped and flattened: this is a diagnosis, not a transcript, and the
// excerpt exists to be read next to its twin.
std::string excerpt(const std::string& text, std::size_t limit = 200) {
    std::string flat = text.substr(0, limit);
    for (char& character : flat) {
        if (character == '\n' || character == '\r' || character == '\t') { character = ' '; }
    }
    if (text.size() > limit) { flat += "..."; }
    return "'" + flat + "'";
}

// The flip itself, with the numbers that decide what it means. Margin is best-minus-second-best in
// nats: a flip behind a margin of a thousandth is a coin tipped by any perturbation at all, a flip
// against a wide margin is a decision the reuse path actually moved. Position 0 comes along too,
// because d(first) in the report is measured there and is only readable against the margin beside
// it. Values come straight from the request log; a server whose log predates the margin field
// prints n/a rather than borrowing a margin from a neighbouring position.
void print_divergence_detail(const std::vector<SuitePrompt>& prompts,
                             const std::vector<ScrQualitySample>& samples,
                             const ScrQualityReport& scored) {
    std::cerr << "\nFIRST-FLIP DETAIL  (measurement: --scr vs --no-scr)\n"
                 "\n  margin = best-minus-second-best logit gap, in nats.\n";
    for (std::size_t i = 0; i < scored.prompts.size() && i < samples.size(); ++i) {
        const ScrQualityPromptResult& result = scored.prompts[i];
        const ScrQualitySample& sample        = samples[i];
        const std::size_t overlap =
            std::min({sample.baseline_tokens.size(), sample.scr_tokens.size(),
                      sample.baseline_logprobs.size(), sample.scr_logprobs.size()});
        const std::string name =
            i < prompts.size() ? prompts[i].name : "prompt " + std::to_string(i);
        std::cerr << "\n  " << name;
        if (overlap == 0) {
            std::cerr << ": no comparable tokens\n";
            continue;
        }
        std::cerr << "\n    pos0    logprob --no-scr " << metric(sample.baseline_logprobs[0])
                  << "  --scr " << metric(sample.scr_logprobs[0])
                  << "   margin --no-scr " << margin_at(sample.baseline_margins, 0)
                  << "  --scr " << margin_at(sample.scr_margins, 0);
        const std::size_t flip = result.first_divergence_index;
        if (flip < overlap) {
            std::cerr << "\n    flip@" << flip << "  --no-scr token " << sample.baseline_tokens[flip]
                      << " logprob " << metric(sample.baseline_logprobs[flip]) << " margin "
                      << margin_at(sample.baseline_margins, flip) << "   |  --scr token "
                      << sample.scr_tokens[flip] << " logprob "
                      << metric(sample.scr_logprobs[flip]) << " margin "
                      << margin_at(sample.scr_margins, flip);
        } else {
            std::cerr << "\n    no flip within the " << overlap << " comparable token(s)";
        }
        std::cerr << "\n";
    }
}

// The two paths' prose, side by side. The scorer reports where the token streams parted; this
// reports what that was worth -- whether both runs still answered the question the same way, and
// the first character at which the text reads differently.
//
// The primed replies are checked too: they seed the resident KV, so a primed reply that differs
// between phases means the two sides did not start from the same state and the pair is not a clean
// A/B. The control is checked against the baseline for the same reason in reverse: two identical
// configurations that produce different prose would make the floor a text question as well as a
// logprob one.
void print_output_comparison(const std::vector<SuitePrompt>& prompts,
                             const std::vector<PhaseExchange>& baseline,
                             const std::vector<PhaseExchange>& scr,
                             const std::vector<PhaseExchange>& control) {
    std::cerr << "\nGENERATED OUTPUT  (--no-scr vs --scr, edited request)\n";
    std::size_t identical      = 0;
    std::size_t primed_same    = 0;
    std::size_t control_same   = 0;
    const std::size_t measured =
        std::min({prompts.size(), baseline.size(), scr.size()});
    for (std::size_t i = 0; i < measured; ++i) {
        const ChatReply& base = baseline[i].edited;
        const ChatReply& test = scr[i].edited;
        const bool same = base.content == test.content && base.reasoning == test.reasoning;
        const bool primed_match = baseline[i].primed.content == scr[i].primed.content &&
                                  baseline[i].primed.reasoning == scr[i].primed.reasoning;
        const bool control_match = i < control.size() &&
                                   control[i].edited.content == base.content &&
                                   control[i].edited.reasoning == base.reasoning;
        identical += same ? 1U : 0U;
        primed_same += primed_match ? 1U : 0U;
        control_same += control_match ? 1U : 0U;

        const std::size_t diff = first_diff(base.content, test.content);
        std::cerr << "\n  " << prompts[i].name << ": "
                  << (same ? "content identical" : "CONTENT DIFFERS");
        if (diff != std::string::npos) {
            std::cerr << " (first differing character " << diff << ")";
        }
        std::cerr << ", finish --no-scr="
                  << (base.finish_reason.empty() ? "?" : base.finish_reason)
                  << " --scr=" << (test.finish_reason.empty() ? "?" : test.finish_reason);
        if (!primed_match) {
            std::cerr << "\n    PRIMED replies differ between phases: the resident KV was seeded "
                         "differently,\n    so this pair is not a clean A/B";
        }
        if (!control_match) {
            std::cerr << "\n    CONTROL prose differs from baseline: two identical "
                         "configurations produced\n    different text here";
        }
        std::cerr << "\n    --no-scr: " << excerpt(base.content)
                  << "\n    --scr    : " << excerpt(test.content);
        if (!base.reasoning.empty() || !test.reasoning.empty()) {
            std::cerr << "\n    reasoning --no-scr: " << excerpt(base.reasoning)
                      << "\n    reasoning --scr    : " << excerpt(test.reasoning);
        }
    }
    std::cerr << "\n  edited content identical on " << identical << "/" << measured
              << " prompt(s); primed replies identical on " << primed_same << "/" << measured
              << "; control matched baseline on " << control_same << "/" << measured << "\n";
}

// `reused=` in the per-prompt lines is prefix_cache_hit_tokens, which the runtime sets from
// `reuse_base`: the length of the prefix that matches the resident sequence *in full*. An
// in-the-middle edit defeats that match however long the shared head is, so the line reads 0 while
// hundreds of tokens are in fact reused -- the reuse that happened is what `relocated=` counts, and
// the head before the edit is re-prefilled by the relocation plan. Stated once, explicitly,
// because "nothing was reused" is the natural -- and wrong -- reading of the line.
void print_reuse_explainer(const std::vector<ReuseInfo>& scr_reuse) {
    bool relocated_without_prefix_hit = false;
    for (const ReuseInfo& entry : scr_reuse) {
        if (entry.prefix_relocated_tokens > 0 && entry.prefix_cache_hit_tokens == 0) {
            relocated_without_prefix_hit = true;
            break;
        }
    }
    if (!relocated_without_prefix_hit) { return; }
    std::cerr << "\n  On `reused=0`: prefix_cache_hit_tokens is the *exact-prefix* frontier "
                 "(reuse_base) -- the length\n  up to which the new prompt equals the resident "
                 "sequence in full. Any in-the-middle edit defeats\n  that match however long the "
                 "shared head is, so it reports 0 while the reuse that actually\n  happened is in "
                 "`relocated=`/prefix_relocated_tokens; the head before the edit is re-prefilled\n"
                 "  by the relocation plan. Re-run with --scr-diag to see the applied plan, "
                 "including that\n  head range, and any refusal the planner made.\n";
}

// The reuse telemetry as JSON, one entry per prompt per phase. It rides in the report because the
// thresholds are read against it: divergence that grows with the relocated span is a different
// finding from divergence that does not, and the K axis of the roadmap's Figure 13 lives here.
std::string reuse_section(const std::vector<ReuseInfo>& scr, const std::vector<ReuseInfo>& baseline,
                          const std::vector<ReuseInfo>& control) {
    const auto entries = [](const std::vector<ReuseInfo>& phase) {
        Json array = Json::array();
        for (const ReuseInfo& entry : phase) {
            array.push_back(Json{{"name", entry.name},
                                 {"primed_prompt_tokens", entry.primed_prompt_tokens},
                                 {"prompt_tokens", entry.prompt_tokens},
                                 {"edit_offset", entry.edit_offset},
                                 {"prefix_cache_hit_tokens", entry.prefix_cache_hit_tokens},
                                 {"prefix_relocated_tokens", entry.prefix_relocated_tokens},
                                 {"prefix_reuse_path", entry.prefix_reuse_path}});
        }
        return array;
    };
    return Json{{"scr", entries(scr)},
                {"baseline", entries(baseline)},
                {"control", entries(control)}}
        .dump();
}

// The two paths' actual prose, in the report rather than only on stderr: the artifact gets read
// again later, and "both runs still said the same answer" is not derivable from token ids by
// whoever reads it then. `first_diff_char` is -1 when the contents are identical.
std::string outputs_section(const std::vector<SuitePrompt>& prompts,
                            const std::vector<PhaseExchange>& baseline,
                            const std::vector<PhaseExchange>& scr,
                            const std::vector<PhaseExchange>& control) {
    const auto reply = [](const ChatReply& value) {
        return Json{{"content", value.content},
                    {"reasoning", value.reasoning},
                    {"finish_reason", value.finish_reason}};
    };
    Json array = Json::array();
    for (std::size_t i = 0; i < prompts.size() && i < baseline.size() && i < scr.size(); ++i) {
        const ChatReply& base = baseline[i].edited;
        const ChatReply& test = scr[i].edited;
        const std::size_t diff = first_diff(base.content, test.content);
        const std::int64_t diff_char =
            diff == std::string::npos ? -1 : static_cast<std::int64_t>(diff);
        Json entry{{"name", prompts[i].name},
                   {"identical", base.content == test.content && base.reasoning == test.reasoning},
                   {"first_diff_char", diff_char},
                   {"primed_identical", baseline[i].primed.content == scr[i].primed.content &&
                                             baseline[i].primed.reasoning ==
                                                 scr[i].primed.reasoning},
                   {"baseline", reply(base)},
                   {"scr", reply(test)}};
        if (i < control.size()) {
            entry["control_identical"] =
                control[i].edited.content == base.content &&
                control[i].edited.reasoning == base.reasoning;
            entry["control"] = reply(control[i].edited);
        }
        array.push_back(std::move(entry));
    }
    return array.dump();
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
        if (options.scr_diag) {
            // The spawned servers inherit this environment, so one variable turns on the runtime's
            // own account of every relocation: applied plan, page remap, bound block-table row,
            // re-rotation fingerprint, and any refusal the planner made. They print it to this
            // console because they share it with us -- nothing is parsed back out of it; the
            // report's job is to be readable without a second run.
#if defined(_WIN32)
            _putenv_s("NINFER_SCR_DIAG", "1");
#else
            ::setenv("NINFER_SCR_DIAG", "1", 1);
#endif
            std::cerr << "NINFER_SCR_DIAG=1 for the spawned servers (--scr-diag)\n";
        }

        const std::vector<PhaseExchange> scr_exchange =
            run_phase(options, prompts, true, options.port, scr_log, "scr");
        // A separate port keeps each later phase off a socket still in TIME_WAIT.
        const std::vector<PhaseExchange> baseline_exchange =
            run_phase(options, prompts, false, options.port + 1, baseline_log, "baseline");
        // The control is the baseline configuration run a second time. Without it the measurement has
        // no noise floor, and every divergence it reports is unattributable: the same spread is
        // produced by two runs that differ in nothing. The 2026-10-08 sweep is the case in point --
        // d(first) varied from 1.8e-4 to 7.5e-2 nats with no relationship to how much context was
        // reused, which is what a floor of comparable size looks like and not what a relocation error
       // that grows with the reused region looks like.
        const std::vector<PhaseExchange> control_exchange =
            run_phase(options, prompts, false, options.port + 2, control_log, "control");

        const auto load = [&](const std::string& path, const char* phase) {
            const std::vector<ninfer::serve::LoggedCompletion> completions =
                ninfer::serve::parse_completion_log(read_file(path));
            if (completions.size() != prompts.size() * 2) {
                throw std::runtime_error(std::string("the ") + phase +
                                         " request log does not describe the prompts that were sent");
            }
            return completions;
        };
        std::vector<ReuseInfo> scr_reuse;
        std::vector<ReuseInfo> baseline_reuse;
        std::vector<ReuseInfo> control_reuse;
        const std::vector<ninfer::serve::LoggedCompletion> scr_edited =
            select_edited(load(scr_log, "SCR"), "scr", true, prompts, scr_reuse);
        const std::vector<ninfer::serve::LoggedCompletion> baseline_edited =
            select_edited(load(baseline_log, "baseline"), "baseline", false, prompts,
                          baseline_reuse);
        const std::vector<ninfer::serve::LoggedCompletion> control_edited =
            select_edited(load(control_log, "control"), "control", false, prompts, control_reuse);

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
        // Hoisted out of the block below: the first-flip detail printer walks the same samples to
        // show the token each side chose where the streams parted, which the report does not carry.
        const auto measurement_samples =
            ninfer::serve::pair_scr_quality_samples(scr_edited, baseline_edited);
        {
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

        print_divergence_detail(prompts, measurement_samples, scored);
        print_output_comparison(prompts, baseline_exchange, scr_exchange, control_exchange);
        print_reuse_explainer(scr_reuse);

        // The envelope carries the control beside the measurement, because a report that states a
        // verdict without the floor it was measured against is not a report. The inner objects are
        // the target's own format, so the thresholds still exist in exactly one place; the reuse
        // telemetry and the outputs are the harness's own, because neither belongs to the target.
        std::ostringstream report;
        report << "{\"attributable\":" << (attributable ? "true" : "false")
               << ",\"reuse\":" << reuse_section(scr_reuse, baseline_reuse, control_reuse)
               << ",\"outputs\":" << outputs_section(prompts, baseline_exchange, scr_exchange,
                                                       control_exchange)
               << ",\"control\":"
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

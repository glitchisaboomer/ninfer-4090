// `docs/SCR.md` §5: does Suffix Cache Reuse change the model's answer?
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
#include <atomic>
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
#    include <fcntl.h>
#    include <io.h>
#    include <process.h>
#    include <windows.h>
#else
#    include <poll.h>
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
    // A fourth phase: the SCR flags with relocation switched off (`--scr --scr-max-spans 0`). Its
    // difference from the baseline is the cost of the configuration alone, which the run-to-run
    // control cannot see -- two baseline runs share their flags, and the flags are exactly what
    // differs between the two phases being compared.
    bool flag_control = false;
    // Where the console tee writes. Empty means <work-dir>/console.log; "-" disables the tee.
    std::string console_log;
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
              << "  --log <file|->                tee stdout and stderr -- this process and every\n"
                 "                                server it spawns -- to a file as well as the terminal\n"
                 "                                (default: <work-dir>/console.log)\n"
              << "  --keep-logs                   leave the request logs in place on success\n"
              << "  --flag-control                add a phase that runs the SCR flags with relocation\n"
                 "                                disabled, scored against the baseline as the\n"
                 "                                configuration floor\n"
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
        } else if (arg == "--log") {
            options.console_log = value_of(index);
        } else if (arg == "--flag-control") {
            options.flag_control = true;
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

// A tee for this process's console: everything written to stdout or stderr -- the harness's own
// diagnostics *and* the output of every ninfer-serve it spawns -- is shown on the terminal and
// written to one file.
//
// The redirection is at the descriptor level, not a stream-buffer swap. A stream-buffer tee would
// capture only what this process writes through C++ streams, and the lines worth keeping are not
// those: the runtime's own relocation diagnostics (`[scr-diag]`, written inside the child) and the
// server's startup banner -- including the CUDA-graph budget, `graphs=`, which is how a
// phase-dependent capture set becomes visible -- belong to the spawned servers. Pointing fd 1 and 2
// at a pipe captures a child's output whether or not the child knows the tee exists, because it
// inherits those descriptors.
//
// The reader polls rather than blocking on a read, so `stop()` cannot hang behind a descriptor some
// child leaked: after the descriptors are restored the reader takes one last drain and exits.
class ConsoleTee {
public:
    ConsoleTee()                               = default;
    ConsoleTee(const ConsoleTee&)              = delete;
    ConsoleTee& operator=(const ConsoleTee&)   = delete;
    ~ConsoleTee() { stop(); }

    // An empty path or "-" leaves the console alone. On failure the console is left as it was and
    // `error` says why: a harness that cannot log is still a harness that can measure.
    bool start(const std::string& path, std::string& error);
    // Restores the descriptors, drains, and joins the reader. Idempotent, and safe from a destructor.
    void stop() noexcept;

private:
    // True when there is something to read. Non-blocking on purpose: see the class comment.
    [[nodiscard]] bool readable() const;
    // One read-and-forward step. False when the pipe has nothing to give right now.
    bool drain_once();

    std::ofstream file_;
    int saved_stdout_ = -1;
    int saved_stderr_ = -1;
    int read_fd_      = -1;
    bool active_      = false;
    std::thread reader_;
    std::atomic<bool> stopping_{false};
};

bool ConsoleTee::start(const std::string& path, std::string& error) {
    if (path.empty() || path == "-") { return true; }
    file_.open(path, std::ios::binary | std::ios::trunc);
    if (!file_) {
        error = "cannot write the console log " + path;
        return false;
    }
    // Anything already written must reach the terminal before the descriptors move under it.
    std::cerr.flush();
    std::cout.flush();

    int pipe_fds[2] = {-1, -1};
#if defined(_WIN32)
    // Text mode on purpose: the CRT then translates a lone LF to CRLF for both the children and this
    // process, so the file gets the line endings a Windows editor expects and the terminal shows the
    // forwarded bytes as they were written.
    if (::_pipe(pipe_fds, 64 * 1024, _O_TEXT) != 0) {
        error = "cannot create the console pipe";
        file_.close();
        return false;
    }
    saved_stdout_ = ::_dup(1);
    saved_stderr_ = ::_dup(2);
#else
    if (::pipe(pipe_fds) != 0) {
        error = "cannot create the console pipe";
        file_.close();
        return false;
    }
    saved_stdout_ = ::dup(1);
    saved_stderr_ = ::dup(2);
#endif
    if (saved_stdout_ < 0 || saved_stderr_ < 0) {
#if defined(_WIN32)
        ::_close(pipe_fds[0]);
        ::_close(pipe_fds[1]);
        if (saved_stdout_ >= 0) { ::_close(saved_stdout_); }
        if (saved_stderr_ >= 0) { ::_close(saved_stderr_); }
#else
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        if (saved_stdout_ >= 0) { ::close(saved_stdout_); }
        if (saved_stderr_ >= 0) { ::close(saved_stderr_); }
#endif
        saved_stdout_ = -1;
        saved_stderr_ = -1;
        error         = "cannot duplicate the console descriptors";
        file_.close();
        return false;
    }
    // Both streams point at the pipe, so a spawned server is captured on either one, and so is this
    // process's own std::cout/std::cerr. The extra write descriptor is closed here: fd 1 and 2 hold
    // the only references this process needs, and keeping a third would stop the reader from seeing
    // EOF when they are restored.
#if defined(_WIN32)
    ::_dup2(pipe_fds[1], 1);
    ::_dup2(pipe_fds[1], 2);
    ::_close(pipe_fds[1]);
#else
    ::dup2(pipe_fds[1], 1);
    ::dup2(pipe_fds[1], 2);
    ::close(pipe_fds[1]);
#endif
    read_fd_ = pipe_fds[0];
    active_  = true;
    reader_  = std::thread([this] {
        while (true) {
            if (drain_once()) { continue; }
            if (stopping_.load()) {
                // One last drain: bytes can land between the availability check and the flag.
                while (drain_once()) {}
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }
    });
    return true;
}

bool ConsoleTee::readable() const {
    if (read_fd_ < 0) { return false; }
#if defined(_WIN32)
    const HANDLE handle = reinterpret_cast<HANDLE>(::_get_osfhandle(read_fd_));
    if (handle == INVALID_HANDLE_VALUE) { return false; }
    DWORD waiting = 0;
    if (::PeekNamedPipe(handle, nullptr, 0, nullptr, &waiting, nullptr) == 0) { return false; }
    return waiting != 0;
#else
    struct pollfd probe{read_fd_, POLLIN, 0};
    return ::poll(&probe, 1, 0) > 0;
#endif
}

bool ConsoleTee::drain_once() {
    if (!readable()) { return false; }
    char buffer[16384];
#if defined(_WIN32)
    const int bytes = ::_read(read_fd_, buffer, static_cast<unsigned>(sizeof buffer));
#else
    const ssize_t bytes = ::read(read_fd_, buffer, sizeof buffer);
#endif
    if (bytes <= 0) { return false; }
    file_.write(buffer, bytes);
    file_.flush();
    // The terminal gets the bytes through the descriptor this process started with. Everything is
    // forwarded to stdout rather than to stderr, because a run that is being piped is piping stdout;
    // a reader who wants the streams separated has the file and the per-phase request logs.
#if defined(_WIN32)
    (void)::_write(saved_stdout_, buffer, static_cast<unsigned>(bytes));
#else
    (void)::write(saved_stdout_, buffer, static_cast<std::size_t>(bytes));
#endif
    return true;
}

void ConsoleTee::stop() noexcept {
    if (!active_) { return; }
    active_ = false;
    std::cerr.flush();
    std::cout.flush();
    // Restoring the descriptors drops this process's references to the pipe's write end. The reader
    // still writes to the saved descriptors, so they are closed only after it has been joined.
#if defined(_WIN32)
    if (saved_stdout_ >= 0) { ::_dup2(saved_stdout_, 1); }
    if (saved_stderr_ >= 0) { ::_dup2(saved_stderr_, 2); }
#else
    if (saved_stdout_ >= 0) { ::dup2(saved_stdout_, 1); }
    if (saved_stderr_ >= 0) { ::dup2(saved_stderr_, 2); }
#endif
    stopping_.store(true);
    if (reader_.joinable()) { reader_.join(); }
#if defined(_WIN32)
    if (saved_stdout_ >= 0) { ::_close(saved_stdout_); }
    if (saved_stderr_ >= 0) { ::_close(saved_stderr_); }
    if (read_fd_ >= 0) { ::_close(read_fd_); }
#else
    if (saved_stdout_ >= 0) { ::close(saved_stdout_); }
    if (saved_stderr_ >= 0) { ::close(saved_stderr_); }
    if (read_fd_ >= 0) { ::close(read_fd_); }
#endif
    saved_stdout_ = -1;
    saved_stderr_ = -1;
    read_fd_      = -1;
    file_.flush();
    file_.close();
}

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
                                     const char* label,
                                     const std::vector<std::string>& phase_args = {}) {
    std::remove(log_path.c_str());
    const std::vector<std::string> extra = [&] {
        std::vector<std::string> args = split_whitespace(options.server_args);
        if (!scr) {
            const std::vector<std::string> baseline = split_whitespace(options.baseline_server_args);
            args.insert(args.end(), baseline.begin(), baseline.end());
        }
        // Last, so a phase can override the earlier flags: the flag control appends
        // `--scr-max-spans 0` to an otherwise ordinary SCR command line.
        args.insert(args.end(), phase_args.begin(), phase_args.end());
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
//
// The priming request carries its own telemetry because it is not always a cold prefill: the SCR
// phase relocates the priming request too whenever a resident tail matches, so its reply is expected
// to differ from a cold baseline's and the two cases have to be told apart by their paths.
struct ReuseInfo {
    std::string name;
    std::int32_t primed_prompt_tokens     = 0;
    std::int32_t prompt_tokens            = 0;
    // The edited request's token count minus the priming request's -- a *length* delta, not the
    // position of the edit. The built-in suite balances its edit, so this is 0 on every prompt while
    // the edit itself sits in the middle; calling it `edit_offset` invited exactly that misreading.
    std::int32_t length_delta               = 0;
    std::uint32_t prefix_cache_hit_tokens   = 0;
    std::uint32_t prefix_relocated_tokens   = 0;
    std::string prefix_reuse_path;
    std::uint32_t primed_relocated_tokens    = 0;
    std::string primed_reuse_path;
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
        const std::int32_t length_delta   = edited.prompt_tokens - primed.prompt_tokens;
        const bool aligned                = length_delta % kPageTokens == 0;
        const std::size_t prompt_index = index / 2;
        const std::string name = prompt_index < prompts.size()
                                     ? prompts[prompt_index].name
                                     : "prompt " + std::to_string(prompt_index);
        std::cerr << "[" << label << "] " << name
                  << ": prompt_tokens=" << edited.prompt_tokens
                  << " primed=" << primed.prompt_tokens
                  << " length_delta=" << length_delta << (aligned ? "" : " NOT_PAGE_ALIGNED")
                  << " reused=" << edited.prefix_cache_hit_tokens
                  << " relocated=" << edited.prefix_relocated_tokens
                  << " path=" << edited.prefix_reuse_path
                  << " primed_relocated=" << primed.prefix_relocated_tokens
                  << " primed_path=" << primed.prefix_reuse_path << "\n";
        if (!aligned && scr) { misaligned.push_back(prompt_index); }
        reuse.push_back(ReuseInfo{name, primed.prompt_tokens, edited.prompt_tokens, length_delta,
                                  edited.prefix_cache_hit_tokens, edited.prefix_relocated_tokens,
                                  edited.prefix_reuse_path, primed.prefix_relocated_tokens,
                                  primed.prefix_reuse_path});
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
//
// The verdict is the divergence budget alone. Greedy agreement is printed beside it because it says
// where the streams parted, but it does not decide the column: Suffix Cache Reuse is an
// approximation, so "the same word must come out" was never a property it promised, and on this
// suite that metric turned out to read where the prompts happen to sit on a tie. See the note in
// scr_quality.cpp for the sweep that established it.
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

// A reply's comparable text. While the model is thinking, `content` is empty and the whole reply
// lives in `reasoning`, so a comparison over `content` alone compares two empty strings on every
// prompt that hits the output limit -- which is how the 2026-10-08 sweep reported
// first_diff_char -1 beside lines that said the text differed. The fields are compared individually
// in `reply_matches`; this join exists for the character diff, so a reply that finished its thinking
// into `content` is not read as a wording change.
std::string reply_text(const ChatReply& reply) {
    if (reply.content.empty()) { return reply.reasoning; }
    if (reply.reasoning.empty()) { return reply.content; }
    return reply.content + "\n" + reply.reasoning;
}

bool reply_matches(const ChatReply& left, const ChatReply& right) {
    return left.content == right.content && left.reasoning == right.reasoning;
}

// The window around the first difference rather than the head of the text. Two replies that share a
// preamble and part at character 800 read as identical when both are printed from the start, which
// is how a real difference hides next to a line announcing that the text differs. Flattening is one
// character for one, so the index stays valid.
std::string window_around(const std::string& text, std::size_t at, std::size_t before = 70,
                          std::size_t after = 150) {
    std::string flat = text;
    for (char& character : flat) {
        if (character == '\n' || character == '\r' || character == '\t') { character = ' '; }
    }
    if (at >= flat.size()) { return excerpt(flat, before + after); }
    const std::size_t begin = at > before ? at - before : 0;
    const std::size_t end   = std::min(flat.size(), at + after);
    const std::string window = (begin == 0 ? std::string() : std::string("...")) +
                               flat.substr(begin, end - begin) +
                               (end == flat.size() ? std::string() : std::string("..."));
    return "'" + window + "'";
}

// The flip itself, with the numbers that decide what it means. Margin is best-minus-second-best in
// nats: a flip behind a margin of a thousandth is a coin tipped by any perturbation at all, a flip
// against a wide margin is a decision the reuse path actually moved. Position 0 comes along too,
// because d(first) in the report is measured there and is only readable against the margin beside
// it. Values come straight from the request log; a server whose log predates the margin field
// prints n/a rather than borrowing a margin from a neighbouring position.
//
// The same detail is printed for the control as for the measurement. A control flip is the most
// important number in the run -- it is the difference between "the reuse path moved this decision"
// and "this prompt sits on a tie" -- and in the 2026-10-08 sweep every control flip happened behind
// an exactly zero margin, which was visible only in the raw request log. `left_label` names the
// sample's `baseline_*` column and `right_label` its `scr_*` column, because for the control pair
// those columns hold the control phase and the baseline phase respectively, not scr and baseline.
void print_divergence_detail(const char* title, const char* left_label, const char* right_label,
                             const std::vector<SuitePrompt>& prompts,
                             const std::vector<ScrQualitySample>& samples,
                             const ScrQualityReport& scored) {
    std::cerr << "\n" << title
              << "\n\n  margin = best-minus-second-best logit gap, in nats: a flip at a margin of "
                 "zero is\n  a coin tipped by any perturbation at all, including the one between "
                 "two runs of the\n  same configuration.\n";
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
        std::cerr << "\n    pos0    logprob " << left_label << " "
                  << metric(sample.baseline_logprobs[0]) << "  " << right_label << " "
                  << metric(sample.scr_logprobs[0]) << "   margin " << left_label << " "
                  << margin_at(sample.baseline_margins, 0) << "  " << right_label << " "
                  << margin_at(sample.scr_margins, 0);
        const std::size_t flip = result.first_divergence_index;
        if (flip < overlap) {
            std::cerr << "\n    flip@" << flip << "  " << left_label << " token "
                      << sample.baseline_tokens[flip] << " logprob "
                      << metric(sample.baseline_logprobs[flip]) << " margin "
                      << margin_at(sample.baseline_margins, flip) << "   |  " << right_label
                      << " token " << sample.scr_tokens[flip] << " logprob "
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
// The text compared is the reply's `content` *and* its `reasoning`, field by field, and the window
// shown starts at the character that differs. Comparing `content` alone compared two empty strings
// on every prompt of the 2026-10-08 sweep -- a reply that stops at the output limit carries its
// whole text in `reasoning` -- and printing both excerpts from the head made two replies that part
// mid-sentence look the same.
//
// The priming replies are checked on the comparison that can carry a verdict. The SCR phase serves
// its priming request through relocation whenever a resident tail matches (K >= 3 in that sweep), so
// its priming reply is *expected* to differ from a cold baseline's; calling that "the resident KV was
// seeded differently" blamed the feature for working. What must not differ is the control against
// the baseline: same flags, same cold path, so a difference there means the priming request itself
// is not reproducible and every pair built on it inherits the difference.
void print_output_comparison(const std::vector<SuitePrompt>& prompts,
                             const std::vector<PhaseExchange>& baseline,
                             const std::vector<PhaseExchange>& scr,
                             const std::vector<PhaseExchange>& control,
                             const std::vector<ReuseInfo>& scr_reuse) {
    std::cerr << "\nGENERATED OUTPUT  (--no-scr vs --scr, edited request)\n"
                 "\n  Excerpts start at the first differing character, so a pair whose excerpts look "
                 "similar\n  but whose difference is late has parted outside its opening."
                 "\n";
    std::size_t identical           = 0;
    std::size_t primed_same         = 0;
    std::size_t primed_by_reuse     = 0;
    std::size_t primed_control_same = 0;
    std::size_t control_same        = 0;
    const std::size_t measured = std::min({prompts.size(), baseline.size(), scr.size()});
    for (std::size_t i = 0; i < measured; ++i) {
        const ChatReply& base = baseline[i].edited;
        const ChatReply& test = scr[i].edited;
        const bool same           = reply_matches(base, test);
        const bool primed_match   = reply_matches(baseline[i].primed, scr[i].primed);
        const bool primed_control_match =
            i < control.size() && reply_matches(control[i].primed, baseline[i].primed);
        const bool control_match = i < control.size() && reply_matches(control[i].edited, base);
        const std::uint32_t primed_relocated =
            i < scr_reuse.size() ? scr_reuse[i].primed_relocated_tokens : 0U;
        const std::string primed_path =
            i < scr_reuse.size() ? scr_reuse[i].primed_reuse_path : std::string();
        identical += same ? 1U : 0U;
        primed_same += primed_match ? 1U : 0U;
        primed_control_same += primed_control_match ? 1U : 0U;
        control_same += control_match ? 1U : 0U;
        if (!primed_match && primed_relocated != 0) { ++primed_by_reuse; }

        // Which field carries the difference. `reasoning` is preferred while the reply is still
        // thinking, because `content` is empty there and a window into nothing says nothing; both are
        // compared and the field is named wherever a difference is reported.
        const bool content_differs   = base.content != test.content;
        const bool reasoning_differs = base.reasoning != test.reasoning;
        const bool show_reasoning    = reasoning_differs || base.content.empty();
        const std::string& base_text = show_reasoning ? base.reasoning : base.content;
        const std::string& test_text = show_reasoning ? test.reasoning : test.content;
        const char* field            = show_reasoning ? "reasoning" : "content";
        const std::size_t diff       = first_diff(base_text, test_text);

        std::cerr << "\n  " << prompts[i].name << ": "
                  << (same ? "reply identical" : "REPLY DIFFERS");
        if (content_differs || reasoning_differs) {
            std::cerr << " (" << field << " differs at character " << diff << " of "
                      << std::max(base_text.size(), test_text.size()) << ")";
        }
        std::cerr << ", finish --no-scr="
                  << (base.finish_reason.empty() ? "?" : base.finish_reason)
                  << " --scr=" << (test.finish_reason.empty() ? "?" : test.finish_reason);
        if (!primed_control_match) {
            std::cerr << "\n    PRIMING NOT REPRODUCIBLE: the control answered its own priming "
                         "request differently\n    from the baseline (both --no-scr "
                         "--no-prefix-reuse, both cold), so this pair is not\n    a clean A/B and "
                         "the priming request is where to look first";
        } else if (!primed_match && primed_relocated != 0) {
            std::cerr << "\n    priming reply differs by design: the SCR phase relocated "
                      << primed_relocated << " token(s) of its priming request (path="
                      << primed_path << "), which the cold baseline cannot";
        } else if (!primed_match) {
            std::cerr << "\n    PRIMED replies differ with both phases on a cold prefill: the "
                         "resident KV was\n    not built the same way";
        }
        if (!control_match) {
            std::cerr << "\n    CONTROL reply differs from baseline: two identical "
                         "configurations produced\n    different text here";
        }
        if (diff != std::string::npos) {
            std::cerr << "\n    --no-scr " << field << ": " << window_around(base_text, diff)
                      << "\n    --scr    " << field << ": " << window_around(test_text, diff);
        } else {
            std::cerr << "\n    --no-scr " << field << ": " << excerpt(base_text)
                      << "\n    --scr    " << field << ": " << excerpt(test_text);
        }
        std::cerr << "\n";
    }
    std::cerr << "\n  edited replies identical on " << identical << "/" << measured
              << " prompt(s) (content+reasoning); primed replies identical on " << primed_same
              << "/" << measured << "; " << primed_by_reuse << " of the "
              << (measured - primed_same)
              << " that differ did so because the SCR phase relocated the priming request\n"
              << "  (that is the feature working, not a confound); primed control vs baseline "
                 "identical on "
              << primed_control_same << "/" << measured << "; edited control matched baseline on "
              << control_same << "/" << measured << "\n";
}

// The measurement against every floor the run has, per prompt. Two floors, because the two phases
// being compared differ in more than relocation: the run-to-run control shares the baseline's flags
// and cannot see the configuration, while the flag phase carries the SCR flags with relocation
// disabled and cannot see the reuse. A divergence that sits inside or behind either one is not
// evidence about relocation, and this is the table that says so without re-reading three others.
//
// The last row is the one the comparison was after all along: the SCR run against the
// relocation-disabled run, which have identical flags, so whatever it shows is relocation's.
void print_attribution_table(const std::vector<SuitePrompt>& prompts,
                             const std::vector<ScrQualitySample>& measurement,
                             const std::vector<ScrQualitySample>& noise,
                             const std::vector<ScrQualitySample>& config_floor,
                             const std::vector<ScrQualitySample>& relocation_only,
                             const ninfer::targets::qwen3_6::detail::ScrQualityThresholds& thresholds) {
    std::cerr << "\nATTRIBUTION  (the measurement against both floors it has to clear)\n"
                 "\n  Every row is measured against the same baseline run, so `parted@` is comparable\n"
                 "  across rows: a smaller one means the two streams parted sooner.\n";
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        std::cerr << "\n  " << prompts[i].name << "\n";
        // Evaluated by the target's own scorer, so the arithmetic here is not a second implementation
        // of `d(first)`/`d(mean)` that can drift from the report's.
        const auto line = [&](const std::vector<ScrQualitySample>& samples) -> std::string {
            if (i >= samples.size()) { return "n/a"; }
            const auto result = ninfer::targets::qwen3_6::detail::evaluate_scr_prompt(samples[i], thresholds);
            std::ostringstream out;
            out.precision(6);
            out << "d(first) " << std::fixed << result.first_token_divergence << "  parted@ "
                << result.first_divergence_index << "  d(mean) " << result.mean_logprob_divergence;
            return out.str();
        };
        std::cerr << "    measurement      scr vs --no-scr                       " << line(measurement)
                  << "\n";
        std::cerr << "    noise floor      --no-scr vs --no-scr (2nd run)        " << line(noise)
                  << "\n";
        std::cerr << "    config floor     --scr --scr-max-spans 0 vs --no-scr  " << line(config_floor)
                  << "\n";
        std::cerr << "    relocation only  scr vs --scr --scr-max-spans 0        " << line(relocation_only)
                  << "\n";
        if (i < relocation_only.size() && i < config_floor.size()) {
            const auto relocated =
                ninfer::targets::qwen3_6::detail::evaluate_scr_prompt(relocation_only[i], thresholds);
            const auto flagged =
                ninfer::targets::qwen3_6::detail::evaluate_scr_prompt(config_floor[i], thresholds);
            const bool ahead = relocated.first_divergence_index < flagged.first_divergence_index;
            std::cerr << "    verdict          "
                      << (ahead
                              ? "relocation parts the streams sooner than the configuration alone"
                              : "the configuration alone parts them no later -- the flags are the "
                                "candidate, not the reuse")
                      << "\n";
        }
    }
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
                          const std::vector<ReuseInfo>& control,
                          const std::vector<ReuseInfo>& flag) {
    const auto entries = [](const std::vector<ReuseInfo>& phase) {
        Json array = Json::array();
        for (const ReuseInfo& entry : phase) {
            array.push_back(Json{{"name", entry.name},
                                 {"primed_prompt_tokens", entry.primed_prompt_tokens},
                                 {"prompt_tokens", entry.prompt_tokens},
                                 // The edited request's token count minus the priming request's. Not
                                 // the position of the edit: the built-in suite balances its edit,
                                 // so this is 0 on every prompt while the edit sits in the middle.
                                 {"length_delta", entry.length_delta},
                                 {"prefix_cache_hit_tokens", entry.prefix_cache_hit_tokens},
                                 {"prefix_relocated_tokens", entry.prefix_relocated_tokens},
                                 {"prefix_reuse_path", entry.prefix_reuse_path},
                                 {"primed_relocated_tokens", entry.primed_relocated_tokens},
                                 {"primed_reuse_path", entry.primed_reuse_path}});
        }
        return array;
    };
    // The flag phase relocates nothing by construction, so its entries are the assertion that the
    // configuration control really was one -- kept in the report, because that is what a reader
    // checking the run later has to go on.
    return Json{{"scr", entries(scr)},
                {"baseline", entries(baseline)},
                {"control", entries(control)},
                {"flag", entries(flag)}}
        .dump();
}

// The two paths' actual prose, in the report rather than only on stderr: the artifact gets read
// again later, and "both runs still said the same answer" is not derivable from token ids by
// whoever reads it then.
//
// `first_diff_char` is the index of the first differing character *within* `first_diff_field`, which
// names the field it was measured in (`content`, `reasoning`, or `none` when both fields match). It
// used to be measured in `content` alone, which is empty whenever the reply stops while still
// thinking, so it read -1 on every prompt of the 2026-10-08 sweep beside `identical: false`. The two
// `*_differs` flags say which fields moved, so a reader does not have to infer it from the window.
std::string outputs_section(const std::vector<SuitePrompt>& prompts,
                            const std::vector<PhaseExchange>& baseline,
                            const std::vector<PhaseExchange>& scr,
                            const std::vector<PhaseExchange>& control,
                            const std::vector<ReuseInfo>& scr_reuse) {
    const auto reply = [](const ChatReply& value) {
        return Json{{"content", value.content},
                    {"reasoning", value.reasoning},
                    {"finish_reason", value.finish_reason}};
    };
    Json array = Json::array();
    for (std::size_t i = 0; i < prompts.size() && i < baseline.size() && i < scr.size(); ++i) {
        const ChatReply& base = baseline[i].edited;
        const ChatReply& test = scr[i].edited;
        const bool content_differs   = base.content != test.content;
        const bool reasoning_differs = base.reasoning != test.reasoning;
        const bool show_reasoning    = reasoning_differs || base.content.empty();
        const std::string& base_text = show_reasoning ? base.reasoning : base.content;
        const std::string& test_text = show_reasoning ? test.reasoning : test.content;
        const std::size_t diff       = first_diff(base_text, test_text);
        const std::int64_t diff_char =
            diff == std::string::npos ? -1 : static_cast<std::int64_t>(diff);
        Json entry{{"name", prompts[i].name},
                   {"identical", reply_matches(base, test)},
                   {"content_differs", content_differs},
                   {"reasoning_differs", reasoning_differs},
                   {"first_diff_field",
                    (content_differs || reasoning_differs) ? (show_reasoning ? "reasoning" : "content")
                                                           : "none"},
                   {"first_diff_char", diff_char},
                   {"primed_identical", reply_matches(baseline[i].primed, scr[i].primed)},
                   {"primed_relocated_tokens",
                    i < scr_reuse.size() ? scr_reuse[i].primed_relocated_tokens : 0U},
                   {"baseline", reply(base)},
                   {"scr", reply(test)}};
        if (i < control.size()) {
            entry["primed_identical_control"] =
                reply_matches(control[i].primed, baseline[i].primed);
            entry["control_identical"] = reply_matches(control[i].edited, base);
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
        // Installed before the first server exists, so the children inherit the pipe and every phase's
        // console output -- the harness's diagnostics, the runtime's `[scr-diag]` lines, and the
        // server's own banner with its CUDA-graph budget -- is both shown and kept.
        ConsoleTee tee;
        {
            const std::string console_path = options.console_log.empty()
                                                 ? options.work_dir + "/console.log"
                                                 : options.console_log;
            std::string tee_error;
            if (!tee.start(console_path, tee_error)) {
                std::cerr << "warning: " << tee_error << "; continuing without a console log\n";
            } else if (options.console_log != "-") {
                std::cerr << "console log: " << console_path << " (also on the terminal)\n";
            }
        }
        const std::string scr_log      = options.work_dir + "/scr.jsonl";
        const std::string baseline_log = options.work_dir + "/baseline.jsonl";
        const std::string control_log  = options.work_dir + "/control.jsonl";
        const std::string flag_log     = options.work_dir + "/flag.jsonl";
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
        // The flag control, when asked for: the same SCR command line with relocation switched off.
        // The run-to-run control cannot see the difference between "relocation perturbed this" and
        // "the SCR configuration perturbed this", because the two phases it compares have different
        // flags; this phase has the SCR flags and no relocation, so the gap between it and the real
        // SCR run is relocation's alone.
        std::vector<PhaseExchange> flag_exchange;
        if (options.flag_control) {
            flag_exchange = run_phase(options, prompts, true, options.port + 3, flag_log, "flag",
                                      {"--scr-max-spans", "0"});
        }

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
        std::vector<ReuseInfo> flag_reuse;
        std::vector<ninfer::serve::LoggedCompletion> flag_edited;
        if (options.flag_control) {
            flag_edited = select_edited(load(flag_log, "flag"), "flag", false, prompts, flag_reuse);
            // The phase is only a configuration control if relocation really did not run. Checked
            // rather than assumed: `--scr-max-spans 0` is what is supposed to disable it, and a
            // silent regression there would turn this phase into a second baseline run that reads
            // like a clean floor.
            for (const ReuseInfo& entry : flag_reuse) {
                if (entry.prefix_relocated_tokens != 0 || entry.primed_relocated_tokens != 0) {
                    throw std::runtime_error(
                        "the flag control relocated " +
                        std::to_string(entry.prefix_relocated_tokens) + " token(s) on " + entry.name +
                        " (primed " + std::to_string(entry.primed_relocated_tokens) +
                        "); --scr-max-spans 0 did not disable relocation, so this phase is not a "
                        "configuration control");
                }
            }
        }

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
        // Kept past the floor loop below: the first-flip detail is worth exactly as much on the
        // control as on the measurement, and a control flip behind a 0.0-nat margin is the difference
        // between "the reuse path moved this decision" and "this prompt sits on a tie". In the
        // 2026-10-08 sweep every control flip was the latter, and it was visible only in the raw
        // request log because only the measurement reached the detail printer.
        const auto control_samples =
            ninfer::serve::pair_scr_quality_samples_for_control(baseline_edited, control_edited);
        for (std::size_t i = 0; i < scored.prompts.size() && i < control_samples.size(); ++i) {
            scored.prompts[i] = ninfer::targets::qwen3_6::detail::evaluate_scr_prompt_against_control(
                measurement_samples[i], control_samples[i], thresholds);
        }
        const std::vector<std::uint32_t> unattributable =
            ninfer::targets::qwen3_6::detail::scr_quality_control_divergent_prompts(control);
        const bool attributable =
            ninfer::targets::qwen3_6::detail::scr_quality_control_is_clean(control);

        // The flag phase scored against the baseline, and the two pairings the attribution table
        // needs. `pair_*_for_control` puts its first argument in `scr_*` and its second in
        // `baseline_*`, so each call below names the order it wants on purpose.
        ninfer::targets::qwen3_6::detail::ScrQualityReport flag_report;
        std::vector<ScrQualitySample> config_floor_samples;
        std::vector<ScrQualitySample> relocation_only_samples;
        if (options.flag_control) {
            flag_report = ninfer::serve::evaluate_scr_quality_control(baseline_edited, flag_edited,
                                                                      thresholds);
            config_floor_samples =
                ninfer::serve::pair_scr_quality_samples_for_control(baseline_edited, flag_edited);
            relocation_only_samples =
                ninfer::serve::pair_scr_quality_samples_for_control(scr_edited, flag_edited);
        }

        print_curve("MEASUREMENT  scr vs baseline", prompts, scr_edited, scored, true);
        print_curve("CONTROL  baseline vs baseline (the noise floor)", prompts, baseline_edited,
                    control, false);
        if (options.flag_control) {
            print_curve("FLAG CONTROL  --scr --scr-max-spans 0 vs baseline (relocation off, flags on)",
                        prompts, flag_edited, flag_report, false);
        }

        std::cerr << "\n  parted@ is the first greedy token the two runs disagreed on, so the columns "
                     "to its right\n  describe a prefix both runs chose identically and are "
                     "comparable; the agreement column is\n  index-aligned and is not.\n";

        print_divergence_detail("FIRST-FLIP DETAIL  (measurement: --scr vs --no-scr)", "--no-scr",
                                "--scr", prompts, measurement_samples, scored);
        // The columns are named, not implied: these samples hold the baseline phase in the `scr_*`
        // column and the control phase in the `baseline_*` column, because that is the order the
        // control was paired in. A label borrowed from the measurement would print the two runs
        // under each other's names.
        print_divergence_detail("FIRST-FLIP DETAIL  (control: baseline run 2 vs baseline run 1)",
                                "control", "baseline", prompts, control_samples, control);
        if (options.flag_control) {
            print_divergence_detail(
                "FIRST-FLIP DETAIL  (flag control: --scr --scr-max-spans 0 vs --no-scr)", "flag",
                "--no-scr", prompts, config_floor_samples, flag_report);
            print_attribution_table(prompts, measurement_samples, control_samples, config_floor_samples,
                                    relocation_only_samples, thresholds);
        }
        print_output_comparison(prompts, baseline_exchange, scr_exchange, control_exchange, scr_reuse);
        print_reuse_explainer(scr_reuse);
        if (options.flag_control) {
            std::cerr << "\n  The flag control relocated no token on any request (asserted, not assumed): "
                         "it ran the SCR\n  command line with `--scr-max-spans 0`, so its divergence from "
                         "the baseline is what the\n  flags alone cost. Read the measurement against "
                         "both floors above, never against zero.\n";
        }

        // The envelope carries the control beside the measurement, because a report that states a
        // verdict without the floor it was measured against is not a report. The inner objects are
        // the target's own format, so the thresholds still exist in exactly one place; the reuse
        // telemetry and the outputs are the harness's own, because neither belongs to the target.
        std::ostringstream report;
        report << "{\"attributable\":" << (attributable ? "true" : "false")
               << ",\"reuse\":" << reuse_section(scr_reuse, baseline_reuse, control_reuse, flag_reuse)
               << ",\"outputs\":" << outputs_section(prompts, baseline_exchange, scr_exchange,
                                                       control_exchange, scr_reuse)
               << ",\"control\":"
               << ninfer::targets::qwen3_6::detail::format_scr_quality_json(control, thresholds)
               << ",\"measurement\":"
               << ninfer::targets::qwen3_6::detail::format_scr_quality_json(scored, thresholds);
        // Only when the phase ran: an empty "flag" section would read as a flag control that came
        // back clean rather than as one that was never taken.
        if (options.flag_control) {
            report << ",\"flag\":"
                   << ninfer::targets::qwen3_6::detail::format_scr_quality_json(flag_report, thresholds);
        }
        report << "}\n";

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
            if (options.flag_control) { std::remove(flag_log.c_str()); }
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

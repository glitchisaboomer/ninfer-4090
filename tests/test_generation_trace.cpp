// The generation trace is an opt-in diagnostic. This pins the exact rule the executor uses: unset,
// empty and '0' are off; any other value is on; the decision is cached after the first read so it
// cannot change mid-request. Host-only, so the rule is checked even where no GPU is available.

#include "runtime/engine/generation_trace.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string_view>

namespace {

int failures = 0;

// A streamable operand with an observable formatting side effect. The disabled trace must not call
// this: formatting a diagnostic line is work the normal request path should not pay for.
struct Formatted {
    int* counter;
};

std::ostream& operator<<(std::ostream& out, const Formatted& value) {
    ++*value.counter;
    return out;
}

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void set_env(const char* value) {
#if defined(_WIN32)
    // MSVC has no setenv/unsetenv. An empty value is one of the documented off states, which is
    // enough for the latch check; CMake -E env --unset covers the true unset case on every host.
    _putenv_s("NINFER_GENERATION_TRACE", value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv("NINFER_GENERATION_TRACE");
    } else {
        setenv("NINFER_GENERATION_TRACE", value, 1);
    }
#endif
}

} // namespace

int main() {
    using ninfer::runtime::generation_trace_enabled;

    // CTest invokes this binary once per setting. Each process reads the environment exactly once,
    // so the case selected here also proves the value stays latched for that process: the cache is
    // checked against the environment again after several later set_env calls.
    const char* value = std::getenv("NINFER_GENERATION_TRACE");
    const std::string_view setting = value == nullptr ? "<unset>" : value;
    bool expected = true;
    if (setting == "1" || setting == "on") {
        expected = true;
    } else if (setting == "0") {
        expected = false;
    } else if (setting == "") {
        expected = false;
    } else if (setting == "<unset>") {
        expected = false;
    } else if (setting == "yes") {
        expected = true;
    } else {
        std::cerr << "FAIL: unrecognised NINFER_GENERATION_TRACE test mode: " << setting << '\n';
        return 1;
    }

    expect(generation_trace_enabled() == expected, "trace switch does not match its setting");
    set_env(expected ? "0" : "1");
    expect(generation_trace_enabled() == expected, "trace switch must stay latched for the process");
    set_env(nullptr);
    expect(generation_trace_enabled() == expected, "trace switch must not be re-read after unset");
    set_env(value);

    if (failures != 0) { return failures; }

    // Separate invocations cover the off side, the non-numeric truthy forms, and the enabled line.
    // Each mode is read once per process, so each case gets its own invocation. The disabled line is
    // checked here by redirecting stderr: with the switch latched off it must neither format its
    // operands nor print. Formatted above has an observable formatting side effect for that check.
    if (setting == "<unset>") {
        expect(value == nullptr, "an unset variable must read as null");
        std::ostringstream observed;
        std::streambuf* saved = std::cerr.rdbuf(observed.rdbuf());
        int formatted = 0;
        {
            ninfer::runtime::GenerationTraceLine line;
            line << Formatted{&formatted};
        }
        std::cerr.rdbuf(saved);
        expect(formatted == 0, "a disabled trace must not format its operands");
        expect(observed.str().empty(), "a disabled trace must not print");
    } else if (setting == "1" || setting == "on") {
        std::ostringstream observed;
        std::streambuf* saved = std::cerr.rdbuf(observed.rdbuf());
        {
            ninfer::runtime::GenerationTraceLine line;
            line << "before step lane=" << 3;
        }
        std::cerr.rdbuf(saved);
        expect(generation_trace_enabled(), "the trace must be on");
        expect(observed.str() == "GENERATION_TRACE: before step lane=3\n",
               "an enabled trace must print one prefixed line");
    }

    if (failures != 0) {
        std::cerr << failures << " generation trace checks failed (" << setting << ")\n";
        return failures;
    }
    std::cout << "generation trace checks passed\n";
    return 0;
}

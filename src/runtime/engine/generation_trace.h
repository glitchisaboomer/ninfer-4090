#pragma once

// Generation-path diagnostic switch. The executor's traces used to be unconditional stderr writes,
// which made every normal build and every request print them. They are now opt-in like the other
// runtime diagnostics: NINFER_GENERATION_TRACE=1 turns them on, unset/empty/0 leaves them off.
//
// `generation_trace` is a null sink when disabled, so call sites build no trace text in production.
// Both the switch and the sink are cached for the process lifetime: a running request's trace mode
// cannot change under it, and the environment must be set before starting the Engine. This follows
// the same rule as NINFER_SCR_DIAG and NINFER_PREFILL_TRACE.
//
// Kept out of concurrent_executor.h so a host test can include the policy without pulling in the
// target runtime and its device headers.

#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace ninfer::runtime {

[[nodiscard]] inline bool generation_trace_enabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* value = std::getenv("NINFER_GENERATION_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

// One line of a generation trace. The enabled form buffers a line and prints it on destruction, so
// a site can stream values into it with `<<` exactly as it did with std::cerr. The disabled form is
// an empty object whose `operator<<` discards its argument; there is no stream state to flush.
class GenerationTraceLine {
public:
    GenerationTraceLine() = default;

    GenerationTraceLine(const GenerationTraceLine&)            = delete;
    GenerationTraceLine& operator=(const GenerationTraceLine&) = delete;

    ~GenerationTraceLine() {
        if (line_) { std::cerr << "GENERATION_TRACE: " << line_->str() << '\n'; }
    }

    template <class T>
    GenerationTraceLine& operator<<(T&& value) {
        if (line_) { *line_ << std::forward<T>(value); }
        return *this;
    }

private:
    std::optional<std::ostringstream> line_ =
        generation_trace_enabled() ? std::optional<std::ostringstream>(std::in_place)
                                   : std::nullopt;
};

} // namespace ninfer::runtime

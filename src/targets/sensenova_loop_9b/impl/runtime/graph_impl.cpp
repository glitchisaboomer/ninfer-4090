#include "targets/sensenova_loop_9b/impl/runtime/graph_impl.h"

#include "core/device.h"

#include <exception>
#include <stdexcept>

namespace ninfer::targets::sensenova_loop_9b::detail {
namespace {

// A stream left in capture mode makes every later launch on it fail, so a failed attempt must
// drain the capture state even when the failure came from instantiation rather than the body.
// Ending a capture that is not active is itself an error; both cases are reported through the
// runner diagnostic, and the sticky per-thread error is cleared so the eager fallback is not
// poisoned by the attempt.
void recover_stream(cudaStream_t stream) noexcept {
    cudaGraph_t discarded = nullptr;
    (void)cudaStreamEndCapture(stream, &discarded);
    if (discarded != nullptr) { (void)cudaGraphDestroy(discarded); }
    (void)cudaGetLastError();
}

} // namespace

bool DecodeGraphRunner::try_capture(cudaStream_t stream,
                                    const std::function<void()>& body) noexcept {
    reset();
    attempted_ = true;
    try {
        definition_.capture(stream, body);
        executable_.instantiate(definition_);
        executable_.upload(stream);
    } catch (const std::exception& error) {
        diagnostic_ = error.what();
        executable_.reset();
        definition_.reset();
        recover_stream(stream);
        return false;
    } catch (...) {
        diagnostic_ = "unknown failure while capturing a decode graph";
        executable_.reset();
        definition_.reset();
        recover_stream(stream);
        return false;
    }
    ready_ = true;
    return true;
}

void DecodeGraphRunner::launch(cudaStream_t stream) {
    if (!ready_) { throw std::logic_error("sensenova_loop_9b: decode graph is not available"); }
    executable_.launch(stream);
}

void DecodeGraphRunner::reset() noexcept {
    executable_.reset();
    definition_.reset();
    ready_     = false;
    attempted_ = false;
    diagnostic_.clear();
}

} // namespace ninfer::targets::sensenova_loop_9b::detail

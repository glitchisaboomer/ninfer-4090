#pragma once

// Stage E6: one captured decode step for one lane.
//
// The 56-visit pass is entirely device work: the attention and router assets are prepared at
// Program creation, the state images and decode workspace never move, and the mHC leave reduction
// is a device Op rather than a host round-trip. The only per-token inputs are the token id and the
// position, which live at fixed addresses in the lane's staging slot. That makes the whole pass
// capturable once and replayable for every later decode step on that lane.
//
// This wrapper exists instead of using core::DecodeGraphDefinition directly because a failed
// capture must be a recorded diagnostic, not an exception: capture depends on the stream and on
// every launch in the body being capture-legal, and a decode step has to fall back to eager
// execution rather than fail the request. A failed capture is never silently retried per token.

#include "core/decode_graph.h"

#include <cuda_runtime.h>

#include <functional>
#include <string>

namespace ninfer::targets::sensenova_loop_9b::detail {

class DecodeGraphRunner {
public:
    DecodeGraphRunner() = default;
    ~DecodeGraphRunner() = default;

    DecodeGraphRunner(const DecodeGraphRunner&)            = delete;
    DecodeGraphRunner& operator=(const DecodeGraphRunner&) = delete;
    DecodeGraphRunner(DecodeGraphRunner&&) noexcept            = default;
    DecodeGraphRunner& operator=(DecodeGraphRunner&&) noexcept = default;

    // Captures `body` on `stream`, instantiates it, and uploads it. Returns true only when the
    // runner can launch. A failed attempt leaves the runner unavailable, records the reason in
    // `diagnostic()`, restores the stream, and returns false; the caller runs the body eagerly.
    [[nodiscard]] bool try_capture(cudaStream_t stream,
                                   const std::function<void()>& body) noexcept;

    [[nodiscard]] bool ready() const noexcept { return ready_; }
    [[nodiscard]] bool attempted() const noexcept { return attempted_; }
    [[nodiscard]] const std::string& diagnostic() const noexcept { return diagnostic_; }

    // Launches one replay. Throws std::logic_error when the runner is not ready: an unavailable
    // graph is a caller bug, whereas a capture failure is an expected fallback.
    void launch(cudaStream_t stream);

    void reset() noexcept;

private:
    DecodeGraphDefinition definition_;
    DecodeGraphExecutable executable_;
    bool ready_     = false;
    bool attempted_ = false;
    std::string diagnostic_;
};

} // namespace ninfer::targets::sensenova_loop_9b::detail

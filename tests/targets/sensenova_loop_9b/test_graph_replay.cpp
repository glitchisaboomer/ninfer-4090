// Stage E6: capture and replay mechanism for one SenseNova decode step.
//
// The cases here use real device nodes (device-to-device copies and memsets) so capture, replay,
// live-input reads, and the eager fallback are exercised on the actual runtime. They deliberately
// do not execute the model: whole-model token/state equality additionally needs the licensed
// artifact, and no fixture harness exists in this repository.

#include "targets/sensenova_loop_9b/impl/runtime/graph_impl.h"

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace sn = ninfer::targets::sensenova_loop_9b::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::vector<std::uint8_t> read_bytes(const ninfer::DeviceBuffer& buffer, std::size_t bytes) {
    std::vector<std::uint8_t> host(bytes);
    buffer.copy_to_host(host.data(), host.size());
    return host;
}

void test_capture_replay_reads_live_inputs(ninfer::DeviceContext& context) {
    constexpr std::size_t bytes = 4096;
    ninfer::DeviceBuffer source(bytes);
    ninfer::DeviceBuffer destination(bytes);
    const std::vector<std::uint8_t> first(bytes, 0x11);
    const std::vector<std::uint8_t> second(bytes, 0x22);
    source.copy_from_host(first.data(), first.size());

    sn::DecodeGraphRunner runner;
    expect(!runner.attempted() && !runner.ready(), "graph runner starts idle");

    const bool captured = runner.try_capture(context.stream, [&] {
        CUDA_CHECK(cudaMemcpyAsync(destination.p, source.p, bytes, cudaMemcpyDeviceToDevice,
                                   context.stream));
    });
    expect(captured, "capturing a real device copy succeeds");
    expect(runner.attempted() && runner.ready(), "a captured runner is ready");
    expect(runner.diagnostic().empty(), "a successful capture has no diagnostic");

    runner.launch(context.stream);
    context.synchronize();
    expect(read_bytes(destination, bytes) == first,
           "replay reproduces the captured device work");

    // The captured copy addresses stable pointers, but the data it reads is live: changing the
    // source outside the graph must be visible to the next replay. That is the contract the
    // per-token staging slot relies on.
    source.copy_from_host(second.data(), second.size());
    runner.launch(context.stream);
    context.synchronize();
    expect(read_bytes(destination, bytes) == second,
           "replay reads its inputs at execution time");
}

void test_replay_matches_eager(ninfer::DeviceContext& context) {
    constexpr std::size_t bytes = 2048;
    ninfer::DeviceBuffer eager(bytes);
    ninfer::DeviceBuffer replayed(bytes);

    CUDA_CHECK(cudaMemsetAsync(eager.p, 0x5C, bytes, context.stream));
    context.synchronize();

    sn::DecodeGraphRunner runner;
    const bool captured = runner.try_capture(context.stream, [&] {
        CUDA_CHECK(cudaMemsetAsync(replayed.p, 0x5C, bytes, context.stream));
    });
    expect(captured, "capturing a memset node succeeds");
    runner.launch(context.stream);
    context.synchronize();

    expect(read_bytes(eager, bytes) == read_bytes(replayed, bytes),
           "graph replay equals eager execution byte for byte");
}

void test_capture_failure_preserves_stream(ninfer::DeviceContext& context) {
    sn::DecodeGraphRunner runner;
    const bool captured = runner.try_capture(context.stream, [] {
        throw std::runtime_error("capture body failed");
    });
    expect(!captured, "a failing capture body is reported, not thrown");
    expect(runner.attempted() && !runner.ready(),
           "a failed capture leaves the runner unavailable");
    expect(!runner.diagnostic().empty(), "a failed capture records a diagnostic");

    bool launch_rejected = false;
    try {
        runner.launch(context.stream);
    } catch (const std::logic_error&) {
        launch_rejected = true;
    }
    expect(launch_rejected, "launching an unavailable graph is a caller error");

    // The failed attempt must not leave the stream in capture mode: a later capture on the same
    // runner and stream has to succeed, which is what lets a lane retry after a real failure.
    ninfer::DeviceBuffer destination(64);
    const bool recovered = runner.try_capture(context.stream, [&] {
        CUDA_CHECK(cudaMemsetAsync(destination.p, 0x7E, 64, context.stream));
    });
    expect(recovered && runner.ready(), "the stream is usable for a later capture");
    runner.launch(context.stream);
    context.synchronize();
    const std::vector<std::uint8_t> bytes = read_bytes(destination, 64);
    expect(std::all_of(bytes.begin(), bytes.end(),
                       [](std::uint8_t value) { return value == 0x7E; }),
           "the recovered graph runs");

    runner.reset();
    expect(!runner.attempted() && !runner.ready(), "reset clears the runner");
}

} // namespace

int main() {
    std::unique_ptr<ninfer::DeviceContext> context;
    try {
        context = std::make_unique<ninfer::DeviceContext>(0);
    } catch (const std::exception&) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    test_capture_replay_reads_live_inputs(*context);
    test_replay_matches_eager(*context);
    test_capture_failure_preserves_stream(*context);

    std::cout << (failures ? "FAIL" : "OK") << " sensenova decode graph replay\n";
    return failures ? 1 : 0;
}

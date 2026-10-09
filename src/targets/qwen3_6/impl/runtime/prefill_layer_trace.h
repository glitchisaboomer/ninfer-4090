#pragma once

// Per-layer residual-stream trace of one prefill chunk, for localizing the non-reproducible prefill
// described in docs/ISSUE_prefill-nondeterminism.md.
//
// The symptom: two cold prefills of one prompt through one Engine disagree at position 0 by
// 1.7e-4..5.1e-3 nats, varying per call and present at every prompt length. A token-count sweep cannot
// name the layer that diverged -- it changes the data and the schedule at the same time and reports one
// scalar -- so this hashes the residual tensor after every layer instead, and compares the hashes
// against the previous prefill. The first layer whose bytes differ had bit-identical inputs, so
// whatever diverged is inside that layer.
//
// The fingerprint and the trace's memory live in prefill_trace.h, which is device-free and unit-tested.
//
// Diagnostic only, and NINFER_PREFILL_TRACE gates it. Each captured layer copies its whole residual
// tile to the host and synchronises on it, so it is never on a request path; a 664-token chunk is
// ~6.8 MB per layer, ~325 MB per prefill. `NINFER_PREFILL_TRACE=2` also prints the layers that did not
// move.
//
// The residual tile is the prefill chunk's `x` tensor: [hidden, tokens] BF16, the same buffer every
// layer reads and writes, captured after each layer's MLP tail. Chunk identity is recovered from the
// chunk's first RoPE position, which is 0 only for the first chunk of a prompt that starts at position
// 0; a prefill that appends to an existing cache therefore resets the comparison at a chunk boundary it
// cannot see. Chunks are compared only against a previous chunk of the same token width and layer
// count, so anything that cannot be lined up prints `baseline=yes` instead of reporting every layer of
// one prompt as divergent from another.
//
// The comparison window is thread_local, and every Engine runs its requests on a worker thread of its
// own, so the window is per-Engine in practice and an Engine's *first* prefill is always a baseline.
// That makes the two-Engine case -- two cold materializations of one prompt in one process, which is
// the case docs/ISSUE_prefill-nondeterminism.md could not name a layer for -- invisible to the
// same-instance path. `NINFER_PREFILL_TRACE_CROSS=1` closes it: a completed chunk is published to a
// process-wide table keyed by shape, and an instance with no baseline of its own compares against the
// last chunk of that shape published by *any* instance. Every compared line then says
// `scope=this_instance` or
// `scope=other_instance`, so a reader never has to guess which comparison they are looking at.
//
// The cross lookup stays off by default because the table cannot know whether two same-width chunks are
// two runs of one prompt; two different prompts of one width would be reported as fully divergent. The
// test's two-Engine case deliberately prefills one prompt through both Engines, which is the guarantee
// the flag relies on.
//
// Declared in `detail` rather than in the variant's runtime namespace: nothing here is
// target-dependent, and keeping it out of the macro's namespace is what lets a host test include it.

#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen3_6/impl/runtime/prefill_trace.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

inline bool prefill_layer_trace_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_PREFILL_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

inline bool prefill_layer_trace_verbose() {
    static const bool verbose = [] {
        const char* value = std::getenv("NINFER_PREFILL_TRACE");
        return value != nullptr && value[0] == '2';
    }();
    return verbose;
}

// Cross-instance baselines: an Engine's first prefill compares against the last chunk of the same
// shape published by any Engine in the process. Separate from NINFER_PREFILL_TRACE because it is the
// only mode that can report divergence between two *different* prompts as if it were a repeat.
inline bool prefill_layer_trace_cross() {
    static const bool cross = [] {
        const char* value = std::getenv("NINFER_PREFILL_TRACE_CROSS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return cross;
}

namespace ninfer::targets::qwen3_6::detail {

// The tap TextContext::run_layers drives. Copyable and stateless; everything it remembers lives in the
// thread_local store, because a tap is constructed per prefill call.
//
// `run_layers` calls begin(), then capture_layer() once per layer, then capture_positions(), and
// consume_prefill_chunk() closes the chunk. The last is where the whole chunk is compared and printed,
// because that is the first point at which the chunk's position is known and nothing is still arriving.
struct PrefillLayerTrace {
    static constexpr bool enabled = true;

    void begin(const Tensor& value) {
        prefill_trace::Store& state = prefill_trace::store();
        state.pending.hash.clear();
        state.pending.sum_abs.clear();
        state.chunk_tokens = value.ne[1];
        if (value.dtype != DType::BF16 || value.data == nullptr) {
            prefill_trace::warn_once(state, "the residual tile is not BF16 device memory");
        }
    }

    void capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
        prefill_trace::Store& state = prefill_trace::store();
        if (layer < 0) { return; }
        if (value.dtype != DType::BF16 || value.data == nullptr || !value.is_contiguous()) {
            prefill_trace::warn_once(state, "the residual tile is not contiguous BF16 device memory");
            return;
        }

        const std::size_t bytes = value.bytes();
        state.scratch.resize(bytes);
        CUDA_CHECK(cudaMemcpyAsync(state.scratch.data(), value.data, bytes, cudaMemcpyDeviceToHost,
                                   stream));
        // Drain before computing: the copy is on the compute stream, and the layer that produced it
        // is not finished until this returns. Without it the hash could describe the previous layer's
        // residual, which is the exact confusion this diagnostic exists to remove.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const prefill_trace::TileFingerprint fingerprint =
            prefill_trace::fingerprint_bf16_tile(state.scratch.data(), bytes);

        const std::size_t slot = static_cast<std::size_t>(layer);
        if (state.pending.hash.size() <= slot) {
            state.pending.hash.resize(slot + 1, prefill_trace::kHashSeed);
            state.pending.sum_abs.resize(slot + 1, 0.0);
        }
        state.pending.hash[slot]    = fingerprint.hash;
        state.pending.sum_abs[slot] = fingerprint.sum_abs;
    }

    // Called once per chunk after the layer walk. Only its first element is read, and only to answer
    // "is this the first chunk of a prompt that starts at position 0".
    void capture_positions(const Tensor& source, cudaStream_t stream) {
        prefill_trace::Store& state = prefill_trace::store();
        state.starts_at_zero = false;
        if (source.dtype != DType::I32 || source.data == nullptr || source.numel() < 1) {
            prefill_trace::warn_once(state, "prefill positions are not I32");
            return;
        }
        std::int32_t first = -1;
        CUDA_CHECK(
            cudaMemcpyAsync(&first, source.data, sizeof(first), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        state.starts_at_zero = first == 0;
    }

    void consume_prefill_chunk(std::int32_t tokens, bool turn_checkpoint) {
        (void)tokens;
        (void)turn_checkpoint;
        prefill_trace::Store& state = prefill_trace::store();

        // A chunk that starts at position 0 begins a new prompt, so the comparison window rolls: what
        // was in progress becomes the baseline for this one.
        if (state.starts_at_zero) {
            state.previous = std::move(state.current);
            state.current.clear();
            state.chunk_ordinal = 0;
        }
        if (state.current.size() <= state.chunk_ordinal) {
            state.current.resize(static_cast<std::size_t>(state.chunk_ordinal) + 1);
        }
        state.current[state.chunk_ordinal]        = state.pending;
        state.current[state.chunk_ordinal].tokens = state.chunk_tokens;
        state.pending.hash.clear();
        state.pending.sum_abs.clear();

        const prefill_trace::ChunkTrace& trace = state.current[state.chunk_ordinal];
        // A comparison is declined, not approximated, when the candidate has a different width or
        // layer count: two different prompts are not two runs of one prompt.
        const bool local_baseline = state.chunk_ordinal >= state.previous.size() ||
                                    state.previous[state.chunk_ordinal].tokens != state.chunk_tokens ||
                                    state.previous[state.chunk_ordinal].hash.size() != trace.hash.size();
        const prefill_trace::ChunkTrace* before =
            local_baseline ? nullptr : &state.previous[state.chunk_ordinal];

        // Nothing to compare against in this instance's window: this is where a second Engine's first
        // prefill lands, and where the same-instance path has nothing to say. Ask the process-wide
        // table instead, and record whether the baseline came from this instance or another one -- the
        // two are different claims, so the line cannot leave it implicit.
        bool other_instance = false;
        if (before == nullptr && prefill_layer_trace_cross()) {
            std::uint64_t owner = 0;
            if (prefill_trace::registry_lookup(state.chunk_tokens, trace.hash.size(),
                                              state.cross_baseline, &owner)) {
                before         = &state.cross_baseline;
                other_instance = owner != state.instance;
            }
        }
        const bool baseline = before == nullptr;

        // Published whether or not a comparison happened: this chunk *is* the baseline the next
        // instance's first prefill is looking for. Done after the lookup so a chunk is never compared
        // against itself.
        prefill_trace::registry_publish(trace, state.instance);

        std::size_t divergent       = 0;
        std::size_t first_divergent = 0;
        bool have_first             = false;

        std::ostringstream rows;
        for (std::size_t layer = 0; layer < trace.hash.size(); ++layer) {
            // Out of range on either side means the two prefills did not walk the same layers, so the
            // layer is reported as differing rather than assumed equal.
            const bool same = before != nullptr && layer < before->hash.size() &&
                              before->hash[layer] == trace.hash[layer];
            if (before != nullptr && !same) {
                if (!have_first) {
                    have_first      = true;
                    first_divergent = layer;
                }
                ++divergent;
            }
            // In the default mode only the layers that moved are printed; the full table is one
            // `NINFER_PREFILL_TRACE=2` away. A baseline chunk has nothing to differ from, so it
            // contributes no rows and is announced by `baseline=yes` in the summary instead.
            if (!prefill_layer_trace_verbose() && (before == nullptr || same)) { continue; }
            rows << "[prefill-trace] chunk=" << (state.chunk_ordinal + 1) << " layer=" << layer
                 << " hash=0x" << std::hex << trace.hash[layer] << std::dec
                 << " sum_abs=" << trace.sum_abs[layer] << " prev=";
            if (before == nullptr || layer >= before->hash.size()) {
                rows << "n/a same=n/a\n";
            } else {
                rows << "0x" << std::hex << before->hash[layer] << std::dec
                     << " prev_sum_abs=" << before->sum_abs[layer]
                     << " same=" << (same ? "yes" : "no") << '\n';
            }
        }

        std::cerr << "[prefill-trace] chunk=" << (state.chunk_ordinal + 1)
                  << " tokens=" << state.chunk_tokens << " layers=" << trace.hash.size()
                  << " baseline=" << (baseline ? "yes" : "no")
                  << " scope="
                  << (baseline ? "none" : (other_instance ? "other_instance" : "this_instance"));
        if (baseline) {
            std::cerr << " divergent_layers=n/a first_divergent_layer=n/a";
        } else {
            std::cerr << " divergent_layers=" << divergent << " first_divergent_layer="
                      << (have_first ? std::to_string(first_divergent) : std::string("none"));
        }
        std::cerr << '\n' << rows.str();
        std::cerr.flush();

        ++state.chunk_ordinal;
    }
};

} // namespace ninfer::targets::qwen3_6::detail

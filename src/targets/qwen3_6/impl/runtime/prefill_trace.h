#pragma once

// Device-free state and fingerprints for the per-layer prefill residual trace
// (prefill_layer_trace.h). Kept free of both the variant namespace macro and the CUDA runtime so the
// fold and the trace bookkeeping can be compiled and driven without a device.
//
// Why a hash rather than a magnitude: sum|x| is an aggregate that two distinct states can share, so it
// can show that two traces differ but never that they agree, which is exactly the weakness
// docs/ISSUE_prefill-nondeterminism.md records against the `[scr-diag] gdn` totals. The fold answers
// both questions; the FP64 sum of |x| rides along only to say how large a difference is.
//
// Within one tile length the fold is injective in every input word: each step is the splitmix64
// finalizer applied to the previous state xored with the word, and that finalizer is a bijection. So a
// change to any byte reaches the final hash, and "the hashes differ" is exact rather than
// probabilistic. Across lengths there is no such claim, which is why the trace only ever compares two
// chunks of the same width.
//
// It is a change detector on one machine, not a portable checksum: words are read in memory order, so
// two machines with different endianness would disagree where two runs on one machine will not.

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::prefill_trace {

// SplitMix64's finalizer.
inline std::uint64_t mix64(std::uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

// FNV's 64-bit offset basis, so an all-zero tile does not hash to zero.
inline constexpr std::uint64_t kHashSeed = 0xcbf29ce484222325ULL;

struct TileFingerprint {
    std::uint64_t hash = kHashSeed;
    double sum_abs     = 0.0;
};

// BF16 is the top 16 bits of an FP32, so the magnitude sum widens by a shift rather than a decode,
// matching LinearAttentionStatePool::fingerprint_slot. The widening is exact, so the sum is exact in
// double for any tile whose values are individually exact -- which is what the unit test pins.
inline TileFingerprint fingerprint_bf16_tile(const std::uint8_t* bytes, std::size_t size) {
    TileFingerprint result;

    const std::size_t words = size / 8;
    for (std::size_t word_index = 0; word_index < words; ++word_index) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes + word_index * 8, sizeof(word));
        result.hash = mix64(result.hash ^ word);
    }
    // Tail, when the tile is not a whole number of 8-byte words. A [hidden, tokens] BF16 residual
    // always is, but a fold that silently dropped the last bytes would report agreement between two
    // tiles that differ only there.
    const std::size_t tail = words * 8;
    if (tail != size) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes + tail, size - tail);
        result.hash = mix64(result.hash ^ word);
    }

    const std::size_t elements = size / 2;
    for (std::size_t element = 0; element < elements; ++element) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes + element * 2, sizeof(bits));
        const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
        float value              = 0.0F;
        std::memcpy(&value, &wide, sizeof(value));
        // The magnitude, not a signed sum: two elements moving in opposite directions must not be
        // able to cancel, which is the same reason scr_quality.h scores divergence as a mean absolute
        // difference. fabs is also what makes a -0.0 word contribute zero rather than -0.0.
        result.sum_abs += std::fabs(static_cast<double>(value));
    }
    return result;
}

// One prefill chunk's per-layer fingerprints, indexed by global layer.
struct ChunkTrace {
    std::vector<std::uint64_t> hash;
    std::vector<double> sum_abs;
    // Chunk width, so a trace is only ever compared against another chunk of the same width. A
    // harness that prefills several prompts in a row would otherwise diff prompt B against prompt A
    // and report every layer as divergent, which is true but useless.
    std::int32_t tokens = 0;
};

// --- Cross-instance baseline registry ---------------------------------------
//
// The trace's memory is thread_local (store(), below), and every Engine serves its requests on a
// worker thread of its own (runtime/engine/concurrent_executor.h), so two Engines in one process never
// share a trace: each one's first prefill has nothing to compare against and prints `baseline=yes`.
// That is precisely the case docs/ISSUE_prefill-nondeterminism.md cannot name a layer for, so a
// completed chunk is also published to a process-wide table keyed by shape, and looked up there when
// this instance has no baseline of its own.
//
// Off unless NINFER_PREFILL_TRACE_CROSS is set, because the table cannot know whether two same-width
// chunks are two runs of one prompt: two *different* prompts of one width would be reported as fully
// divergent, which is the approximation the same-instance path deliberately declines.
//
// One slot per (tokens, layers): the most recent chunk of that shape wins. The chunks of one prefill
// carry different widths, so an ordinal's cross baseline is the previous instance's chunk of the same
// width.
struct PublishedChunk {
    std::int32_t tokens = 0;
    std::size_t layers  = 0;
    std::vector<std::uint64_t> hash;
    std::vector<double> sum_abs;
    // Identity of the Store that published it -- which Engine, i.e. which worker thread.
    std::uint64_t owner = 0;
};

// Identity for a Store. Not its address: a ThreadPool thread's TLS block is regularly handed straight
// to the next thread the OS creates, so two Engines built one after the other reuse one address and an
// address comparison would label a cross-Engine comparison `this_instance`. Monotonic and per process.
inline std::uint64_t next_instance_id() {
    static std::atomic<std::uint64_t> counter{0};
    return ++counter;
}

inline std::mutex& registry_mutex() {
    static std::mutex instance;
    return instance;
}

inline std::vector<PublishedChunk>& registry() {
    static std::vector<PublishedChunk> instance;
    return instance;
}

// Copy the published digests for `tokens` x `layers` into `out`; false when that shape has never been
// published. The lock covers a vector copy and nothing else -- it is never held across a device call,
// which is what keeps it a leaf.
inline bool registry_lookup(std::int32_t tokens, std::size_t layers, ChunkTrace& out,
                            std::uint64_t* owner) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (const PublishedChunk& entry : registry()) {
        if (entry.tokens != tokens || entry.layers != layers) { continue; }
        out.hash    = entry.hash;
        out.sum_abs = entry.sum_abs;
        out.tokens  = entry.tokens;
        if (owner != nullptr) { *owner = entry.owner; }
        return true;
    }
    return false;
}

inline void registry_publish(const ChunkTrace& trace, std::uint64_t owner) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (PublishedChunk& entry : registry()) {
        if (entry.tokens != trace.tokens || entry.layers != trace.hash.size()) { continue; }
        entry.hash    = trace.hash;
        entry.sum_abs = trace.sum_abs;
        entry.owner   = owner;
        return;
    }
    registry().push_back(PublishedChunk{trace.tokens, trace.hash.size(), trace.hash, trace.sum_abs,
                                        owner});
}

// The trace's memory between calls: the chunk being built, the prefill in progress, and the previous
// prefill. Held in a thread_local below rather than in the tap, which is constructed per call.
struct Store {
    // Identifies this window -- one Engine, in practice. See next_instance_id().
    std::uint64_t instance = next_instance_id();
    // Reused between layers so a capture does not allocate 6.8 MB per layer per chunk.
    std::vector<std::uint8_t> scratch;
    ChunkTrace pending;
    // Per chunk ordinal: the prefill in progress, and the last prefill that completed. Ordinals only
    // line up between two prefills that had the same chunking, which identical prompts do.
    std::vector<ChunkTrace> current;
    std::vector<ChunkTrace> previous;
    // Copied out of the registry, because the registry's slot for this shape is overwritten by this
    // very chunk once it is published.
    ChunkTrace cross_baseline;
    std::uint32_t chunk_ordinal = 0;
    std::int32_t chunk_tokens   = 0;
    bool starts_at_zero         = false;
    bool warned                 = false;
};

// Per thread, matching the function-local side-stream context the attention input projection already
// keeps, so two request threads cannot interleave their traces.
inline Store& store() {
    static thread_local Store instance;
    return instance;
}

// Warn once and carry on. A diagnostic that throws would take down the run it was switched on to
// explain, and a capture that cannot be taken is a wiring problem, not a data condition.
inline void warn_once(Store& state, const char* problem) {
    if (state.warned) { return; }
    state.warned = true;
    std::cerr << "[prefill-trace] layer capture disabled: " << problem << '\n';
    std::cerr.flush();
}

} // namespace ninfer::targets::qwen3_6::detail::prefill_trace

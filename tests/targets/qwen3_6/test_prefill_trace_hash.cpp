// Fingerprint checks for the per-layer prefill residual trace. Pure host code: no device, no Engine,
// no artifact, so this runs wherever the rest of the CTest suite runs.
//
// The trace decides "did this layer's output move" by comparing one 64-bit fold, so the fold is the
// load-bearing part and it gets an oracle. The expected values are declared in FP64 next to the BF16
// words they widen to rather than computed from the code under test, so an error in the widening
// cannot be confirmed by the same error in the expectation. The widening is exact (BF16 is the top
// half of an FP32), and the words are chosen so the sum is exact in double.
//
// The cross-instance registry is device-free too, so it is pinned here: shape-keyed lookup, the latest
// publish winning, that a looked-up chunk is a copy, and that an instance identity is not an address.
//
// What this does not cover: the tap that compares two chunks and declines a comparison between different
// widths. That needs device tensors and a prefill, so it is exercised by the engine test under
// NINFER_PREFILL_TRACE rather than here.

#include "targets/qwen3_6/impl/runtime/prefill_trace.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace trace = ninfer::targets::qwen3_6::detail::prefill_trace;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::vector<std::uint8_t> tile_bytes(const std::vector<std::uint16_t>& words) {
    std::vector<std::uint8_t> bytes(words.size() * sizeof(std::uint16_t));
    std::size_t offset = 0;
    for (const std::uint16_t word : words) {
        std::memcpy(bytes.data() + offset, &word, sizeof(word));
        offset += sizeof(word);
    }
    return bytes;
}

trace::TileFingerprint fingerprint(const std::vector<std::uint16_t>& words) {
    const std::vector<std::uint8_t> bytes = tile_bytes(words);
    return trace::fingerprint_bf16_tile(bytes.data(), bytes.size());
}

// 1.0, 2.0, 0.5, 1.5, -2.0 and -0.0 as their exact BF16 encodings. 0x3F80 is the top half of the
// FP32 1.0f, and so on down the list; each is checked below against the value it is declared with.
void test_widening_oracle() {
    const std::vector<std::uint16_t> words = {0x3F80, 0xC000, 0x3F00, 0x8000, 0x0000, 0x3FC0};
    // |1.0| + |-2.0| + |0.5| + |-0.0| + |0.0| + |1.5|
    const double expected = 1.0 + 2.0 + 0.5 + 0.0 + 0.0 + 1.5;
    expect(fingerprint(words).sum_abs == expected,
           "sum|x| must equal the declared FP64 sum of the widened words");
}

void test_magnitude_ignores_sign() {
    const trace::TileFingerprint negative = fingerprint({0x8000});
    expect(negative.sum_abs == 0.0, "a -0.0 word must contribute zero");
    // The magnitude is what makes opposing errors unable to cancel, which is the same reason
    // scr_quality.h scores divergence as a mean absolute difference rather than a signed mean. A
    // signed aggregate here would report a layer as unchanged when two of its elements moved in
    // opposite directions.
    expect(!std::signbit(negative.sum_abs), "the magnitude sum must never be negative");
}

void test_determinism() {
    const std::vector<std::uint16_t> words = {0x3F80, 0xC000, 0x3F00, 0x0000};
    const trace::TileFingerprint first     = fingerprint(words);
    const trace::TileFingerprint second    = fingerprint(words);
    expect(first.hash == second.hash, "the same bytes must hash the same");
    expect(first.sum_abs == second.sum_abs, "the same bytes must sum the same");
}

// The fold is a chain of bijections, so a change to any word must reach the final hash. Tested per
// position rather than once, because a fold that reads only the first word, or only the last, would
// pass a single-position test by accident.
void test_every_word_reaches_the_hash() {
    // Six words: one whole 8-byte word plus a 4-byte tail, so the tail fold is on the path too.
    const std::vector<std::uint16_t> even = {0x3F80, 0xC000, 0x3F00, 0x4000, 0x3FC0, 0x3E80};
    // Five words: the tail is a single 2-byte word.
    const std::vector<std::uint16_t> odd = {0x3F80, 0xC000, 0x3F00, 0x4000, 0x3FC0};

    for (const std::vector<std::uint16_t>* words : {&even, &odd}) {
        const std::uint64_t baseline = fingerprint(*words).hash;
        for (std::size_t index = 0; index < words->size(); ++index) {
            std::vector<std::uint16_t> moved = *words;
            moved[index]                     = 0x3F81;
            expect(fingerprint(moved).hash != baseline,
                   "changing word " + std::to_string(index) + " of " +
                       std::to_string(words->size()) + " must change the hash");
        }
    }
}

// One BF16 ULP in one element, with the arithmetic spelled out. 0x3F81 widens to 1.0078125: exponent
// 0 with one mantissa bit set, since the mantissa is 0x010000 of 2^23.
void test_single_ulp_change() {
    const trace::TileFingerprint before = fingerprint({0x3F80, 0xC000, 0x3F00, 0x8000});
    const trace::TileFingerprint after  = fingerprint({0x3F81, 0xC000, 0x3F00, 0x8000});
    expect(before.hash != after.hash, "a one-ULP change must change the hash");
    expect(before.sum_abs == 3.5, "baseline magnitude");
    expect(after.sum_abs == 1.0078125 + 2.0 + 0.5,
           "sum|x| must move by the widened value of the changed word");
}

void test_empty_tile() {
    const trace::TileFingerprint empty = trace::fingerprint_bf16_tile(nullptr, 0);
    expect(empty.hash == trace::kHashSeed, "an empty tile must hash to the seed");
    expect(empty.sum_abs == 0.0, "an empty tile must sum to zero");
}

trace::ChunkTrace make_chunk(std::int32_t tokens, const std::vector<std::uint64_t>& hash) {
    trace::ChunkTrace chunk;
    chunk.tokens = tokens;
    chunk.hash   = hash;
    chunk.sum_abs.assign(hash.size(), 1.0);
    return chunk;
}

// The registry is what lets one Engine's first prefill be compared against another Engine's, because
// two Engines never share the thread_local window the same-instance path uses. It is device-free, so
// its contract is pinned here rather than left to the engine test that needs a device.
void test_registry_round_trip() {
    const trace::ChunkTrace published = make_chunk(101, {0x11, 0x22, 0x33});
    trace::registry_publish(published, 7);

    trace::ChunkTrace found;
    std::uint64_t owner = 0;
    expect(trace::registry_lookup(101, 3, found, &owner), "a published shape must be found");
    expect(found.hash == published.hash, "the published digests must come back unchanged");
    expect(found.sum_abs == published.sum_abs, "the published magnitudes must come back unchanged");
    expect(found.tokens == 101, "the returned chunk must carry its width");
    expect(owner == 7, "the returned owner must name the publishing instance");

    // The copy is a copy: a caller that edits what it got back must not be editing the table.
    found.hash[0] = 0xBAD;
    trace::ChunkTrace again;
    expect(trace::registry_lookup(101, 3, again, nullptr), "the shape is still published");
    expect(again.hash[0] == 0x11, "mutating a looked-up chunk must not edit the registry");
}

// The key is the shape, so a width or a layer count that has never been published is a miss -- and a
// miss is what makes the caller print `baseline=yes` instead of comparing against the wrong chunk.
void test_registry_is_shape_keyed() {
    trace::registry_publish(make_chunk(202, {0x44, 0x55}), 1);

    trace::ChunkTrace found;
    expect(trace::registry_lookup(202, 2, found, nullptr), "the published shape is present");
    expect(!trace::registry_lookup(203, 2, found, nullptr), "a different width must miss");
    expect(!trace::registry_lookup(202, 3, found, nullptr), "a different layer count must miss");
    expect(trace::registry_lookup(202, 2, found, nullptr), "the hit must be repeatable");
}

// One slot per shape: the most recent chunk of that shape wins. A stale entry that survived a newer
// publish would make an Engine compare against a prefill that is two prefills old.
void test_registry_keeps_the_latest() {
    trace::registry_publish(make_chunk(303, {0x66, 0x77}), 1);
    trace::registry_publish(make_chunk(303, {0x88, 0x99}), 2);

    trace::ChunkTrace found;
    std::uint64_t owner = 0;
    expect(trace::registry_lookup(303, 2, found, &owner), "the shape is present");
    expect(found.hash[0] == 0x88 && found.hash[1] == 0x99, "the latest publish must win");
    expect(owner == 2, "the latest publisher must own the slot");
}

// Two Engines built one after the other are two windows, and the identity that says so must not be the
// window's address: an exited thread's TLS block is handed to the next thread, so consecutive Engines
// reuse the address and a real cross-Engine comparison would be labelled `this_instance`.
void test_instance_identity_is_not_an_address() {
    const std::uint64_t first  = trace::next_instance_id();
    const std::uint64_t second = trace::next_instance_id();
    expect(first != second, "two windows must not share an identity");
    expect(second > first, "identities must be handed out monotonically");
    expect(trace::Store{}.instance != trace::Store{}.instance,
           "two stores must be distinguishable by their instance id");
}

} // namespace

int main() {
    test_widening_oracle();
    test_magnitude_ignores_sign();
    test_determinism();
    test_every_word_reaches_the_hash();
    test_single_ulp_change();
    test_empty_tile();
    test_registry_round_trip();
    test_registry_is_shape_keyed();
    test_registry_keeps_the_latest();
    test_instance_identity_is_not_an_address();
    if (failures != 0) {
        std::cerr << failures << " prefill trace fingerprint checks failed\n";
        return 1;
    }
    std::cout << "prefill trace fingerprint checks passed\n";
    return 0;
}

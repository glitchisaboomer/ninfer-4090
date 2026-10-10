// Test for the rope-delta layout arithmetic: the shape and sizing of the per-logical-page delta plane
// that lets attention read a relocated page as if it had been prefilled at its new position, and the
// per-page delta rule that fills it.
//
// This is the half of docs/SCR_QUANTIZED_KV_RELOCATION.md Step 1 that needs no CUDA toolchain. The
// pool header that consumes these numbers includes <cuda_runtime_api.h>, so this arithmetic was pulled
// into a CUDA-free header specifically so it can be tested somewhere without nvcc -- and so that an
// off-by-one in the page rounding or a half-page span is caught here rather than on a GPU run where
// it presents as "relocation silently did nothing".
//
// The two properties that are easy to get wrong and silent when wrong:
//
//   * page rounding. A context that is not a whole number of pages still needs its last partial page
//     reserved, and 64 and 65*64 are where `ceil` and `page_count` disagree if you get it wrong.
//   * span boundaries. A page wholly inside the span takes the shift; a page half inside must take
//     none, because shifting half a page's keys moves them by a fractional position, which is not a
//     rotation at all.

#include "core/paged_kv_delta_layout.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>

namespace {

using ninfer::kPagedKVPageSize;
using ninfer::kRopeDeltaNone;
using ninfer::RopeDeltaLayout;
using ninfer::relocated_span_is_page_aligned;
using ninfer::resolve_rope_delta_layout;
using ninfer::rope_delta_for_span;
using ninfer::rope_delta_layout_is_supported;
using ninfer::rope_delta_logical_pages;
using ninfer::rope_delta_plane_bytes;

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("  FAIL: %s\n", what);
        ++g_failures;
    }
}

void test_page_rounding() {
    std::printf("logical page count rounds up, and the boundaries are where they should be\n");
    check(rope_delta_logical_pages(0) == 0, "no context needs no pages");
    check(rope_delta_logical_pages(-1) == 0, "an invalid context is rejected, not wrapped");
    check(rope_delta_logical_pages(1) == 1, "one token still needs one page");
    check(rope_delta_logical_pages(63) == 1, "63 tokens fit one page");
    check(rope_delta_logical_pages(64) == 1, "exactly one page is still one page");
    check(rope_delta_logical_pages(65) == 2, "one token past a page needs the next one");
    check(rope_delta_logical_pages(128) == 2, "two whole pages are two");
    check(rope_delta_logical_pages(8192) == 128, "8k context is 128 pages");
    check(rope_delta_logical_pages(65536) == 1024, "65k is 1024 pages");
    check(rope_delta_logical_pages(524288) == 8192, "500k is 8192 pages");

    // Off-by-one sweep: every length in [1, 320] must equal ceil(len/64) exactly, which is the whole
    // rule, applied without the author having written the answer down twice.
    for (std::int32_t len = 1; len <= 320; ++len) {
        const std::int32_t expected = (len + kPagedKVPageSize - 1) / kPagedKVPageSize;
        if (rope_delta_logical_pages(len) != expected) {
            std::printf("  FAIL: len=%d gave %d, expected %d\n", len, rope_delta_logical_pages(len),
                        expected);
            ++g_failures;
        }
    }
}

void test_plane_sizing() {
    std::printf("plane sizing matches the figures in the roadmap\n");
    check(rope_delta_plane_bytes(1, 128) == 512, "8k / concurrency 1 is 0.5 KiB");
    check(rope_delta_plane_bytes(4, 1024) == 16 * 1024, "65k / concurrency 4 is 16 KiB");
    check(rope_delta_plane_bytes(8, 4096) == 128 * 1024, "256k / concurrency 8 is 128 KiB");
    check(rope_delta_plane_bytes(8, 8192) == 256 * 1024, "500k / concurrency 8 is 256 KiB");

    // The plane must never silently shrink. A negative extent is a caller error and must not produce a
    // huge unsigned product that then passes a size check.
    check(rope_delta_plane_bytes(-1, 128) == 0, "a negative row count produces no plane");
    check(rope_delta_plane_bytes(4, -1) == 0, "a negative page count produces no plane");
}

void test_support_rules() {
    std::printf("support rules\n");
    check(rope_delta_layout_is_supported(1, 128, 1, 128), "a layout matching its block table is fine");
    check(rope_delta_layout_is_supported(8, 8192, 1, 128),
          "a plane larger than its block table is fine");
    check(!rope_delta_layout_is_supported(0, 128, 0, 128), "no rows is not supported");
    check(!rope_delta_layout_is_supported(1, 0, 1, 0), "no pages is not supported");
    check(!rope_delta_layout_is_supported(-1, 128, -1, 128), "a negative row count is not supported");
    // The failure mode this exists to catch: a delta plane that cannot cover the block table beside it
    // reads as "this page has no delta" forever, which is indistinguishable from "relocation did
    // nothing" rather than as an error.
    check(!rope_delta_layout_is_supported(1, 64, 1, 128),
          "a plane smaller than the block table is rejected");
    check(!rope_delta_layout_is_supported(4, 128, 8, 128),
          "fewer rows than the block table is rejected");
    // And the int32 bound, so a huge table cannot produce a plane whose entries are unreachable.
    check(!rope_delta_layout_is_supported(0x20000, 0x20000, 0x20000, 0x20000),
          "a plane that overflows int32 positions is rejected");

    const auto good = resolve_rope_delta_layout(8192, 4);
    check(good.logical_pages == 128 && good.table_rows == 4, "8k resolves to 128 pages and 4 rows");
    check(good.plane_bytes == 4 * 128 * 4, "and its plane is 2 KiB");
    const auto bad = resolve_rope_delta_layout(0, 4);
    check(bad.logical_pages == 0 && bad.table_rows == 0 && bad.plane_bytes == 0,
          "an unsupported context resolves to nothing rather than to a bogus layout");
}

void test_span_alignment() {
    std::printf("span alignment\n");
    check(relocated_span_is_page_aligned(0, 64), "a page-aligned head is aligned");
    check(relocated_span_is_page_aligned(448, 192), "the 3-block span from the quality suite is aligned");
    check(!relocated_span_is_page_aligned(1, 64), "a one-token offset is not aligned");
    check(!relocated_span_is_page_aligned(0, 63), "a partial trailing page is not aligned");
    check(!relocated_span_is_page_aligned(0, 65), "a page plus one token is not aligned");
    check(!relocated_span_is_page_aligned(0, 0), "an empty span is not aligned");
}

void test_per_page_delta() {
    std::printf("per-page delta: inside, outside, and the half case\n");

    // Every case below uses the quality suite's span shape: source [448, 640), three 64-token pages,
    // so logical pages 7, 8 and 9. The returned shift is `dest_begin - source_begin`, so a span moved
    // left is negative -- which is the reasoning-token-stripping shape, the survivor shifting back.

    // The zero-delta case from the quality suite: every page keeps its own shift.
    for (std::int32_t page = 7; page < 10; ++page) {
        check(rope_delta_for_span(448, 192, 448, page) == 0, "an unmoved span shifts nothing");
    }

    // Moved left by two blocks, so every page of the span carries -128.
    for (std::int32_t page = 7; page < 10; ++page) {
        check(rope_delta_for_span(448, 192, 320, page) == -128,
              "a left shift is negative, and the same for every page of the span");
    }

    // Moved right by one block.
    check(rope_delta_for_span(448, 192, 512, 7) == 64, "a right shift is positive");

    // Outside the span: no delta at all, which is what keeps the vast majority of pages on the
    // existing attention path.
    check(rope_delta_for_span(448, 192, 320, 0) == kRopeDeltaNone, "before the span there is none");
    check(rope_delta_for_span(448, 192, 320, 6) == kRopeDeltaNone, "the page just before has none");
    check(rope_delta_for_span(448, 192, 320, 10) == kRopeDeltaNone, "the page just after has none");
    check(rope_delta_for_span(448, 192, 320, 129) == kRopeDeltaNone, "a far page has none");
    check(rope_delta_for_span(448, 192, 320, -1) == kRopeDeltaNone, "a negative page index is rejected");

    // The boundaries, in tokens rather than pages. Page 7 covers [448, 512) and page 9
    // [576, 640); the span is [448, 640), so both ends are inclusive and the page outside each end
    // is not.
    check(rope_delta_for_span(448, 192, 320, 7) == -128, "the first page of the span is included");
    check(rope_delta_for_span(448, 192, 320, 9) == -128, "the last page of the span is included");
    check(rope_delta_for_span(448, 192, 320, 6) == kRopeDeltaNone, "the page below it is not");
    check(rope_delta_for_span(448, 192, 320, 10) == kRopeDeltaNone, "the page above it is not");

    // A span that does not fill a page. Shifting it would move half a page's keys, which is not a
    // rotation, so the rule must return none rather than a partial shift.
    check(rope_delta_for_span(448 + 32, 128, 320, 7) == kRopeDeltaNone,
          "a page only half inside the span takes no delta");
    check(rope_delta_for_span(448, 128, 320, 9) == kRopeDeltaNone,
          "a page whose tail falls past the span takes no delta");

    // The int32 bound: a shift beyond the largest representable position must be rejected, not
    // wrapped into a small negative number that would quietly rotate the wrong way.
    check(rope_delta_for_span(0, 64, 0x80000000u, 0) == kRopeDeltaNone,
          "a shift past int32 is rejected, not wrapped");
    check(rope_delta_for_span(0, 64, 0x7fffffffu, 0) == 0x7fffffff,
          "the largest representable shift is accepted");
}

// `plan_paged_kv_pool` sizes the delta plane by reconstructing a context length from the pool's
// logical page count and asking this header for the layout. That round trip has to be exact: if
// logical_pages * 64 does not resolve back to logical_pages, the plane's extents silently stop
// matching the block table it parallels, and every delta read then fetches the wrong page's shift.
// Pinning it here keeps the planner's derivation honest without needing a CUDA build.
//
// Powers of two alone would not show much: they are the values where a rounding rule that is wrong
// only on a remainder still agrees. Odd and prime page counts are where ceil and floor part company.
void test_implied_context_round_trip() {
    std::printf("implied-context round trip matches the block table it parallels\n");
    const std::int32_t page_counts[] = {1,   2,   3,   5,   7,   8,   13,  31,   63,
                                        64,  65,  127, 128, 255, 1023, 1025, 4095, 8192};
    for (const std::int32_t pages : page_counts) {
        const std::int32_t implied = static_cast<std::int32_t>(
            static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(kPagedKVPageSize));
        for (const std::int32_t rows : {std::int32_t{1}, std::int32_t{4}, std::int32_t{8}}) {
            const RopeDeltaLayout rope = resolve_rope_delta_layout(implied, rows);
            if (rope.logical_pages != pages || rope.table_rows != rows) {
                std::printf("  FAIL: pages=%d rows=%d resolved to %d/%d\n", pages, rows,
                            rope.logical_pages, rope.table_rows);
                ++g_failures;
            }
        }
    }
    // A capacity that is not a whole number of pages must still resolve, because the pool reserves
    // the page the partial tail lives in.
    check(resolve_rope_delta_layout(65 * kPagedKVPageSize, 4).logical_pages == 65,
          "a context one token past a page boundary still resolves");
    // And an empty pool resolves to nothing rather than to a one-page plane. plan_paged_kv_pool
    // reads this as "unsupported" and throws, which is what makes the single equality check there
    // cover both an unsupported layout and a mismatched one.
    check(resolve_rope_delta_layout(0, 4).logical_pages == 0, "no context resolves to no plane");
    // The top of the int32 range: the rounding must stay defined rather than wrap, and the round trip
    // must still close. `max_context + 63` overflows signed at this value, which is why the widening
    // in rope_delta_logical_pages is there.
    check(rope_delta_logical_pages(std::numeric_limits<std::int32_t>::max()) == 33554432,
          "the largest int32 context rounds without wrapping");
    check(resolve_rope_delta_layout(std::numeric_limits<std::int32_t>::max(), 8).logical_pages ==
              33554432,
          "the round trip closes at the top of the int32 range");
}

} // namespace

int main() {
    std::printf("rope-delta layout arithmetic\n");

    // Compile-time pins, so a change to the constants fails the build rather than the run.
    static_assert(ninfer::rope_delta_logical_pages(64) == 1);
    static_assert(ninfer::rope_delta_logical_pages(65) == 2);
    static_assert(ninfer::rope_delta_plane_bytes(4, 1024) == 16 * 1024);
    static_assert(ninfer::rope_delta_for_span(448, 192, 320, 7) == -128);
    static_assert(ninfer::rope_delta_for_span(448, 192, 320, 6) == 0);
    static_assert(ninfer::relocated_span_is_page_aligned(448, 192));

    test_page_rounding();
    test_plane_sizing();
    test_support_rules();
    test_span_alignment();
    test_per_page_delta();
    test_implied_context_round_trip();

    if (g_failures != 0) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

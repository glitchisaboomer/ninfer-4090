// Test for the rope-delta layout arithmetic: the shape and sizing of the per-logical-page delta plane
// that lets attention read a relocated page as if it had been prefilled at its new position, and the
// per-page delta rule that fills it.
//
// This is the half of docs/SCR.md Step 1 that needs no CUDA toolchain. The
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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::kPagedKVPageSize;
using ninfer::kRopeDeltaNone;
using ninfer::RopeDeltaLayout;
using ninfer::relocated_span_is_page_aligned;
using ninfer::RopeDeltaRowSpan;
using ninfer::rope_delta_row_span_i32;
using ninfer::resolve_rope_delta_layout;
using ninfer::rope_delta_for_span;
using ninfer::rope_delta_layout_is_supported;
using ninfer::rope_delta_logical_pages;
using ninfer::rope_delta_plane_bytes;
using ninfer::kPagedKVAquirePage;
using ninfer::plan_relocated_page_moves;
using ninfer::plan_relocated_page_substitution;
using ninfer::RelocatedCopySpan;
using ninfer::RelocatedPageMove;
using ninfer::RelocatedPageSubstitution;
using ninfer::RopeDeltaPageRange;
using ninfer::plan_rope_delta_page_ranges;
using ninfer::encode_rope_delta_page_ranges;
using ninfer::pack_rope_delta_page_range;
using ninfer::rope_delta_page_range_begin;
using ninfer::rope_delta_page_range_end;
using ninfer::rope_delta_page_range_is_empty;
using ninfer::rope_delta_page_range_is_partial;
using ninfer::kRopeDeltaRangeEmpty;
using ninfer::kRopeDeltaRangeMaxOffset;

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
    // the page the partial tail lives in. `65 * 64` would be exactly 65 whole pages and would not
    // exercise the ceiling at all; one token more does.
    check(resolve_rope_delta_layout(64 * kPagedKVPageSize + 1, 4).logical_pages == 65,
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

// `PagedKVPool::clear_rope_delta_row` zeroes one table row with a single contiguous memset, which is
// only safe if the plane's rows are byte-disjoint and together cover it exactly. That property is
// what a reviewer got wrong once -- they assumed the usual row-major convention, with dim 0
// outermost, under which a dim-1 slice is strided and the memset would zero part of a neighbouring
// row. This asserts the actual property rather than restating the convention, so the next reader has
// a test to run instead of a claim to trust.
//
// Disjointness alone is not enough: rows that were disjoint but left gaps would mean some entry is
// never cleared, which is the same bug arriving from the other side. Covering is asserted too.
void test_row_span_tiles_the_plane() {
    std::printf("row spans are disjoint and jointly tile the delta plane\n");
    const struct {
        std::int32_t pages;
        std::int32_t rows;
    } shapes[] = {
        {1, 1},  {10, 2},   {64, 1},   {64, 8},   {128, 4},
        {1024, 4}, {255, 3}, {4096, 8}, {8192, 8}, {65, 8},
    };
    for (const auto& shape : shapes) {
        const std::size_t total_bytes =
            static_cast<std::size_t>(shape.pages) * shape.rows * sizeof(std::int32_t);
        std::vector<char> covered(total_bytes / sizeof(std::int32_t), 0);
        for (std::int32_t row = 0; row < shape.rows; ++row) {
            const RopeDeltaRowSpan span =
                rope_delta_row_span_i32(shape.pages, shape.rows, row);
            if (span.bytes != static_cast<std::size_t>(shape.pages) * sizeof(std::int32_t)) {
                std::printf("  FAIL: pages=%d rows=%d row=%d has %zu bytes\n", shape.pages,
                            shape.rows, row, span.bytes);
                ++g_failures;
                continue;
            }
            if (span.offset % sizeof(std::int32_t) != 0) {
                std::printf("  FAIL: pages=%d rows=%d row=%d starts at %zu, not element-aligned\n",
                            shape.pages, shape.rows, row, span.offset);
                ++g_failures;
            }
            const std::size_t first = span.offset / sizeof(std::int32_t);
            const std::size_t last  = first + shape.pages;
            if (last > covered.size()) {
                std::printf("  FAIL: pages=%d rows=%d row=%d overruns the plane\n", shape.pages,
                            shape.rows, row);
                ++g_failures;
                continue;
            }
            for (std::size_t i = first; i < last; ++i) {
                if (covered[i]) {
                    std::printf("  FAIL: pages=%d rows=%d row=%d overlaps element %zu\n", shape.pages,
                                shape.rows, row, i);
                    ++g_failures;
                    break;
                }
                covered[i] = true;
            }
        }
        for (std::size_t i = 0; i < covered.size(); ++i) {
            if (!covered[i]) {
                std::printf("  FAIL: pages=%d rows=%d leaves element %zu uncovered\n", shape.pages,
                            shape.rows, i);
                ++g_failures;
                break;
            }
        }
    }

    // Out-of-range rows resolve to an empty span, which the caller turns into a throw. An empty span
    // is not a valid byte length to memset, so "bytes == 0" has to be distinguishable from a real
    // one -- and it is, because the plane always has at least one page per row.
    check(rope_delta_row_span_i32(128, 4, 4).bytes == 0, "one past the last row is empty");
    check(rope_delta_row_span_i32(128, 4, -1).bytes == 0, "a negative row is empty");
    check(rope_delta_row_span_i32(128, 0, 0).bytes == 0, "no rows is empty");
    check(rope_delta_row_span_i32(0, 4, 0).bytes == 0, "no pages is empty");
    // The exact figures, so a change to the stride or the element size is a test failure rather
    // than something to reason about.
    check(rope_delta_row_span_i32(10, 2, 0).offset == 0 && rope_delta_row_span_i32(10, 2, 0).bytes == 40,
          "row 0 of a {10,2} plane starts at 0 and is 40 bytes");
    check(rope_delta_row_span_i32(10, 2, 1).offset == 40 && rope_delta_row_span_i32(10, 2, 1).bytes == 40,
          "row 1 of a {10,2} plane starts at byte 40");
    check(rope_delta_row_span_i32(1024, 8, 5).offset == 5 * 4096,
          "row 5 of a {1024,8} plane starts at byte 20480");
    // Pins that the helper stays a constant expression, so a change that makes it non-constexpr
    // fails here rather than at whichever call site next needs it in a constant context.
    static_assert(rope_delta_row_span_i32(10, 2, 1).offset == 40);
}

// A copied relocation moves a run whose *source* need not be page aligned into a destination that is.
// Two things about the decomposition can be wrong in ways a GPU run would report as "the relocated
// pages hold the wrong tokens" long after the fact: the number of moves per destination page, and the
// split of a destination page across the source page boundary. Both are checked here by rebuilding the
// source token each destination token came from and comparing it to the arithmetic that defines the
// move, over every offset instead of a hand-picked pair.
void test_copied_span_moves() {
    std::printf("copied spans decompose into whole destination pages, split at source page ends\n");

    const auto moves_of = [](std::uint32_t source, std::uint32_t dest, std::uint32_t tokens,
                             std::size_t pages) {
        const RelocatedCopySpan span{source, dest, tokens};
        return plan_relocated_page_moves(std::span<const RelocatedCopySpan>(&span, 1), pages);
    };

    // An aligned source is one move per destination page: the 64 tokens fit inside one source page.
    {
        const std::vector<RelocatedPageMove> moves = moves_of(448, 320, 192, 16);
        check(moves.size() == 3, "three aligned destination pages are three moves");
        for (std::size_t index = 0; index < moves.size() && index < 3; ++index) {
            check(moves[index].dest_page == static_cast<std::int32_t>(5 + index),
                  "the aligned destination pages ascend from the first");
            check(moves[index].source_page == static_cast<std::int32_t>(7 + index),
                  "the aligned source pages ascend with them");
            check(moves[index].dest_token == 0 && moves[index].source_token == 0,
                  "an aligned move starts at both page origins");
            check(moves[index].tokens == 64, "an aligned move carries a whole page");
        }
    }

    // Off the grid: the source starts 17 tokens into its page, so every destination page is split --
    // the head of it in one source page and the tail in the next.
    {
        const std::vector<RelocatedPageMove> moves = moves_of(465, 320, 128, 16);
        check(moves.size() == 4, "two off-grid destination pages are four moves");
        if (moves.size() == 4) {
            check(moves[0].dest_page == 5 && moves[0].dest_token == 0 && moves[0].tokens == 47,
                  "the first destination page opens with the 47 tokens left in its source page");
            check(moves[1].dest_page == 5 && moves[1].dest_token == 47 && moves[1].tokens == 17,
                  "and closes with the 17 from the next source page");
            check(moves[2].dest_page == 6 && moves[2].dest_token == 0 && moves[2].tokens == 47,
                  "the second destination page is split the same way");
            check(moves[3].dest_page == 6 && moves[3].dest_token == 47 && moves[3].tokens == 17,
                  "including its tail");
            check(moves[1].source_page == moves[0].source_page + 1 && moves[1].source_token == 0,
                  "the tail comes from the start of the next source page");
        }
    }

    // The property, over every source offset in a page and lengths from one page to four: each
    // destination token in the run is carried exactly once, from the source token that the run itself
    // says it came from. Nothing here re-states the decomposition; it rebuilds the run from the moves
    // and compares against `source_begin + (dest - dest_begin)`.
    for (std::uint32_t offset = 0; offset < kPagedKVPageSize; ++offset) {
        for (std::uint32_t tokens : {1u, 17u, 64u, 128u, 192u, 256u}) {
            for (std::uint32_t dest_begin : {0u, 1u, 31u, 63u, 64u, 704u}) {
                const std::uint32_t source_begin = 448u + offset;
                const std::uint32_t dest_end     = dest_begin + tokens;
                const std::vector<RelocatedPageMove> moves =
                    moves_of(source_begin, dest_begin, tokens, 32);

                std::vector<bool> carried(dest_end, false);
                for (const RelocatedPageMove& move : moves) {
                    for (std::uint32_t i = 0; i < move.tokens; ++i) {
                        const std::uint32_t dest_token = move.dest_token + i;
                        const std::uint32_t source_token = move.source_token + i;
                        const std::size_t dest_position =
                            static_cast<std::size_t>(move.dest_page) *
                                static_cast<std::size_t>(kPagedKVPageSize) +
                            dest_token;
                        const std::size_t source_position =
                            static_cast<std::size_t>(move.source_page) *
                                static_cast<std::size_t>(kPagedKVPageSize) +
                            source_token;
                        if (dest_position < dest_begin || dest_position >= dest_end) {
                            std::printf("  FAIL: offset=%u tokens=%u dest_begin=%u moves outside the "
                                        "destination run (%zu)\n",
                                        offset, tokens, dest_begin, dest_position);
                            ++g_failures;
                            continue;
                        }
                        if (source_position != source_begin + (dest_position - dest_begin)) {
                            std::printf("  FAIL: offset=%u tokens=%u dest_begin=%u dest token %zu "
                                        "came from %zu, not %zu\n",
                                        offset, tokens, dest_begin, dest_position, source_position,
                                        source_begin + (dest_position - dest_begin));
                            ++g_failures;
                        }
                        if (carried[dest_position]) {
                            std::printf("  FAIL: offset=%u tokens=%u dest_begin=%u dest token %zu "
                                        "carried twice\n",
                                        offset, tokens, dest_begin, dest_position);
                            ++g_failures;
                        }
                        carried[dest_position] = true;
                    }
                }
                for (std::uint32_t position = dest_begin; position < dest_end; ++position) {
                    if (!carried[position]) {
                        std::printf("  FAIL: offset=%u tokens=%u dest_begin=%u dest token %u "
                                    "carried by no move\n",
                                    offset, tokens, dest_begin, position);
                        ++g_failures;
                        break;
                    }
                }
            }
        }
    }

    // Rejections. Destination boundaries are now token-granular; only empty or out-of-mapping spans
    // are invalid.
    const auto rejected = [&](const char* what, std::uint32_t source, std::uint32_t dest,
                              std::uint32_t tokens, std::size_t pages) {
        const RelocatedCopySpan span{source, dest, tokens};
        try {
            (void)plan_relocated_page_moves(std::span<const RelocatedCopySpan>(&span, 1), pages);
            std::printf("  FAIL: accepted %s\n", what);
            ++g_failures;
        } catch (const std::invalid_argument&) {
        }
    };
    rejected("an empty run", 448, 0, 0, 16);
    rejected("a run past the destination mapping", 448, 1024, 64, 16);
    rejected("a source past the current mapping", 1024, 0, 64, 16);
    rejected("a source that runs off the end of the mapping", 960, 0, 128, 16);

    {
        const std::vector<RelocatedPageMove> moves = moves_of(448, 1, 128, 16);
        std::vector<bool> written(16 * kPagedKVPageSize, false);
        for (const RelocatedPageMove& move : moves) {
            for (std::uint32_t i = 0; i < move.tokens; ++i) {
                const std::uint32_t destination =
                    static_cast<std::uint32_t>(move.dest_page) * kPagedKVPageSize +
                    move.dest_token + i;
                const std::uint32_t source =
                    static_cast<std::uint32_t>(move.source_page) * kPagedKVPageSize +
                    move.source_token + i;
                check(source == 448 + destination - 1,
                      "destination edge copy preserves the source token correspondence");
                check(!written[destination], "destination edge token is written only once");
                written[destination] = true;
            }
        }
        for (std::uint32_t destination = 1; destination < 129; ++destination) {
            check(written[destination], "every token in an unaligned destination span is copied");
        }
    }

    // A relocation of nothing is not a relocation: the caller has to learn that rather than publish an
    // empty move list as success.
    try {
        (void)plan_relocated_page_moves(std::span<const RelocatedCopySpan>(), 16);
        std::printf("  FAIL: accepted a relocation with no spans\n");
        ++g_failures;
    } catch (const std::invalid_argument&) {
    }
}

void test_rope_delta_page_ranges() {
    std::printf("RoPE delta ranges represent relocated sub-page intervals\n");
    const std::uint32_t packed_range = pack_rope_delta_page_range(7, 53);
    check(!rope_delta_page_range_is_empty(packed_range), "a valid interval has a nonempty encoding");
    check(rope_delta_page_range_begin(packed_range) == 7 &&
          rope_delta_page_range_end(packed_range) == 53,
        "packed page offsets round-trip exactly");
    check(rope_delta_page_range_is_empty(kRopeDeltaRangeEmpty),
        "the empty page-range sentinel is recognized");
    check(pack_rope_delta_page_range(0, 64) == (64u << 6),
        "a full page uses a valid packed interval");
    check(rope_delta_page_range_is_empty(pack_rope_delta_page_range(10, 10)),
        "an empty interval is rejected by the packer");

    // Every representable interval, not a sample. begin and end share one uint32, so the encoding
    // is only correct if the two half-widths never overlap: any pair that decodes to something
    // other than what it encoded is a bug the sampled cases above can miss, and the ones that
    // miss it are the boundary values at the ends of a mixed page.
    {
        std::size_t mismatches   = 0;
        std::size_t rejected     = 0;
        for (std::uint32_t begin = 0; begin < kRopeDeltaRangeMaxOffset; ++begin) {
            for (std::uint32_t end = begin + 1; end <= kRopeDeltaRangeMaxOffset; ++end) {
                const std::uint32_t packed = pack_rope_delta_page_range(begin, end);
                if (rope_delta_page_range_is_empty(packed)) { ++rejected; continue; }
                if (rope_delta_page_range_begin(packed) != begin ||
                    rope_delta_page_range_end(packed) != end) {
                    if (mismatches < 4) {
                        std::printf("    FAIL: (%u,%u) decodes as (%u,%u)\n", begin, end,
                                    rope_delta_page_range_begin(packed),
                                    rope_delta_page_range_end(packed));
                    }
                    ++mismatches;
                }
            }
        }
        check(rejected == 0,
              "every valid interval encodes, and none is silently turned into the sentinel");
        check(mismatches == 0,
              "every representable offset pair round-trips exactly");
        // A pair the packer must refuse, so the caller learns it rather than reading a wrong
        // interval back out. `end <= begin` is empty or inverted; `end > begin` below the page size is
        // valid and encoded.
        for (std::uint32_t begin = 0; begin <= kRopeDeltaRangeMaxOffset; ++begin) {
            check(rope_delta_page_range_is_empty(pack_rope_delta_page_range(begin, begin)),
                  "a zero-width interval decodes as empty");
            if (begin > 0) {
                check(rope_delta_page_range_is_empty(pack_rope_delta_page_range(begin, begin - 1)),
                      "an inverted interval decodes as empty");
            }
        }
        check(rope_delta_page_range_is_empty(
                  pack_rope_delta_page_range(0, kRopeDeltaRangeMaxOffset + 1)),
              "an interval past the end of a page decodes as empty");
    }

    // The boundary page shapes the kernels actually branch on. A page whose interval is the whole
    // page takes the whole-page delta route; anything narrower must be recognized as partial, or
    // the boundary score path never runs for it.
    {
        check(rope_delta_page_range_is_partial(pack_rope_delta_page_range(0, 1)),
              "a one-token interval at the start of a page is partial");
        check(rope_delta_page_range_is_partial(pack_rope_delta_page_range(63, 64)),
              "a one-token interval at the end of a page is partial");
        check(rope_delta_page_range_is_partial(pack_rope_delta_page_range(7, 8)),
              "a single interior token is partial");
        check(!rope_delta_page_range_is_partial(pack_rope_delta_page_range(0, 64)),
              "a whole-page interval is not partial");
        check(!rope_delta_page_range_is_partial(kRopeDeltaRangeEmpty),
              "the empty sentinel is not partial");
        check(rope_delta_page_range_begin(kRopeDeltaRangeEmpty) == 63 &&
                  rope_delta_page_range_end(kRopeDeltaRangeEmpty) == 127,
              "the sentinel decodes as a nonsensical interval, which is why callers must test "
              "emptiness first");
    }
    const std::vector<RelocatedCopySpan> spans{{465, 321, 100}};
    const std::vector<RopeDeltaPageRange> ranges = plan_rope_delta_page_ranges(spans, 16);
    check(ranges.size() == 16, "the range table parallels logical pages");
    check(ranges[5].begin == 1 && ranges[5].end == 64 && ranges[5].delta == -144,
          "the first boundary page records its relocated suffix");
    check(ranges[6].begin == 0 && ranges[6].end == 37 && ranges[6].delta == -144,
          "the last boundary page records its relocated prefix");
    check(ranges[4].begin == 0 && ranges[4].end == 0 && ranges[4].delta == 0,
          "untouched pages have an empty zero-delta interval");

    const std::vector<RelocatedCopySpan> whole_page{{448, 320, 128}};
    const std::vector<RopeDeltaPageRange> whole_ranges =
        plan_rope_delta_page_ranges(whole_page, 16);
    check(whole_ranges[5].begin == 0 && whole_ranges[5].end == 64 &&
              whole_ranges[5].delta == -128,
          "a fully relocated page uses the same interval representation");
    check(whole_ranges[6].begin == 0 && whole_ranges[6].end == 64 &&
              whole_ranges[6].delta == -128,
          "every full interior page has the span shift");

    const std::vector<RelocatedCopySpan> overlapping{{448, 321, 20}, {512, 350, 12}};
    try {
        (void)plan_rope_delta_page_ranges(overlapping, 16);
        check(false, "two relocation spans sharing a page are rejected");
    } catch (const std::invalid_argument&) {
    }

    // The two planes are published together and neither alone means anything, so they have to
    // name the same move. The runtime derives the scalar delta plane from this same table and
    // ignores a page whose interval is empty. A bug that split the two derivations would leave a
    // page with a range but a stale delta -- which no runtime check catches, because the kernel
    // reads the range to pick the score path and the delta to pick the angle, and only disagrees
    // with itself on a mixed page, producing plausible wrong attention.
    {
        // Destination pages 5, 10, 11, 12: 5/10/11 wholly relocated, 12 crossed by the tail of
        // the second span. Chosen so the two planes can be compared on whole and partial pages
        // alike, and so no two spans share a destination page, which the planner refuses.
        const std::vector<RelocatedCopySpan> spans{{448, 320, 64}, {896, 640, 130}};
        const std::size_t pages = 16;
        const std::vector<RopeDeltaPageRange> ranges = plan_rope_delta_page_ranges(spans, pages);
        const std::vector<std::uint32_t> packed =
            encode_rope_delta_page_ranges(spans, pages);

        std::size_t relocated_pages = 0;
        for (std::size_t page = 0; page < pages; ++page) {
            const RopeDeltaPageRange& range = ranges[page];
            const bool has_interval = range.begin != range.end;

            // The scalar plane, exactly as ProgramImplCore builds it.
            const std::int32_t scalar_delta = has_interval ? range.delta : 0;

            const std::uint32_t this_packed = packed[page];

            check(rope_delta_page_range_is_empty(this_packed) == !has_interval,
                  "the packed plane is empty exactly where the table has no interval");
            check(has_interval || scalar_delta == 0,
                  "a page with no interval keeps the ordinary-page delta");
            if (!has_interval) { continue; }

            ++relocated_pages;
            check(rope_delta_page_range_begin(this_packed) == range.begin &&
                      rope_delta_page_range_end(this_packed) == range.end,
                  "the packed interval decodes back to the table's interval");
            check(scalar_delta == range.delta,
                  "the scalar delta equals the range table's shift wherever a range exists");
            check(rope_delta_page_range_is_partial(this_packed) ==
                      (range.begin != 0 || range.end != kPagedKVPageSize),
                  "the partial predicate matches the table's interval shape");
        }
        check(relocated_pages == 4,
              "the scenario relocated four pages: three whole and one crossed");
        check(!rope_delta_page_range_is_partial(packed[5]) &&
                  !rope_delta_page_range_is_partial(packed[10]) &&
                  !rope_delta_page_range_is_partial(packed[11]) &&
                  rope_delta_page_range_is_partial(packed[12]),
              "only the page the span's tail crosses reads as partial");
    }
}

// The ownership half: a copied relocation must never hand a page it is reading straight back as the
// destination it is writing, and must not drop a page before the copy that reads it has run. The first
// is an overlapping device copy, whose behaviour the CUDA runtime does not define; the second hands a
// page to another lane's prefill while it is still being read.
void test_relocated_page_substitution() {
    std::printf("a copied relocation never reuses a page it is reading\n");

    // Two pages dropped, both read: neither can be a destination, so both come from the pool and both
    // are held until the copies have run.
    {
        const std::vector<std::int32_t> current{10, 11, 12, 13};
        const std::vector<std::int32_t> mapping{10, 11, kPagedKVAquirePage, kPagedKVAquirePage};
        const std::vector<std::int32_t> protected_pages{12, 13};
        const RelocatedPageSubstitution out =
            plan_relocated_page_substitution(current, mapping, protected_pages);
        check(out.final_ids.size() == 4 && out.final_ids[0] == 10 && out.final_ids[1] == 11,
              "the pages the mapping keeps are kept");
        check(out.final_ids[2] == kPagedKVAquirePage && out.final_ids[3] == kPagedKVAquirePage,
              "a page the copies read is not proposed as a destination");
        check(out.from_pool == 2, "both destinations have to come from the pool");
        check(out.held_for_copy.size() == 2, "both sources are held until the copies have run");
    }

    // The same shape with only one page read: the other dropped page is free to be a destination, and
    // it is exactly one, so the pool is asked for one page rather than two.
    {
        const std::vector<std::int32_t> current{10, 11, 12, 13};
        const std::vector<std::int32_t> mapping{10, 11, kPagedKVAquirePage, kPagedKVAquirePage};
        const std::vector<std::int32_t> protected_pages{12};
        const RelocatedPageSubstitution out =
            plan_relocated_page_substitution(current, mapping, protected_pages);
        check(out.from_pool == 1, "one unread dropped page pays for one of the two destinations");
        check(out.final_ids[2] == 13, "the unread dropped page becomes a destination");
        check(out.final_ids[3] == kPagedKVAquirePage, "the remaining one comes from the pool");
        check(out.held_for_copy.size() == 1 && out.held_for_copy[0] == 12,
              "and only the page that is read is held");
    }

    // Nothing read and exactly as many dropped pages as destinations: the pool is never asked.
    {
        const std::vector<std::int32_t> current{10, 11, 12, 13};
        const std::vector<std::int32_t> mapping{10, 11, kPagedKVAquirePage, kPagedKVAquirePage};
        const RelocatedPageSubstitution out = plan_relocated_page_substitution(current, mapping, {});
        check(out.from_pool == 0, "dropped pages the copies do not read pay for the destinations");
        check(out.held_for_copy.empty(), "and nothing is left to release late");
        check(out.final_ids[2] == 12 && out.final_ids[3] == 13,
              "the reuse is the pages the mapping dropped, in ascending order");
    }

    // A page the mapping still names stays mapped, so reading it is not a reason to hold it: the copy
    // reads it where it already is, and the pool never has to supply a replacement for it either.
    {
        const std::vector<std::int32_t> current{10, 11, 12, 13};
        const std::vector<std::int32_t> mapping{10, 11, 12, kPagedKVAquirePage};
        const std::vector<std::int32_t> protected_pages{12, 13};
        const RelocatedPageSubstitution out =
            plan_relocated_page_substitution(current, mapping, protected_pages);
        check(out.from_pool == 1, "the kept source page is not a destination and not a saving");
        check(out.held_for_copy.size() == 1 && out.held_for_copy[0] == 13,
              "the dropped source page is held for the copy");
        check(out.final_ids[2] == 12, "the kept source page is still mapped");
    }

    // Determinism, because the pages chosen become the published block table: the same inputs must give
    // the same physical pages, which an iteration over an unordered container would not guarantee.
    {
        const std::vector<std::int32_t> current{40, 7, 19, 3, 12, 25};
        const std::vector<std::int32_t> mapping{40, kPagedKVAquirePage, 19, 3, kPagedKVAquirePage,
                                                kPagedKVAquirePage};
        const std::vector<std::int32_t> protected_pages{7};
        const RelocatedPageSubstitution first =
            plan_relocated_page_substitution(current, mapping, protected_pages);
        for (int repeat = 0; repeat < 8; ++repeat) {
            const RelocatedPageSubstitution again =
                plan_relocated_page_substitution(current, mapping, protected_pages);
            if (again.final_ids != first.final_ids || again.from_pool != first.from_pool ||
                again.held_for_copy != first.held_for_copy) {
                std::printf("  FAIL: the substitution is not deterministic\n");
                ++g_failures;
                break;
            }
        }
    }

    // Conservation, over a sweep of shapes: every page the allocation owned is either named by the
    // mapping or accounted for as held for the copy, exactly as many times as it was owned. A leak here
    // is a page the pool believes is free while the allocation still believes it owns it; a duplicate
    // is two logical pages aliasing one physical page.
    for (std::size_t protected_count = 0; protected_count <= 4; ++protected_count) {
        for (std::uint32_t acquires = 1; acquires <= 4; ++acquires) {
            std::vector<std::int32_t> current{10, 11, 12, 13, 14, 15};
            std::vector<std::int32_t> mapping{10, 11};
            mapping.resize(2 + acquires, kPagedKVAquirePage);
            std::vector<std::int32_t> protected_pages{12, 13, 14, 15};
            protected_pages.resize(protected_count);

            const RelocatedPageSubstitution out =
                plan_relocated_page_substitution(current, mapping, protected_pages);

            std::vector<std::int32_t> accounted = out.held_for_copy;
            std::size_t acquire_slots = 0;
            for (const std::int32_t page_id : out.final_ids) {
                if (page_id == kPagedKVAquirePage) {
                    ++acquire_slots;
                } else {
                    accounted.push_back(page_id);
                }
            }
            if (acquire_slots != out.from_pool) {
                std::printf("  FAIL: protected=%zu acquires=%u leaves %zu slots for %u pool pages\n",
                            protected_count, acquires, acquire_slots, out.from_pool);
                ++g_failures;
            }
            std::sort(accounted.begin(), accounted.end());
            std::sort(current.begin(), current.end());

            // The pool supplies `from_pool` pages of its own, so they are the one part of the ledger
            // that is not in `accounted`; every other owned page has to be, exactly once.
            std::vector<std::int32_t> owned_still_held(accounted);
            std::vector<std::int32_t> expected(current);
            // `accounted` is the mapping's kept pages plus the held ones; `current` is everything that
            // was owned. The difference is what was dropped and reused as a destination, which the
            // acquire slots above do not name because the substituted ids are already in `final_ids`.
            if (owned_still_held != expected) {
                std::printf("  FAIL: protected=%zu acquires=%u accounts for %zu of %zu owned pages\n",
                            protected_count, acquires, owned_still_held.size(), expected.size());
                ++g_failures;
            }
            for (const std::int32_t page_id : out.final_ids) {
                if (page_id == kPagedKVAquirePage) { continue; }
                if (std::count(out.final_ids.begin(), out.final_ids.end(), page_id) != 1) {
                    std::printf("  FAIL: protected=%zu acquires=%u maps page %d more than once\n",
                                protected_count, acquires, page_id);
                    ++g_failures;
                }
            }
        }
    }

    // Rejections: a mapping that names a page the allocation does not own, and a copy that reads one,
    // are both caller errors. Silently ignoring either would leave the caller believing a relocation
    // happened against a page that was never its to relocate.
    const auto rejected = [&](const char* what, const std::vector<std::int32_t>& current,
                              const std::vector<std::int32_t>& mapping,
                              const std::vector<std::int32_t>& protected_pages) {
        try {
            (void)plan_relocated_page_substitution(current, mapping, protected_pages);
            std::printf("  FAIL: accepted %s\n", what);
            ++g_failures;
        } catch (const std::invalid_argument&) {
        }
    };
    rejected("a mapping naming a page the allocation does not own", {10, 11}, {10, 99}, {});
    rejected("a mapping naming the same owned page twice", {10, 11}, {10, 10}, {});
    rejected("a copy reading a page the allocation does not own", {10, 11}, {10, 11}, {99});

    // Reading one page from two moves is not an error, it is the ordinary shape of an off-grid span:
    // the destination page that starts inside a source page reads its head from that page, and the
    // destination page after it reads that same page's tail. What matters is that the page is read at
    // all, so both occurrences are stripped of their surplus in one go and nothing is handed back
    // while the copies are still reading it.
    {
        const std::vector<std::int32_t> current{10, 11, 12};
        const std::vector<std::int32_t> mapping{10, kPagedKVAquirePage, kPagedKVAquirePage};
        const std::vector<std::int32_t> protected_pages{11, 11};
        const RelocatedPageSubstitution out =
            plan_relocated_page_substitution(current, mapping, protected_pages);
        check(out.from_pool == 1, "a page read twice still cannot take a destination slot");
        check(out.held_for_copy.size() == 1 && out.held_for_copy[0] == 11,
              "and it is held once, as the page it is");
        check(out.final_ids[1] == 12 && out.final_ids[2] == kPagedKVAquirePage,
              "so the other dropped page pays for one destination and the pool for the other");
    }
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
    test_row_span_tiles_the_plane();
    test_copied_span_moves();
    test_rope_delta_page_ranges();
    test_relocated_page_substitution();

    if (g_failures != 0) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

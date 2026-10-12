#pragma once

// The delta plane's shape, sizing and per-page delta rule, as pure integer arithmetic.
//
// This header is deliberately free of CUDA and of `Tensor`, so the relocation's *layout arithmetic*
// can be built and tested where there is no CUDA toolchain. `core/paged_kv_cache.h` owns the device
// binding and includes this, so nothing downstream of it changes.
//
// Why it exists as its own translation-unit-free header: `paged_kv_cache.h` includes
// <cuda_runtime_api.h>, so the header that declares the pool cannot be compiled, let alone tested, in a
// sandbox with no nvcc. Everything below is arithmetic over `table_rows`, `logical_pages` and
// `sizeof(std::int32_t)` and depends on no CUDA type, which makes it the part of the pool-layout work
// that can be verified off-machine. See docs/SCR_QUANTIZED_KV_RELOCATION.md §"Step 1a".
//
// The plane it describes is what lets attention read a relocated page as if it had been prefilled at
// its new position: instead of rewriting the quantized K plane, attention reads this per-logical-page
// delta and counter-rotates the query tile by it.

#include <cstddef>
#include <cstdint>

// The namespace is `ninfer`, not `ninfer::core`, because that is where
// `kPagedKVPageSize` lived before this header was extracted: every consumer
// (`layouts_impl.h`, `request_plan_impl.h`, `dflash_context_impl.h`, the
// attention kernels) names it unqualified inside `ninfer`, and moving it into a
// nested namespace would break all of them at once.
namespace ninfer {

// One paged KV page holds this many tokens. Moved here from core/paged_kv_cache.h so the arithmetic
// below can name it without pulling in the device-side header; the two must not both define it.
inline constexpr std::int32_t kPagedKVPageSize = 64;

// A page with a delta of this is an ordinary page and the attention path takes its existing route.
inline constexpr std::int32_t kRopeDeltaNone = 0;

struct RopeDeltaLayout {
    std::int32_t logical_pages = 0;
    std::int32_t table_rows    = 0;
    std::size_t  plane_bytes   = 0;
};

// Logical pages a `max_context`-token context needs. Rounds up, so a context that is not a whole
// number of pages still reserves every page its last partial page lives in.
//
// The arithmetic is widened before the division because `max_context` is int32: at the top of the
// range `max_context + 63` overflows signed, which is UB and, on the compilers this project uses,
// wraps negative and returns a negative page count. Nothing in the product reaches that far -- real
// contexts are capped by the variant's maximum, orders of magnitude below -- but a rounding helper
// that is undefined at the top of its own domain is a trap rather than a guard, and a test does now
// call it there deliberately, so the case has to behave rather than merely not be hit.
// Byte span of one table row's slice of the delta plane: `logical_pages` contiguous int32 values at
// `row * logical_pages * sizeof(std::int32_t)`.
//
// Extracted from `PagedKVPool::clear_rope_delta_row` for the same reason `q5_linear_add_catalog.h`
// was extracted from its planner: the arithmetic that must not be wrong lives where it can be tested
// without a CUDA toolchain. It is also the part a reviewer got wrong once already -- see the row-span
// test, which proves the rows are disjoint and jointly tile the plane.
//
// The name carries the element size because the plane's dtype does. A RoPE delta is a signed position
// count and the plane is `DType::I32`; a caller that ever changes that has to change this too, and
// the name is what makes the coupling visible rather than something to discover from a stride bug.
struct RopeDeltaRowSpan {
    std::size_t offset = 0;
    std::size_t bytes  = 0;
};

// The plane is `{logical_pages, table_rows}` -- the parameter order matches the tensor's own shape,
// so the call site reads `span(plane.ne[0], plane.ne[1], row)` -- with dim 0 innermost, so a row is
// contiguous and rows are `logical_pages` elements apart.
//
// Returns an all-zero span when the row is out of range. The degenerate plane shapes are not
// reachable from a pool this repo builds: `plan_paged_kv_pool` requires `table_rows > 0` and derives
// `logical_pages` through `checked_i32`, which rejects zero. They are rejected here anyway so the
// caller gets a span it can refuse rather than a wrapped offset, but the caller checks the row itself
// so that its diagnostic names the right cause.
[[nodiscard]] constexpr RopeDeltaRowSpan rope_delta_row_span_i32(std::int32_t logical_pages,
                                                                std::int32_t table_rows,
                                                                std::int32_t row) noexcept {
    if (table_rows <= 0 || logical_pages <= 0 || row < 0 || row >= table_rows) {
        return RopeDeltaRowSpan{0, 0};
    }
    const std::size_t stride =
        static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t);
    return RopeDeltaRowSpan{static_cast<std::size_t>(row) * stride, stride};
}

[[nodiscard]] constexpr std::int32_t rope_delta_logical_pages(std::int32_t max_context) noexcept {
    if (max_context <= 0) { return 0; }
    const std::uint64_t pages = (static_cast<std::uint64_t>(max_context) +
                                 static_cast<std::uint64_t>(kPagedKVPageSize) - 1) /
                                static_cast<std::uint64_t>(kPagedKVPageSize);
    return static_cast<std::int32_t>(pages);
}

[[nodiscard]] constexpr std::size_t rope_delta_plane_bytes(std::int32_t table_rows,
                                                          std::int32_t logical_pages) noexcept {
    if (table_rows < 0 || logical_pages < 0) { return 0; }
    return static_cast<std::size_t>(table_rows) * static_cast<std::size_t>(logical_pages) *
           sizeof(std::int32_t);
}

// A layout is supported when both extents are positive, a single page delta plane fits in int32
// positions, and the plane is no smaller than the block table it parallels. The last condition is the
// one that matters: a delta plane that understates the block table would leave pages whose delta can
// never be read, which reads as "relocation silently did nothing" rather than as an error.
[[nodiscard]] constexpr bool rope_delta_layout_is_supported(std::int32_t table_rows,
                                                           std::int32_t logical_pages,
                                                           std::int32_t block_table_rows,
                                                           std::int32_t block_table_pages) noexcept {
    if (table_rows <= 0 || logical_pages <= 0) { return false; }
    if (block_table_rows <= 0 || block_table_pages <= 0) { return false; }
    if (static_cast<std::uint64_t>(table_rows) * logical_pages > 0x7fffffffu) { return false; }
    return table_rows >= block_table_rows && logical_pages >= block_table_pages;
}

[[nodiscard]] constexpr RopeDeltaLayout
resolve_rope_delta_layout(std::int32_t max_context, std::int32_t table_rows) noexcept {
    const std::int32_t pages = rope_delta_logical_pages(max_context);
    if (!rope_delta_layout_is_supported(table_rows, pages, table_rows, pages)) {
        return RopeDeltaLayout{0, 0, 0};
    }
    return RopeDeltaLayout{pages, table_rows, rope_delta_plane_bytes(table_rows, pages)};
}

// Whether a relocated span is expressible as whole-page deltas. Spans are block-aligned by
// construction -- `diff_prompt_edit` hashes whole 64-token pages and `plan_tail_relocation` trims
// spans back to whole blocks -- so a span that is not aligned is a caller error, not a case to
// tolerate. The runtime must assert this before writing the plane. An empty span is rejected for the
// same reason: it relocates nothing, so accepting it would let the caller believe it did.
[[nodiscard]] constexpr bool relocated_span_is_page_aligned(std::uint32_t begin,
                                                            std::uint32_t tokens) noexcept {
    if (tokens == 0) { return false; }
    return begin % static_cast<std::uint32_t>(kPagedKVPageSize) == 0u &&
           tokens % static_cast<std::uint32_t>(kPagedKVPageSize) == 0u;
}

// Whether a *destination* range is expressible as whole-page deltas. This is the half of the
// alignment rule that survives when the payload is copied instead of the page table being
// re-pointed: a page carries exactly one delta, so a page shared by relocated and freshly prefilled
// tokens could carry none, and the destination has to stay on the grid. The source is deliberately
// absent from this predicate -- a copied source is a token offset like any other, and its page may
// hold the head of the run in one physical page and the tail in the next.
[[nodiscard]] constexpr bool relocated_dest_is_page_aligned(std::uint32_t dest_begin,
                                                           std::uint32_t tokens) noexcept {
    if (tokens == 0) { return false; }
    return dest_begin % static_cast<std::uint32_t>(kPagedKVPageSize) == 0u &&
           tokens % static_cast<std::uint32_t>(kPagedKVPageSize) == 0u;
}

// The delta for one logical page of one relocated span, or `kRopeDeltaNone` when the page is not
// wholly inside the span's source range. Because spans are page-aligned, a page is either entirely
// inside or entirely outside; the partial case is therefore a boundary error and returns none rather
// than a shift that would move half a page's keys.
//
// The sign is the one the attention kernels' Q counter-rotation documents: the
// caller supplies `new_position - old_position`, so a span moved left is negative.
[[nodiscard]] constexpr std::int32_t rope_delta_for_span(std::uint32_t source_begin,
                                                        std::uint32_t source_tokens,
                                                        std::uint32_t dest_begin,
                                                        std::int32_t logical_page) noexcept {
    if (logical_page < 0) { return kRopeDeltaNone; }
    const std::uint64_t page_first =
        static_cast<std::uint64_t>(logical_page) * static_cast<std::uint64_t>(kPagedKVPageSize);
    const std::uint64_t page_last    = page_first + static_cast<std::uint64_t>(kPagedKVPageSize);
    const std::uint64_t source_first = static_cast<std::uint64_t>(source_begin);
    const std::uint64_t source_last  = source_first + static_cast<std::uint64_t>(source_tokens);
    if (page_first < source_first || page_last > source_last) { return kRopeDeltaNone; }

    const std::int64_t delta =
        static_cast<std::int64_t>(dest_begin) - static_cast<std::int64_t>(source_begin);
    if (delta < -0x7fffffffLL || delta > 0x7fffffffLL) { return kRopeDeltaNone; }
    return static_cast<std::int32_t>(delta);
}

// The delta for one *destination* logical page of a copied span, or `kRopeDeltaNone` when that page
// is not wholly inside the span's destination range.
//
// The value is the same as `rope_delta_for_span`'s -- the span's own shift, which is uniform over
// every page it holds, because the destination grid is whole pages and the source is one contiguous
// run. What changes is the containment test: it has to be in destination coordinates, because that
// is what the plane is indexed by, and a copied source may start anywhere inside a page (so a
// source-page test would reject pages that are perfectly expressible).
[[nodiscard]] constexpr std::int32_t rope_delta_for_dest_page(std::uint32_t source_begin,
                                                             std::uint32_t dest_begin,
                                                             std::uint32_t dest_tokens,
                                                             std::int32_t logical_page) noexcept {
    if (logical_page < 0) { return kRopeDeltaNone; }
    const std::uint64_t page_first =
        static_cast<std::uint64_t>(logical_page) * static_cast<std::uint64_t>(kPagedKVPageSize);
    const std::uint64_t page_last = page_first + static_cast<std::uint64_t>(kPagedKVPageSize);
    const std::uint64_t dest_last =
        static_cast<std::uint64_t>(dest_begin) + static_cast<std::uint64_t>(dest_tokens);
    if (page_first < static_cast<std::uint64_t>(dest_begin) || page_last > dest_last) {
        return kRopeDeltaNone;
    }

    const std::int64_t delta =
        static_cast<std::int64_t>(dest_begin) - static_cast<std::int64_t>(source_begin);
    if (delta < -0x7fffffffLL || delta > 0x7fffffffLL) { return kRopeDeltaNone; }
    return static_cast<std::int32_t>(delta);
}

} // namespace ninfer

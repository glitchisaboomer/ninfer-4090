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
// that can be verified off-machine. See docs/SCR.md §"Step 1a".
//
// The plane it describes is what lets attention read a relocated page as if it had been prefilled at
// its new position: instead of rewriting the quantized K plane, attention reads this per-logical-page
// delta and counter-rotates the query tile by it.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

// Sentinel for one logical page in a relocated or reassigned mapping that must be filled with a
// physical page acquired from the pool. Every real physical page id is non-negative, so -1 cannot
// collide with an owned page.
//
// Moved here from `core/paged_kv_cache.h` for the same reason `kPagedKVPageSize` was: the arithmetic
// that plans a relocation's substitution and its copies names it, and that arithmetic has to be
// buildable and testable without the device-side header. The two must not both define it.
inline constexpr std::int32_t kPagedKVAquirePage = -1;

// One surviving run as the payload copier needs it: a whole-page destination and an arbitrary source
// offset. The fields mirror the relocation planner's `RelocatedSpan`, but this type lives here
// because core cannot include a target runtime's header, and the copier is a core concern -- it owns
// the page table and the pool the sources have to stay in.
struct RelocatedCopySpan {
    std::uint32_t source_begin = 0;
    std::uint32_t dest_begin   = 0;
    std::uint32_t tokens       = 0;

    [[nodiscard]] constexpr std::uint32_t source_end() const noexcept {
        return source_begin + tokens;
    }
    [[nodiscard]] constexpr std::uint32_t dest_end() const noexcept { return dest_begin + tokens; }
};

struct RopeDeltaPageRange {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    std::int32_t delta = 0;
};

inline constexpr std::uint32_t kRopeDeltaRangeEmpty = 0xffffffffu;

// The largest token offset a packed interval can name. `end` is inclusive of it because a whole
// page's relocated interval ends at its last token plus one, so this is `kPagedKVPageSize`.
inline constexpr std::uint32_t kRopeDeltaRangeMaxOffset =
    static_cast<std::uint32_t>(kPagedKVPageSize);

// Pure packing/unpacking arithmetic over one uint32_t, called from both host planning code and the
// boundary-aware device kernels. `__host__ __device__` keeps them constexpr-evaluable on the host
// (the layout tests depend on that) while legal to inline into device code; a host-only constexpr
// here is an nvcc error at every kernel call site. The annotation collapses to nothing in plain
// C++ translation units, where CUDA's specifiers do not exist.
#ifdef __CUDACC__
#define NINFER_ROPE_DELTA_RANGE_HOST_DEVICE __host__ __device__
#else
#define NINFER_ROPE_DELTA_RANGE_HOST_DEVICE
#endif

[[nodiscard]] NINFER_ROPE_DELTA_RANGE_HOST_DEVICE constexpr std::uint32_t pack_rope_delta_page_range(
    std::uint32_t begin, std::uint32_t end) noexcept {
    if (begin >= end || end > static_cast<std::uint32_t>(kPagedKVPageSize)) {
        return kRopeDeltaRangeEmpty;
    }
    return begin | (end << 6);
}

[[nodiscard]] NINFER_ROPE_DELTA_RANGE_HOST_DEVICE constexpr bool rope_delta_page_range_is_empty(
    std::uint32_t packed) noexcept {
    return packed == kRopeDeltaRangeEmpty;
}

[[nodiscard]] NINFER_ROPE_DELTA_RANGE_HOST_DEVICE constexpr std::uint32_t rope_delta_page_range_begin(
    std::uint32_t packed) noexcept {
    return packed & 0x3fu;
}

[[nodiscard]] NINFER_ROPE_DELTA_RANGE_HOST_DEVICE constexpr std::uint32_t rope_delta_page_range_end(
    std::uint32_t packed) noexcept {
    return (packed >> 6) & 0x7fu;
}

// A page whose packed interval is a strict subset of the page: part of its tokens were
// relocated and part were not, so no single whole-page shift can describe either half and only the
// boundary-aware score path can read it correctly.
//
// This is the single definition of "the boundary-aware score path applies". Every kernel must take
// it from here rather than re-deriving it from the packed fields: `rope_delta_page_range_begin`
// reads the empty sentinel as offset 63 and `rope_delta_page_range_end` as 127, so a derivation
// that forgets the emptiness guard mistakes an ordinary page for a partial one. That failure is
// silent -- both halves rotate plausibly -- which is why it gets a tested definition instead of
// being spelled out at each use.
[[nodiscard]] NINFER_ROPE_DELTA_RANGE_HOST_DEVICE constexpr bool rope_delta_page_range_is_partial(
    std::uint32_t packed) noexcept {
    if (rope_delta_page_range_is_empty(packed)) { return false; }
    return rope_delta_page_range_begin(packed) != 0 ||
           rope_delta_page_range_end(packed) != static_cast<std::uint32_t>(kPagedKVPageSize);
}

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

// One device-to-device move a copied relocation issues: `tokens` tokens of one destination page that
// come from one source page. `dest_page` and `source_page` are *logical* page indices -- of the
// mapping being published and of the mapping being replaced, respectively -- because the physical
// pages behind them are exactly what the substitution below is computing. A move ends at either page
// boundary; a partial destination page may therefore require up to three segments when both boundaries
// fall inside the span at different offsets.
//
// This is arithmetic, not device code, and it lives here for the reason the rest of this header does:
// the statement "a destination page needs at most two moves, and here they are" is the part of the
// copier that must not be wrong, and it is checkable without a GPU.
struct RelocatedPageMove {
    std::int32_t dest_page    = -1;
    std::uint32_t dest_token  = 0;
    std::int32_t source_page  = -1;
    std::uint32_t source_token = 0;
    std::uint32_t tokens      = 0;
};

// The two logical page counts a copied relocation spans: `source` is the mapping standing now,
// which the copies read, and `destination` the mapping being published. Named rather than
// positional because the two are the same type -- and, for a relocation that does not resize,
// the same value -- so a transposition of two `std::size_t` arguments would compile and stay
// invisible.
struct RelocatedPageCounts {
    std::size_t source      = 0;
    std::size_t destination = 0;

    // The common no-resize case, where both mappings are the same length.
    [[nodiscard]] static constexpr RelocatedPageCounts same(std::size_t pages) noexcept {
        return RelocatedPageCounts{.source = pages, .destination = pages};
    }
};

// Decomposes copied spans into the moves that carry them, validating each end against the mapping
// that end is indexed in: the sources are logical pages of the mapping standing now, which is the
// copy's source, and the destinations logical pages of the mapping being published.
//
// Two counts, not one, because a relocation *can* resize. An edit that deletes a region shortens the
// prompt, so the published mapping is shorter than the one it replaces, while a surviving run's
// source still lies in the part of the old mapping the new one does not cover -- the reused prefix is
// exactly that part. Bounding the sources by the destination's count refuses those edits for a reason
// that is not true of them: the source never left its own mapping. Callers whose two mappings are the
// same length pass `RelocatedPageCounts::same(pages)`.
//
// Splits at both source and destination page boundaries. Throws `std::invalid_argument` for a span
// outside either mapping or for a relocation with no moves.
[[nodiscard]] inline std::vector<RelocatedPageMove>
plan_relocated_page_moves(std::span<const RelocatedCopySpan> spans, RelocatedPageCounts mappings) {
    constexpr std::uint32_t page = static_cast<std::uint32_t>(kPagedKVPageSize);
    std::vector<RelocatedPageMove> moves;
    moves.reserve(spans.size() * 4);
    for (const RelocatedCopySpan& span : spans) {
        const std::uint64_t dest_end = static_cast<std::uint64_t>(span.dest_begin) + span.tokens;
        const std::uint64_t source_end = static_cast<std::uint64_t>(span.source_begin) + span.tokens;
        const std::uint64_t dest_tokens =
            static_cast<std::uint64_t>(mappings.destination) * page;
        const std::uint64_t source_tokens = static_cast<std::uint64_t>(mappings.source) * page;
        if (span.tokens == 0 || dest_end > dest_tokens) {
            throw std::invalid_argument("Paged KV relocation span reaches past its mapping");
        }
        if (source_end > source_tokens) {
            throw std::invalid_argument("Paged KV relocation source reaches past its mapping");
        }

        std::uint64_t dest_token = span.dest_begin;
        while (dest_token < dest_end) {
            const std::uint64_t source_token =
                static_cast<std::uint64_t>(span.source_begin) + (dest_token - span.dest_begin);
            const std::uint32_t dest_offset = static_cast<std::uint32_t>(dest_token % page);
            const std::uint32_t source_offset = source_token % page;
            const std::uint32_t take = static_cast<std::uint32_t>(std::min({
                static_cast<std::uint64_t>(page - dest_offset),
                static_cast<std::uint64_t>(page - source_offset), dest_end - dest_token}));

            RelocatedPageMove move;
            move.dest_page    = static_cast<std::int32_t>(dest_token / page);
            move.dest_token   = dest_offset;
            move.source_page  = static_cast<std::int32_t>(source_token / page);
            move.source_token = source_offset;
            move.tokens       = take;
            moves.push_back(move);
            dest_token += take;
        }
    }
    if (moves.empty()) {
        throw std::invalid_argument("Paged KV relocation carries no moves");
    }
    return moves;
}

// The page ownership transition a copied relocation performs, as pure arithmetic over page ids.
struct RelocatedPageSubstitution {
    // Parallel to `mapping`, with every acquire slot the dropped pages can fill already filled. The
    // slots still holding `kPagedKVAquirePage` are the ones the caller must fill from the pool, in
    // order, and there are exactly `from_pool` of them.
    std::vector<std::int32_t> final_ids;
    // Dropped pages the caller may return to the pool only *after* the copies have run: the ones the
    // copies read, plus every dropped page past the slots they could fill.
    std::vector<std::int32_t> held_for_copy;
    std::uint32_t from_pool = 0;
};

// Computes that transition. `current_ids` is the mapping being replaced (its pages, in logical order,
// with repeats for a page mapped more than once), `mapping` the mapping being published, and
// `protected_pages` the physical pages the copies will read. A page appears as many times as the
// moves read it -- an off-grid span reads the page its destination page starts in twice, once for the
// head of the destination page and once for the tail of the one before it -- so this is membership,
// not a count: what matters is that the page is read at all, and handing back *any* occurrence of it
// frees it for another lane while the copies are still reading it.
//
// The one difference from `PagedKVAllocation::reassign_pages`' substitution is the point of this
// function: a page the copies read is never a destination, so it is held out of the reuse and handed
// back afterwards instead. Without that, the substitution prefers to give a dropped page straight back
// as the new physical page -- which for a copied relocation means the destination *is* the source, an
// overlapping device copy, whose behaviour the CUDA runtime does not define.
//
// Deterministic: pages are considered in ascending id order, so the same inputs always produce the
// same physical pages, which is what makes the published block table reproducible.
[[nodiscard]] inline RelocatedPageSubstitution plan_relocated_page_substitution(
    std::span<const std::int32_t> current_ids, std::span<const std::int32_t> mapping,
    std::span<const std::int32_t> protected_pages) {
    std::map<std::int32_t, std::uint32_t> owned;
    for (const std::int32_t page_id : current_ids) { ++owned[page_id]; }

    std::map<std::int32_t, std::uint32_t> remaining(owned);
    std::uint32_t acquire_count = 0;
    for (const std::int32_t page_id : mapping) {
        if (page_id == kPagedKVAquirePage) {
            ++acquire_count;
            continue;
        }
        const auto found = remaining.find(page_id);
        if (found == remaining.end() || found->second == 0) {
            throw std::invalid_argument(
                "Paged KV relocation names a page this allocation does not own");
        }
        if (--found->second == 0) { remaining.erase(found); }
    }

    // The copies read these pages, so they are checked against what the allocation owns rather than
    // against what is left over -- a page the mapping still names is read while mapped, and a
    // protected page the allocation does not own at all is a caller error, not a silent no-op. The
    // check is per *page*, not per occurrence: a page read by two moves is still one owned page.
    std::set<std::int32_t> protected_pages_set;
    for (const std::int32_t page_id : protected_pages) {
        if (owned.find(page_id) == owned.end()) {
            throw std::invalid_argument(
                "Paged KV relocation copies from a page this allocation does not own");
        }
        protected_pages_set.insert(page_id);
    }

    std::vector<std::int32_t> free_surplus;
    std::vector<std::int32_t> held;
    for (const auto& entry : remaining) {
        const bool read_by_a_copy = protected_pages_set.count(entry.first) != 0;
        for (std::uint32_t occurrence = 0; occurrence < entry.second; ++occurrence) {
            if (read_by_a_copy) {
                held.push_back(entry.first);
                continue;
            }
            free_surplus.push_back(entry.first);
        }
    }

    const std::uint32_t reused =
        std::min(acquire_count, static_cast<std::uint32_t>(free_surplus.size()));

    RelocatedPageSubstitution out;
    out.final_ids.assign(mapping.begin(), mapping.end());
    std::size_t slot = 0;
    for (std::int32_t& page_id : out.final_ids) {
        if (page_id == kPagedKVAquirePage && slot < reused) { page_id = free_surplus[slot++]; }
    }
    for (std::size_t index = reused; index < free_surplus.size(); ++index) {
        held.push_back(free_surplus[index]);
    }
    out.held_for_copy = std::move(held);
    out.from_pool     = acquire_count - reused;
    return out;
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

[[nodiscard]] inline std::vector<RopeDeltaPageRange> plan_rope_delta_page_ranges(
    std::span<const RelocatedCopySpan> spans, std::size_t mapping_pages) {
    constexpr std::uint32_t page = static_cast<std::uint32_t>(kPagedKVPageSize);
    std::vector<RopeDeltaPageRange> ranges(mapping_pages);
    std::vector<bool> claimed(mapping_pages, false);
    for (const RelocatedCopySpan& span : spans) {
        if (span.tokens == 0 || static_cast<std::uint64_t>(span.dest_begin) + span.tokens >
                                    static_cast<std::uint64_t>(mapping_pages) * page) {
            throw std::invalid_argument("Paged KV relocation span is outside its destination mapping");
        }
        const std::int64_t delta = static_cast<std::int64_t>(span.dest_begin) - span.source_begin;
        if (delta < -0x7fffffffLL || delta > 0x7fffffffLL) {
            throw std::invalid_argument("Paged KV relocation delta does not fit int32");
        }
        const std::uint32_t end = span.dest_begin + span.tokens;
        for (std::uint32_t token = span.dest_begin; token < end;) {
            const std::uint32_t logical_page = token / page;
            const std::uint32_t page_begin = logical_page * page;
            const std::uint32_t range_end = std::min(end, page_begin + page);
            if (claimed[logical_page]) {
                throw std::invalid_argument("Multiple relocation spans share a logical page");
            }
            claimed[logical_page] = true;
            ranges[logical_page] = RopeDeltaPageRange{
                .begin = token - page_begin,
                .end = range_end - page_begin,
                .delta = static_cast<std::int32_t>(delta),
            };
            token = range_end;
        }
    }
    return ranges;
}

[[nodiscard]] inline std::vector<std::uint32_t> encode_rope_delta_page_ranges(
    std::span<const RelocatedCopySpan> spans, std::size_t mapping_pages) {
    const std::vector<RopeDeltaPageRange> ranges =
        plan_rope_delta_page_ranges(spans, mapping_pages);
    std::vector<std::uint32_t> packed(mapping_pages, kRopeDeltaRangeEmpty);
    for (std::size_t page = 0; page < ranges.size(); ++page) {
        if (ranges[page].begin != ranges[page].end) {
            packed[page] = pack_rope_delta_page_range(ranges[page].begin, ranges[page].end);
        }
    }
    return packed;
}

} // namespace ninfer

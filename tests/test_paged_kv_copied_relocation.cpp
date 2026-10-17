// Oracle test for PagedKVAllocation::relocate_copied_pages, the mechanism the copied relocation
// path uses when a surviving run's source is off the page grid and the destination page therefore
// has to be written rather than re-pointed.
//
// The property this file exists for is the one that made the path unreachable for the edits it was
// built to serve: a copied relocation is asked to publish a mapping that is *shorter* than the one
// standing now. A deletion shortens the prompt, so the new prompt needs fewer pages than the
// resident mapping holds, while the surviving run's source pages lie above the new mapping's end.
// The allocation used to refuse that -- it required the two mappings to be the same length, which
// made every deletion a 500 -- and the arithmetic never needed it: a page above the published
// mapping is a dropped page like any other, so it pays for a destination unless a copy reads it, in
// which case it is held and returned once the copies have run.
//
// The invariants, derived from the contract in paged_kv_cache.h rather than from the implementation:
//
//   C1  the published mapping is the requested one, with its length, in order
//   C2  every page the copies read is still exactly what it was -- a source is never a destination,
//       because a device copy of overlapping ranges is not defined by the CUDA runtime
//   C3  every destination page holds its source page's bytes, byte for byte and over the whole page
//   C4  no page outside the moved ranges changed
//   C5  the ownership partition holds: free pages + mapped pages == total page groups, so a page the
//       call dropped is returned and a page it took is not leaked
//   C6  a rejection leaves the mapping and the pool exactly as they were
//
// The byte comparisons are whole-page because the scenario's moves are whole pages: the plane is
// required to be dense (asserted below), so a move of 64 tokens at offset 0 carries every byte of
// its page and the two pages must be identical afterwards. No tolerance and no range arithmetic --
// this is the copier's own claim, checked against the pool's bytes rather than against a re-derived
// offset table that would duplicate the code under test.

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_delta_layout.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::cerr << "  FAIL: " << what << '\n';
    ++failures;
}

void note(const std::string& what) { std::cout << "  " << what << '\n'; }

void check(bool condition, const std::string& what) {
    if (!condition) { fail(what); }
}

struct PlannedPool {
    ninfer::PagedKVPoolLayout layout;
    std::size_t bytes = 0;
};

PlannedPool plan_pool(std::uint32_t page_groups, std::uint32_t logical_pages, std::int32_t rows) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = page_groups,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .planes = {
                                                           {ninfer::DType::BF16, 16, 1}}});
    return PlannedPool{std::move(layout), builder.finish(256)};
}

std::vector<std::int32_t> ids_of(const ninfer::PagedKVAllocation& alloc) {
    return std::vector<std::int32_t>(alloc.page_ids().begin(), alloc.page_ids().end());
}

std::vector<std::uint32_t> read_u32(const ninfer::Tensor& row, std::size_t count) {
    std::vector<std::uint32_t> values(count);
    CUDA_CHECK(cudaMemcpy(values.data(), row.data, count * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    return values;
}

// A relocation's moves, predicted independently of the call under test but from the same spans and
// the same two mapping lengths: the sources are logical pages of the mapping standing now, the
// destinations logical pages of the one being published. The counts are named so a case can still
// ask what the *old* single-count bound would have accepted, which is how the change this file
// covers is recorded rather than assumed.
std::vector<ninfer::RelocatedPageMove> moves_of(std::span<const ninfer::RelocatedCopySpan> spans,
                                                ninfer::RelocatedPageCounts mappings) {
    return ninfer::plan_relocated_page_moves(spans, mappings);
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver ||
        (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    ninfer::DeviceContext ctx(0);

    constexpr std::uint32_t kPageGroups  = 12;
    constexpr std::uint32_t kEntitlement = 8;
    constexpr std::int32_t kRows         = 1;

    auto plan = plan_pool(kPageGroups, kPageGroups, kRows);
    ninfer::DeviceArena arena(plan.bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);

    std::cout << "PagedKVAllocation::relocate_copied_pages oracle checks\n";

    // The whole-page byte comparisons below are only sound if a plane's page is dense: a move of 64
    // tokens at offset 0 then carries every byte of the page. Assert that rather than assume it,
    // because a padded plane would make them report a difference that is padding, not payload.
    const ninfer::Tensor& plane  = pool.plane(0);
    const std::size_t page_bytes = pool.page_bytes(0);
    check(plane.nb[0] == static_cast<std::int64_t>(ninfer::dtype_size(plane.dtype)),
          "the plane's leading dimension is dense, so a token's elements are contiguous");
    check(static_cast<std::size_t>(plane.nb[1]) * ninfer::kPagedKVPageSize *
              static_cast<std::size_t>(plane.ne[2]) == page_bytes,
          "the plane has no padding, so a whole-page move carries the whole page");
    note("plane 0: ne{" + std::to_string(plane.ne[0]) + "," + std::to_string(plane.ne[1]) + "," +
         std::to_string(plane.ne[2]) + "}, token stride " + std::to_string(plane.nb[1]) +
         ", page " + std::to_string(page_bytes) + " bytes");

    // A recognizable byte pattern per physical page, so "the destination holds its source" and
    // "nothing else moved" are both checkable against the pool rather than against a model.
    std::vector<std::vector<std::byte>> pattern(kPageGroups);
    for (std::size_t page = 0; page < kPageGroups; ++page) {
        pattern[page].resize(page_bytes);
        for (std::size_t i = 0; i < page_bytes; ++i) {
            pattern[page][i] = static_cast<std::byte>((page * 71 + i * 23) & 0xFF);
        }
    }
    const auto read_page = [&](std::int32_t page) {
        std::vector<std::byte> bytes(page_bytes);
        pool.copy_page_to_host(0, page, bytes.data(), ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        return bytes;
    };
    const auto write_page = [&](std::int32_t page, const std::vector<std::byte>& bytes) {
        pool.copy_page_from_host(0, page, bytes.data(), ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    };

    auto alloc = pool.reserve(kEntitlement);
    alloc.materialize_pages(6);
    const std::vector<std::int32_t> resident = ids_of(alloc);
    for (const std::int32_t page : resident) {
        write_page(page, pattern[static_cast<std::size_t>(page)]);
    }
    alloc.bind_row(0, ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // ---------------------------------------------------------------- the shrinking relocation --
    // Six resident pages, four published: the edit deleted a region. The surviving run is two whole
    // pages long and sits at the end of the *resident* mapping, above the published one entirely.
    {
        const std::uint32_t prompt_tokens = 4 * ninfer::kPagedKVPageSize;
        const std::vector<ninfer::RelocatedCopySpan> spans{
            ninfer::RelocatedCopySpan{.source_begin = 4 * ninfer::kPagedKVPageSize,
                                      .dest_begin   = ninfer::kPagedKVPageSize,
                                      .tokens       = 2 * ninfer::kPagedKVPageSize}};
        std::vector<std::int32_t> mapping(resident.begin(), resident.begin() + 1);
        mapping.resize(4, ninfer::kPagedKVAquirePage);
        const std::vector<std::int32_t> deltas(mapping.size(), ninfer::kRopeDeltaNone);

        // What the change is: the sources are bounded by the *resident* mapping's length. The bound
        // the runtime used to pass -- the published length for both ends -- refuses this run, and
        // that refusal is recorded here so the case cannot silently stop being the one under test.
        const std::vector<ninfer::RelocatedPageMove> moves = moves_of(
            spans, {.source = resident.size(), .destination = mapping.size()});
        check(moves.size() == 2, "a shrinking relocation splits into one move per destination page");
        for (std::size_t index = 0; index < moves.size() && index < 2; ++index) {
            check(moves[index].dest_page == static_cast<std::int32_t>(1 + index),
                  "its destinations ascend from the first published page it covers");
            check(moves[index].source_page == static_cast<std::int32_t>(4 + index),
                  "and its sources ascend in the resident mapping, above the published one's end");
        }
        bool old_bound_refused = false;
        try {
            (void)moves_of(spans, ninfer::RelocatedPageCounts::same(mapping.size()));
        } catch (const std::invalid_argument&) {
            old_bound_refused = true;
        }
        check(old_bound_refused,
              "bounding both ends by the published length refuses a source above its end, which is "
              "the refusal this case exists to keep out of the runtime");

        // The pages the copies read: the last two resident pages. They are the surplus the mapping
        // drops, so the substitution has to hold them out of the acquire slots, or a destination
        // would be a page the copies are reading.
        const std::vector<std::int32_t> sources{resident[4], resident[5]};
        const std::uint32_t free_before = pool.free_pages();

        std::vector<std::byte> src0 = read_page(resident[4]);
        std::vector<std::byte> src1 = read_page(resident[5]);
        std::vector<std::byte> keep0 = read_page(resident[0]);
        std::vector<std::byte> keep3 = read_page(resident[3]);

        alloc.relocate_copied_pages(mapping, deltas, spans, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        const std::vector<std::int32_t> published = ids_of(alloc);
        check(published.size() == mapping.size(),
              "C1 the published mapping is the requested length");
        check(published[0] == mapping[0], "C1 the named slot is published exactly");
        for (std::size_t index = 0; index < published.size(); ++index) {
            if (published[index] == sources[0] || published[index] == sources[1]) {
                fail("C2 a page the copies read was published as a destination");
            }
        }
        for (std::size_t i = 0; i < published.size(); ++i) {
            for (std::size_t j = i + 1; j < published.size(); ++j) {
                if (published[i] == published[j]) { fail("C1 a page is mapped twice"); }
            }
        }

        check(read_page(resident[4]) == src0,
              "C2 the first source page is untouched by the copy that reads it");
        check(read_page(resident[5]) == src1,
              "C2 the second source page is untouched by the copy that reads it");
        check(read_page(resident[0]) == keep0, "C4 the carried prefix page did not move");
        check(read_page(resident[3]) == keep3,
              "C4 a page neither moved into nor read did not change");

        check(published[1] != resident[4] && published[2] != resident[4] &&
                  published[3] != resident[4],
              "C2 the first source page is not published as a destination");
        const auto dest_for_source = [&](std::int32_t source_page) {
            for (const ninfer::RelocatedPageMove& move : moves) {
                if (move.source_page == source_page) {
                    return published[static_cast<std::size_t>(move.dest_page)];
                }
            }
            return std::int32_t{-1};
        };
        const std::int32_t dest_of      = dest_for_source(4);
        const std::int32_t dest_of_next = dest_for_source(5);
        check(dest_of >= 0 && dest_of_next >= 0, "both moves name a published destination");
        if (dest_of >= 0) {
            check(read_page(dest_of) == src0,
                  "C3 the first destination holds its whole source page, byte for byte");
        }
        if (dest_of_next >= 0) {
            check(read_page(dest_of_next) == src1,
                  "C3 the second destination holds its whole source page, byte for byte");
        }

        // C5. Nothing came from the pool -- three dropped pages paid for three destinations -- and
        // the two pages the copies read were returned afterwards, so the pool ends two pages richer
        // and owned + free still accounts for every page group.
        check(pool.free_pages() == free_before + 2,
              "C5 the two pages the copies read were returned and nothing else was taken");
        check(pool.free_pages() + published.size() == kPageGroups,
              "C5 the ownership partition balances after a shrinking relocation");
        note("shrink: ok (6 pages -> " + std::to_string(published.size()) + " pages, " +
             std::to_string(pool.free_pages()) + " free)");

        // The planes the call published in the same pass as the mapping, so no reader can see one
        // without the other: the block table row is the published mapping, and the interval plane
        // carries the shift for each page a span covers and nothing for the rest.
        const std::vector<std::int32_t> row = [&] {
            std::vector<std::int32_t> ids(published.size());
            CUDA_CHECK(cudaMemcpy(ids.data(), alloc.block_table().data,
                                  published.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
            return ids;
        }();
        check(row == published, "the republished block-table row is the published mapping");
        const std::vector<std::uint32_t> intervals =
            read_u32(pool.rope_delta_range_row(0), mapping.size());
        check(intervals[0] == ninfer::kRopeDeltaRangeEmpty &&
                  intervals[3] == ninfer::kRopeDeltaRangeEmpty,
              "a page no span covers carries no relocated interval");
        check(intervals[1] != ninfer::kRopeDeltaRangeEmpty &&
                  intervals[2] != ninfer::kRopeDeltaRangeEmpty,
              "both pages a span covers carry their interval");
    }

    // ------------------------------------------------------- an equal-length relocation still --
    // The shape the path was verified for before this change, kept as the regression control: a
    // mapping the same length as the resident one, one whole page relocated, one destination bought
    // from the pool because the page it reads cannot serve as one.
    {
        const std::vector<std::int32_t> before = ids_of(alloc);
        std::vector<std::int32_t> mapping = before;
        mapping[1] = ninfer::kPagedKVAquirePage;
        const std::vector<ninfer::RelocatedCopySpan> spans{
            ninfer::RelocatedCopySpan{.source_begin = 3 * ninfer::kPagedKVPageSize,
                                      .dest_begin   = ninfer::kPagedKVPageSize,
                                      .tokens       = ninfer::kPagedKVPageSize}};
        const std::vector<std::int32_t> deltas(mapping.size(), ninfer::kRopeDeltaNone);

        const std::vector<std::byte> source_bytes = read_page(before[3]);

        alloc.relocate_copied_pages(mapping, deltas, spans, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        const std::vector<std::int32_t> published = ids_of(alloc);
        check(published.size() == before.size(), "C1 an equal-length relocation keeps its length");
        check(published[0] == before[0] && published[2] == before[2] && published[3] == before[3],
              "C1 the pages the mapping keeps are kept");
        check(published[1] != before[3],
              "C2 the page the copy reads is not its own destination");
        check(read_page(before[1]) == source_bytes,
              "C3 the destination holds its source page, byte for byte");
        check(read_page(before[3]) == source_bytes,
              "C2 the source page is still itself after the copy");
        check(pool.free_pages() + published.size() == kPageGroups,
              "C5 the ownership partition balances after an equal-length relocation");
        note("equal length: ok (" + std::to_string(pool.free_pages()) + " free)");
    }

    // ------------------------------------------------------------------ refusals still refused --
    // Each of these must leave the mapping and the pool exactly as they were. A refusal that moved
    // a page, or that published half a mapping, is worse than the failure it reports.
    {
        const std::vector<std::int32_t> before      = ids_of(alloc);
        const std::uint32_t free_before             = pool.free_pages();
        const std::vector<ninfer::RelocatedCopySpan> spans{
            ninfer::RelocatedCopySpan{.source_begin = 3 * ninfer::kPagedKVPageSize,
                                      .dest_begin   = ninfer::kPagedKVPageSize,
                                      .tokens       = ninfer::kPagedKVPageSize}};

        const auto expect_reject = [&](const char* what,
                                       const std::vector<std::int32_t>& mapping,
                                       std::span<const std::int32_t> deltas,
                                       std::span<const ninfer::RelocatedCopySpan> plan_spans) {
            bool threw = false;
            try {
                alloc.relocate_copied_pages(mapping, deltas, plan_spans, ctx.stream);
            } catch (const std::exception&) {
                threw = true;
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            check(threw, std::string(what) + " must be refused");
            if (ids_of(alloc) != before) {
                fail(std::string(what) + ": a refusal changed the mapping");
            }
            if (pool.free_pages() != free_before) {
                fail(std::string(what) + ": a refusal changed the free page count");
            }
        };

        const std::vector<std::int32_t> ok(before.size(), ninfer::kPagedKVAquirePage);
        const std::vector<std::int32_t> ok_deltas(ok.size(), ninfer::kRopeDeltaNone);
        const std::vector<std::int32_t> short_deltas(ok.size() - 1, ninfer::kRopeDeltaNone);
        const std::vector<std::int32_t> oversized(kEntitlement + 1, ninfer::kPagedKVAquirePage);
        const std::vector<std::int32_t> oversized_deltas(oversized.size(), ninfer::kRopeDeltaNone);
        const std::vector<std::int32_t> empty_mapping;

        expect_reject("an empty mapping", empty_mapping, std::span<const std::int32_t>(),
                      spans);
        expect_reject("a mapping past the entitlement", oversized, oversized_deltas, spans);
        expect_reject("a delta plane not parallel to the mapping", ok, short_deltas, spans);
        expect_reject("no spans to copy", ok, ok_deltas,
                      std::span<const ninfer::RelocatedCopySpan>());
        note("refusals: ok (each left the mapping and the pool untouched)");
    }

    // ------------------------------------------------------------------ a growing relocation --
    // The opposite reshape to the shrink this file exists for: the published mapping is *longer*
    // than the resident one, so its extra destinations cannot come from dropped pages and have to
    // be bought from the pool. Nothing in the copier's arithmetic bounds a destination by the
    // source count, so this is expected to work -- but the copied path was only ever verified for
    // an equal-length mapping, and a length the path never saw is worth recording rather than
    // assuming.
    {
        const std::vector<std::int32_t> before = ids_of(alloc);
        const std::uint32_t free_before        = pool.free_pages();
        std::vector<std::int32_t> mapping      = before;
        mapping.resize(before.size() + 2, ninfer::kPagedKVAquirePage);
        const std::vector<std::int32_t> deltas(mapping.size(), ninfer::kRopeDeltaNone);
        // One whole resident page copied into the first new destination, so the growth still
        // exercises the copier rather than only the acquire.
        const std::vector<ninfer::RelocatedCopySpan> spans{
            ninfer::RelocatedCopySpan{.source_begin = 1 * ninfer::kPagedKVPageSize,
                                      .dest_begin   = 4 * ninfer::kPagedKVPageSize,
                                      .tokens       = ninfer::kPagedKVPageSize}};

        const std::vector<std::byte> source_bytes = read_page(before[1]);

        alloc.relocate_copied_pages(mapping, deltas, spans, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        const std::vector<std::int32_t> published = ids_of(alloc);
        check(published.size() == mapping.size(),
              "C1 a growing relocation lengthens the mapping to the requested size");
        check(published[0] == before[0] && published[1] == before[1] &&
                  published[2] == before[2] && published[3] == before[3],
              "C1 the pages the mapping keeps are kept");
        check(published[4] != before[0] && published[4] != before[1] &&
                  published[4] != before[2] && published[4] != before[3] &&
                  published[5] != before[0] && published[5] != before[1] &&
                  published[5] != before[2] && published[5] != before[3],
              "C1 the new destinations are pool pages, not resident ones named twice");
        check(published[4] != published[5], "C1 a page is not mapped twice");
        check(read_page(published[4]) == source_bytes,
              "C3 the new destination holds the source page it was copied from, byte for byte");
        check(read_page(before[1]) == source_bytes,
              "C2 the source page is still itself after the copy");
        check(pool.free_pages() + published.size() == kPageGroups,
              "C5 the ownership partition balances after a growing relocation");
        check(pool.free_pages() == free_before - 2,
              "C5 both new destinations were bought from the pool and nothing else moved");
        note("growth: ok (" + std::to_string(published.size()) + " pages, " +
             std::to_string(pool.free_pages()) + " free)");
    }

    if (failures != 0) {
        std::cout << "PagedKVAllocation::relocate_copied_pages oracle checks FAILED (" << failures
                  << ")\n";
        return 1;
    }
    std::cout << "PagedKVAllocation::relocate_copied_pages oracle checks passed\n";
    return 0;
}

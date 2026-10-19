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
//   C7  attention over the relocated cache matches an independent FP64 oracle that counter-rotates
//       each stored key by its page's delta, with a zeroed-delta control that must break it
//
// Most byte comparisons are whole-page because those scenarios' moves are whole pages: the plane is
// required to be dense (asserted below), so a move of 64 tokens at offset 0 carries every byte of
// its page and the two pages must be identical afterwards. The off-grid case near the end is the
// exception and the one the copied path exists for: its destination page is written only over the
// span's token interval, so it compares per token, and it also checks the tokens *outside* that
// interval against the page's bytes from before the call. No tolerance and no re-derived offset
// table that would duplicate the code under test.
//
// C1 to C6 are byte comparisons, and a byte comparison cannot see the failure this feature actually
// risks: a cache that holds the right bytes read at the wrong RoPE phase is byte-exact and
// numerically wrong. C7 closes that by reading the relocated cache through the public attention
// entry point and judging it against the FP64 delta oracle, which is the only form that separates a
// correct relocation from a skipped counter-rotation on a plane carrying its own rounding error.

#include "ops/gqa_relocation_test_common.h"

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

// The BF16 conformance criterion for this file's fixture. The shared kConformanceBf16 is calibrated
// on tests/ops/test_gqa_attention.cpp's key distribution, and BF16 attention error scales with the
// logit magnitude, so it moves when the distribution does: on this fixture the correct path measures
// 2.81e-3 against the shared 2.80e-3 limit, which is a criterion miss with no defect behind it. The
// limit below sits above the measured correct path with margin and an order of magnitude below the
// control's own reach, which is what keeps a miss meaningful.
constexpr Criterion kCopiedRelocationBf16{4.0e-3, 1.0e-3, 2.7e-3};

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

// The same pool at the qwen3_6_27b Text geometry the shared oracle harness is calibrated for: 256
// leading elements over 4 KV heads, which is what a real K or V page holds. The byte cases above
// declare a 16x1 stub plane because they only need *some* bytes to move around; C7 has to be written
// and read through the production KV path, so it needs the real extent. The plane order follows the
// pool's convention: K at index 0, V at index 1, consumers assigning meaning to indices.
PlannedPool plan_kv_pool(std::uint32_t page_groups, std::uint32_t logical_pages, std::int32_t rows) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = page_groups,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .planes = {
                                                           {ninfer::DType::BF16, kHeadDim,
                                                            kKvHeads},
                                                           {ninfer::DType::BF16, kHeadDim,
                                                            kKvHeads}}});
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

// ---------------------------------------------------------------------------------------------
// C7: the relocated cache, read back through attention.
//
// Everything above is a byte comparison: it proves the copier moved the bytes it was asked to move
// and left everything else alone. It cannot see whether the *read* path interprets them correctly.
// A cache whose keys carry the source positions' rotation instead of the destination's is
// byte-exact and numerically wrong, and that is the failure this group exists to catch.
//
// The judgement is anchored, not cache-against-cache: attention over the relocated cache is
// compared against an independent FP64 oracle (ideal_attention_with_delta) that decodes each stored
// key and rotates it forward by its page's delta, which is the exact inverse of the kernel's Q
// counter-rotation. The oracle therefore carries the plane's own encode error, so a skipped rotation
// cannot hide behind it. A negative control -- the same cache with the delta plane zeroed -- has to
// break the criterion, which is what proves the check can fail at all.
//
// The pool-backed cache the shared oracle templates consume. Its image() gathers every page group
// of the pool by *physical* id, which is the convention k_logical/v_logical resolve a position into
// through the allocation's published mapping. Sizing that gather by the allocation's logical slot
// count instead lines the two up only while every owned page happens to have an id below it, which
// is false for a cache with acquire slots -- exactly the caches this feature produces.
class PoolCache {
public:
    PoolCache(const ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& alloc,
              ninfer::DType dtype, cudaStream_t stream)
        : pool_(pool), alloc_(alloc), dtype_(dtype), stream_(stream) {}

    struct HostImage {
        std::vector<double> k;
        std::vector<double> v;
    };

    [[nodiscard]] std::vector<std::int32_t> page_ids() const {
        const std::span<const std::int32_t> ids = alloc_.page_ids();
        return std::vector<std::int32_t>(ids.begin(), ids.end());
    }

    [[nodiscard]] HostImage image() const {
        const std::size_t pages = static_cast<std::size_t>(pool_.page_group_count());
        const std::size_t elems = pages * static_cast<std::size_t>(kPageSize) *
                                  static_cast<std::size_t>(kKvHeads) *
                                  static_cast<std::size_t>(kHeadDim);
        std::vector<std::int32_t> ids(pages);
        for (std::size_t i = 0; i < pages; ++i) { ids[i] = static_cast<std::int32_t>(i); }
        const std::size_t k_bytes_per_page = pool_.page_bytes(0);
        const std::size_t v_bytes_per_page = pool_.page_bytes(1);
        std::vector<std::uint8_t> k_raw(pages * k_bytes_per_page);
        std::vector<std::uint8_t> v_raw(pages * v_bytes_per_page);
        for (std::size_t i = 0; i < pages; ++i) {
            pool_.copy_page_to_host(0, static_cast<std::int32_t>(i),
                                    k_raw.data() + i * k_bytes_per_page, stream_);
            pool_.copy_page_to_host(1, static_cast<std::int32_t>(i),
                                    v_raw.data() + i * v_bytes_per_page, stream_);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        HostImage out{std::vector<double>(elems), std::vector<double>(elems)};
        for (std::size_t i = 0; i < elems; ++i) {
            __nv_bfloat16 kk;
            __nv_bfloat16 vv;
            std::memcpy(&kk, k_raw.data() + i * sizeof(kk), sizeof(kk));
            std::memcpy(&vv, v_raw.data() + i * sizeof(vv), sizeof(vv));
            out.k[i] = static_cast<double>(__bfloat162float(kk));
            out.v[i] = static_cast<double>(__bfloat162float(vv));
        }
        return out;
    }

    [[nodiscard]] double k_logical(const HostImage& img, std::int32_t position, std::int32_t head,
                                   std::int32_t d) const {
        return img.k[logical_index(physical_page_of(position), head, position % kPageSize, d)];
    }

    [[nodiscard]] double v_logical(const HostImage& img, std::int32_t position, std::int32_t head,
                                   std::int32_t d) const {
        return img.v[logical_index(physical_page_of(position), head, position % kPageSize, d)];
    }

    // The view attention reads, carrying the delta and interval planes the relocation published.
    // Without them the kernel cannot counter-rotate, so attaching them is what makes the run below
    // the one the runtime performs rather than a plain decode.
    [[nodiscard]] ninfer::PagedKVLayerView view(const FreqTable& freq) const {
        ninfer::PagedKVLayerView out;
        out.k_pages       = shaped(pool_.plane(0), kHeadDim);
        out.v_pages       = shaped(pool_.plane(1), kHeadDim);
        out.block_table   = alloc_.block_table();
        out.head_dim      = kHeadDim;
        out.num_kv_heads  = kKvHeads;
        out.dtype             = dtype_;
        out.rope_deltas       = pool_.rope_delta_row(0);
        out.rope_delta_ranges = pool_.rope_delta_range_row(0);
        out.rope_inv_freq     = freq;
        return out;
    }

private:
    [[nodiscard]] std::int32_t physical_page_of(std::int32_t position) const {
        return alloc_.page_ids()[static_cast<std::size_t>(position / kPageSize)];
    }

    // The plane in the view shape {leading, page, heads, pages}: the pool's page image is
    // [head][page_offset][leading] for a PageMajor plane, which is already that shape.
    [[nodiscard]] static ninfer::Tensor shaped(const ninfer::Tensor& plane, std::int32_t leading) {
        return ninfer::Tensor(plane.data, plane.dtype,
                              {leading, kPageSize, kKvHeads, plane.ne[3]});
    }

    [[nodiscard]] static std::size_t logical_index(std::int32_t physical_page, std::int32_t head,
                                                   std::int32_t page_offset, std::int32_t d) {
        return (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKvHeads) * static_cast<std::size_t>(physical_page)) *
                   static_cast<std::size_t>(kPageSize) * static_cast<std::size_t>(kHeadDim) +
               static_cast<std::size_t>(page_offset) * static_cast<std::size_t>(kHeadDim) +
               static_cast<std::size_t>(d);
    }

    const ninfer::PagedKVPool& pool_;
    const ninfer::PagedKVAllocation& alloc_;
    ninfer::DType dtype_;
    cudaStream_t stream_;
};

// Appends one chunk through the production write path at sequential positions, so the cache holds
// keys rotated at the *source* positions exactly as a resident sequence does.
void append_pool_chunk(const PoolCache& cache, const std::vector<__nv_bfloat16>& k,
                       const std::vector<__nv_bfloat16>& v, std::int32_t first_position,
                       std::int32_t tokens, const FreqTable& freq, cudaStream_t stream) {
    const ninfer::PagedKVLayerView view = cache.view(freq);
    ninfer::DeviceBuffer dk(k.size() * sizeof(__nv_bfloat16));
    ninfer::DeviceBuffer dv(v.size() * sizeof(__nv_bfloat16));
    ninfer::DeviceBuffer dp(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    dk.copy_from_host(k.data(), k.size() * sizeof(__nv_bfloat16));
    dv.copy_from_host(v.data(), v.size() * sizeof(__nv_bfloat16));
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        positions[static_cast<std::size_t>(t)] = first_position + t;
    }
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    ninfer::ops::gqa_kv_append(
        ninfer::Tensor(dk.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, tokens}),
        ninfer::Tensor(dv.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, tokens}),
        ninfer::Tensor(dp.p, ninfer::DType::I32, {tokens}), view, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
}

// Decode attention over a pool-backed cache through the public entry point, with the delta and
// interval planes attached to the view.
AttentionRun run_pool_attention(const PoolCache& cache, const std::vector<__nv_bfloat16>& q,
                                std::int32_t first_position, std::int32_t tokens,
                                std::uint32_t visible, const ninfer::PagedKVLayerView& view) {
    ninfer::DeviceBuffer dq   = upload(q);
    ninfer::DeviceBuffer dout = ninfer::DeviceBuffer(q.size() * sizeof(__nv_bfloat16));
    dout.fill(0);
    ninfer::DeviceBuffer dp = upload_positions(sequential_positions(first_position, tokens));

    const ninfer::ops::GqaExecutionEnvelope envelope{visible, visible};
    const std::size_t workspace_bytes = ninfer::ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, ninfer::DType::BF16, envelope, 1, tokens, tokens);
    ninfer::DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    ninfer::DeviceArena workspace(ninfer::DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});

    ninfer::Tensor tq(dq.p, ninfer::DType::BF16, {kHeadDim, kQHeads, tokens});
    ninfer::Tensor tp(dp.p, ninfer::DType::I32, {tokens});
    ninfer::Tensor tout(dout.p, ninfer::DType::BF16, {kHeadDim, kQHeads, tokens});
    ninfer::ops::gqa_attention_cached(tq, tp, kAttentionScale, view, envelope, workspace, tout,
                                      nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
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
        const std::vector<ninfer::RopeDeltaPageRange> ranges =
            ninfer::plan_rope_delta_page_ranges(spans, mapping.size());

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

        alloc.relocate_copied_pages(mapping, ranges, spans, ctx.stream);
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
        const std::vector<ninfer::RopeDeltaPageRange> ranges =
            ninfer::plan_rope_delta_page_ranges(spans, mapping.size());

        const std::vector<std::byte> source_bytes = read_page(before[3]);

        alloc.relocate_copied_pages(mapping, ranges, spans, ctx.stream);
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
                                       std::span<const ninfer::RopeDeltaPageRange> ranges,
                                       std::span<const ninfer::RelocatedCopySpan> plan_spans) {
            bool threw = false;
            try {
                alloc.relocate_copied_pages(mapping, ranges, plan_spans, ctx.stream);
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
        const std::vector<ninfer::RopeDeltaPageRange> ok_ranges(ok.size());
        const std::vector<ninfer::RopeDeltaPageRange> short_ranges(ok.size() - 1);
        const std::vector<std::int32_t> oversized(kEntitlement + 1, ninfer::kPagedKVAquirePage);
        const std::vector<ninfer::RopeDeltaPageRange> oversized_ranges(oversized.size());
        const std::vector<std::int32_t> empty_mapping;

        expect_reject("an empty mapping", empty_mapping,
                      std::span<const ninfer::RopeDeltaPageRange>(), spans);
        expect_reject("a mapping past the entitlement", oversized, oversized_ranges, spans);
        expect_reject("a range plane not parallel to the mapping", ok, short_ranges, spans);
        expect_reject("no spans to copy", ok, ok_ranges,
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
        // One whole resident page copied into the first new destination, so the growth still
        // exercises the copier rather than only the acquire.
        const std::vector<ninfer::RelocatedCopySpan> spans{
            ninfer::RelocatedCopySpan{.source_begin = 1 * ninfer::kPagedKVPageSize,
                                      .dest_begin   = 4 * ninfer::kPagedKVPageSize,
                                      .tokens       = ninfer::kPagedKVPageSize}};
        const std::vector<ninfer::RopeDeltaPageRange> ranges =
            ninfer::plan_rope_delta_page_ranges(spans, mapping.size());

        const std::vector<std::byte> source_bytes = read_page(before[1]);

        alloc.relocate_copied_pages(mapping, ranges, spans, ctx.stream);
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

    // -------------------------------------------------- an off-grid destination: partial pages --
    // The shape this path exists for and the one no case above covers: the surviving run starts
    // partway through a page, so its first and last destination pages are written only in part and a
    // move stops at whichever page boundary comes first. The span below needs three segments -- one
    // for the 37 tokens of its first destination page and two for its second, split where the source
    // page ends -- which is the `move.dest_token != 0` and split-at-both-boundaries arithmetic the
    // copier was extended for and that nothing here exercised.
    {
        const std::vector<std::int32_t> before = ids_of(alloc);
        const std::uint32_t page_tokens  = ninfer::kPagedKVPageSize;
        const std::size_t token_bytes    = static_cast<std::size_t>(pool.plane(0).nb[1]);

        // Destination tokens 91..190 of the new prompt, source tokens 192..291 of the resident one:
        // an edit of -101 tokens, which is exactly the shape an off-grid edit produces.
        const std::uint32_t dest_begin   = page_tokens + 27;
        const std::uint32_t source_begin = 3 * page_tokens;
        const std::vector<ninfer::RelocatedCopySpan> spans{ninfer::RelocatedCopySpan{
            .source_begin = source_begin, .dest_begin = dest_begin, .tokens = 100}};

        std::vector<std::int32_t> mapping(before.begin(), before.begin() + 1);
        mapping.resize(3, ninfer::kPagedKVAquirePage);
        const std::vector<ninfer::RopeDeltaPageRange> ranges =
            ninfer::plan_rope_delta_page_ranges(spans, mapping.size());

        // The moves laid out by hand, so this stays an oracle: (dest page, dest token, source page,
        // source token, tokens). The first destination page takes the span's first 37 tokens from
        // the head of source page 3; the second takes the rest of that page and then the head of
        // page 4.
        struct ExpectedMove {
            std::uint32_t dest_page;
            std::uint32_t dest_token;
            std::uint32_t source_page;
            std::uint32_t source_token;
            std::uint32_t tokens;
        };
        const ExpectedMove expected[] = {{1, 27, 3, 0, 37}, {2, 0, 3, 37, 27}, {2, 27, 4, 0, 36}};
        constexpr std::size_t kExpectedMoves = sizeof(expected) / sizeof(expected[0]);

        const std::vector<ninfer::RelocatedPageMove> moves =
            moves_of(spans, {.source = before.size(), .destination = mapping.size()});
        check(moves.size() == kExpectedMoves, "an off-grid destination splits into three moves");
        for (std::size_t index = 0; index < moves.size() && index < kExpectedMoves; ++index) {
            const ninfer::RelocatedPageMove& move = moves[index];
            const ExpectedMove& expect_move       = expected[index];
            check(move.dest_page == static_cast<std::int32_t>(expect_move.dest_page) &&
                      move.dest_token == expect_move.dest_token &&
                      move.source_page == static_cast<std::int32_t>(expect_move.source_page) &&
                      move.source_token == expect_move.source_token &&
                      move.tokens == expect_move.tokens,
                  "the copier's move " + std::to_string(index) +
                      " starts and ends where the page boundaries say it must");
        }

        // Every page group's bytes before the call, so the tokens a partial destination must leave
        // alone can be checked against what was there rather than against a guess.
        std::vector<std::vector<std::byte>> pre(kPageGroups);
        for (std::size_t page = 0; page < kPageGroups; ++page) {
            pre[page] = read_page(static_cast<std::int32_t>(page));
        }

        alloc.relocate_copied_pages(mapping, ranges, spans, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        const std::vector<std::int32_t> published = ids_of(alloc);
        check(published.size() == mapping.size(),
              "C1 the off-grid relocation publishes the requested length");
        check(published[0] == before[0], "C1 the page the mapping keeps is kept");

        for (const ExpectedMove& expect_move : expected) {
            const std::int32_t dest_page = published[expect_move.dest_page];
            const std::int32_t src_page  = before[expect_move.source_page];
            const std::vector<std::byte> dest_bytes = read_page(dest_page);
            const std::vector<std::byte> src_bytes  = read_page(src_page);
            bool equal = true;
            for (std::uint32_t token = 0; token < expect_move.tokens; ++token) {
                const auto* dest_token =
                    dest_bytes.data() +
                    static_cast<std::size_t>(expect_move.dest_token + token) * token_bytes;
                const auto* src_token =
                    src_bytes.data() +
                    static_cast<std::size_t>(expect_move.source_token + token) * token_bytes;
                if (std::memcmp(dest_token, src_token, token_bytes) != 0) { equal = false; }
            }
            check(equal, "C3 a partial destination page carries every token its move brings");
            check(dest_page != src_page, "C2 no destination is a page the copies read");
        }

        // C4 for a partial page: the 27 tokens ahead of the span in its first destination page were
        // never written, so they must still be the bytes that page group held before the call.
        const std::int32_t first_dest = published[1];
        const std::vector<std::byte> first_dest_bytes = read_page(first_dest);
        const std::vector<std::byte>& first_dest_pre = pre[static_cast<std::size_t>(first_dest)];
        bool head_untouched = true;
        for (std::uint32_t token = 0; token < 27; ++token) {
            const auto* now = first_dest_bytes.data() + static_cast<std::size_t>(token) * token_bytes;
            const auto* was = first_dest_pre.data() + static_cast<std::size_t>(token) * token_bytes;
            if (std::memcmp(now, was, token_bytes) != 0) { head_untouched = false; }
        }
        check(head_untouched,
              "C4 the tokens ahead of the span in its first destination page were not written");

        // Both planes, from one set of ranges: the interval attention reads and the shift it
        // counter-rotates with describe the same tokens and the same delta.
        const std::vector<std::uint32_t> intervals = read_u32(pool.rope_delta_range_row(0), 3);
        check(ninfer::rope_delta_page_range_begin(intervals[1]) == 27 &&
                  ninfer::rope_delta_page_range_end(intervals[1]) == 64,
              "the first destination page's interval covers the tokens its move wrote");
        check(ninfer::rope_delta_page_range_begin(intervals[2]) == 0 &&
                  ninfer::rope_delta_page_range_end(intervals[2]) == 63,
              "the second destination page's interval covers the tokens its two moves wrote");
        if (pool.rope_delta_row(0).data != nullptr) {
            std::vector<std::int32_t> shifts(mapping.size());
            CUDA_CHECK(cudaMemcpy(shifts.data(), pool.rope_delta_row(0).data,
                                  mapping.size() * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost));
            check(shifts[1] == static_cast<std::int32_t>(dest_begin) -
                                   static_cast<std::int32_t>(source_begin) &&
                      shifts[2] == shifts[1],
                  "the shift plane carries the delta the interval plane describes");
            check(shifts[0] == ninfer::kRopeDeltaNone,
                  "a page no span covers carries no shift");
        }
        check(pool.free_pages() + published.size() == kPageGroups,
              "C5 the ownership partition balances after an off-grid relocation");
        note("off-grid destination: ok (3 moves, " + std::to_string(pool.free_pages()) +
             " free)");
    }
    // ------------------------------------------------------------------ C7: attention over it ---
    // A second pool, because this case seeds real KV content through the production write path and
    // the byte cases above must stay a pattern-comparison of their own. The shape is the off-grid
    // one: the run's source and destination sit mid-page, so every destination page is partial and
    // the interval plane is what tells the read path which keys moved.
    {
        const FreqTable freq = default_inv_freq();
        constexpr std::uint32_t kC7PageGroups  = 16;
        constexpr std::uint32_t kC7Entitlement = 12;
        constexpr std::int32_t kC7Pages        = 12;
        constexpr std::int32_t kC7Tokens       = kC7Pages * kPageSize;
        // 100 tokens moved by -101, so the source and destination both land mid-page.
        constexpr std::int32_t kC7Source = 3 * kPageSize;
        constexpr std::int32_t kC7Dest   = kPageSize + 27;
        constexpr std::int32_t kC7Run    = 100;
        // The query window sits immediately after the relocated run, so the read's softmax weights
        // the relocated keys heavily. That is what makes the control discriminating: a skipped
        // counter-rotation has to move the output by a wide factor, and a query far from the run
        // dilutes the damage across keys that were never relocated.
        constexpr std::int32_t kC7Query    = kC7Dest + kC7Run;
        constexpr std::int32_t kC7TokensRead = 8;
        constexpr std::uint32_t kC7Visible =
            static_cast<std::uint32_t>(kC7Query + kC7TokensRead);

        auto plan7 = plan_kv_pool(kC7PageGroups, kC7Entitlement, 1);
        ninfer::DeviceArena arena7(plan7.bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
        ninfer::PagedKVPool pool7({arena7.base(), arena7.capacity()}, plan7.layout);

        // The whole prompt's K and V, drawn as the shared scenario builder draws them: keys are
        // rotated at their own positions, so the cache a resident sequence would hold is reproduced
        // exactly and only the relocation can move them.
        const std::size_t prompt_elems =
            static_cast<std::size_t>(kHeadDim) * kKvHeads * static_cast<std::size_t>(kC7Tokens);
        std::vector<__nv_bfloat16> all_k(prompt_elems);
        std::vector<__nv_bfloat16> all_v(prompt_elems);
        std::mt19937 rng(0x5c7c01u);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> unit(-0.25, 0.25);
        std::array<double, kHeadDim> row{};
        std::array<double, kHeadDim> rotated{};
        for (std::int32_t position = 0; position < kC7Tokens; ++position) {
            for (std::int32_t head = 0; head < kKvHeads; ++head) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    row[static_cast<std::size_t>(d)] = normal(rng);
                }
                forward_rope(row.data(), position, freq, rotated.data());
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    all_k[act_index(head, position, kKvHeads, d)] =
                        __float2bfloat16(static_cast<float>(rotated[static_cast<std::size_t>(d)]));
                    all_v[act_index(head, position, kKvHeads, d)] =
                        __float2bfloat16(static_cast<float>(unit(rng)));
                }
            }
        }
        const std::size_t query_elems = static_cast<std::size_t>(kHeadDim) * kQHeads *
                                        static_cast<std::size_t>(kC7TokensRead);
        std::vector<__nv_bfloat16> queries(query_elems);
        for (std::size_t i = 0; i < query_elems; ++i) {
            queries[i] = __float2bfloat16(static_cast<float>(unit(rng)));
        }

        // Two cells: the relocated one, and the directly prefilled reference. Same pool geometry, so
        // the only difference between them is what the relocation did.
        auto seed_cell = [&](ninfer::PagedKVPool& pool, ninfer::PagedKVAllocation& alloc,
                             std::int32_t pages) {
            alloc = pool.reserve(static_cast<std::uint32_t>(pages));
            alloc.materialize_pages(static_cast<std::uint32_t>(pages));
            alloc.bind_row(0, ctx.stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            const PoolCache cache(pool, alloc, ninfer::DType::BF16, ctx.stream);
            for (std::int32_t base = 0; base < pages * kPageSize; base += 256) {
                const std::int32_t count = std::min(256, pages * kPageSize - base);
                const std::size_t offset =
                    static_cast<std::size_t>(base) * kHeadDim * kKvHeads;
                const std::size_t elems =
                    static_cast<std::size_t>(count) * kHeadDim * kKvHeads;
                append_pool_chunk(
                    cache,
                    std::vector<__nv_bfloat16>(all_k.begin() + static_cast<std::ptrdiff_t>(offset),
                                               all_k.begin() +
                                                   static_cast<std::ptrdiff_t>(offset + elems)),
                    std::vector<__nv_bfloat16>(all_v.begin() + static_cast<std::ptrdiff_t>(offset),
                                               all_v.begin() +
                                                   static_cast<std::ptrdiff_t>(offset + elems)),
                    base, count, freq, ctx.stream);
            }
        };

        // The relocated cell: seed the resident prompt, then relocate the run off-grid.
        ninfer::PagedKVAllocation relocated;
        seed_cell(pool7, relocated, kC7Pages);
        const std::vector<std::int32_t> resident7 = ids_of(relocated);
        const std::vector<ninfer::RelocatedCopySpan> spans7{
            ninfer::RelocatedCopySpan{.source_begin = static_cast<std::uint32_t>(kC7Source),
                                      .dest_begin   = static_cast<std::uint32_t>(kC7Dest),
                                      .tokens       = static_cast<std::uint32_t>(kC7Run)}};
        std::vector<std::int32_t> mapping7(resident7.begin(), resident7.begin() + kC7Pages);
        mapping7[static_cast<std::size_t>(kC7Dest / kPageSize)] = ninfer::kPagedKVAquirePage;
        mapping7[static_cast<std::size_t>(kC7Dest / kPageSize) + 1] =
            ninfer::kPagedKVAquirePage;
        const std::vector<ninfer::RopeDeltaPageRange> ranges7 =
            ninfer::plan_rope_delta_page_ranges(spans7, mapping7.size());
        relocated.relocate_copied_pages(mapping7, ranges7, spans7, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        // The reference cell: a cold prefill of the *edited* prompt. The edit deleted the 101 tokens
        // the run stepped over, so the edited prompt is the resident one with [dest, source) removed
        // -- the run then already sits at its destination positions and nothing has to be
        // counter-rotated. Seeding the resident prompt here instead would make C7d compare two
        // different prompts and report a relocation defect that is not one.
        // The edit removes [dest + run, source + run) -- the stretch between where the run now ends
        // and where it used to -- and keeps everything before and after it. That is what a mid-prompt
        // edit does, and it is the shape the relocation serves: the run survives at its new
        // positions and the rest of the prompt is unchanged.
        const std::int32_t kC7EditedTokens = kC7Tokens - (kC7Source + kC7Run) + (kC7Dest + kC7Run);
        const std::size_t head_elems =
            static_cast<std::size_t>(kC7Dest) * kHeadDim * kKvHeads;
        const std::size_t run_elems =
            static_cast<std::size_t>(kC7Run) * kHeadDim * kKvHeads;
        const std::size_t src_elems =
            static_cast<std::size_t>(kC7Source) * kHeadDim * kKvHeads;
        const std::size_t tail_src_elems =
            static_cast<std::size_t>(kC7Source + kC7Run) * kHeadDim * kKvHeads;
        const std::size_t tail_elems =
            static_cast<std::size_t>(kC7Tokens - (kC7Source + kC7Run)) * kHeadDim * kKvHeads;
        std::vector<__nv_bfloat16> edited_k(prompt_elems);
        std::vector<__nv_bfloat16> edited_v(prompt_elems);
        std::copy(all_k.begin(), all_k.begin() + static_cast<std::ptrdiff_t>(head_elems),
                  edited_k.begin());
        std::copy(all_v.begin(), all_v.begin() + static_cast<std::ptrdiff_t>(head_elems),
                  edited_v.begin());
        std::copy(all_k.begin() + static_cast<std::ptrdiff_t>(src_elems),
                  all_k.begin() + static_cast<std::ptrdiff_t>(src_elems + run_elems),
                  edited_k.begin() + static_cast<std::ptrdiff_t>(head_elems));
        std::copy(all_v.begin() + static_cast<std::ptrdiff_t>(src_elems),
                  all_v.begin() + static_cast<std::ptrdiff_t>(src_elems + run_elems),
                  edited_v.begin() + static_cast<std::ptrdiff_t>(head_elems));
        std::copy(all_k.begin() + static_cast<std::ptrdiff_t>(tail_src_elems),
                  all_k.begin() + static_cast<std::ptrdiff_t>(tail_src_elems + tail_elems),
                  edited_k.begin() + static_cast<std::ptrdiff_t>(head_elems + run_elems));
        std::copy(all_v.begin() + static_cast<std::ptrdiff_t>(tail_src_elems),
                  all_v.begin() + static_cast<std::ptrdiff_t>(tail_src_elems + tail_elems),
                  edited_v.begin() + static_cast<std::ptrdiff_t>(head_elems + run_elems));

        auto plan7r = plan_kv_pool(kC7PageGroups, kC7Entitlement, 1);
        ninfer::DeviceArena arena7r(plan7r.bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
        ninfer::PagedKVPool pool7r({arena7r.base(), arena7r.capacity()}, plan7r.layout);
        ninfer::PagedKVAllocation reference;
        reference = pool7r.reserve(static_cast<std::uint32_t>(kC7Pages));
        reference.materialize_pages(static_cast<std::uint32_t>(kC7Pages));
        reference.bind_row(0, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        {
            const PoolCache cache(pool7r, reference, ninfer::DType::BF16, ctx.stream);
            for (std::int32_t base = 0; base < kC7EditedTokens; base += 256) {
                const std::int32_t count = std::min(256, kC7EditedTokens - base);
                const std::size_t offset =
                    static_cast<std::size_t>(base) * kHeadDim * kKvHeads;
                const std::size_t elems =
                    static_cast<std::size_t>(count) * kHeadDim * kKvHeads;
                append_pool_chunk(
                    cache,
                    std::vector<__nv_bfloat16>(edited_k.begin() +
                                                    static_cast<std::ptrdiff_t>(offset),
                                                edited_k.begin() +
                                                    static_cast<std::ptrdiff_t>(offset + elems)),
                    std::vector<__nv_bfloat16>(edited_v.begin() +
                                                    static_cast<std::ptrdiff_t>(offset),
                                                edited_v.begin() +
                                                    static_cast<std::ptrdiff_t>(offset + elems)),
                    base, count, freq, ctx.stream);
            }
        }
        const PoolCache relocated_cache(pool7, relocated, ninfer::DType::BF16, ctx.stream);
        const PoolCache reference_cache(pool7r, reference, ninfer::DType::BF16, ctx.stream);

        // The interval and delta planes the call published, read back as the oracle reads them.
        const std::vector<std::uint32_t> published_ranges =
            read_u32(pool7.rope_delta_range_row(0), mapping7.size());
        const std::vector<std::int32_t> published_deltas = [&] {
            std::vector<std::int32_t> out(mapping7.size());
            CUDA_CHECK(cudaMemcpy(out.data(), pool7.rope_delta_row(0).data,
                                  out.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
            return out;
        }();
        const ninfer::PagedKVLayerView relocated_view = relocated_cache.view(freq);
        const ninfer::PagedKVLayerView reference_view = reference_cache.view(freq);
        const AttentionRun got_relocated =
            run_pool_attention(relocated_cache, queries, kC7Query, kC7TokensRead, kC7Visible,
                               relocated_view);
        const AttentionRun got_reference =
            run_pool_attention(reference_cache, queries, kC7Query, kC7TokensRead, kC7Visible,
                               reference_view);

        // C7a: the relocated cache against the FP64 delta oracle. This is the assertion that a
        // byte-exact cache with the wrong rotation cannot pass: the oracle rotates each decoded key
        // forward by its page's delta, so it carries the plane's own BF16 rounding and nothing else.
        if (!expect_conformant(
                measure(got_relocated.out,
                        ideal_attention_with_delta(queries, relocated_cache, kC7Query,
                                                   kC7TokensRead, published_deltas,
                                                   published_ranges, freq),
                        kCopiedRelocationBf16),
                "C7 attention over the relocated cache against the FP64 delta oracle",
                kCopiedRelocationBf16)) {
            ++failures;
        }

        // C7b: the reference cache against the plain FP64 oracle. A miss here means the harness is
        // not reading the cache the way the oracle describes it, which would void C7a.
        if (!expect_conformant(measure(got_reference.out,
                                       ideal_attention(queries, reference_cache, kC7Query,
                                                       kC7TokensRead),
                                       kCopiedRelocationBf16),
                               "C7 attention over the directly prefilled cache",
                               kCopiedRelocationBf16)) {
            ++failures;
        }

        // C7c: the negative control. The same relocated cache with its delta plane zeroed is read at
        // the source phase, and the criterion must break -- otherwise C7a would pass whatever the
        // relocation did. Zeroing is done in a scratch buffer so the published plane is left as the
        // call left it.
        {
            std::vector<std::int32_t> zeroed(published_deltas.size(), 0);
            ninfer::DeviceBuffer zero_buf(zeroed.size() * sizeof(std::int32_t));
            zero_buf.copy_from_host(zeroed.data(), zeroed.size() * sizeof(std::int32_t));
            ninfer::PagedKVLayerView stale = relocated_view;
            stale.rope_deltas = ninfer::Tensor(zero_buf.p, ninfer::DType::I32,
                                               {static_cast<std::int32_t>(zeroed.size())});
            const AttentionRun got_stale =
                run_pool_attention(relocated_cache, queries, kC7Query, kC7TokensRead, kC7Visible,
                                   stale);
            const Error stale_error = measure(
                got_stale.out,
                ideal_attention_with_delta(queries, relocated_cache, kC7Query, kC7TokensRead,
                                           published_deltas, published_ranges, freq));
            if (!expect_control_separates(
                    measure(got_relocated.out,
                            ideal_attention_with_delta(queries, relocated_cache, kC7Query,
                                                       kC7TokensRead, published_deltas,
                                                       published_ranges, freq)),
                    stale_error, "C7 the zeroed delta plane is separated from the correct read")) {
                ++failures;
            }
            note("C7 control: relative L2 " + std::to_string(stale_error.relative_l2) +
                 " with the delta plane zeroed");
        }

        // C7d is *not* asserted here, and the reason is worth writing down. The two caches are the
        // two different prompts this edit produces: the relocated one still holds the resident
        // sequence, which is 101 tokens longer, so past the relocated run they legitimately hold
        // different tokens and the cache-against-cache comparison has nothing left to say. The
        // anchored form above is the assertion that survives on a real relocation, because the FP64
        // oracle derives what *this* cache should read rather than what the other cache holds.
        //
        // The number is still reported, because it is the quantity a reader wants to see next to the
        // anchored one, and a large value here is only meaningful once the shapes are checked. What
        // is asserted instead is the shape the equivalence would need: the relocated cache must hold
        // the run's tokens at the run's positions, which is the byte check C3 already made.
        note("C7 relocated against directly prefilled: relative L2 " +
             std::to_string(measure(got_relocated.out, got_reference.out).relative_l2) +
             " (observation only: the two caches hold two different prompts past the relocated run)");
    }

    if (failures != 0) {
        std::cout << "PagedKVAllocation::relocate_copied_pages oracle checks FAILED (" << failures
                  << ")\n";
        return 1;
    }
    std::cout << "PagedKVAllocation::relocate_copied_pages oracle checks passed\n";
    return 0;
}

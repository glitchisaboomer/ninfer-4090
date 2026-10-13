#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct PlannedPagedCache {
    ninfer::PagedKVPoolLayout layout;
    std::size_t bytes = 0;
};

PlannedPagedCache
plan_paged_cache(std::uint32_t pages, std::uint32_t logical_pages, std::int32_t rows,
                 std::vector<ninfer::PagedKVPlaneSpec> planes,
                 ninfer::PagedKVPlaneOrder order = ninfer::PagedKVPlaneOrder::PageMajor) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = pages,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .plane_order           = order,
                                                       .planes                = std::move(planes)});
    return PlannedPagedCache{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& tensor, const std::int32_t (&expected)[4],
                const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got "
                      << tensor.ne[i] << '\n';
        }
    }
    return failures;
}

int expect_page_ids(std::span<const std::int32_t> actual,
                    std::initializer_list<std::int32_t> expected, const char* label) {
    if (actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) {
        return 0;
    }
    std::cerr << label << " page IDs differ\n";
    return 1;
}

int expect_device_page_ids(const ninfer::Tensor& row, std::initializer_list<std::int32_t> expected,
                           const char* label) {
    std::vector<std::int32_t> actual(expected.size());
    const cudaError_t err = cudaMemcpy(
        actual.data(), row.data, actual.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    return expect_page_ids(actual, expected, label);
}

// The RoPE delta plane is the one tensor in this pool whose reader has no other source of truth: if
// `clear_rope_delta_row` zeroes the wrong bytes, attention silently counter-rotates a page that was
// never relocated. That is why the host test pins the arithmetic and this one pins the effect.
//
// It fills the whole plane with a non-zero pattern, clears one row, and requires that *exactly* that
// row changed and every other row is byte-identical to before. Checking only "row 0 is zero now"
// would pass with a stride that also clobbers row 1, which is the bug the arithmetic was extracted to
// prevent.
int expect_rope_delta_row_clear(ninfer::PagedKVPool& pool, std::int32_t row, cudaStream_t stream,
                                const char* label) {
    const ninfer::Tensor& deltas = pool.rope_deltas();
    const ninfer::Tensor& ranges = pool.rope_delta_ranges();
    if (deltas.ne[0] <= 0 || deltas.ne[1] != pool.table_row_count()) {
        std::cerr << label << " plane shape is inconsistent\n";
        return 1;
    }
    if (ranges.ne[0] != deltas.ne[0] || ranges.ne[1] != deltas.ne[1]) {
        std::cerr << label << " range plane shape is inconsistent\n";
        return 1;
    }
    const std::size_t rows      = static_cast<std::size_t>(deltas.ne[1]);
    const std::size_t row_bytes = static_cast<std::size_t>(deltas.ne[0]) * sizeof(std::int32_t);

    // A non-zero fill: 0xCC as int32, so every entry reads as a plausible-looking shift rather than
    // as an obviously-poisoned value.
    std::vector<std::int32_t> pattern(deltas.ne[0] * deltas.ne[1], 0);
    const std::int32_t poison = static_cast<std::int32_t>(0xCCCCCC00u);
    std::fill(pattern.begin(), pattern.end(), poison);
    cudaError_t err = cudaMemcpyAsync(deltas.data, pattern.data(), pattern.size() * sizeof(std::int32_t),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << label << " fill failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    std::vector<std::uint32_t> range_pattern(pattern.size(), 0xccccccccu);
    err = cudaMemcpyAsync(ranges.data, range_pattern.data(),
                          range_pattern.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) {
        std::cerr << label << " range fill failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " fill sync failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    pool.clear_rope_delta_row(row, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " clear sync failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::vector<std::int32_t> after(pattern.size(), 0);
    err = cudaMemcpy(after.data(), deltas.data, after.size() * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " readback failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    std::vector<std::uint32_t> range_after(range_pattern.size(), 0);
    err = cudaMemcpy(range_after.data(), ranges.data, range_after.size() * sizeof(std::uint32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " range readback failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    int failures = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::int32_t p = 0; p < deltas.ne[0]; ++p) {
            const std::size_t index = r * static_cast<std::size_t>(deltas.ne[0]) +
                                      static_cast<std::size_t>(p);
            const bool cleared = r == static_cast<std::size_t>(row);
            const std::int32_t want = cleared ? 0 : poison;
            if (after[index] != want) {
                if (failures < 4) {
                    std::cerr << label << " row " << r << " page " << p << " is " << after[index]
                              << ", expected " << want << '\n';
                }
                ++failures;
            }
            const std::uint32_t range_want = cleared ? ninfer::kRopeDeltaRangeEmpty : 0xccccccccu;
            if (range_after[index] != range_want) {
                if (failures < 4) {
                    std::cerr << label << " row " << r << " page " << p
                              << " has range " << range_after[index] << ", expected "
                              << range_want << '\n';
                }
                ++failures;
            }
        }
    }
    if (failures != 0) {
        std::cerr << label << " " << failures << " of " << pattern.size()
                  << " entries wrong after clearing row " << row << " (row is " << row_bytes
                  << " bytes)\n";
    }
    return failures == 0 ? 0 : 1;
}

// A row is recycled between sequences. The delta plane must not carry a previous sequence's
// relocations into the next one's attention: the failure mode is a query counter-rotated by a
// stale shift for a page that was never moved, which no type error catches and no test that
// ignores the recycling sequence will see.
//
// This test exercises the sequence: write deltas into row 0 -> release the row -> bind a fresh
// allocation to row 0 -> assert row 0's deltas are all zero *before* anything writes them.
int expect_rope_delta_recycling(ninfer::PagedKVPool& pool, cudaStream_t stream, const char* label) {
    const ninfer::Tensor& deltas = pool.rope_deltas();
    const std::int32_t page_count = deltas.ne[0];
    if (page_count <= 0 || pool.table_row_count() < 1) {
        std::cerr << label << " pool shape is inconsistent for recycling test\n";
        return 1;
    }

    // 1. Reserve the whole row's worth of entitlement and materialize that many distinct pages.
    //    The pool this test runs against has page_group_count == logical_page_capacity == 10
    //    (see `plan_paged_cache(10, 10, 2, ...)` in main), and no earlier test in main holds
    //    entitlement on paged_pool at the point this function runs, so reserving `page_count`
    //    pages succeeds. The distinct-pages requirement is not optional: `reassign_pages` rejects
    //    a mapping that names the same owned physical page more than once.
    const std::uint32_t entitlement = static_cast<std::uint32_t>(page_count);
    auto first = pool.reserve(entitlement);
    first.materialize_pages(entitlement, stream);
    first.bind_row(0, stream);

    // 2. Write nonzero deltas into every logical page of row 0, one distinct physical page each.
    //    `first.page_ids()` is now exactly `page_count` distinct pages, so it can be used as the
    //    mapping directly — no repetition, no padding.
    std::vector<std::int32_t> mapping(first.page_ids().begin(), first.page_ids().end());
    if (static_cast<std::int32_t>(mapping.size()) != page_count) {
        std::cerr << label << " allocation materialized " << mapping.size()
                  << " pages, expected " << page_count << '\n';
        return 1;
    }
    std::vector<std::int32_t> deltas_in(static_cast<std::size_t>(page_count), 17);
    first.reassign_pages(mapping, deltas_in, stream);
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " reassign sync failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    // 3. Read back and confirm the row is all-17. If this fails, the test cannot say anything
    //    about recycling.
    auto read_row = [&](std::vector<std::int32_t>& out) -> cudaError_t {
        out.assign(static_cast<std::size_t>(page_count), 0);
        ninfer::Tensor row = pool.rope_delta_row(0);
        return cudaMemcpy(out.data(), row.data,
                          static_cast<std::size_t>(page_count) * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost);
    };
    std::vector<std::int32_t> after_write;
    err = read_row(after_write);
    if (err != cudaSuccess) {
        std::cerr << label << " first readback failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    for (std::int32_t p = 0; p < page_count; ++p) {
        if (after_write[p] != 17) {
            std::cerr << label << " precondition failed: row 0 page " << p << " is "
                      << after_write[p] << ", expected 17\n";
            return 1;
        }
    }

    // 4. Release the row: unbind the allocation and drop it.
    first.unbind_row();
    first.release();

    // 5. Bind a *different* allocation to the same row. bind_row is what must clear the row.
    auto second = pool.reserve(1);
    second.bind_row(0, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " second bind sync failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    // 6. Read back. Every entry must be zero.
    std::vector<std::int32_t> after_rebind;
    err = read_row(after_rebind);
    if (err != cudaSuccess) {
        std::cerr << label << " second readback failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    int failures = 0;
    for (std::int32_t p = 0; p < page_count; ++p) {
        if (after_rebind[p] != 0) {
            if (failures < 4) {
                std::cerr << label << " recycled row 0 page " << p << " is " << after_rebind[p]
                          << ", expected 0\n";
            }
            ++failures;
        }
    }
    second.unbind_row();
    second.release();
    return failures == 0 ? 0 : 1;
}

// Step 3a of docs/SCR_QUANTIZED_KV_RELOCATION.md: after a reassign that carries a delta
// plane, the plane must hold the shift for each relocated page and kRopeDeltaNone
// elsewhere. This is the pool-side half of the writer; the runtime-side half (the loop
// that computes the deltas from relocated spans) is in program_impl.h and needs a GPU.
//
// What this test can prove without a GPU is that `reassign_pages` stores and publishes
// what it was handed, and that non-relocated pages default to zero.
int expect_reassign_publishes_deltas(ninfer::PagedKVPool& pool, cudaStream_t stream,
                                     const char* label) {
    const ninfer::Tensor& deltas = pool.rope_deltas();
    const std::int32_t page_count = deltas.ne[0];
    if (page_count < 4 || pool.table_row_count() < 1) {
        std::cerr << label << " pool is too small for this test\n";
        return 1;
    }

    // Reserve enough distinct pages to map every logical page, then bind to row 0.
    // Requires page_group_count >= logical_page_capacity (10 for the pool in main).
    const std::uint32_t entitlement = static_cast<std::uint32_t>(page_count);
    auto allocation = pool.reserve(entitlement);
    allocation.materialize_pages(entitlement, stream);
    allocation.bind_row(0, stream);

    std::vector<std::int32_t> mapping(allocation.page_ids().begin(), allocation.page_ids().end());
    if (static_cast<std::int32_t>(mapping.size()) != page_count) {
        std::cerr << label << " materialized " << mapping.size() << " pages, expected " << page_count
                  << '\n';
        return 1;
    }

    // Mark logical pages 1 and 3 as relocated, everything else as not. Two pages with
    // *different* nonzero deltas, so a bug that writes the same value everywhere is
    // visible, and one gap between them so an "everything after the first write gets the
    // same value" bug is visible too.
    std::vector<std::int32_t> deltas_in(static_cast<std::size_t>(page_count),
                                         ninfer::kRopeDeltaNone);
    deltas_in[1] = 17;
    deltas_in[3] = -42;

    allocation.reassign_pages(mapping, deltas_in, stream);
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " reassign sync failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    // Read back through the pool's own row accessor, the same path a reader would use.
    ninfer::Tensor row = pool.rope_delta_row(0);
    std::vector<std::int32_t> after(static_cast<std::size_t>(page_count), 0);
    err = cudaMemcpy(after.data(), row.data,
                     static_cast<std::size_t>(page_count) * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " readback failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    int failures = 0;
    for (std::int32_t p = 0; p < page_count; ++p) {
        const std::int32_t want = deltas_in[static_cast<std::size_t>(p)];
        if (after[static_cast<std::size_t>(p)] != want) {
            if (failures < 4) {
                std::cerr << label << " page " << p << " is "
                          << after[static_cast<std::size_t>(p)] << ", expected " << want << '\n';
            }
            ++failures;
        }
    }
    allocation.unbind_row();
    allocation.release();
    return failures == 0 ? 0 : 1;
}

int expect_zeroed_pages(ninfer::PagedKVPool& pool, ninfer::PagedKVPlaneOrder order,
                        std::span<const std::int32_t> pages, cudaStream_t stream,
                        const char* label) {
    const ninfer::Tensor& plane = pool.plane(0);
    cudaError_t err             = cudaMemsetAsync(plane.data, 0x5a, plane.bytes(), stream);
    if (err != cudaSuccess) {
        std::cerr << label << " setup failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    pool.zero_pages(pages, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " synchronization failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::vector<unsigned char> actual(plane.bytes());
    err = cudaMemcpy(actual.data(), plane.data, actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    std::vector<unsigned char> expected(actual.size(), 0x5a);
    for (const std::int32_t page : pages) {
        if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
            const std::size_t begin = static_cast<std::size_t>(page * plane.nb[3]);
            std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                      expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[3]), 0);
        } else {
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                const std::size_t begin =
                    static_cast<std::size_t>(head * plane.nb[3] + page * plane.nb[2]);
                std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                          expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[2]), 0);
            }
        }
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " cleared bytes outside the selected physical pages\n";
    return 1;
}

int expect_page_host_roundtrip(ninfer::PagedKVPool& pool, ninfer::PagedKVPlaneOrder order,
                               cudaStream_t stream, const char* label) {
    const std::size_t planes = pool.plane_count();

    // 1. Fill entire device pool with a known background pattern (0xAA)
    for (std::size_t p = 0; p < planes; ++p) {
        const ninfer::Tensor& plane = pool.plane(p);
        CUDA_CHECK(cudaMemsetAsync(plane.data, 0xaa, plane.bytes(), stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 2. Generate unique deterministic data for physical page 3 across all planes
    const std::int32_t target_page = 3;
    std::vector<std::vector<std::byte>> original_page_data(planes);
    for (std::size_t p = 0; p < planes; ++p) {
        const std::size_t bytes = pool.page_bytes(p);
        original_page_data[p].resize(bytes);
        for (std::size_t i = 0; i < bytes; ++i) {
            original_page_data[p][i] = static_cast<std::byte>((p * 37 + i * 13 + target_page) & 0xFF);
        }
        // Write to GPU using copy_page_from_host
        pool.copy_page_from_host(p, target_page, original_page_data[p].data(), stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 3. Read back using copy_page_to_host into a clean buffer
    std::vector<std::vector<std::byte>> readback_data(planes);
    for (std::size_t p = 0; p < planes; ++p) {
        const std::size_t bytes = pool.page_bytes(p);
        readback_data[p].resize(bytes, std::byte{0});
        pool.copy_page_to_host(p, target_page, readback_data[p].data(), stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 4. Verify byte-exact equality for target page across all planes
    for (std::size_t p = 0; p < planes; ++p) {
        if (readback_data[p] != original_page_data[p]) {
            std::cerr << label << " plane " << p << " roundtrip data mismatch for page "
                      << target_page << '\n';
            return 1;
        }
    }

    // 5. Verify that neighboring physical page 2 was not overwritten
    for (std::size_t p = 0; p < planes; ++p) {
        const std::size_t bytes = pool.page_bytes(p);
        std::vector<std::byte> neighbor(bytes);
        pool.copy_page_to_host(p, 2, neighbor.data(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < bytes; ++i) {
            if (static_cast<unsigned char>(neighbor[i]) != 0xaa) {
                std::cerr << label << " plane " << p << " corrupted neighbor page 2 at byte " << i
                          << '\n';
                return 1;
            }
        }
    }

    return 0;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    auto paged_plan = plan_paged_cache(10, 10, 2,
                                       {{ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena paged_arena(paged_plan.bytes);
    // Drain default-stream / WDDM background work so the non-blocking test stream
    // sees fully committed backing memory (D3D12 residency race on Windows).
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::PagedKVPool paged_pool({paged_arena.base(), paged_arena.capacity()}, paged_plan.layout);
    failures += expect_size(paged_pool.plane_count(), 4, "paged plane count");
    failures += check_shape(paged_pool.plane(0), {64, 64, 2, 10}, "paged code plane");
    failures += check_shape(paged_pool.plane(2), {1, 64, 2, 10}, "paged scale plane");
    failures += check_shape(paged_pool.block_table_row(0), {10, 1, 1, 1}, "paged block-table row");
    if (!paged_pool.plane(0).is_contiguous() || !paged_pool.plane(2).is_contiguous() ||
        !paged_pool.block_table_row(0).is_contiguous()) {
        ++failures;
        std::cerr << "Paged planes or block-table row are not contiguous\n";
    }
    const std::size_t expected_payload =
        2 * ninfer::Tensor(nullptr, ninfer::DType::I8, {64, 64, 2, 10}).bytes() +
        2 * ninfer::Tensor(nullptr, ninfer::DType::FP16, {1, 64, 2, 10}).bytes();
    failures +=
        expect_size(paged_plan.layout.payload_bytes(), expected_payload, "paged payload bytes");
    // Metadata is the sum of the block table, delta plane and boundary-range plane sizes, matching how
    // `payload_bytes` accounts the storage planes. The alignment padding LayoutBuilder inserts
    // between them is deliberately not included -- this is a tensor-size total, not an allocation
    // size, and the only thing that reads it is this assertion.
    const std::size_t expected_metadata = 3 * 10 * 2 * sizeof(std::int32_t);
    failures += expect_size(paged_plan.layout.metadata_bytes(), expected_metadata,
                            "paged metadata bytes");
    failures += check_shape(paged_pool.rope_deltas(), {10, 2, 1, 1}, "paged rope-delta plane");
    failures += check_shape(paged_pool.rope_delta_ranges(), {10, 2, 1, 1},
                            "paged rope-delta range plane");
    // The effect, not just the arithmetic: filling the plane with a plausible shift and clearing one
    // row must leave that row zero and the other byte-identical.
    failures += expect_rope_delta_row_clear(paged_pool, 0, ctx.stream, "rope-delta row 0 clear");
    failures += expect_rope_delta_row_clear(paged_pool, 1, ctx.stream, "rope-delta row 1 clear");
    failures += expect_rope_delta_recycling(paged_pool, ctx.stream, "rope-delta recycling");
    failures += expect_reassign_publishes_deltas(paged_pool, ctx.stream,
                                                 "rope-delta reassign publish");
    if (paged_pool.rope_deltas().ne[0] != paged_pool.block_tables().ne[0] ||
        paged_pool.rope_deltas().ne[1] != paged_pool.block_tables().ne[1]) {
        ++failures;
        std::cerr << "Paged rope-delta plane does not parallel its block table\n";
    }
    const std::int32_t selected_pages[] = {1, 2, 6};
    failures += expect_zeroed_pages(paged_pool, ninfer::PagedKVPlaneOrder::PageMajor,
                                    selected_pages, ctx.stream, "page-major selective zero");
    failures += expect_page_host_roundtrip(paged_pool, ninfer::PagedKVPlaneOrder::PageMajor,
                                           ctx.stream, "page-major page host roundtrip");

    auto head_major_plan = plan_paged_cache(10, 10, 1, {{ninfer::DType::BF16, 128, 8}},
                                            ninfer::PagedKVPlaneOrder::HeadMajor);
    ninfer::DeviceArena head_major_arena(head_major_plan.bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::PagedKVPool head_major_pool({head_major_arena.base(), head_major_arena.capacity()},
                                        head_major_plan.layout);
    failures += check_shape(head_major_pool.plane(0), {128, 64, 10, 8}, "head-major paged plane");
    failures += expect_zeroed_pages(head_major_pool, ninfer::PagedKVPlaneOrder::HeadMajor,
                                    selected_pages, ctx.stream, "head-major selective zero");
    failures += expect_page_host_roundtrip(head_major_pool, ninfer::PagedKVPlaneOrder::HeadMajor,
                                           ctx.stream, "head-major page host roundtrip");

    auto allocation_a = paged_pool.reserve(3);
    auto allocation_b = paged_pool.reserve(3);
    allocation_a.materialize_pages(3);
    allocation_b.materialize_pages(3);
    allocation_a.bind_row(0);
    allocation_b.bind_row(1);
    failures += expect_device_page_ids(allocation_a.block_table(), {0, 1, 2}, "allocation A");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "allocation B");
    allocation_a.release();

    auto allocation_c = paged_pool.reserve(6);
    allocation_c.materialize_pages(6);
    allocation_c.bind_row(0);
    failures +=
        expect_page_ids(allocation_c.page_ids(), {0, 1, 2, 6, 7, 8}, "fragmented allocation C");
    failures +=
        expect_device_page_ids(allocation_c.block_table(), {0, 1, 2, 6, 7, 8}, "fragmented row C");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "isolated row B");

    allocation_c.trim_pages(2);
    if (paged_pool.can_reserve(2)) {
        ++failures;
        std::cerr << "Unmapped entitlement was exposed as reservable capacity\n";
    }
    allocation_c.cancel_unmapped_entitlement();
    if (!paged_pool.can_reserve(5)) {
        ++failures;
        std::cerr << "Cancelled entitlement did not return reservable capacity\n";
    }

    allocation_c.unbind_row();
    const std::int32_t retained_prefix[] = {allocation_c.page_ids()[0], allocation_c.page_ids()[1]};
    const ninfer::PagedKVResize claim[]  = {
        {.allocation = &allocation_c, .mapped_pages = 2, .page_entitlement = 5}};
    ninfer::resize_paged_kv_bundle(claim);
    allocation_c.bind_row(0);
    allocation_c.materialize_pages(5);
    if (allocation_c.page_ids()[0] != retained_prefix[0] ||
        allocation_c.page_ids()[1] != retained_prefix[1]) {
        ++failures;
        std::cerr << "Retained allocation moved its prefix pages\n";
    }
    allocation_c.trim_tokens(65);
    failures += expect_size(allocation_c.mapped_page_count(), 2, "partial-tail mapped pages");
    allocation_c.trim_tokens(64);
    failures += expect_size(allocation_c.mapped_page_count(), 1, "page-aligned mapped pages");

    auto main_plan    = plan_paged_cache(4, 4, 1, {{ninfer::DType::BF16, 16, 1}});
    auto backend_plan = plan_paged_cache(2, 2, 1, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena main_arena(main_plan.bytes);
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool main_pool({main_arena.base(), main_arena.capacity()}, main_plan.layout);
    ninfer::PagedKVPool backend_pool({backend_arena.base(), backend_arena.capacity()},
                                     backend_plan.layout);
    const ninfer::PagedKVReservation impossible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 3},
        {.pool = &backend_pool, .page_entitlement = 3},
    };
    try {
        auto unused = ninfer::reserve_paged_kv_bundle(impossible_bundle);
        (void)unused;
        ++failures;
        std::cerr << "Impossible multi-pool reservation succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(main_pool.entitled_pages(), 0, "failed bundle main entitlement");
    failures += expect_size(backend_pool.entitled_pages(), 0, "failed bundle backend entitlement");

    const ninfer::PagedKVReservation possible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 2},
        {.pool = &backend_pool, .page_entitlement = 2},
    };
    auto bundle = ninfer::reserve_paged_kv_bundle(possible_bundle);
    bundle[0].materialize_pages(2);
    bundle[1].materialize_pages(2);
    const ninfer::PagedKVResize impossible_resize[] = {
        {.allocation = &bundle[0], .mapped_pages = 1, .page_entitlement = 1},
        {.allocation = &bundle[1], .mapped_pages = 1, .page_entitlement = 3},
    };
    try {
        ninfer::resize_paged_kv_bundle(impossible_resize);
        ++failures;
        std::cerr << "Impossible multi-pool resize succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(bundle[0].mapped_page_count(), 2, "failed resize main mapping");
    failures += expect_size(bundle[0].page_entitlement(), 2, "failed resize main entitlement");
    failures += expect_size(bundle[1].mapped_page_count(), 2, "failed resize backend mapping");
    failures += expect_size(bundle[1].page_entitlement(), 2, "failed resize backend entitlement");

    // A Suffix Cache Reuse plan asks for an entitlement below the lane's current mapping, because a
    // shrinking edit makes the new prompt smaller while the resident pages above the frontier stay
    // mapped until the page remap publishes. The runtime does not pass that figure through: it asks
    // for `max(requested, mapped)`. That substitution is only sound if the unclamped request is
    // refused *before* anything mutates, so this pins both halves -- the refusal and its atomicity,
    // then the acceptance of the clamped form.
    const ninfer::PagedKVResize shrink_below_mapping[] = {
        {.allocation       = &bundle[0],
         .mapped_pages     = bundle[0].mapped_page_count(),
         .page_entitlement = 1},
    };
    bool shrink_refused = false;
    try {
        ninfer::resize_paged_kv_bundle(shrink_below_mapping);
    } catch (const std::invalid_argument&) { shrink_refused = true; }
    if (!shrink_refused) {
        ++failures;
        std::cerr << "A resize to an entitlement below the mapping was accepted\n";
    }
    failures += expect_size(bundle[0].mapped_page_count(), 2, "refused shrink left the mapping");
    failures += expect_size(bundle[0].page_entitlement(), 2, "refused shrink left the entitlement");

    const ninfer::PagedKVResize clamped_shrink[] = {
        {.allocation       = &bundle[0],
         .mapped_pages     = bundle[0].mapped_page_count(),
         .page_entitlement = std::max<std::uint32_t>(1, bundle[0].mapped_page_count())},
    };
    ninfer::resize_paged_kv_bundle(clamped_shrink);
    failures += expect_size(bundle[0].mapped_page_count(), 2, "clamped shrink mapping");
    failures += expect_size(bundle[0].page_entitlement(), 2, "clamped shrink entitlement");

    return failures == 0 ? 0 : fail("kv cache test failed");
}

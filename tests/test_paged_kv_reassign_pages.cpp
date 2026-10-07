// Oracle test for PagedKVAllocation::reassign_pages, the mechanism suffix relocation uses to
// re-point destination blocks at surviving pages without copying KV.
//
// The oracle is derived from the contract in paged_kv_cache.h, not from the implementation. It models
// pool ownership as a page -> owner map and asserts the invariants that must hold whatever choice the
// implementation makes about which page fills an acquire slot:
//
//   I1  every page group is either free or owned by exactly one allocation, never both
//   I2  free pages + owned pages == total page groups
//   I3  no allocation holds a duplicate page id
//   I4  named slots in the requested mapping are published exactly, in order
//   I5  acquire slots are filled only from pages the call released or newly took
//   I6  a rejected reassignment leaves the previous mapping and the pool untouched
//
// It also checks the property that makes the Op worth having: KV payloads are never copied, so the
// bytes at a surviving physical page are still that page's bytes after the logical mapping moves.

#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <new>
#include <set>
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

// Independent model of pool ownership: which allocation owns which page, and how many are free.
struct PoolOracle {
    std::size_t total      = 0;
    std::vector<int> owner = {}; // -1 means free
    std::uint32_t free     = 0;

    explicit PoolOracle(std::size_t page_groups) : total(page_groups), owner(page_groups, -1),
                                                   free(static_cast<std::uint32_t>(page_groups)) {}

    void own(std::span<const std::int32_t> pages, int allocation) {
        for (const std::int32_t page : pages) {
            if (owner[static_cast<std::size_t>(page)] != -1) {
                fail("oracle: a page was claimed while another allocation still owned it");
            }
            owner[static_cast<std::size_t>(page)] = allocation;
            --free;
        }
    }

    void drop(std::span<const std::int32_t> pages, int allocation) {
        for (const std::int32_t page : pages) {
            if (owner[static_cast<std::size_t>(page)] != allocation) {
                fail("oracle: a page was released that this allocation did not own");
            }
            owner[static_cast<std::size_t>(page)] = -1;
            ++free;
        }
    }

    [[nodiscard]] std::set<std::int32_t> free_pages() const {
        std::set<std::int32_t> pages;
        for (std::size_t page = 0; page < owner.size(); ++page) {
            if (owner[page] == -1) { pages.insert(static_cast<std::int32_t>(page)); }
        }
        return pages;
    }
};

std::vector<std::int32_t> ids_of(const ninfer::PagedKVAllocation& alloc) {
    return std::vector<std::int32_t>(alloc.page_ids().begin(), alloc.page_ids().end());
}

// I1 and I2: ownership partitions the page groups, and the pool agrees with the model.
void check_partition(const PoolOracle& oracle, ninfer::PagedKVPool& pool, const char* label) {
    std::size_t owned_total = 0;
    for (const int holder : oracle.owner) {
        if (holder != -1) { ++owned_total; }
    }
    if (owned_total + oracle.free != oracle.total) {
        fail(std::string(label) + ": owned + free != total page groups");
    }
    if (pool.free_pages() != oracle.free) {
        fail(std::string(label) + ": the pool reports " + std::to_string(pool.free_pages()) +
             " free pages, the model says " + std::to_string(oracle.free));
    }
}

// I3: no allocation holds a duplicate page id.
void check_no_duplicates(std::span<const std::int32_t> ids, const char* label) {
    const std::set<std::int32_t> unique(ids.begin(), ids.end());
    if (unique.size() != ids.size()) {
        fail(std::string(label) + ": the mapping contains a duplicate page id");
    }
}

// I4 and I5: compare the published mapping against the requested one. `surplus` is the pages the
// allocation owned but the mapping did not name; `previously_free` is what the pool could offer. An
// acquire slot may be filled from either, and from nothing else.
void check_mapping(std::span<const std::int32_t> actual, std::span<const std::int32_t> requested,
                   const std::set<std::int32_t>& surplus,
                   const std::set<std::int32_t>& previously_free, const char* label) {
    if (actual.size() != requested.size()) {
        fail(std::string(label) + ": published " + std::to_string(actual.size()) +
             " pages for a " + std::to_string(requested.size()) + " page mapping");
        return;
    }
    check_no_duplicates(actual, label);
    for (std::size_t i = 0; i < requested.size(); ++i) {
        if (requested[i] == ninfer::kPagedKVAquirePage) {
            if (surplus.count(actual[i]) == 0 && previously_free.count(actual[i]) == 0) {
                fail(std::string(label) + ": acquire slot " + std::to_string(i) +
                     " was filled with page " + std::to_string(actual[i]) +
                     ", which was neither released by this call nor free beforehand");
            }
            continue;
        }
        if (actual[i] != requested[i]) {
            fail(std::string(label) + ": named slot " + std::to_string(i) + " published page " +
                 std::to_string(actual[i]) + ", requested " + std::to_string(requested[i]));
        }
    }
}

// A reassignment the contract allows. Derives the prediction from the mapping and the current
// ownership, then validates the published result and resynchronises the model with it.
void expect_reassign(ninfer::PagedKVPool& pool, PoolOracle& oracle, ninfer::PagedKVAllocation& alloc,
                     int allocation_id, std::span<const std::int32_t> mapping, const char* label) {
    const std::vector<std::int32_t> before = ids_of(alloc);
    const std::set<std::int32_t> previously_free = oracle.free_pages();

    // Named slots consume ownership; what remains is the surplus, which covers as many acquire slots
    // as it can so that only the shortfall has to come from the pool.
    std::multiset<std::int32_t> remaining(before.begin(), before.end());
    std::uint32_t acquire_count = 0;
    bool valid                  = true;
    for (const std::int32_t page : mapping) {
        if (page == ninfer::kPagedKVAquirePage) {
            ++acquire_count;
            continue;
        }
        const auto found = remaining.find(page);
        if (found == remaining.end()) {
            valid = false;
            break;
        }
        remaining.erase(found);
    }
    if (!valid) {
        fail(std::string(label) + ": the model rejected a mapping the implementation accepted");
        return;
    }
    const std::uint32_t surplus = static_cast<std::uint32_t>(remaining.size());
    const std::uint32_t from_pool = acquire_count - std::min(acquire_count, surplus);
    if (from_pool > previously_free.size()) {
        fail(std::string(label) + ": the model expected this to need " + std::to_string(from_pool) +
             " free pages but only " + std::to_string(previously_free.size()) + " were free");
        return;
    }

    alloc.reassign_pages(mapping, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    const std::vector<std::int32_t> actual = ids_of(alloc);
    oracle.drop(before, allocation_id);
    oracle.own(actual, allocation_id);
    check_mapping(actual, mapping, std::set<std::int32_t>(remaining.begin(), remaining.end()),
                  previously_free, label);
    check_partition(oracle, pool, label);
    note(std::string(label) + ": ok (" + std::to_string(actual.size()) + " pages, " +
         std::to_string(pool.free_pages()) + " free)");
}

// I6: a rejection must leave the mapping and the pool exactly as they were.
template <typename Fn>
void expect_reject(ninfer::PagedKVPool& pool, PoolOracle& oracle, ninfer::PagedKVAllocation& alloc,
                   int allocation_id, Fn fn, const char* label) {
    const std::vector<std::int32_t> before = ids_of(alloc);
    const std::uint32_t free_before        = pool.free_pages();

    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw) {
        fail(std::string(label) + ": expected a rejection");
        return;
    }
    const std::vector<std::int32_t> after = ids_of(alloc);
    if (after != before) {
        fail(std::string(label) + ": a rejected reassignment changed the mapping");
    }
    if (pool.free_pages() != free_before) {
        fail(std::string(label) + ": a rejected reassignment changed the free page count");
    }
    oracle.drop(before, allocation_id);
    oracle.own(after, allocation_id);
    check_partition(oracle, pool, label);
    note(std::string(label) + ": ok (rejected, nothing changed)");
}

std::vector<std::int32_t> read_row(const ninfer::Tensor& row, std::size_t count) {
    std::vector<std::int32_t> ids(count);
    CUDA_CHECK(
        cudaMemcpy(ids.data(), row.data, count * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    return ids;
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

    constexpr std::uint32_t kPageGroups = 12;
    constexpr std::uint32_t kEntitlement = 8;
    constexpr std::int32_t kRows        = 1;

    auto plan = plan_pool(kPageGroups, kPageGroups, kRows);
    ninfer::DeviceArena arena(plan.bytes);
    // Drain default-stream / WDDM background work so the non-blocking test stream sees fully
    // committed backing memory (the D3D12 residency race on Windows).
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    PoolOracle oracle(kPageGroups);

    // A recognizable byte pattern per physical page, so "no KV was copied" is checkable.
    const std::size_t page_bytes = pool.page_bytes(0);
    std::vector<std::vector<std::byte>> pattern(kPageGroups);
    for (std::size_t page = 0; page < kPageGroups; ++page) {
        pattern[page].resize(page_bytes);
        for (std::size_t i = 0; i < page_bytes; ++i) {
            pattern[page][i] = static_cast<std::byte>((page * 61 + i * 29) & 0xFF);
        }
        pool.copy_page_from_host(0, static_cast<std::int32_t>(page), pattern[page].data(),
                                 ctx.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    const auto verify_payload = [&](std::int32_t page, const char* label) {
        std::vector<std::byte> readback(page_bytes);
        pool.copy_page_to_host(0, page, readback.data(), ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        if (readback != pattern[static_cast<std::size_t>(page)]) {
            fail(std::string(label) + ": page " + std::to_string(page) +
                 " no longer holds its own KV payload");
        }
    };

    std::cout << "PagedKVAllocation::reassign_pages oracle checks\n";

    // One allocation owns most of the pool's entitlement for the whole run, so each case can grow or
    // shrink freely without a second allocation perturbing the free count. The pool is larger than
    // the entitlement so a second allocation can still be reserved for the rejection cases.
    auto alloc = pool.reserve(kEntitlement);
    alloc.materialize_pages(4);
    oracle.own(std::span<const std::int32_t>(alloc.page_ids()), 0);

    // --- 1. Permutation: the same pages in a different order. ---
    {
        const std::vector<std::int32_t> original = ids_of(alloc);
        const std::vector<std::int32_t> reversed(original.rbegin(), original.rend());
        expect_reassign(pool, oracle, alloc, 0, reversed, "permutation");
        if (ids_of(alloc) != reversed) {
            fail("permutation: the mapping was not published in the requested order");
        }
        if (pool.free_pages() != kPageGroups - 4) {
            fail("permutation: a permutation must not change the free page count");
        }        verify_payload(original[0], "permutation");
    }

    // --- 2. Grow with a page from the pool. ---
    {
        const std::vector<std::int32_t> current   = ids_of(alloc);
        const std::vector<std::int32_t> grown     = {current[0], current[1],
                                                     ninfer::kPagedKVAquirePage, current[2],
                                                     current[3]};
        const std::uint32_t free_before          = pool.free_pages();
        expect_reassign(pool, oracle, alloc, 0, grown, "grow");
        if (pool.free_pages() + 1 != free_before) {
            fail("grow: growing by one page did not consume exactly one free page");
        }
        if (alloc.mapped_page_count() != 5 ||
            alloc.mapped_token_capacity() != 5u * ninfer::kPagedKVPageSize) {
            fail("grow: the mapped page or token count is wrong");
        }
        for (const std::int32_t page : alloc.page_ids()) { verify_payload(page, "grow"); }
    }

    // --- 3. Shrink: the pages the mapping does not name go back to the pool. ---
    {
        const std::vector<std::int32_t> current = ids_of(alloc);
        const std::vector<std::int32_t> shrunk = {current[0], current[1]};
        const std::uint32_t free_before        = pool.free_pages();
        expect_reassign(pool, oracle, alloc, 0, shrunk, "shrink");
        if (pool.free_pages() != free_before + 3) {
            fail("shrink: dropping three pages did not return exactly three free pages");
        }
        if (alloc.mapped_page_count() != 2) {
            fail("shrink: mapped page count is not 2");
        }
        if (alloc.mapped_token_capacity() != 2u * ninfer::kPagedKVPageSize) {
            fail("shrink: mapped token capacity is not two pages worth of tokens");
        }
    }

    // --- 4. Reclaim a whole entitlement through the mapping itself. ---
    // Nothing is released here, so the pages beyond the current two have to come from the pool.
    {
        const std::vector<std::int32_t> claim(kEntitlement, ninfer::kPagedKVAquirePage);
        expect_reassign(pool, oracle, alloc, 0, claim, "grow to the whole entitlement");
        if (pool.free_pages() != kPageGroups - kEntitlement) {
            fail("reclaim: taking the whole entitlement left free pages behind");
        }
    }

    // --- 5. Shrink one region while growing another, served entirely from the surplus. ---
    // This is the shape suffix relocation produces. The three acquire slots are covered by the five
    // pages the mapping stopped naming, so the pool is never asked for anything: it is not paid twice
    // for a page it already gave back.
    {
        const std::vector<std::int32_t> current   = ids_of(alloc);
        const std::vector<std::int32_t> remapped  = {current[5], current[4], current[3],
                                                      ninfer::kPagedKVAquirePage,
                                                      ninfer::kPagedKVAquirePage,
                                                      ninfer::kPagedKVAquirePage};
        expect_reassign(pool, oracle, alloc, 0, remapped, "shrink-and-grow with a small free pool");
        if (pool.free_pages() != kPageGroups - kEntitlement + 2) {
            fail("shrink-and-grow: expected the two unused surplus pages to return to the pool, "
                 "the pool reports " +
                 std::to_string(pool.free_pages()));
        }
        for (const std::int32_t page : alloc.page_ids()) {
            verify_payload(page, "shrink-and-grow");
        }
    }

    // --- 6. Rejections, each of which must leave the mapping and the pool untouched. ---
    {
        auto guarded = pool.reserve(kPageGroups - kEntitlement);
        guarded.materialize_pages(1);
        oracle.own(std::span<const std::int32_t>(guarded.page_ids()), 1);
        const std::int32_t mine = guarded.page_ids()[0];

        expect_reject(pool, oracle, guarded, 1, [&] { guarded.reassign_pages({}, nullptr); },
                      "empty mapping");
        expect_reject(
            pool, oracle, guarded, 1,
            [&] {
                const std::vector<std::int32_t> oversized(kPageGroups - kEntitlement + 1, mine);
                guarded.reassign_pages(oversized, nullptr);
            },
            "mapping larger than the entitlement");
        expect_reject(
            pool, oracle, guarded, 1,
            [&] {
                const std::vector<std::int32_t> foreign = {mine, alloc.page_ids()[0]};
                guarded.reassign_pages(foreign, nullptr);
            },
            "naming a page the allocation does not own");
        expect_reject(
            pool, oracle, guarded, 1,
            [&] {
                const std::vector<std::int32_t> twice = {mine, mine};
                guarded.reassign_pages(twice, nullptr);
            },
            "naming an owned page twice");

        // Release it so the accounting below describes a single allocation and the expected counts
        // are not perturbed by a second owner.
        oracle.drop(std::span<const std::int32_t>(guarded.page_ids()), 1);
        guarded.release();
        check_partition(oracle, pool, "rejections teardown");
    }

    // --- 7. The pool can never be over-committed through this path, so bad_alloc is unreachable. ---
    // reassign_pages requests at most (entitlement - owned) pages beyond the surplus, while
    // materialize_pages caps ownership at the entitlement and reserve caps the sum of entitlements at
    // page_group_count. Together those bound free pages below by (own_entitlement - owned), so the
    // shortfall can never exceed what is free. This asserts that bound holds at the end of the run
    // rather than trying to provoke a throw that the public API cannot reach.
    {
        const std::uint32_t shortfall = kEntitlement - alloc.mapped_page_count();
        if (pool.free_pages() < shortfall) {
            fail("accounting: the pool holds fewer free pages than the worst-case growth needs, so "
                 "reassign_pages could throw std::bad_alloc on a legal mapping");
        }
        // And the strongest legal growth the allocation can ask for must still succeed.
        const std::vector<std::int32_t> maximum(kEntitlement, ninfer::kPagedKVAquirePage);
        expect_reassign(pool, oracle, alloc, 0, maximum, "maximum legal growth");
        if (alloc.mapped_page_count() != kEntitlement) {
            fail("accounting: the allocation did not reach its full entitlement");
        }
        note("accounting: ok (the pool cannot be over-committed)");
    }

    // --- 8. A default-constructed allocation has nothing to reassign. ---
    {
        ninfer::PagedKVAllocation empty;
        const std::vector<std::int32_t> one_page = {0};
        expect_reject(pool, oracle, empty, -2,
                      [&] { empty.reassign_pages(one_page, nullptr); },
                      "reassign on an empty allocation");
    }

    // --- 9. The republished block-table row. ---
    // The re-rotation Op resolves a destination page through this row, so it has to track the
    // reassignment exactly and immediately, without a separate bind.
    {
        auto published_plan = plan_pool(4, 4, 1);
        ninfer::DeviceArena published_arena(published_plan.bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
        ninfer::PagedKVPool published_pool({published_arena.base(), published_arena.capacity()},
                                           published_plan.layout);
        PoolOracle published_oracle(4);

        auto bound = published_pool.reserve(4);
        bound.materialize_pages(3);
        published_oracle.own(std::span<const std::int32_t>(bound.page_ids()), 0);
        bound.bind_row(0, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        if (read_row(bound.block_table(), 3) != ids_of(bound)) {
            fail("published row: the initial row does not match the materialized pages");
        }

        const std::vector<std::int32_t> remapped = {2, 0, ninfer::kPagedKVAquirePage};
        expect_reassign(published_pool, published_oracle, bound, 0, remapped, "remap while bound");
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const std::vector<std::int32_t> row      = read_row(bound.block_table(), 3);
        const std::vector<std::int32_t> published = ids_of(bound);
        if (row != published) {
            fail("published row: the device row does not match the reassigned mapping");
        }
        if (row[0] != 2 || row[1] != 0) {
            fail("published row: the named slots were not republished in order");
        }
        note("published row: ok");
    }

    if (failures != 0) {
        std::cout << "PagedKVAllocation::reassign_pages oracle checks FAILED (" << failures << ")\n";
        return 1;
    }
    std::cout << "PagedKVAllocation::reassign_pages oracle checks passed\n";
    return 0;
}

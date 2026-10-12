#include "core/paged_kv_cache.h"

#include "core/device.h"
#include "ops/kernel/kv_paged_staging.cuh"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer {
namespace {

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must fit positive int32");
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    if (tokens == 0) { return 0; }
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

void validate_distinct_pools(std::span<const PagedKVReservation> reservations) {
    for (std::size_t i = 0; i < reservations.size(); ++i) {
        if (reservations[i].pool == nullptr || reservations[i].page_entitlement == 0) {
            throw std::invalid_argument("Paged KV reservation must name a pool and pages");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (reservations[i].pool == reservations[j].pool) {
                throw std::invalid_argument("Paged KV bundle contains the same pool twice");
            }
        }
    }
}

} // namespace

PagedKVPoolLayout plan_paged_kv_pool(LayoutBuilder& builder, const PagedKVPoolSpec& spec) {
    const std::int32_t physical_pages = checked_i32(spec.page_group_count, "Paged KV page count");
    const std::int32_t logical_pages =
        checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity");
    if (spec.table_rows <= 0) {
        throw std::invalid_argument("Paged KV table row count must be positive");
    }
    if (spec.planes.empty()) { throw std::invalid_argument("Paged KV pool must contain planes"); }

    PagedKVPoolLayout layout;
    layout.spec = spec;
    layout.planes.reserve(spec.planes.size());
    for (std::size_t index = 0; index < spec.planes.size(); ++index) {
        const PagedKVPlaneSpec& plane = spec.planes[index];
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Paged KV plane extents must be positive");
        }
        const std::string label = "Paged KV plane " + std::to_string(index);
        PagedKVPlaneLayout planned;
        planned.spec = plane;
        if (spec.plane_order == PagedKVPlaneOrder::PageMajor) {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, plane.head_extent, physical_pages},
                plane.alignment, label);
        } else {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, physical_pages, plane.head_extent},
                plane.alignment, label);
        }
        layout.planes.push_back(planned);
    }
    layout.block_tables = builder.add_tensor(DType::I32, {logical_pages, spec.table_rows}, 256,
                                             "Paged KV block tables");

    // The RoPE delta plane, sized by the same arithmetic the host test pins rather than by a second
    // hand-written formula. It is allocated unconditionally: table_rows is max_concurrency, capped at
    // kMaximumConcurrency (8), so the plane is between 0.5 KiB at 8k context and 256 KiB at 500k --
    // against a KV cache measured in GiB, a flag that could be forgotten is not worth the risk.
    //
    // The extents are cross-checked rather than assumed to match, because the mismatch is the failure
    // this plane exists to avoid: a delta plane smaller than the block table beside it leaves pages
    // whose delta can never be read, which presents as "relocation silently did nothing" rather than
    // as an error. Both are derived from one source here, so the check is structural rather than
    // load-bearing -- it is what catches the day the sources diverge.
    //
    // No separate `rope_delta_layout_is_supported` call follows: `resolve_rope_delta_layout`
    // already returns an all-zero layout when the extents are unsupported, and an all-zero layout
    // fails the equality below, so one check covers both. Passing the block table's extents to the
    // four-argument form as well would compare values just shown equal, which is the tautology this
    // whole check was added to remove.
    // Pages are already int32 from `checked_i32`, so the product can exceed int32 while each factor
    // fits. The bound being tested is `logical_pages * kPagedKVPageSize`, not the page count: past
    // this the reconstructed context no longer round-trips, and the equality below would reject a
    // silently-wrapped value rather than a detected one.
    const std::uint64_t implied_context =
        static_cast<std::uint64_t>(logical_pages) * static_cast<std::uint64_t>(kPagedKVPageSize);
    if (implied_context > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Paged KV logical page count is too large to reconstruct a "
                                    "context length for the rope-delta plane");
    }
    const RopeDeltaLayout rope =
        resolve_rope_delta_layout(static_cast<std::int32_t>(implied_context), spec.table_rows);
    if (rope.logical_pages != logical_pages || rope.table_rows != spec.table_rows) {
        throw std::logic_error("Paged KV rope-delta plane does not cover its block table");
    }
    layout.rope_deltas = builder.add_tensor(DType::I32, {logical_pages, spec.table_rows}, 256,
                                            "Paged KV rope deltas");
    return layout;
}

std::size_t PagedKVPoolLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const PagedKVPlaneLayout& plane : planes) { total += plane.storage.region.bytes; }
    return total;
}

// Sum of the metadata tensors' own sizes, the same accounting `payload_bytes` uses for the storage
// planes. It deliberately excludes the alignment padding `LayoutBuilder` inserts between tensors, so
// it is *not* an allocation size -- `LayoutBuilder::finish` is, from its own peak, and no caller
// sizes anything from here. Reporting the span from the layout start instead would fold in the entire
// payload, which is what a first attempt at "include the padding" did and what the kv-cache test
// caught: 169296 bytes of planes reported as metadata.
std::size_t PagedKVPoolLayout::metadata_bytes() const noexcept {
    return block_tables.region.bytes + rope_deltas.region.bytes;
}

PagedKVPool::PagedKVPool(DeviceSpan backing, const PagedKVPoolLayout& layout)
    : spec_(layout.spec), block_tables_(layout.block_tables.bind(backing)),
      rope_deltas_(layout.rope_deltas.bind(backing)),
      row_in_use_(static_cast<std::size_t>(layout.spec.table_rows), false) {
    if (layout.planes.size() != spec_.planes.size() || layout.planes.empty()) {
        throw std::invalid_argument("Paged KV layout plane inventory is inconsistent");
    }
    if (block_tables_.dtype != DType::I32 ||
        block_tables_.ne[0] !=
            checked_i32(spec_.logical_page_capacity, "Paged KV logical page capacity") ||
        block_tables_.ne[1] != spec_.table_rows) {
        throw std::logic_error("Paged KV block-table layout is inconsistent");
    }
    // The delta plane is read with the block table's index, so it must carry the block table's shape.
    // A mismatch here is not a size question, it is an addressing question: every delta read would
    // fetch the wrong page's shift. All four dims, not just the two the block-table check above
    // looks at, because ne[2] and ne[3] being anything but 1 would change the element stride.
    if (rope_deltas_.dtype != DType::I32 ||
        rope_deltas_.ne[0] != block_tables_.ne[0] || rope_deltas_.ne[1] != block_tables_.ne[1] ||
        rope_deltas_.ne[2] != 1 || rope_deltas_.ne[3] != 1) {
        throw std::logic_error("Paged KV rope-delta layout is inconsistent with its block table");
    }

    planes_.reserve(layout.planes.size());
    for (std::size_t index = 0; index < layout.planes.size(); ++index) {
        const PagedKVPlaneLayout& plane = layout.planes[index];
        if (plane.spec.dtype != spec_.planes[index].dtype ||
            plane.spec.leading_extent != spec_.planes[index].leading_extent ||
            plane.spec.head_extent != spec_.planes[index].head_extent) {
            throw std::logic_error("Paged KV plane layout does not match its spec");
        }
        planes_.push_back(plane.storage.bind(backing));
    }

    std::vector<ops::PagedKVPlaneDescriptor> h_planes;
    h_planes.reserve(planes_.size());
    uint32_t running_page_offset_uint4s = 0;
    for (std::size_t p = 0; p < planes_.size(); ++p) {
        const Tensor& plane = planes_[p];
        const std::size_t p_bytes = page_bytes(p);
        ops::PagedKVPlaneDescriptor desc{};
        desc.plane_base = plane.data;
        if (spec_.plane_order == PagedKVPlaneOrder::HeadMajor) {
            desc.num_heads          = static_cast<uint32_t>(plane.ne[3]);
            desc.page_head_uint4s   = static_cast<uint32_t>(p_bytes / desc.num_heads / 16);
            desc.head_stride_uint4s = static_cast<uint32_t>(plane.nb[3] / 16);
            desc.page_stride_uint4s = static_cast<uint32_t>(plane.nb[2] / 16);
        } else {
            desc.num_heads          = static_cast<uint32_t>(plane.ne[2]);
            desc.page_head_uint4s   = static_cast<uint32_t>(p_bytes / desc.num_heads / 16);
            desc.head_stride_uint4s = static_cast<uint32_t>(plane.nb[2] / 16);
            desc.page_stride_uint4s = static_cast<uint32_t>(plane.nb[3] / 16);
        }
        desc.plane_page_offset_uint4s = running_page_offset_uint4s;
        running_page_offset_uint4s += desc.num_heads * desc.page_head_uint4s;
        max_page_head_uint4s_ = std::max(max_page_head_uint4s_, desc.page_head_uint4s);
        max_num_heads_        = std::max(max_num_heads_, desc.num_heads);
        h_planes.push_back(desc);
    }
    for (auto& desc : h_planes) {
        desc.single_page_uint4s = running_page_offset_uint4s;
    }
    if (!h_planes.empty()) {
        CUDA_CHECK(cudaMalloc(&d_planes_, h_planes.size() * sizeof(ops::PagedKVPlaneDescriptor)));
        CUDA_CHECK(cudaMemcpy(d_planes_, h_planes.data(), h_planes.size() * sizeof(ops::PagedKVPlaneDescriptor), cudaMemcpyHostToDevice));
    }

    free_page_ids_.reserve(spec_.page_group_count);
    for (std::uint32_t page = 0; page < spec_.page_group_count; ++page) {
        free_page_ids_.push_back(static_cast<std::int32_t>(page));
    }
}

PagedKVPool::~PagedKVPool() {
    if (d_planes_) {
        cudaFree(d_planes_);
        d_planes_ = nullptr;
    }
}

std::uint32_t PagedKVPool::page_group_count() const noexcept { return spec_.page_group_count; }

std::uint32_t PagedKVPool::logical_page_capacity() const noexcept {
    return spec_.logical_page_capacity;
}

std::int32_t PagedKVPool::table_row_count() const noexcept { return spec_.table_rows; }

std::size_t PagedKVPool::plane_count() const noexcept { return planes_.size(); }

const Tensor& PagedKVPool::plane(std::size_t index) const { return planes_.at(index); }

const Tensor& PagedKVPool::block_tables() const noexcept { return block_tables_; }

const Tensor& PagedKVPool::rope_deltas() const noexcept { return rope_deltas_; }

void PagedKVPool::clear_rope_delta_row(std::int32_t row, cudaStream_t stream) {
    // Own precondition, so the diagnostic names the actual cause. The degenerate-plane case is not
    // reachable from a pool this repo builds -- `plan_paged_kv_pool` rejects `table_rows <= 0` and
    // derives `logical_pages` through `checked_i32`, which rejects zero -- so reaching it would mean
    // the pool was constructed from a spec that already failed. Say that, rather than reporting a
    // row as out of range because the plane is empty.
    if (row < 0 || row >= table_row_count()) {
        throw std::out_of_range("Paged KV rope-delta row out of range");
    }
    const RopeDeltaRowSpan span = rope_delta_row_span_i32(rope_deltas_.ne[0], table_row_count(), row);
    if (span.bytes == 0) {
        throw std::logic_error("Paged KV rope-delta plane is degenerate");
    }
    // `kRopeDeltaNone` is zero, so clearing the row *is* the reset. This runs on the caller's stream
    // beside `publish_mapping`, not on the default stream, because a default-stream write does not
    // order against the non-blocking stream that will read it.
    //
    // The offset comes from the tested `rope_delta_row_span_i32` rather than from a `Tensor::slice`,
    // so the arithmetic that decides *which bytes get zeroed* is host-verifiable. It reads as though
    // it should need a strided write -- the plane is `{logical_pages, table_rows}` and the row is
    // dim 1 -- but dim 0 is innermost here, so a row is contiguous. `publish_range` proves the
    // convention: it indexes a block-table row as `row.data + first_page` and memcpys contiguous
    // bytes, which only works under it. The row-span test proves the rows are disjoint and jointly
    // tile the plane, which is what makes one memset over one row safe.
    auto* row_base = static_cast<unsigned char*>(rope_deltas_.data) + span.offset;
    CUDA_CHECK(cudaMemsetAsync(row_base, 0, span.bytes, stream));
}

Tensor PagedKVPool::block_table_row(std::int32_t row) const {
    if (row < 0 || row >= table_row_count()) {
        throw std::out_of_range("Paged KV block-table row out of range");
    }
    return block_tables_.slice(1, row, 1).view(
        {static_cast<std::int32_t>(logical_page_capacity())});
}

Tensor PagedKVPool::rope_delta_row(std::int32_t row) const {
    if (row < 0 || row >= table_row_count()) {
        throw std::out_of_range("Paged KV rope-delta row out of range");
    }
    return rope_deltas_.slice(1, row, 1).view(
        {static_cast<std::int32_t>(logical_page_capacity())});
}

// Tensor::slice advances the view's `data` pointer by `start * nb[dim]` and Tensor
// carries no separate offset field, so a block-table row slice's `data` already points
// at its first element inside `block_tables_`. The row index therefore comes from the
// raw pointer difference alone, with no offset compensation.
Tensor PagedKVPool::rope_delta_row_for_block_table(Tensor block_table) const {
    if (block_table.data == nullptr) { return Tensor{}; }
    const auto* base = static_cast<const unsigned char*>(block_tables_.data);
    const auto* ptr  = static_cast<const unsigned char*>(block_table.data);
    const auto  span = static_cast<std::size_t>(block_tables_.numel()) * sizeof(std::int32_t);
    if (ptr < base || ptr >= base + span) {
        throw std::out_of_range("block-table slice does not belong to this pool");
    }
    const auto offset_bytes = static_cast<std::size_t>(ptr - base);
    const auto stride_bytes =
        static_cast<std::size_t>(logical_page_capacity()) * sizeof(std::int32_t);
    if (stride_bytes == 0 || offset_bytes % stride_bytes != 0) {
        throw std::invalid_argument("block-table slice is not row-aligned");
    }
    const std::int32_t row = static_cast<std::int32_t>(offset_bytes / stride_bytes);
    return rope_delta_row(row);
}

std::uint32_t PagedKVPool::entitled_pages() const noexcept { return entitled_pages_; }

std::uint32_t PagedKVPool::mapped_pages() const noexcept { return mapped_pages_; }

std::uint32_t PagedKVPool::free_pages() const noexcept {
    return static_cast<std::uint32_t>(free_page_ids_.size());
}

bool PagedKVPool::can_reserve(std::uint32_t page_entitlement) const noexcept {
    return page_entitlement != 0 && page_entitlement <= logical_page_capacity() &&
           page_entitlement <= page_group_count() - entitled_pages_;
}

bool PagedKVPool::can_replace_entitlement(std::uint32_t old_pages,
                                          std::uint32_t new_pages) const noexcept {
    return old_pages <= entitled_pages_ && new_pages <= logical_page_capacity() &&
           new_pages <= page_group_count() - (entitled_pages_ - old_pages);
}

PagedKVAllocation PagedKVPool::reserve(std::uint32_t page_entitlement) {
    if (!can_reserve(page_entitlement)) { throw std::bad_alloc(); }
    add_entitlement(page_entitlement);
    return PagedKVAllocation(*this, page_entitlement);
}

void PagedKVPool::zero_pages(std::span<const std::int32_t> page_ids, cudaStream_t stream) {
    if (page_ids.empty()) { return; }
    for (std::size_t p = 0; p < planes_.size(); ++p) {
        const Tensor& plane       = planes_[p];
        auto* base                = static_cast<unsigned char*>(plane.data);
        const std::size_t p_bytes = page_bytes(p);

        std::size_t i = 0;
        while (i < page_ids.size()) {
            std::size_t run = 1;
            while (i + run < page_ids.size() && page_ids[i + run] == page_ids[i] + static_cast<std::int32_t>(run)) {
                ++run;
            }

            const std::int32_t first_page = page_ids[i];
            if (spec_.plane_order == PagedKVPlaneOrder::PageMajor) {
                auto* dst = base + static_cast<std::int64_t>(first_page) * plane.nb[3];
                CUDA_CHECK(cudaMemsetAsync(dst, 0, run * plane.nb[3], stream));
            } else {
                for (std::int64_t head = 0; head < plane.ne[3]; ++head) {
                    auto* dst = base + head * plane.nb[3] +
                                static_cast<std::int64_t>(first_page) * plane.nb[2];
                    CUDA_CHECK(cudaMemsetAsync(dst, 0, run * plane.nb[2], stream));
                }
            }

            i += run;
        }
    }
}

std::size_t PagedKVPool::page_bytes(std::size_t plane_index) const {
    const Tensor& plane = planes_.at(plane_index);
    if (spec_.plane_order == PagedKVPlaneOrder::PageMajor) {
        return static_cast<std::size_t>(plane.nb[3]);
    }
    return static_cast<std::size_t>(plane.ne[3]) * static_cast<std::size_t>(plane.nb[2]);
}

std::size_t PagedKVPool::total_page_bytes() const noexcept {
    std::size_t bytes = 0;
    for (std::size_t p = 0; p < planes_.size(); ++p) {
        bytes += page_bytes(p);
    }
    return bytes;
}

void PagedKVPool::copy_page_to_host(std::size_t plane_index, std::int32_t page_id, void* dst,
                                    cudaStream_t stream) const {
    const std::array<std::int32_t, 1> page_ids{page_id};
    copy_pages_to_host(plane_index, page_ids, dst, stream);
}

void PagedKVPool::copy_page_from_host(std::size_t plane_index, std::int32_t page_id,
                                      const void* src, cudaStream_t stream) {
    const std::array<std::int32_t, 1> page_ids{page_id};
    copy_pages_from_host(plane_index, page_ids, src, stream);
}

void PagedKVPool::copy_pages_to_host(std::size_t plane_index, std::span<const std::int32_t> page_ids,
                                     void* dst, cudaStream_t stream) const {
    if (page_ids.empty() || dst == nullptr) { return; }
    const Tensor& plane = planes_.at(plane_index);
    const auto* base    = static_cast<const unsigned char*>(plane.data);
    auto* dst_ptr       = static_cast<unsigned char*>(dst);
    const std::size_t p_bytes = page_bytes(plane_index);

    std::size_t i = 0;
    while (i < page_ids.size()) {
        std::size_t run = 1;
        while (i + run < page_ids.size() && page_ids[i + run] == page_ids[i] + static_cast<std::int32_t>(run)) {
            ++run;
        }

        const std::int32_t first_page = page_ids[i];
        if (spec_.plane_order == PagedKVPlaneOrder::PageMajor) {
            const auto* src = base + static_cast<std::int64_t>(first_page) * plane.nb[3];
            CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src, run * plane.nb[3], cudaMemcpyDeviceToHost, stream));
        } else {
            const std::size_t width  = static_cast<std::size_t>(plane.nb[2]) * run;
            const std::size_t height = static_cast<std::size_t>(plane.ne[3]);
            const auto* src          = base + static_cast<std::int64_t>(first_page) * plane.nb[2];
            CUDA_CHECK(cudaMemcpy2DAsync(dst_ptr, width, src, plane.nb[3], width, height,
                                         cudaMemcpyDeviceToHost, stream));
        }

        dst_ptr += run * p_bytes;
        i += run;
    }
}

void PagedKVPool::copy_pages_from_host(std::size_t plane_index, std::span<const std::int32_t> page_ids,
                                       const void* src, cudaStream_t stream) {
    if (page_ids.empty() || src == nullptr) { return; }
    const Tensor& plane = planes_.at(plane_index);
    auto* base          = static_cast<unsigned char*>(plane.data);
    const auto* src_ptr = static_cast<const unsigned char*>(src);
    const std::size_t p_bytes = page_bytes(plane_index);

    std::size_t i = 0;
    while (i < page_ids.size()) {
        std::size_t run = 1;
        while (i + run < page_ids.size() && page_ids[i + run] == page_ids[i] + static_cast<std::int32_t>(run)) {
            ++run;
        }

        const std::int32_t first_page = page_ids[i];
        if (spec_.plane_order == PagedKVPlaneOrder::PageMajor) {
            auto* dst = base + static_cast<std::int64_t>(first_page) * plane.nb[3];
            CUDA_CHECK(cudaMemcpyAsync(dst, src_ptr, run * plane.nb[3], cudaMemcpyHostToDevice, stream));
        } else {
            const std::size_t width  = static_cast<std::size_t>(plane.nb[2]) * run;
            const std::size_t height = static_cast<std::size_t>(plane.ne[3]);
            auto* dst                = base + static_cast<std::int64_t>(first_page) * plane.nb[2];
            CUDA_CHECK(cudaMemcpy2DAsync(dst, plane.nb[3], src_ptr, width, width, height,
                                         cudaMemcpyHostToDevice, stream));
        }

        src_ptr += run * p_bytes;
        i += run;
    }
}

void PagedKVPool::gather_to_contiguous_device(std::span<const std::int32_t> page_ids, void* d_staging,
                                              cudaStream_t stream) const {
    if (page_ids.empty() || d_staging == nullptr || d_planes_ == nullptr) { return; }

    int32_t* d_page_ids = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_page_ids, page_ids.size() * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_page_ids, page_ids.data(), page_ids.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));

    dim3 block_dim(256);
    dim3 grid_dim((max_page_head_uint4s_ + 255) / 256, max_num_heads_, static_cast<uint32_t>(page_ids.size()));
    ops::gather_paged_kv_kernel<<<grid_dim, block_dim, 0, stream>>>(
        static_cast<const ops::PagedKVPlaneDescriptor*>(d_planes_),
        static_cast<uint32_t>(planes_.size()),
        d_page_ids, static_cast<uint32_t>(page_ids.size()),
        static_cast<uint4*>(d_staging));

    CUDA_CHECK(cudaFreeAsync(d_page_ids, stream));
}

void PagedKVPool::scatter_from_contiguous_device(std::span<const std::int32_t> page_ids, const void* d_staging,
                                                cudaStream_t stream) {
    if (page_ids.empty() || d_staging == nullptr || d_planes_ == nullptr) { return; }

    int32_t* d_page_ids = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_page_ids, page_ids.size() * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_page_ids, page_ids.data(), page_ids.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));

    dim3 block_dim(256);
    dim3 grid_dim((max_page_head_uint4s_ + 255) / 256, max_num_heads_, static_cast<uint32_t>(page_ids.size()));
    ops::scatter_paged_kv_kernel<<<grid_dim, block_dim, 0, stream>>>(
        static_cast<const ops::PagedKVPlaneDescriptor*>(d_planes_),
        static_cast<uint32_t>(planes_.size()),
        d_page_ids, static_cast<uint32_t>(page_ids.size()),
        static_cast<const uint4*>(d_staging));

    CUDA_CHECK(cudaFreeAsync(d_page_ids, stream));
}

std::vector<std::int32_t> PagedKVPool::take_pages(std::uint32_t count,
                                                  std::int32_t preferred_first) {
    if (count == 0) { return {}; }
    if (count > free_page_ids_.size()) {
        throw std::logic_error("Paged KV entitlement could not be materialized");
    }

    const auto run_at = [&](std::size_t begin) {
        if (begin + count > free_page_ids_.size()) { return false; }
        for (std::uint32_t offset = 1; offset < count; ++offset) {
            if (free_page_ids_[begin + offset] !=
                free_page_ids_[begin] + static_cast<std::int32_t>(offset)) {
                return false;
            }
        }
        return true;
    };

    std::size_t selected = free_page_ids_.size();
    if (preferred_first >= 0) {
        const auto it =
            std::lower_bound(free_page_ids_.begin(), free_page_ids_.end(), preferred_first);
        if (it != free_page_ids_.end() && *it == preferred_first) {
            const auto begin = static_cast<std::size_t>(it - free_page_ids_.begin());
            if (run_at(begin)) { selected = begin; }
        }
    }
    if (selected == free_page_ids_.size()) {
        for (std::size_t begin = 0; begin + count <= free_page_ids_.size(); ++begin) {
            if (run_at(begin)) {
                selected = begin;
                break;
            }
        }
    }

    std::vector<std::int32_t> out;
    out.reserve(count);
    if (selected != free_page_ids_.size()) {
        const auto first = free_page_ids_.begin() + static_cast<std::ptrdiff_t>(selected);
        const auto last  = first + static_cast<std::ptrdiff_t>(count);
        out.insert(out.end(), first, last);
        free_page_ids_.erase(first, last);
    } else {
        const auto last = free_page_ids_.begin() + static_cast<std::ptrdiff_t>(count);
        out.insert(out.end(), free_page_ids_.begin(), last);
        free_page_ids_.erase(free_page_ids_.begin(), last);
    }
    mapped_pages_ += count;
    return out;
}

void PagedKVPool::return_pages(std::span<const std::int32_t> pages) noexcept {
    if (pages.empty()) { return; }
    free_page_ids_.insert(free_page_ids_.end(), pages.begin(), pages.end());
    std::sort(free_page_ids_.begin(), free_page_ids_.end());
    mapped_pages_ -= static_cast<std::uint32_t>(pages.size());
}

void PagedKVPool::add_entitlement(std::uint32_t pages) noexcept { entitled_pages_ += pages; }

void PagedKVPool::replace_entitlement(std::uint32_t old_pages, std::uint32_t new_pages) noexcept {
    entitled_pages_ = entitled_pages_ - old_pages + new_pages;
}

void PagedKVPool::acquire_row(std::int32_t row) {
    if (row < 0 || row >= table_row_count()) {
        throw std::out_of_range("Paged KV block-table row out of range");
    }
    if (row_in_use_[static_cast<std::size_t>(row)]) {
        throw std::logic_error("Paged KV block-table row is already bound");
    }
    row_in_use_[static_cast<std::size_t>(row)] = true;
}

void PagedKVPool::release_row(std::int32_t row) noexcept {
    row_in_use_[static_cast<std::size_t>(row)] = false;
}

PagedKVAllocation::PagedKVAllocation(PagedKVPool& pool, std::uint32_t page_entitlement)
    : pool_(&pool), page_entitlement_(page_entitlement) {
    page_ids_.reserve(page_entitlement);
    rope_deltas_.reserve(page_entitlement);
}

PagedKVAllocation::~PagedKVAllocation() { release(); }

PagedKVAllocation::PagedKVAllocation(PagedKVAllocation&& other) noexcept
    : pool_(other.pool_), page_ids_(std::move(other.page_ids_)),
      rope_deltas_(std::move(other.rope_deltas_)),
      page_entitlement_(other.page_entitlement_), bound_row_(other.bound_row_) {
    other.pool_             = nullptr;
    other.page_entitlement_ = 0;
    other.bound_row_        = -1;
}

PagedKVAllocation& PagedKVAllocation::operator=(PagedKVAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    release();
    pool_                   = other.pool_;
    page_ids_               = std::move(other.page_ids_);
    rope_deltas_            = std::move(other.rope_deltas_);
    page_entitlement_       = other.page_entitlement_;
    bound_row_              = other.bound_row_;
    other.pool_             = nullptr;
    other.page_entitlement_ = 0;
    other.bound_row_        = -1;
    return *this;
}

bool PagedKVAllocation::valid() const noexcept { return pool_ != nullptr; }

std::uint32_t PagedKVAllocation::page_entitlement() const noexcept { return page_entitlement_; }

std::uint32_t PagedKVAllocation::mapped_page_count() const noexcept {
    return static_cast<std::uint32_t>(page_ids_.size());
}

std::uint32_t PagedKVAllocation::mapped_token_capacity() const noexcept {
    return mapped_page_count() * static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::int32_t PagedKVAllocation::bound_row() const noexcept { return bound_row_; }

std::span<const std::int32_t> PagedKVAllocation::page_ids() const noexcept { return page_ids_; }

bool PagedKVAllocation::belongs_to(const PagedKVPool& pool) const noexcept {
    return pool_ == &pool;
}

void PagedKVAllocation::set_page_entitlement(std::uint32_t pages) {
    if (!valid() || pages < mapped_page_count()) {
        throw std::invalid_argument("Paged KV entitlement is smaller than mapped pages");
    }
    if (!pool_->can_replace_entitlement(page_entitlement_, pages)) { throw std::bad_alloc(); }
    page_ids_.reserve(pages);
    pool_->replace_entitlement(page_entitlement_, pages);
    page_entitlement_ = pages;
}

void PagedKVAllocation::cancel_unmapped_entitlement() noexcept {
    if (!valid()) { return; }
    const std::uint32_t mapped = mapped_page_count();
    pool_->replace_entitlement(page_entitlement_, mapped);
    page_entitlement_ = mapped;
}

void PagedKVAllocation::materialize_pages(std::uint32_t pages, cudaStream_t stream) {
    if (!valid() || pages < mapped_page_count() || pages > page_entitlement_) {
        throw std::invalid_argument("Paged KV materialize extent is outside entitlement");
    }
    const std::uint32_t old_count = mapped_page_count();
    const std::uint32_t count     = pages - old_count;
    if (count == 0) { return; }
    const std::int32_t preferred =
        page_ids_.empty() ? -1 : static_cast<std::int32_t>(page_ids_.back() + 1);
    std::vector<std::int32_t> acquired = pool_->take_pages(count, preferred);
    page_ids_.insert(page_ids_.end(), acquired.begin(), acquired.end());
    rope_deltas_.insert(rope_deltas_.end(), count, kRopeDeltaNone);
    if (bound_row_ >= 0) { publish_range(old_count, count, stream); }
}

void PagedKVAllocation::materialize_tokens(std::uint32_t tokens, cudaStream_t stream) {
    materialize_pages(pages_for_tokens(tokens), stream);
}

void PagedKVAllocation::trim_pages(std::uint32_t pages) {
    if (!valid()) { throw std::logic_error("Cannot trim an empty Paged KV allocation"); }
    if (pages > mapped_page_count()) {
        throw std::invalid_argument("Paged KV trim extent exceeds mapped pages");
    }
    if (pages == mapped_page_count()) { return; }
    pool_->return_pages(std::span<const std::int32_t>(
        page_ids_.data() + pages, static_cast<std::size_t>(mapped_page_count() - pages)));
    page_ids_.resize(pages);
    rope_deltas_.resize(pages);
}

void PagedKVAllocation::trim_tokens(std::uint32_t tokens) { trim_pages(pages_for_tokens(tokens)); }

void PagedKVAllocation::reassign_pages(std::span<const std::int32_t> mapping,
                                       std::span<const std::int32_t> deltas,
                                       cudaStream_t stream) {
    if (!valid()) { throw std::logic_error("Cannot reassign an empty Paged KV allocation"); }
    if (mapping.empty()) {
        throw std::invalid_argument("Paged KV reassignment must map at least one page");
    }
    if (mapping.size() > page_entitlement_) {
        throw std::invalid_argument("Paged KV reassignment exceeds the page entitlement");
    }
    if (deltas.size() != mapping.size()) {
        throw std::invalid_argument(
            "Paged KV reassignment delta plane does not parallel its mapping");
    }

    // Remaining ownership, by page. An entry the mapping names consumes one occurrence, so a page
    // owned once cannot be claimed by two logical pages.
    std::unordered_map<std::int32_t, std::uint32_t> remaining;
    remaining.reserve(page_ids_.size());
    for (const std::int32_t page : page_ids_) { ++remaining[page]; }

    std::uint32_t acquire_count = 0;
    for (const std::int32_t page : mapping) {
        if (page == kPagedKVAquirePage) {
            ++acquire_count;
            continue;
        }
        const auto found = remaining.find(page);
        if (found == remaining.end() || found->second == 0) {
            throw std::invalid_argument(
                "Paged KV reassignment names a page this allocation does not own");
        }
        if (--found->second == 0) { remaining.erase(found); }
    }

    // Occurrences the mapping did not claim are surplus. Those pages can themselves serve as the
    // newly acquired physical pages, so only a shortfall beyond them has to come from the pool. A
    // remap that shrinks one region while growing another therefore does not need free pool pages
    // for the growth: the pool is never asked for pages the release already paid for.
    std::vector<std::int32_t> surplus;
    surplus.reserve(page_ids_.size());
    for (const auto& entry : remaining) {
        for (std::uint32_t copy = 0; copy < entry.second; ++copy) { surplus.push_back(entry.first); }
    }

    const std::uint32_t surplus_reused =
        std::min(acquire_count, static_cast<std::uint32_t>(surplus.size()));
    const std::uint32_t from_pool = acquire_count - surplus_reused;
    if (from_pool > pool_->free_pages()) { throw std::bad_alloc(); }

    // Acquire before releasing anything: once this returns no later step can fail, so the pool
    // never observes a page that is both still owned here and already free.
    std::vector<std::int32_t> acquired;
    if (from_pool != 0) {
        const std::int32_t preferred =
            page_ids_.empty() ? -1 : static_cast<std::int32_t>(page_ids_.back() + 1);
        acquired = pool_->take_pages(from_pool, preferred);
    }

    std::vector<std::int32_t> final_ids(mapping.begin(), mapping.end());
    std::size_t surplus_slot = 0;
    std::size_t pooled_slot  = 0;
    for (std::int32_t& page : final_ids) {
        if (page != kPagedKVAquirePage) { continue; }
        if (surplus_slot < surplus_reused) {
            page = surplus[surplus_slot++];
        } else {
            page = acquired[pooled_slot++];
        }
    }
    if (pooled_slot != acquired.size()) {
        // Unreachable: surplus_reused + from_pool == acquire_count, which is exactly how many slots
        // were substituted. Hand the pages back rather than leak them if that ever stops holding.
        if (!acquired.empty()) { pool_->return_pages(acquired); }
        throw std::logic_error("Paged KV reassignment acquired more pages than it mapped");
    }

    // Surplus occurrences the mapping did not reuse are genuinely dropped.
    if (surplus.size() > surplus_reused) {
        pool_->return_pages(std::span<const std::int32_t>(surplus.data() + surplus_reused,
                                                          surplus.size() - surplus_reused));
    }

    page_ids_ = std::move(final_ids);
    rope_deltas_.assign(deltas.begin(), deltas.end());
    if (bound_row_ >= 0) { publish_mapping(stream); }
}

void PagedKVAllocation::relocate_copied_pages(std::span<const std::int32_t> mapping,
                                              std::span<const std::int32_t> deltas,
                                              std::span<const RelocatedCopySpan> spans,
                                              cudaStream_t stream) {
    if (!valid()) { throw std::logic_error("Cannot relocate into an empty Paged KV allocation"); }
    if (mapping.empty()) {
        throw std::invalid_argument("Paged KV relocation must map at least one page");
    }
    if (mapping.size() > page_entitlement_) {
        throw std::invalid_argument("Paged KV relocation exceeds the page entitlement");
    }
    if (deltas.size() != mapping.size()) {
        throw std::invalid_argument("Paged KV relocation delta plane does not parallel its mapping");
    }
    if (spans.empty()) {
        throw std::invalid_argument("Paged KV relocation carries no spans to copy");
    }

    // The mapping as it stands is the copy's source, and it stays valid until the copies have run.
    const std::vector<std::int32_t> sources(page_ids_.begin(), page_ids_.end());
    if (mapping.size() != sources.size()) {
        throw std::invalid_argument("Paged KV relocation does not resize its mapping");
    }

    // The moves, then the pages they read. Both are pure arithmetic over page ids -- in
    // `core/paged_kv_delta_layout.h`, so they are testable without a GPU -- and both refuse a span that
    // does not fit the mapping, which is what keeps a half-copied relocation from being published.
    const std::vector<RelocatedPageMove> moves = plan_relocated_page_moves(spans, sources.size());
    std::vector<std::int32_t> protected_pages;
    protected_pages.reserve(moves.size());
    for (const RelocatedPageMove& move : moves) {
        protected_pages.push_back(sources[static_cast<std::size_t>(move.source_page)]);
    }

    // The substitution `reassign_pages` performs, with the one difference that a page the copies read
    // is never a destination. That path is hardware-verified and keeps its own copy of the arithmetic;
    // this is the same transition with the sources held out of the reuse.
    const RelocatedPageSubstitution substitution =
        plan_relocated_page_substitution(sources, mapping, protected_pages);
    if (substitution.from_pool > pool_->free_pages()) { throw std::bad_alloc(); }

    // Acquire before releasing anything, as `reassign_pages` does: once the pool has answered, no later
    // step can fail, so the pool never observes a page that is both still owned here and already free.
    std::vector<std::int32_t> acquired;
    if (substitution.from_pool != 0) {
        const std::int32_t preferred =
            page_ids_.empty() ? -1 : static_cast<std::int32_t>(page_ids_.back() + 1);
        acquired = pool_->take_pages(substitution.from_pool, preferred);
    }

    std::vector<std::int32_t> final_ids = substitution.final_ids;
    std::size_t pooled_slot = 0;
    for (std::int32_t& page : final_ids) {
        if (page != kPagedKVAquirePage) { continue; }
        page = acquired[pooled_slot++];
    }
    if (pooled_slot != acquired.size()) {
        // Unreachable: `from_pool` is exactly the number of acquire slots the substitution left, which
        // is the count this loop filled. Hand the pages back rather than leak them if that stops
        // holding.
        if (!acquired.empty()) { pool_->return_pages(acquired); }
        throw std::logic_error("Paged KV relocation acquired more pages than it mapped");
    }

    // The copies. A destination page's payload is contiguous per head and the head pitch is the same
    // on both sides, so the tokens a move carries are one 2-D copy per plane: `width` is the byte span
    // of those tokens, `height` the head count, and the pitch the head's stride. Two moves per
    // destination page where the source page boundary falls inside it, one where it does not.
    const bool page_major = pool_->spec_.plane_order == PagedKVPlaneOrder::PageMajor;
    for (std::size_t plane_index = 0; plane_index < pool_->plane_count(); ++plane_index) {
        const Tensor& plane = pool_->plane(plane_index);
        if (plane.ne[1] != kPagedKVPageSize || plane.ne[0] <= 0) {
            throw std::logic_error("a KV plane is not paged over its token dimension");
        }
        // The strides are read from the plane rather than derived, because the two plane orders put
        // the page on opposite ends of the shape: `plan_paged_kv_pool` builds
        // {leading, page, heads, pages} or {leading, page, pages, heads}, and a hand-derived page
        // stride is right for exactly one of them. A dense leading dimension is required -- without it
        // a token is not a contiguous run and a move is not a 2-D strided copy.
        if (plane.nb[0] != static_cast<std::int64_t>(dtype_size(plane.dtype)) || plane.nb[1] <= 0) {
            throw std::logic_error("a KV plane's leading dimension is not dense");
        }
        const std::size_t token_bytes = static_cast<std::size_t>(plane.nb[1]);
        const std::size_t heads       = static_cast<std::size_t>(page_major ? plane.ne[2] : plane.ne[3]);
        const std::size_t head_stride = static_cast<std::size_t>(page_major ? plane.nb[2] : plane.nb[3]);
        const std::size_t page_stride = static_cast<std::size_t>(page_major ? plane.nb[3] : plane.nb[2]);
        if (heads == 0 || head_stride == 0 || page_stride == 0) {
            throw std::logic_error("a KV plane's page or head stride is empty");
        }
        auto* base = static_cast<std::uint8_t*>(plane.data);

        for (const RelocatedPageMove& move : moves) {
            const auto* source =
                base +
                static_cast<std::size_t>(sources[static_cast<std::size_t>(move.source_page)]) *
                    page_stride +
                static_cast<std::size_t>(move.source_token) * token_bytes;
            auto* dest =
                base +
                static_cast<std::size_t>(final_ids[static_cast<std::size_t>(move.dest_page)]) *
                    page_stride +
                static_cast<std::size_t>(move.dest_token) * token_bytes;
            CUDA_CHECK(cudaMemcpy2DAsync(dest, head_stride, source, head_stride,
                                         static_cast<std::size_t>(move.tokens) * token_bytes, heads,
                                         cudaMemcpyDeviceToDevice, stream));
        }
    }

    // Hold the sources until their content has been read. The pool is shared and another lane takes a
    // released page on its own stream, with nothing ordering against these copies, so releasing first
    // would let a prefill overwrite a page mid-copy. The drain costs the copy and nothing else.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!substitution.held_for_copy.empty()) { pool_->return_pages(substitution.held_for_copy); }

    page_ids_ = std::move(final_ids);
    rope_deltas_.assign(deltas.begin(), deltas.end());
    if (bound_row_ >= 0) { publish_mapping(stream); }
}

void PagedKVAllocation::reassign_pages(std::span<const std::int32_t> mapping,
                                       cudaStream_t stream) {
    // Temporary shim for the pre-Step-3 caller in program_impl.h, which rerotates the K plane
    // itself and has no deltas to supply. All-zero deltas are correct for that caller: it never
    // relies on the delta plane. Step 3 converts that call site and deletes this overload.
    const std::vector<std::int32_t> none(mapping.size(), kRopeDeltaNone);
    reassign_pages(mapping, none, stream);
}

void PagedKVAllocation::bind_row(std::int32_t row, cudaStream_t stream) {
    if (!valid()) { throw std::logic_error("Cannot bind an empty Paged KV allocation"); }
    if (bound_row_ >= 0) { throw std::logic_error("Paged KV allocation is already bound"); }
    // Claim the row *before* clearing its deltas, and give it back if the clear fails.
    //
    // The order matters and the obvious one is wrong. Clearing first means that when `acquire_row`
    // rejects the row because another allocation still owns it, that other allocation's deltas have
    // already been zeroed: it is still bound, still believes its relocated pages carry shifts, and
    // attention now reads `kRopeDeltaNone` for pages that were actually moved. That is the same
    // silent-corruption class as a stale shift -- a plausible zero instead of a plausible number --
    // and it is not recoverable, because nothing downstream can tell.
    //
    // Acquiring first converts that into the failure the previous revision was trying to avoid, a row
    // left marked in use, and the catch below then undoes it. A row stuck in use is loud: the next
    // bind to that row throws. Choose the failure that reports itself.
    pool_->acquire_row(row);
    try {
        pool_->clear_rope_delta_row(row, stream);
    } catch (...) {
        pool_->release_row(row);
        throw;
    }
    bound_row_ = row;
    publish_mapping(stream);
}

void PagedKVAllocation::publish_mapping(cudaStream_t stream) const {
    if (bound_row_ < 0) { throw std::logic_error("Paged KV allocation is not bound"); }
    publish_range(0, mapped_page_count(), stream);
}

void PagedKVAllocation::publish_range(std::uint32_t first_page, std::uint32_t page_count,
                                      cudaStream_t stream) const {
    if (page_count == 0) { return; }
    if (rope_deltas_.size() != page_ids_.size()) {
        throw std::logic_error("Paged KV delta plane does not parallel its block table");
    }

    Tensor block_row = pool_->block_table_row(bound_row_);
    auto* dst_block = static_cast<std::int32_t*>(block_row.data) + first_page;
    const auto* src_block = page_ids_.data() + first_page;
    CUDA_CHECK(cudaMemcpyAsync(dst_block, src_block,
                               static_cast<std::size_t>(page_count) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));

    Tensor delta_row = pool_->rope_delta_row(bound_row_);
    auto* dst_delta = static_cast<std::int32_t*>(delta_row.data) + first_page;
    const auto* src_delta = rope_deltas_.data() + first_page;
    CUDA_CHECK(cudaMemcpyAsync(dst_delta, src_delta,
                               static_cast<std::size_t>(page_count) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
}

void PagedKVAllocation::unbind_row() noexcept {
    if (bound_row_ < 0) { return; }
    pool_->release_row(bound_row_);
    bound_row_ = -1;
}

Tensor PagedKVAllocation::block_table() const {
    if (bound_row_ < 0) { throw std::logic_error("Paged KV allocation is not bound"); }
    return pool_->block_table_row(bound_row_);
}

void PagedKVAllocation::release() noexcept {
    if (!valid()) { return; }
    unbind_row();
    pool_->return_pages(page_ids_);
    pool_->replace_entitlement(page_entitlement_, 0);
    page_ids_.clear();
    rope_deltas_.clear();
    page_entitlement_ = 0;
    pool_             = nullptr;
}

std::vector<PagedKVAllocation>
reserve_paged_kv_bundle(std::span<const PagedKVReservation> reservations) {
    validate_distinct_pools(reservations);
    for (const PagedKVReservation& reservation : reservations) {
        if (!reservation.pool->can_reserve(reservation.page_entitlement)) {
            throw std::bad_alloc();
        }
    }

    std::vector<PagedKVAllocation> allocations;
    allocations.reserve(reservations.size());
    for (const PagedKVReservation& reservation : reservations) {
        allocations.push_back(reservation.pool->reserve(reservation.page_entitlement));
    }
    return allocations;
}

void resize_paged_kv_bundle(std::span<const PagedKVResize> changes) {
    for (std::size_t i = 0; i < changes.size(); ++i) {
        const PagedKVResize& change = changes[i];
        if (change.allocation == nullptr || !change.allocation->valid()) {
            throw std::invalid_argument("Paged KV resize must name a live allocation");
        }
        if (change.mapped_pages > change.allocation->mapped_page_count() ||
            change.mapped_pages > change.page_entitlement) {
            throw std::invalid_argument("Paged KV resize extents are inconsistent");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (change.allocation == changes[j].allocation ||
                change.allocation->pool_ == changes[j].allocation->pool_) {
                throw std::invalid_argument("Paged KV resize contains the same pool twice");
            }
        }
        if (!change.allocation->pool_->can_replace_entitlement(change.allocation->page_entitlement_,
                                                               change.page_entitlement)) {
            throw std::bad_alloc();
        }
    }
    // Complete every potentially throwing host allocation before changing any pool accounting.
    for (const PagedKVResize& change : changes) {
        change.allocation->page_ids_.reserve(change.page_entitlement);
    }
    for (const PagedKVResize& change : changes) {
        change.allocation->trim_pages(change.mapped_pages);
        change.allocation->set_page_entitlement(change.page_entitlement);
    }
}

} // namespace ninfer

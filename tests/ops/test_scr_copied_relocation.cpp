// Device oracle for the copied relocation: a run whose *source* is off the
// 64-token page grid, relocated by payload copy, held bit-for-bit against the
// pages it was copied from and read back through the public attention entry
// points.
//
// Why this is its own file rather than an extension of
// test_scr_relocation_integration.cpp: the copier is a method on
// PagedKVAllocation inside a real PagedKVPool, so the harness has to be
// pool-backed, while the integration test is BF16-only and hand-builds its own
// planes through gqa_relocation_test_common.h's Cache. The shared oracle
// pattern is reused by inclusion instead -- ideal_attention_with_delta,
// expect_conformant, expect_control_outside/expect_control_separates and the
// criteria are templates and constants over CacheType::HostImage /
// k_logical / v_logical, and the pool-backed adapter below satisfies that
// without touching the shared header.
//
// The scenario is not hand-built. Two synthetic prompts drive the real
// detector -> selector -> copied-remap -> move-decomposition chain
// (diff_prompt_edit_shifted, plan_shifted_relocations, plan_copied_page_remap,
// rope_delta_for_dest_page), and the planner's output is held against the
// geometry this file verified independently, so a planner change is a failure
// rather than a silent re-shape of the test. The prompt pair leaves three
// survivors: an off-grid source (two-move destination pages), an aligned source
// (the single-move branch), and a run three destination pages long, over a
// 1408-token (22-page) prompt whose first 576 tokens are prefix reuse.
//
// The assertion groups, in order:
//   1. bytes: every moved range bit-identical to its source, per plane (codes
//      and both scale planes), per head, with no tolerance at all;
//   2. nothing else moved: every page equals its pre-copy image with exactly
//      the moved ranges overwritten;
//   3. what was published: page_ids() is the mapping with acquire slots
//      filled, the block-table row agrees with it, and the pool's delta row
//      carries the relocation's shifts -- the two published in one call;
//   4. pool accounting: the pages the copies read are held until after the
//      drain and then returned, the pool demand was taken, and no destination
//      page is left free;
//   5. operational equivalence: attention over the copied cache through the
//      public entry points, judged by the anchored FP64 delta oracle
//      (ideal_attention_with_delta), with the directly prefilled cache as the
//      anchor and a zeroed delta plane as the control that must break the
//      criterion;
//   6. aliasing: the same relocation on a pool sized so the free pages are
//      exactly the relocation's demand, where a substitution that handed a
//      source page back as a destination would produce an overlapping
//      cudaMemcpy2DAsync;
//   7. refusals: a non-parallel delta plane, empty spans, a resized mapping, a
//      mapping past the entitlement, and a pool that cannot afford the demand
//      -- each throws before any page moves.
//
// One limitation is recorded rather than papered over: the matrix of group 5
// is PageMajor only. The attention Ops address pages page-major
// (paged_kv_page_head_offset in src/ops/kernel/paged_kv_address.cuh), so a
// HeadMajor plane neither passes the wrapper's shape validation nor can be
// written by ops::gqa_kv_append. HeadMajor cells therefore seed through a
// scratch PageMajor pool of the same spec (every byte still comes from the
// production writer) and run groups 1-4, 6 and 7, which are exactly the groups
// that exercise the copier's order-independent stride arithmetic.

#include "ops/gqa_relocation_test_common.h"

#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/prompt_edit.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using ninfer::targets::qwen3_6::detail::PromptEditDiff;
using ninfer::targets::qwen3_6::detail::RelocatedSpan;
using ninfer::targets::qwen3_6::detail::diff_prompt_edit_shifted;
using ninfer::targets::qwen3_6::detail::plan_copied_page_remap;
using ninfer::targets::qwen3_6::detail::plan_shifted_relocations;

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
}

void note(const std::string& what) { std::printf("  %s\n", what.c_str()); }

// ---------------------------------------------------------------- geometry ---

// 22 owned pages + 7 the relocation takes from the pool + one page of slack, so
// the main cell is not accidentally tight. The tight cell is sized below.
constexpr std::uint32_t kPoolPages      = 30;
constexpr std::uint32_t kTightPoolPages = 29; // free == the relocation's demand

constexpr std::int32_t kPromptPages  = 22;
constexpr std::int32_t kPromptTokens = kPromptPages * kPageSize;
constexpr std::int32_t kPrefixTokens = 576; // the prefix-reuse frontier

// Queries sit inside the last relocated destination page, so all three spans'
// keys are visible and every page they read was written in both caches.
constexpr std::int32_t kQueryBase   = 1056;
constexpr std::int32_t kQueryTokens = 16;
constexpr std::int32_t kVisible     = kQueryBase + kQueryTokens;
// The A1 window is the page after the relocated region, which prefill itself
// would write. Its width is deliberately wider than kMaximumVerifyTokens (16)
// and kSmallTChunkTokens (8), so the public call resolves to the Prompt route --
// the route the delta plane is qualified on -- rather than to a verify route.
constexpr std::int32_t kTailFirst  = 1088;
constexpr std::int32_t kTailTokens = 32;

// Writes go through the production fill path in chunks, the way prefill does.
constexpr std::int32_t kChunkTokens = 256;

// The shifted detector's anchor size: a survivor shorter than this cannot be
// anchored at all, and all three survivors here are whole pages long.
constexpr std::uint32_t kPromptEditAnchor = 16;

// The expected planner geometry, held against what the real chain produces.
struct ExpectedSpan {
    std::int32_t source_begin, dest_begin, tokens;
};
constexpr ExpectedSpan kExpectedSpans[] = {
    {741, 576, 128},  // off-grid source: two-move destination pages
    {960, 768, 64},   // aligned source: the single-move branch
    {1216, 896, 192}, // three destination pages
};

// The expected move decomposition, declared in RelocatedPageMove's own field order and with its
// field types, so the literals below can be read against the table in the header and the comparison
// carries no sign conversions. The order is load-bearing: grouping the two int32 fields together
// instead makes `{9, 0, 11, 37, 27}` mean source_page 0 and dest_token 11, which is not the move it
// spells out, and every comparison then fails on a correctly decomposed plan.
struct ExpectedMove {
    std::int32_t dest_page;
    std::uint32_t dest_token;
    std::int32_t source_page;
    std::uint32_t source_token;
    std::uint32_t tokens;
};
constexpr ExpectedMove kExpectedMoves[] = {
    {9, 0, 11, 37, 27},  {9, 27, 12, 0, 37},  {10, 0, 12, 37, 27},
    {10, 27, 13, 0, 37}, {12, 0, 15, 0, 64},  {14, 0, 19, 0, 64},
    {15, 0, 20, 0, 64},  {16, 0, 21, 0, 64},
};

// The two in-scope variants, with the plane inventory
// src/targets/qwen3_6/impl/state/decoder_state.cpp builds for each.
struct Variant {
    const char* name;
    DType dtype;
    bool packed_v, rotate_k, rotate_v, packed_k, e8_lattice;
    std::array<PagedKVPlaneSpec, 4> planes;
    std::size_t plane_count;
};

constexpr Variant kVariants[] = {
    {"bf16",
     DType::BF16,
     false,
     false,
     false,
     false,
     false,
     {{{DType::BF16, kHeadDim, kKvHeads}, {DType::BF16, kHeadDim, kKvHeads}}},
     2},
    {"rk4v4-e8",
     DType::I8,
     true,
     true,
     true,
     true,
     true,
     {{{DType::U8, kHeadDim / 2, kKvHeads},
       {DType::U8, kHeadDim / 2, kKvHeads},
       {DType::FP16, kQuantGroups, kKvHeads},
       {DType::FP16, kQuantGroups, kKvHeads}}},
     4},
};

constexpr PagedKVPlaneOrder kOrders[] = {PagedKVPlaneOrder::PageMajor,
                                         PagedKVPlaneOrder::HeadMajor};

// ------------------------------------------------------------------ content ---

// Pre-RoPE keys and values for one whole prompt, in activation order
// [head][token][d]. V carries no rotation, so a token's V is the same in every
// prompt it appears in.
struct TokenContent {
    std::vector<double> raw_k;
    std::vector<double> v;
};

TokenContent make_content(std::uint32_t seed) {
    TokenContent content;
    const std::size_t elems =
        static_cast<std::size_t>(kHeadDim) * kKvHeads * static_cast<std::size_t>(kPromptTokens);
    content.raw_k.resize(elems);
    content.v.resize(elems);
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unit(-0.25, 0.25);
    for (std::size_t i = 0; i < elems; ++i) {
        content.raw_k[i] = normal(rng);
        content.v[i]     = unit(rng);
    }
    return content;
}

// The current prompt's content: fresh random tokens, with every relocated
// destination position re-sourced from the token the source run carries there.
// That is what makes the directly prefilled cache the oracle of this scenario:
// the same tokens at the destination positions, rotated at the destination.
TokenContent reedited_content(const TokenContent& base,
                              const std::vector<RelocatedSpan>& spans) {
    TokenContent content = make_content(0x5eed1234u);
    const auto copy_span = [&content, &base](std::int32_t from, std::int32_t to,
                                             std::int32_t tokens) {
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t head = 0; head < kKvHeads; ++head) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    content.raw_k[act_index(head, to + t, kKvHeads, d)] =
                        base.raw_k[act_index(head, from + t, kKvHeads, d)];
                    content.v[act_index(head, to + t, kKvHeads, d)] =
                        base.v[act_index(head, from + t, kKvHeads, d)];
                }
            }
        }
    };
    for (const RelocatedSpan& span : spans) {
        copy_span(static_cast<std::int32_t>(span.source_begin),
                  static_cast<std::int32_t>(span.dest_begin),
                  static_cast<std::int32_t>(span.tokens));
    }
    // The shared prefix is identical in both prompts by construction.
    copy_span(0, 0, kPrefixTokens);
    return content;
}

// The BF16 activations the production writer consumes: bf16(rope(raw, position)).
std::vector<__nv_bfloat16> rope_tokens(const TokenContent& content, const FreqTable& freq) {
    const std::size_t elems = content.raw_k.size();
    std::vector<__nv_bfloat16> out(elems);
    std::array<double, kHeadDim> row{};
    std::array<double, kHeadDim> rotated{};
    for (std::int32_t position = 0; position < kPromptTokens; ++position) {
        for (std::int32_t head = 0; head < kKvHeads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                row[static_cast<std::size_t>(d)] =
                    content.raw_k[act_index(head, position, kKvHeads, d)];
            }
            forward_rope(row.data(), position, freq, rotated.data());
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                out[act_index(head, position, kKvHeads, d)] =
                    __float2bfloat16(static_cast<float>(rotated[static_cast<std::size_t>(d)]));
            }
        }
    }
    return out;
}

// ------------------------------------------------------------------ prompts ---

// The two prompts and the spans the real detector and selector find in them.
// The region ids are chosen so no two regions share a token id: a run cannot
// grow past its own region, and no block of one region can match a block of
// another. The previous prompt is P R1 S1 R2 S2 R3 S3; the current one is
// P S1 E2 S2 E3 S3 E4.
struct PromptPair {
    std::vector<TokenId> previous;
    std::vector<TokenId> current;
    std::vector<RelocatedSpan> spans;
    std::uint32_t prefix_reuse = 0;
};

PromptPair build_prompt_pair() {
    PromptPair pair;
    pair.previous.assign(static_cast<std::size_t>(kPromptTokens), 0);
    pair.current.assign(static_cast<std::size_t>(kPromptTokens), 0);

    const auto fill = [](std::vector<TokenId>& tokens, std::int32_t begin, std::int32_t end,
                         TokenId base) {
        for (std::int32_t i = begin; i < end; ++i) {
            tokens[static_cast<std::size_t>(i)] = base + static_cast<TokenId>(i - begin);
        }
    };
    fill(pair.previous, 0, 576, 0);
    fill(pair.previous, 576, 741, 1000);
    fill(pair.previous, 741, 869, 500);
    fill(pair.previous, 869, 960, 2000);
    fill(pair.previous, 960, 1024, 700);
    fill(pair.previous, 1024, 1216, 3000);
    fill(pair.previous, 1216, 1408, 900);
    fill(pair.current, 0, 576, 0);
    fill(pair.current, 576, 704, 500);
    fill(pair.current, 704, 768, 4000);
    fill(pair.current, 768, 832, 700);
    fill(pair.current, 832, 896, 5000);
    fill(pair.current, 896, 1088, 900);
    fill(pair.current, 1088, 1408, 6000);

    const PromptEditDiff diff =
        diff_prompt_edit_shifted(pair.previous, pair.current, kPromptEditAnchor);
    if (diff.spans.size() != 4) {
        fail("prompt fixture: the shifted detector found " + std::to_string(diff.spans.size()) +
             " survivors, expected the prefix plus three");
        return pair;
    }
    pair.prefix_reuse = diff.leading_prefix_tokens();
    if (pair.prefix_reuse != static_cast<std::uint32_t>(kPrefixTokens)) {
        fail("prompt fixture: prefix reuse is " + std::to_string(pair.prefix_reuse) + ", expected " +
             std::to_string(kPrefixTokens));
        return pair;
    }
    pair.spans = plan_shifted_relocations(diff, static_cast<std::uint32_t>(kPromptTokens),
                                          pair.prefix_reuse,
                                          static_cast<std::uint32_t>(kPromptTokens), std::nullopt, 6);
    if (pair.spans.size() != std::size(kExpectedSpans)) {
        fail("prompt fixture: the selector returned " + std::to_string(pair.spans.size()) +
             " spans, expected " + std::to_string(std::size(kExpectedSpans)));
        return pair;
    }
    for (std::size_t i = 0; i < pair.spans.size(); ++i) {
        const RelocatedSpan& got = pair.spans[i];
        const ExpectedSpan& want = kExpectedSpans[i];
        if (got.source_begin != static_cast<std::uint32_t>(want.source_begin) ||
            got.dest_begin != static_cast<std::uint32_t>(want.dest_begin) ||
            got.tokens != static_cast<std::uint32_t>(want.tokens)) {
            fail("prompt fixture: span " + std::to_string(i) + " is src=" +
                 std::to_string(got.source_begin) + " dst=" + std::to_string(got.dest_begin) +
                 " tokens=" + std::to_string(got.tokens) + ", expected src=" +
                 std::to_string(want.source_begin) + " dst=" + std::to_string(want.dest_begin) +
                 " tokens=" + std::to_string(want.tokens));
        }
    }
    return pair;
}

// ------------------------------------------------------------------- pool ----

// A pool, its arena, and one bound allocation inside it. Owned together because
// the views and page images are non-owning.
struct PoolCell {
    const Variant* variant  = nullptr;
    PagedKVPlaneOrder order = PagedKVPlaneOrder::PageMajor;
    PagedKVPoolLayout layout;
    std::optional<DeviceArena> arena;
    std::unique_ptr<PagedKVPool> pool;
    PagedKVAllocation alloc;

    PoolCell() = default;
    PoolCell(const PoolCell&)            = delete;
    PoolCell& operator=(const PoolCell&) = delete;
};

void build_cell(PoolCell& cell, const Variant& variant, PagedKVPlaneOrder order,
                std::uint32_t page_groups, cudaStream_t stream) {
    cell.variant = &variant;
    cell.order   = order;
    LayoutBuilder builder;
    PagedKVPoolSpec spec;
    spec.page_group_count      = page_groups;
    spec.logical_page_capacity = kPromptPages;
    spec.table_rows            = 1;
    spec.plane_order           = order;
    for (std::size_t i = 0; i < variant.plane_count; ++i) {
        spec.planes.push_back(variant.planes[i]);
    }
    cell.layout      = plan_paged_kv_pool(builder, spec);
    cell.arena.emplace(builder.finish(256));
    // Drain default-stream and WDDM background work before publishing the arena
    // to the non-blocking stream this test runs on (the residency race on
    // Windows; harmless elsewhere).
    CUDA_CHECK(cudaDeviceSynchronize());
    cell.pool = std::make_unique<PagedKVPool>(DeviceSpan{cell.arena->base(), cell.arena->capacity()},
                                              cell.layout);
    cell.alloc = cell.pool->reserve(kPromptPages);
    cell.alloc.materialize_tokens(static_cast<std::uint32_t>(kPromptTokens), stream);
    // Bind before any append: the bind publishes the block table and clears the
    // delta row, which is the row the relocation then publishes into.
    cell.alloc.bind_row(0, stream);
}

// The plane presented in the view shape {leading, page, heads, pages}. A
// PageMajor plane already has that shape. A HeadMajor plane has the same bytes
// with the head and page extents transposed, re-described here with the
// strides swapped -- the same fields the copier reads, never a shared constant.
Tensor view_shaped(const Tensor& plane, std::int32_t leading, std::int32_t heads,
                   PagedKVPlaneOrder order) {
    if (order == PagedKVPlaneOrder::PageMajor) {
        return Tensor(plane.data, plane.dtype, {leading, kPageSize, heads, plane.ne[3]});
    }
    Tensor out;
    out.data      = plane.data;
    out.dtype     = plane.dtype;
    out.ne[0]     = leading;
    out.ne[1]     = kPageSize;
    out.ne[2]     = heads;
    out.ne[3]     = plane.ne[2];
    out.nb[0]     = plane.nb[0];
    out.nb[1]     = plane.nb[1];
    out.nb[2]     = plane.nb[3]; // head stride
    out.nb[3]     = plane.nb[2]; // page stride
    return out;
}

[[nodiscard]] std::int32_t plane_leading(const Variant& variant, std::size_t plane_index) {
    return variant.planes[plane_index].leading_extent;
}

// ------------------------------------------------------------ pool adapter ---

// The pool-backed cache the shared oracle templates consume: host images of the
// planes decoded as the read path decodes them (full-width signed codes, packed
// int4 nibbles through unpack_i4, half scales per 64-element group, and the
// inverse Hadamard on a rotated variant), resolved to logical positions through
// the allocation's published mapping.
class PoolCache {
public:
    // The stream is the one the whole cell runs on: image() reads the pool's
    // pages on it and drains it, so the shared oracle template's argument-less
    // image() still sees the pool as the relocation left it.
    PoolCache(const Variant& variant, const PoolCell& cell, cudaStream_t stream)
        : variant_(variant), cell_(cell), order_(cell.order), stream_(stream) {}

    struct HostImage {
        std::vector<double> k;
        std::vector<double> v;
    };

    [[nodiscard]] std::vector<std::int32_t> page_ids() const {
        const std::span<const std::int32_t> ids = cell_.alloc.page_ids();
        return std::vector<std::int32_t>(ids.begin(), ids.end());
    }

    [[nodiscard]] HostImage image() const {
        // The image is indexed by *physical* page id: that is the convention the shared oracle
        // template's Cache documents ("[physical page][kv head][page offset][d]") and the one
        // k_logical/v_logical resolve a position into through page_ids(). `copy_pages_to_host`
        // writes the pages it is given contiguously, in the order it is given them, so the plane
        // has to be gathered over the pool's page groups by id. Sizing it by the allocation's
        // logical slot count instead lines the two up only while every owned page happens to have
        // an id below that count -- true for a cache that was never relocated, false for one whose
        // acquire slots came from the pool, where each page id past the allocation's own pages
        // indexes 32 KiB further past the end of the image and reads unmapped heap.
        const std::size_t pages = static_cast<std::size_t>(cell_.pool->page_group_count());
        std::vector<std::int32_t> ids(pages);
        for (std::size_t i = 0; i < pages; ++i) {
            ids[i] = static_cast<std::int32_t>(i);
        }
        const std::size_t elems = pages * kPageSize * kKvHeads * kHeadDim;
        const std::size_t scale_elems = pages * kPageSize * kKvHeads * kQuantGroups;

        const bool quantized = variant_.dtype == DType::I8;
        const std::int32_t k_leading = quantized && variant_.packed_k ? kHeadDim / 2 : kHeadDim;
        const std::int32_t v_leading = quantized && variant_.packed_v ? kHeadDim / 2 : kHeadDim;

        // Every plane is byte storage, so the host vector's element type carries
        // the interpretation exactly as the published view declares it. The
        // buffers are sized from the plane's own leading extent and element
        // size, so a packed plane is half the bytes of a full-width one.
        const std::size_t k_elem = (quantized && variant_.packed_k) ? 1u : dtype_size(variant_.dtype);
        const std::size_t v_elem = (quantized && variant_.packed_v) ? 1u : dtype_size(variant_.dtype);
        const std::size_t row_bytes = pages * kPageSize * kKvHeads;
        std::vector<std::uint8_t> k_bytes(row_bytes * static_cast<std::size_t>(k_leading) * k_elem);
        std::vector<std::uint8_t> v_bytes(row_bytes * static_cast<std::size_t>(v_leading) * v_elem);
        std::vector<__half> k_scale(scale_elems, __half{});
        std::vector<__half> v_scale(scale_elems, __half{});
        cell_.pool->copy_pages_to_host(0, ids, k_bytes.data(), stream_);
        cell_.pool->copy_pages_to_host(1, ids, v_bytes.data(), stream_);
        if (quantized) {
            cell_.pool->copy_pages_to_host(2, ids, k_scale.data(), stream_);
            cell_.pool->copy_pages_to_host(3, ids, v_scale.data(), stream_);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        HostImage out{std::vector<double>(elems), std::vector<double>(elems)};
        for (std::size_t page = 0; page < pages; ++page) {
            for (std::int32_t head = 0; head < kKvHeads; ++head) {
                for (std::int32_t offset = 0; offset < kPageSize; ++offset) {
                    // The pool's page image is [head][page_offset][leading] for
                    // both plane orders.
                    const std::size_t row =
                        (page * static_cast<std::size_t>(kKvHeads) + static_cast<std::size_t>(head)) *
                            kPageSize +
                        static_cast<std::size_t>(offset);
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        const std::size_t elem  = row * kHeadDim + static_cast<std::size_t>(d);
                        const std::int32_t group = d / kQuantGroup;
                        const std::size_t scale  = row * kQuantGroups + static_cast<std::size_t>(group);
                        if (!quantized) {
                            // A full-width plane is one bf16 per element, so the
                            // byte offset is twice the element index.
                            __nv_bfloat16 kk;
                            __nv_bfloat16 vv;
                            const std::size_t k_at =
                                (row * kHeadDim + static_cast<std::size_t>(d)) * sizeof(kk);
                            const std::size_t v_at =
                                (row * kHeadDim + static_cast<std::size_t>(d)) * sizeof(vv);
                            std::memcpy(&kk, &k_bytes[k_at], sizeof(kk));
                            std::memcpy(&vv, &v_bytes[v_at], sizeof(vv));
                            out.k[elem] = static_cast<double>(__bfloat162float(kk));
                            out.v[elem] = static_cast<double>(__bfloat162float(vv));
                            continue;
                        }
                        const double ks = static_cast<double>(__half2float(k_scale[scale]));
                        const double vs = static_cast<double>(__half2float(v_scale[scale]));
                        out.k[elem]     = decode_code(k_bytes, row, k_leading, d,
                                                      variant_.packed_k ? 1 : 0) *
                                          ks;
                        out.v[elem] = decode_code(v_bytes, row, v_leading, d,
                                                  variant_.packed_v ? 1 : 0) *
                                      vs;
                    }
                }
            }
        }
        if (variant_.rotate_k) { inverse_hadamard_plane(out.k); }
        if (variant_.rotate_v) { inverse_hadamard_plane(out.v); }
        return out;
    }

    [[nodiscard]] PagedKVLayerView view(const FreqTable& freq) const {
        PagedKVLayerView out;
        out.k_pages       = view_shaped(cell_.pool->plane(0), plane_leading(variant_, 0),
                                        kKvHeads, order_);
        out.v_pages       = view_shaped(cell_.pool->plane(1), plane_leading(variant_, 1),
                                        kKvHeads, order_);
        out.block_table   = cell_.alloc.block_table();
        out.head_dim      = kHeadDim;
        out.num_kv_heads  = kKvHeads;
        out.dtype         = variant_.dtype;
        out.quant_group   = variant_.dtype == DType::I8 ? kQuantGroup : 0;
        out.packed_v      = variant_.packed_v;
        out.rotate_k      = variant_.rotate_k;
        out.rotate_v      = variant_.rotate_v;
        out.packed_k      = variant_.packed_k;
        out.e8_lattice    = variant_.e8_lattice;
        out.e8_root       = false;
        out.rope_deltas   = cell_.pool->rope_delta_row(0);
        out.rope_inv_freq = freq;
        if (variant_.dtype == DType::I8) {
            out.k_scale_pages = view_shaped(cell_.pool->plane(2), plane_leading(variant_, 2),
                                            kKvHeads, order_);
            out.v_scale_pages = view_shaped(cell_.pool->plane(3), plane_leading(variant_, 3),
                                            kKvHeads, order_);
        }
        return out;
    }

    [[nodiscard]] PagedKVBatchLayerView batch_view(const FreqTable& freq) const {
        const PagedKVLayerView direct = view(freq);
        PagedKVBatchLayerView out;
        out.k_pages       = direct.k_pages;
        out.v_pages       = direct.v_pages;
        out.k_scale_pages = direct.k_scale_pages;
        out.v_scale_pages = direct.v_scale_pages;
        out.block_tables  = direct.block_table.view({direct.block_table.ne[0], 1});
        out.rope_deltas   = Tensor(direct.rope_deltas.data, DType::I32,
                                   {direct.block_table.ne[0], 1});
        out.rope_inv_freq = freq;
        out.head_dim      = direct.head_dim;
        out.num_kv_heads  = direct.num_kv_heads;
        out.dtype         = direct.dtype;
        out.quant_group   = direct.quant_group;
        out.packed_v      = direct.packed_v;
        out.rotate_k      = direct.rotate_k;
        out.rotate_v      = direct.rotate_v;
        out.packed_k      = direct.packed_k;
        out.e8_lattice    = direct.e8_lattice;
        out.e8_root       = direct.e8_root;
        return out;
    }

    [[nodiscard]] double k_logical(const HostImage& img, std::int32_t position, std::int32_t head,
                                   std::int32_t d) const {
        return img.k[logical_index(static_cast<std::int32_t>( cell_.alloc.page_ids()[
                                       static_cast<std::size_t>(position / kPageSize)]),
                                   head, position % kPageSize, d)];
    }

    [[nodiscard]] double v_logical(const HostImage& img, std::int32_t position, std::int32_t head,
                                   std::int32_t d) const {
        return img.v[logical_index(static_cast<std::int32_t>(cell_.alloc.page_ids()[
                                       static_cast<std::size_t>(position / kPageSize)]),
                                   head, position % kPageSize, d)];
    }

private:
    // One code out of a byte plane: the plane's own leading extent names the
    // byte, and a packed plane's low nibble is the even dim, its high nibble the
    // odd one -- the same convention gqa_kv_pack_i4 / gqa_kv_unpack_i4x16 use.
    static double decode_code(const std::vector<std::uint8_t>& bytes, std::size_t row,
                              std::int32_t leading, std::int32_t d, int packed) {
        const std::size_t index = row * static_cast<std::size_t>(leading) +
                                  (packed ? static_cast<std::size_t>(d / 2) : static_cast<std::size_t>(d));
        if (packed == 0) {
            return static_cast<double>(static_cast<std::int8_t>(bytes[index]));
        }
        return static_cast<double>(unpack_i4(bytes[index], d & 1));
    }

    [[nodiscard]] std::size_t logical_index(std::int32_t physical_page, std::int32_t head,
                                            std::int32_t page_offset, std::int32_t d) const {
        return ((static_cast<std::size_t>(physical_page) * kKvHeads +
                 static_cast<std::size_t>(head)) *
                    kPageSize +
                static_cast<std::size_t>(page_offset)) *
                   kHeadDim +
               static_cast<std::size_t>(d);
    }

    const Variant& variant_;
    const PoolCell& cell_;
    PagedKVPlaneOrder order_;
    cudaStream_t stream_ = nullptr;
};

// ----------------------------------------------------------------- seeding ---

// Appends one chunk through the production write path at sequential positions.
void append_chunk(const PoolCell& cell, const std::vector<__nv_bfloat16>& k,
                  const std::vector<__nv_bfloat16>& v, std::int32_t first_position,
                  std::int32_t tokens, const FreqTable& freq, cudaStream_t stream) {
    const PoolCache adapter(*cell.variant, cell, stream);
    const PagedKVLayerView view = adapter.view(freq);
    DeviceBuffer dk(k.size() * sizeof(__nv_bfloat16));
    DeviceBuffer dv(v.size() * sizeof(__nv_bfloat16));
    DeviceBuffer dp(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    dk.copy_from_host(k.data(), k.size() * sizeof(__nv_bfloat16));
    dv.copy_from_host(v.data(), v.size() * sizeof(__nv_bfloat16));
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        positions[static_cast<std::size_t>(t)] = first_position + t;
    }
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    ops::gqa_kv_append(Tensor(dk.p, DType::BF16, {kHeadDim, kKvHeads, tokens}),
                       Tensor(dv.p, DType::BF16, {kHeadDim, kKvHeads, tokens}),
                       Tensor(dp.p, DType::I32, {tokens}), view, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
}

// Seeds a whole run [first, first + tokens) in chunks, the way prefill does.
void append_run(const PoolCell& cell, const std::vector<__nv_bfloat16>& k,
                const std::vector<__nv_bfloat16>& v, std::int32_t first_position,
                std::int32_t tokens, const FreqTable& freq, cudaStream_t stream) {
    for (std::int32_t base = first_position; base < first_position + tokens;
         base += kChunkTokens) {
        const std::int32_t count = std::min(kChunkTokens, first_position + tokens - base);
        const std::size_t offset =
            static_cast<std::size_t>(base - first_position) * kHeadDim * kKvHeads;
        const std::size_t elems = static_cast<std::size_t>(count) * kHeadDim * kKvHeads;
        append_chunk(cell, std::vector<__nv_bfloat16>(k.begin() + static_cast<std::ptrdiff_t>(offset),
                                                     k.begin() +
                                                         static_cast<std::ptrdiff_t>(offset + elems)),
                     std::vector<__nv_bfloat16>(v.begin() + static_cast<std::ptrdiff_t>(offset),
                                                v.begin() +
                                                    static_cast<std::ptrdiff_t>(offset + elems)),
                     base, count, freq, stream);
    }
}

// Page images of every plane of a pool, indexed [plane][page group][byte]. The
// pool's page image is [head][page_offset][leading] in both plane orders, which
// is what lets one byte comparison cover the whole matrix.
using PageImage     = std::vector<std::uint8_t>;
using PlaneSnapshot = std::vector<std::vector<PageImage>>;

PlaneSnapshot snapshot_pages(const PagedKVPool& pool, cudaStream_t stream) {
    PlaneSnapshot out(pool.plane_count());
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const std::size_t bytes = pool.page_bytes(plane);
        out[plane].resize(static_cast<std::size_t>(pool.page_group_count()));
        for (std::uint32_t page = 0; page < pool.page_group_count(); ++page) {
            out[plane][static_cast<std::size_t>(page)].resize(bytes);
            pool.copy_page_to_host(plane, static_cast<std::int32_t>(page),
                                   out[plane][static_cast<std::size_t>(page)].data(), stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
}

std::vector<std::int32_t> read_i32_row(const Tensor& row, cudaStream_t stream) {
    std::vector<std::int32_t> out(static_cast<std::size_t>(row.numel()));
    if (out.empty()) { return out; }
    CUDA_CHECK(cudaMemcpyAsync(out.data(), row.data, out.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
}

// -------------------------------------------------------------- relocation ---

// The published relocation, built the way program_impl.h's copying branch does
// it: the real planner pair, never a hand-built array.
struct Relocation {
    std::vector<std::int32_t> mapping;
    std::vector<std::int32_t> deltas;
    std::vector<RelocatedCopySpan> copy_spans;
};

Relocation plan_relocation(const PoolCell& cell, const std::vector<RelocatedSpan>& spans,
                           std::uint32_t prefix_reuse) {
    Relocation out;
    const std::span<const std::int32_t> ids = cell.alloc.page_ids();
    const std::vector<std::int32_t> current(ids.begin(), ids.end());
    out.mapping = plan_copied_page_remap(current, spans, prefix_reuse,
                                         static_cast<std::uint32_t>(kPromptTokens));
    if (out.mapping.empty()) {
        fail("planner: the copied remap declined a relocation it must express");
        return out;
    }
    out.deltas.assign(out.mapping.size(), kRopeDeltaNone);
    for (const RelocatedSpan& span : spans) {
        for (std::uint32_t offset = 0; offset < span.tokens; offset += kPagedKVPageSize) {
            const std::int32_t dest_page = static_cast<std::int32_t>(
                (span.dest_begin + offset) / kPagedKVPageSize);
            out.deltas[static_cast<std::size_t>(dest_page)] = rope_delta_for_dest_page(
                span.source_begin, span.dest_begin, span.tokens, dest_page);
        }
        out.copy_spans.push_back(RelocatedCopySpan{.source_begin = span.source_begin,
                                                   .dest_begin   = span.dest_begin,
                                                   .tokens       = span.tokens});
    }
    return out;
}

// ------------------------------------------------------- groups 1 and 2 -----

// The expected delta plane: every logical page ordinary except the destination
// pages, which carry the span's own shift. Written down here rather than read
// back from the planner, so the publication carries an independent expectation.
std::vector<std::int32_t> expected_deltas() {
    std::vector<std::int32_t> out(static_cast<std::size_t>(kPromptPages), kRopeDeltaNone);
    for (const ExpectedSpan& span : kExpectedSpans) {
        for (std::int32_t page = span.dest_begin / kPageSize;
             page * kPageSize < span.dest_begin + span.tokens; ++page) {
            out[static_cast<std::size_t>(page)] = span.dest_begin - span.source_begin;
        }
    }
    return out;
}

// 1: every moved range bit-identical to the page it was copied from, per plane,
// per head, no tolerance. The per-head token run is contiguous inside the
// page's head block in both plane orders, and token_bytes comes from the plane
// itself -- the same fields the copier reads.
void check_moved_bytes(const PoolCell& cell, const PlaneSnapshot& before,
                       const PlaneSnapshot& after, const std::vector<std::int32_t>& before_ids,
                       const std::vector<std::int32_t>& after_ids,
                       const std::vector<RelocatedPageMove>& moves, const std::string& label) {
    for (std::size_t plane = 0; plane < cell.pool->plane_count(); ++plane) {
        const std::size_t token_bytes = static_cast<std::size_t>(cell.pool->plane(plane).nb[1]);
        for (const RelocatedPageMove& move : moves) {
            const auto& source =
                before[plane][static_cast<std::size_t>(
                    before_ids[static_cast<std::size_t>(move.source_page)])];
            const auto& dest = after[plane][static_cast<std::size_t>(
                                               after_ids[static_cast<std::size_t>(move.dest_page)])];
            for (std::int32_t head = 0; head < kKvHeads; ++head) {
                const std::size_t src_off =
                    (static_cast<std::size_t>(head) * kPageSize +
                     static_cast<std::size_t>(move.source_token)) *
                    token_bytes;
                const std::size_t dst_off =
                    (static_cast<std::size_t>(head) * kPageSize +
                     static_cast<std::size_t>(move.dest_token)) *
                    token_bytes;
                const std::size_t bytes = static_cast<std::size_t>(move.tokens) * token_bytes;
                if (std::memcmp(source.data() + src_off, dest.data() + dst_off, bytes) != 0) {
                    fail(label + ": plane " + std::to_string(plane) + " move dest_page " +
                         std::to_string(move.dest_page) + " head " + std::to_string(head) +
                         " is not bit-identical to the " + std::to_string(move.tokens) +
                         " tokens it was copied from (source page " +
                         std::to_string(move.source_page) + " offset " +
                         std::to_string(move.source_token) + ")");
                    return;
                }
            }
        }
    }
    note(label + ": moved bytes bit-identical on every plane (moves=" + std::to_string(moves.size()) +
         ", planes=" + std::to_string(cell.pool->plane_count()) + ")");
}

// 2: nothing else moved. Every page equals its pre-copy image with exactly the
// moved ranges overwritten from their sources; every page the mapping keeps is
// byte-for-byte what it was.
void check_nothing_else_moved(const PoolCell& cell, const PlaneSnapshot& before,
                              const PlaneSnapshot& after,
                              const std::vector<std::int32_t>& before_ids,
                              const std::vector<std::int32_t>& after_ids,
                              const std::vector<RelocatedPageMove>& moves,
                              const std::string& label) {
    // Rebuild what every page should hold: its own pre-copy image with exactly
    // the moved ranges replaced from their sources. Starting from the pre-copy
    // image is the point -- a destination page's untouched bytes must still be
    // what they were before the copy.
    PlaneSnapshot expected = before;
    for (std::size_t plane = 0; plane < cell.pool->plane_count(); ++plane) {
        const std::size_t token_bytes = static_cast<std::size_t>(cell.pool->plane(plane).nb[1]);
        for (const RelocatedPageMove& move : moves) {
            const std::size_t src_page = static_cast<std::size_t>(
                before_ids[static_cast<std::size_t>(move.source_page)]);
            const std::size_t dst_page = static_cast<std::size_t>(
                after_ids[static_cast<std::size_t>(move.dest_page)]);
            for (std::int32_t head = 0; head < kKvHeads; ++head) {
                const std::size_t src_off =
                    (static_cast<std::size_t>(head) * kPageSize +
                     static_cast<std::size_t>(move.source_token)) *
                    token_bytes;
                const std::size_t dst_off =
                    (static_cast<std::size_t>(head) * kPageSize +
                     static_cast<std::size_t>(move.dest_token)) *
                    token_bytes;
                std::memcpy(expected[plane][dst_page].data() + dst_off,
                            before[plane][src_page].data() + src_off,
                            static_cast<std::size_t>(move.tokens) * token_bytes);
            }
        }
    }
    for (std::size_t plane = 0; plane < cell.pool->plane_count(); ++plane) {
        for (std::size_t page = 0; page < after[plane].size(); ++page) {
            if (after[plane][page] != expected[plane][page]) {
                fail(label + ": plane " + std::to_string(plane) + " page " +
                     std::to_string(page) +
                     " changed outside the moved ranges, so the copy touched bytes it does not own");
                return;
            }
        }
    }
    note(label + ": no page changed outside the moved ranges");
}

// ------------------------------------------------------- groups 3 and 4 -----

// The pages the copies read, resolved through the pre-relocation mapping: the
// substitution must never name one of these as a destination.
std::set<std::int32_t> source_pages(const std::vector<std::int32_t>& before_ids,
                                    const std::vector<RelocatedPageMove>& moves) {
    std::set<std::int32_t> out;
    for (const RelocatedPageMove& move : moves) {
        out.insert(before_ids[static_cast<std::size_t>(move.source_page)]);
    }
    return out;
}

// The relocation's move decomposition, and the arithmetic that predicts
// its pool demand independently of the implementation: acquire slots minus the
// dropped pages the copies do not read. The pages the copies read are held
// until the drain and returned afterwards, so the held count is the unique
// source set.
std::vector<RelocatedPageMove> relocation_moves(const Relocation& planned) {
    return plan_relocated_page_moves(planned.copy_spans, planned.mapping.size());
}

std::uint32_t expected_from_pool(const Relocation& planned,
                                 const std::vector<std::int32_t>& before_ids) {
    const std::vector<RelocatedPageMove> moves = relocation_moves(planned);
    std::uint32_t acquires                    = 0;
    for (const std::int32_t page : planned.mapping) {
        if (page == kPagedKVAquirePage) { ++acquires; }
    }
    const std::set<std::int32_t> sources = source_pages(before_ids, moves);
    const std::set<std::int32_t> named(planned.mapping.begin(), planned.mapping.end());
    std::uint32_t dropped = 0;
    for (const std::int32_t page : before_ids) {
        if (named.count(page) == 0) { ++dropped; }
    }
    const std::uint32_t usable = dropped > static_cast<std::uint32_t>(sources.size())
                                     ? dropped - static_cast<std::uint32_t>(sources.size())
                                     : 0;
    return acquires > usable ? acquires - usable : 0;
}

// 3: what was published. page_ids() is the mapping with acquire slots filled,
// the block-table row agrees with it, and the pool's delta row carries the
// relocation's shifts -- the mapping and the delta plane in one call, so no
// reader can see one without the other.
void check_published(const PoolCell& cell, const Relocation& relocation,
                     const std::vector<std::int32_t>& before_ids, const std::string& label,
                     cudaStream_t stream) {
    const std::vector<std::int32_t> after_ids = [&cell] {
        const std::span<const std::int32_t> ids = cell.alloc.page_ids();
        return std::vector<std::int32_t>(ids.begin(), ids.end());
    }();
    if (after_ids.size() != relocation.mapping.size()) {
        fail(label + ": published " + std::to_string(after_ids.size()) +
             " pages for a " + std::to_string(relocation.mapping.size()) + " page mapping");
        return;
    }
    const std::set<std::int32_t> owned_before(before_ids.begin(), before_ids.end());
    const std::set<std::int32_t> sources =
        source_pages(before_ids, relocation_moves(relocation));
    std::uint32_t from_pool_observed = 0;
    std::set<std::int32_t> distinct;
    for (std::size_t i = 0; i < relocation.mapping.size(); ++i) {
        const std::int32_t published = after_ids[i];
        if (distinct.count(published) != 0) {
            fail(label + ": logical page " + std::to_string(i) + " and another share physical page " +
                 std::to_string(published));
            return;
        }
        distinct.insert(published);
        if (sources.count(published) != 0) {
            fail(label + ": published destination " + std::to_string(published) +
                 " is a page the copies read, so the copy overlaps itself");
            return;
        }
        if (relocation.mapping[i] == kPagedKVAquirePage) {
            if (owned_before.count(published) == 0) { ++from_pool_observed; }
            continue;
        }
        if (published != relocation.mapping[i]) {
            fail(label + ": named slot " + std::to_string(i) + " published page " +
                 std::to_string(published) + ", the planner named " +
                 std::to_string(relocation.mapping[i]));
            return;
        }
    }
    const std::uint32_t from_pool_expected = expected_from_pool(relocation, before_ids);
    if (from_pool_observed != from_pool_expected) {
        fail(label + ": " + std::to_string(from_pool_observed) +
             " published pages came from the pool, the relocation's demand is " +
             std::to_string(from_pool_expected));
        return;
    }
    const std::vector<std::int32_t> block_row =
        read_i32_row(cell.pool->block_table_row(0), stream);
    if (block_row != after_ids) {
        fail(label + ": the published block-table row does not agree with page_ids()");
        return;
    }
    const std::vector<std::int32_t> delta_row = read_i32_row(cell.pool->rope_delta_row(0), stream);
    const std::vector<std::int32_t> want     = expected_deltas();
    if (delta_row != want) {
        std::string got;
        for (const std::int32_t d : delta_row) { got += std::to_string(d) + " "; }
        fail(label + ": the published delta plane is [" + got + "], expected the planner's shifts");
        return;
    }
    note(label + ": mapping, block table and delta plane all published as planned (pool demand " +
         std::to_string(from_pool_expected) + " pages)");
}

// 4: pool accounting. The pages the copies read come back after the drain, the
// pool demand was taken before the copies, and the free count says exactly
// that without a page leaked or handed out twice.
void check_pool_accounting(const PagedKVPool& pool, const Relocation& planned,
                           const std::vector<std::int32_t>& before_ids, std::uint32_t free_before,
                           const std::string& label) {
    const std::vector<RelocatedPageMove> moves = relocation_moves(planned);
    const std::set<std::int32_t> sources         = source_pages(before_ids, moves);
    const std::uint32_t held                     = static_cast<std::uint32_t>(sources.size());
    const std::uint32_t from_pool                = expected_from_pool(planned, before_ids);
    const std::uint32_t free_after               = pool.free_pages();
    if (free_after != free_before - from_pool + held) {
        fail(label + ": free pages " + std::to_string(free_after) + " after the relocation, expected " +
             std::to_string(free_before) + " - " + std::to_string(from_pool) + " taken + " +
             std::to_string(held) + " returned");
        return;
    }
    if (!pool.can_reserve(free_after) || pool.can_reserve(free_after + 1)) {
        fail(label + ": the pool's free-page count does not match what a reserve can take (" +
             std::to_string(free_after) + " free)");
        return;
    }
    note(label + ": pool accounting holds (" + std::to_string(from_pool) + " taken, " +
         std::to_string(held) + " held and returned, " + std::to_string(free_after) + " free)");
}
// ------------------------------------------------------------ group 5 --------
AttentionRun run_a3(const PoolCell& cell, const PoolCache& cache,
                    const std::vector<__nv_bfloat16>& q, std::int32_t first_position,
                    std::int32_t tokens, std::uint32_t visible, const FreqTable& freq,
                    cudaStream_t stream) {
    const PagedKVLayerView view = cache.view(freq);
    DeviceBuffer dq            = upload(q);
    DeviceBuffer dout          = DeviceBuffer(q.size() * sizeof(__nv_bfloat16));
    dout.fill(0);
    DeviceBuffer dp = upload_positions(sequential_positions(first_position, tokens));
    const ops::GqaExecutionEnvelope envelope{visible, visible};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, cell.variant->dtype, envelope, 1, tokens, tokens);
    DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    DeviceArena workspace(DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});
    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    ops::gqa_attention_cached(tq, tp, kAttentionScale, view, envelope, workspace, tout, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

// A1 through the public entry point, the route prefill itself uses. The width
// is deliberately wider than kMaximumVerifyTokens so the call resolves to the
// Prompt route, which is the route the delta plane is qualified on.
AttentionRun run_a1(const PoolCell& cell, const PoolCache& cache,
                    const std::vector<__nv_bfloat16>& q,
                    const std::vector<__nv_bfloat16>& k, const std::vector<__nv_bfloat16>& v,
                    std::int32_t first_position, std::int32_t tokens, std::uint32_t visible,
                    const FreqTable& freq, cudaStream_t stream) {
    append_chunk(cell, k, v, first_position, tokens, freq, stream);
    const PagedKVBatchLayerView batch = cache.batch_view(freq);
    DeviceBuffer dq                   = upload(q);
    DeviceBuffer dk                   = upload(k);
    DeviceBuffer dv                   = upload(v);
    DeviceBuffer dtable               = DeviceBuffer(sizeof(std::int32_t));
    const std::int32_t table_row      = 0;
    dtable.copy_from_host(&table_row, sizeof(table_row));
    DeviceBuffer dout = DeviceBuffer(q.size() * sizeof(__nv_bfloat16));
    dout.fill(0);
    DeviceBuffer dp = upload_positions(sequential_positions(first_position, tokens));
    const ops::GqaExecutionEnvelope envelope{visible, visible};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, cell.variant->dtype, envelope, 1, tokens, tokens);
    DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    DeviceArena workspace(DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});
    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor ttable(dtable.p, DType::I32, {1});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable, kAttentionScale, batch, envelope,
                       workspace, tout, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    return AttentionRun{download_doubles(dout, q.size())};
}

std::vector<__nv_bfloat16> make_queries(std::uint32_t seed, std::int32_t tokens) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(-0.25, 0.25);
    const std::size_t elems = static_cast<std::size_t>(kHeadDim) * kQHeads * tokens;
    std::vector<__nv_bfloat16> out(elems);
    for (std::size_t i = 0; i < elems; ++i) {
        out[i] = __float2bfloat16(static_cast<float>(unit(rng)));
    }
    return out;
}

// The pages of the new prompt no span relocates: the runtime prefills them,
// so the copied cache is written the same way and the two caches differ only
// in what the relocation itself moved.
void append_reedited_pages(const PoolCell& cell, const std::vector<__nv_bfloat16>& k,
                           const std::vector<__nv_bfloat16>& v, const FreqTable& freq,
                           cudaStream_t stream) {
    const std::int32_t ranges[][2] = {{704, 768}, {832, 896}, {1088, 1408}};
    for (const auto& range : ranges) {
        const std::int32_t first = range[0];
        const std::int32_t tokens = range[1] - range[0];
        const std::size_t offset =
            static_cast<std::size_t>(first) * kHeadDim * kKvHeads;
        const std::size_t elems = static_cast<std::size_t>(tokens) * kHeadDim * kKvHeads;
        append_chunk(cell,
                     std::vector<__nv_bfloat16>(k.begin() + static_cast<std::ptrdiff_t>(offset),
                                                k.begin() + static_cast<std::ptrdiff_t>(offset + elems)),
                     std::vector<__nv_bfloat16>(v.begin() + static_cast<std::ptrdiff_t>(offset),
                                                v.begin() + static_cast<std::ptrdiff_t>(offset + elems)),
                     first, tokens, freq, stream);
    }
}

// 5: operational equivalence. Attention over the copied cache through the
// public entry points, judged by the anchored FP64 delta oracle -- the oracle
// that rotates every decoded key back by its page's published shift, so it
// carries the plane's own encode error exactly. The directly prefilled cache is
// the anchor that keeps the harness itself honest; the zeroed delta plane is
// the control, and the design is only discriminating if the control breaks the
// criterion the correct run is judged by.
// The storage side of the equivalence, measured against two references at once.
//
// Parity with the directly prefilled cache is the design's own anchor and is kept: group 1 shows the
// copy moved the source bytes, this shows whether those bytes, read at their destination through the
// published mapping and corrected by the page's own delta, are the keys a direct prefill wrote.
//
// Parity alone cannot name a side. The pages the relocation never touches are written from the same
// activation at the same position through the same kernel in both caches, so they hold the same bytes
// by construction; a harness that reports a disagreement there has measured itself, not the copier,
// and no parity number can say that. So each side is additionally held against the activation this
// fixture wrote, which is an independent oracle and the only reading here that can name one:
//
//   * on the plane whose writer is exact (bf16 in, bf16 stored, no codec between them) the stored row
//     reproduces that activation bit for bit, so a non-zero column there is a defect rather than
//     rounding. It is the column that decides whether the anchor itself is the current prompt's
//     content, before any parity number is read as a property of the relocation;
//   * on a quantized plane the stored codes are not the activation, so that column is omitted rather
//     than fudged, and only parity is judged.
//
// Every column is per logical page: the query window is one page wide, and one aggregate cannot
// separate the moved pages from the kept ones. The probe rows print both halves of a single row -- the
// rotary dims forward_rope rewrites and the dims it copies verbatim -- because a correction that
// leaves an aggregate unchanged must have reached one of them and not the other, and that asymmetry
// is the difference between a rotation that did nothing and a rotation that was not needed.
void check_stored_keys(const PoolCell& copied_cell, const PoolCache& copied_cache,
                       const PoolCell& reference_cell, const PoolCache& reference_cache,
                       const std::vector<__nv_bfloat16>& curr_k, bool exact_plane,
                       const std::vector<std::int32_t>& page_deltas, const FreqTable& freq,
                       cudaStream_t stream, const std::string& label) {
    const PoolCache::HostImage copied_img = copied_cache.image();
    const PoolCache::HostImage ref_img    = reference_cache.image();
    const std::size_t pages               = static_cast<std::size_t>(kPromptPages);
    std::array<double, kHeadDim> stored{};
    std::array<double, kHeadDim> corrected{};
    std::vector<double> raw_worst(pages, 0.0);
    std::vector<double> corrected_worst(pages, 0.0);
    std::vector<double> anchor_host_worst(pages, 0.0);
    std::vector<double> copied_host_raw_worst(pages, 0.0);
    std::vector<double> copied_host_worst(pages, 0.0);
    std::vector<std::int32_t> copied_host_dim(pages, -1);
    // The magnitude on each side, because a page that holds nothing and a page that holds the wrong
    // thing are the same difference and not the same defect: a side reading zeros shows up here as an
    // empty row, which no deviation column can separate from a row of unrelated keys.
    std::vector<double> copied_magnitude(pages, 0.0);
    std::vector<double> anchor_magnitude(pages, 0.0);
    for (std::int32_t x = 0; x < kVisible; ++x) {
        const std::int32_t page  = x / kPageSize;
        const std::size_t index  = static_cast<std::size_t>(page);
        const std::int32_t delta = page_deltas[index];
        for (std::int32_t head = 0; head < kKvHeads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                stored[static_cast<std::size_t>(d)] = copied_cache.k_logical(copied_img, x, head, d);
            }
            forward_rope(stored.data(), delta, freq, corrected.data());
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t dim  = static_cast<std::size_t>(d);
                const double want      = reference_cache.k_logical(ref_img, x, head, d);
                const double raw_err   = std::fabs(stored[dim] - want);
                const double fixed_err = std::fabs(corrected[dim] - want);
                if (raw_err > raw_worst[index]) { raw_worst[index] = raw_err; }
                if (fixed_err > corrected_worst[index]) { corrected_worst[index] = fixed_err; }
                if (std::fabs(stored[dim]) > copied_magnitude[index]) {
                    copied_magnitude[index] = std::fabs(stored[dim]);
                }
                if (std::fabs(want) > anchor_magnitude[index]) {
                    anchor_magnitude[index] = std::fabs(want);
                }
                if (!exact_plane) { continue; }
                const double truth = static_cast<double>(
                    __bfloat162float(curr_k[act_index(head, x, kKvHeads, d)]));
                const double anchor_err = std::fabs(want - truth);
                const double host_raw   = std::fabs(stored[dim] - truth);
                const double host_fixed = std::fabs(corrected[dim] - truth);
                if (anchor_err > anchor_host_worst[index]) { anchor_host_worst[index] = anchor_err; }
                if (host_raw > copied_host_raw_worst[index]) {
                    copied_host_raw_worst[index] = host_raw;
                }
                if (host_fixed > copied_host_worst[index]) {
                    copied_host_worst[index]  = host_fixed;
                    copied_host_dim[index]    = d;
                }
            }
        }
    }

    // The two probe rows: one on a page the relocation never touches, one on the page the queries sit
    // in, each read through the same correction the aggregate used.
    const std::int32_t probe_positions[2] = {0, kQueryBase};
    for (int i = 0; i < 2; ++i) {
        const std::int32_t x     = probe_positions[i];
        const std::int32_t delta = page_deltas[static_cast<std::size_t>(x / kPageSize)];
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            stored[static_cast<std::size_t>(d)] = copied_cache.k_logical(copied_img, x, 0, d);
        }
        forward_rope(stored.data(), delta, freq, corrected.data());
        for (int j = 0; j < 2; ++j) {
            const std::int32_t d    = j == 0 ? 0 : kRotaryPairs;
            const std::size_t dim   = static_cast<std::size_t>(d);
            const double truth      = exact_plane
                                          ? static_cast<double>(__bfloat162float(
                                                curr_k[act_index(0, x, kKvHeads, d)]))
                                          : std::numeric_limits<double>::quiet_NaN();
            std::printf("\n      probe x=%d head 0 d=%d: copied as stored %.6e, corrected %.6e; anchor "
                        "%.6e; activation %.6e",
                        x, d, stored[dim], corrected[dim],
                        reference_cache.k_logical(ref_img, x, 0, d), truth);
        }
    }

    const auto ids_of = [](const PoolCell& cell) {
        std::string out;
        for (const std::int32_t id : cell.alloc.page_ids()) { out += std::to_string(id) + " "; }
        return out;
    };
    std::printf("\n  %s: stored keys per logical page (parity: as stored / corrected by the delta;"
                " copied against the written activation: as stored / corrected, worst dim; anchor"
                " against the written activation):",
                label.c_str());
    for (std::int32_t page = 0; page < kPromptPages; ++page) {
        const std::size_t index = static_cast<std::size_t>(page);
        const bool quiet = raw_worst[index] < 1.0e-2 && corrected_worst[index] < 1.0e-2 &&
                           anchor_host_worst[index] < 1.0e-2 &&
                           copied_host_worst[index] < 1.0e-2;
        // An empty page on either side is never quiet, whatever the deviation columns say.
        const bool empty = copied_magnitude[index] < 1.0e-6 || anchor_magnitude[index] < 1.0e-6;
        if (quiet && !empty) { continue; }
        if (!exact_plane) {
            std::printf("\n      page %d delta %d: |copied| %.3e |anchor| %.3e ; parity %.3e / %.3e", page,
                        page_deltas[index], copied_magnitude[index], anchor_magnitude[index],
                        raw_worst[index], corrected_worst[index]);
            continue;
        }
        std::printf("\n      page %d delta %d: |copied| %.3e |anchor| %.3e ; parity %.3e / %.3e ; copied"
                    " %.3e / %.3e (d=%d) ; anchor %.3e",
                    page, page_deltas[index], copied_magnitude[index], anchor_magnitude[index],
                    raw_worst[index], corrected_worst[index], copied_host_raw_worst[index],
                    copied_host_worst[index], copied_host_dim[index], anchor_host_worst[index]);
    }
    std::printf("\n      copied page ids:    %s\n      reference page ids: %s\n",
                ids_of(copied_cell).c_str(), ids_of(reference_cell).c_str());

    // The ground truth under both images: the K plane's own bytes on the physical page each logical page
    // resolves to, read straight out of the two pools. A decoded comparison cannot separate a wrong page
    // from a wrong decode, and the difference decides which of the two is being looked at; these bytes
    // are the only reading here that does not go through the image at all.
    const PlaneSnapshot copied_pages    = snapshot_pages(*copied_cell.pool, stream);
    const PlaneSnapshot reference_pages = snapshot_pages(*reference_cell.pool, stream);
    const std::int32_t probe_pages[3]   = {0, 11, 16};
    std::printf("      K-plane bytes per page (kept frontier / re-prefilled / relocated destination):");
    for (const std::int32_t page : probe_pages) {
        const std::int32_t copied_id =
            copied_cell.alloc.page_ids()[static_cast<std::size_t>(page)];
        const std::int32_t reference_id =
            reference_cell.alloc.page_ids()[static_cast<std::size_t>(page)];
        const PageImage& copied_bytes = copied_pages[0][static_cast<std::size_t>(copied_id)];
        const PageImage& reference_bytes = reference_pages[0][static_cast<std::size_t>(reference_id)];
        std::size_t first_diff = copied_bytes.size();
        const std::size_t shared = std::min(copied_bytes.size(), reference_bytes.size());
        for (std::size_t i = 0; i < shared; ++i) {
            if (copied_bytes[i] != reference_bytes[i]) {
                first_diff = i;
                break;
            }
        }
        const auto hex = [](const PageImage& bytes) {
            std::string out;
            char buf[4];
            for (std::size_t i = 0; i < 8 && i < bytes.size(); ++i) {
                std::snprintf(buf, sizeof(buf), "%02x", static_cast<unsigned>(bytes[i]));
                out += buf;
            }
            return out;
        };
        const bool identical = copied_bytes.size() == reference_bytes.size() &&
                               first_diff == copied_bytes.size();
        std::printf("\n      logical page %d: copied id %d %s vs reference id %d %s -> %s (first "
                    "differing byte %zu of %zu)",
                    page, copied_id, hex(copied_bytes).c_str(), reference_id,
                    hex(reference_bytes).c_str(), identical ? "identical" : "differ", first_diff,
                    copied_bytes.size());
    }
    std::printf("\n");

    // Three readings, named as measured rather than as their likely cause, because they are three
    // different defects and only one of them is the copier's: an empty side is the harness failing to
    // read a cache the anchored attention checks in this same cell say holds content; a stored-and-
    // corrected mismatch is a payload/shift disagreement; and a stored match that the correction
    // breaks is a page whose published shift does not describe the row it was published for.
    for (std::int32_t page = 0; page < kPromptPages; ++page) {
        const std::size_t index = static_cast<std::size_t>(page);
        const std::string where =
            " page " + std::to_string(page) + " (delta " + std::to_string(page_deltas[index]) + "): ";
        const bool copied_empty = copied_magnitude[index] < 1.0e-6;
        const bool anchor_empty = anchor_magnitude[index] < 1.0e-6;
        if (copied_empty || anchor_empty) {
            fail(label + where + "the " +
                 std::string(copied_empty ? "copied" : "directly prefilled") +
                 " cache reads an empty row here (magnitudes " +
                 std::to_string(copied_magnitude[index]) + " and " +
                 std::to_string(anchor_magnitude[index]) +
                 "), so this comparison has no second side and measures the harness's read of that "
                 "cache rather than the relocation");
            continue;
        }
        const bool reconstructs = corrected_worst[index] > 3.0e-2;
        const bool already_matches = raw_worst[index] < 3.0e-2;
        if (already_matches && reconstructs) {
            fail(label + where + "the copied cache's stored keys already equal the directly prefilled "
                 "cache's (as stored " + std::to_string(raw_worst[index]) +
                 ") and the published delta is what breaks them (corrected " +
                 std::to_string(corrected_worst[index]) +
                 "), so that page's shift does not describe the row it is paired with");
        } else if (reconstructs) {
            fail(label + where + "the copied cache's stored keys do not reconstruct the directly "
                 "prefilled cache's even with the published delta applied (as stored " +
                 std::to_string(raw_worst[index]) + ", corrected " +
                 std::to_string(corrected_worst[index]) +
                 "), so the payload and the shift it is paired with disagree");
        }
        if (!exact_plane) { continue; }
        if (anchor_host_worst[index] > 3.0e-2) {
            fail(label + ": page " + std::to_string(page) + " (delta " +
                 std::to_string(page_deltas[index]) +
                 ") disagrees by " + std::to_string(anchor_host_worst[index]) +
                 " with the activation this fixture wrote, so the directly prefilled anchor every "
                 "comparison here is anchored to is not the current prompt's content");
        }
        if (copied_host_worst[index] > 3.0e-2) {
            fail(label + ": page " + std::to_string(page) + " (delta " +
                 std::to_string(page_deltas[index]) +
                 ") disagrees by " + std::to_string(copied_host_worst[index]) + " (worst at dim " +
                 std::to_string(copied_host_dim[index]) +
                 ") with the activation this fixture wrote even after its published delta is applied, "
                 "so the relocated payload at that page is not the current prompt's keys");
        }
    }
}

void check_attention(const Variant& variant, const PromptPair& prompts,
                     const std::vector<__nv_bfloat16>& prev_k,
                     const std::vector<__nv_bfloat16>& prev_v,
                     const std::vector<__nv_bfloat16>& curr_k,
                     const std::vector<__nv_bfloat16>& curr_v, const FreqTable& freq,
                     DeviceContext& ctx, const std::string& label) {
    const std::string cell_label = label + " [" + variant.name + "]";
    const bool quantized         = variant.dtype == DType::I8;
    const Criterion& criterion   = quantized ? kConformanceRotatedI8 : kConformanceBf16;

    PoolCell copied;
    PoolCell reference;
    build_cell(copied, variant, PagedKVPlaneOrder::PageMajor, kPoolPages, ctx.stream);
    build_cell(reference, variant, PagedKVPlaneOrder::PageMajor, kPoolPages, ctx.stream);
    // The source run is seeded first, exactly as the resident prompt was.
    append_run(copied, prev_k, prev_v, 0, kPromptTokens, freq, ctx.stream);
    // The reference cache is a direct prefill of the edited prompt.
    append_run(reference, curr_k, curr_v, 0, kPromptTokens, freq, ctx.stream);

    const Relocation planned = plan_relocation(copied, prompts.spans, prompts.prefix_reuse);
    const std::vector<RelocatedPageMove> moves = relocation_moves(planned);
    if (moves.size() != std::size(kExpectedMoves)) {
        fail(cell_label + " A3: the move decomposition produced " + std::to_string(moves.size()) +
             " moves, expected " + std::to_string(std::size(kExpectedMoves)));
        return;
    }
    for (std::size_t i = 0; i < moves.size(); ++i) {
        const RelocatedPageMove& got = moves[i];
        const ExpectedMove& want     = kExpectedMoves[i];
        if (got.dest_page != want.dest_page || got.dest_token != want.dest_token ||
            got.source_page != want.source_page || got.source_token != want.source_token ||
            got.tokens != want.tokens) {
            fail(cell_label + " A3: move " + std::to_string(i) + " is (" +
                 std::to_string(got.dest_page) + "," + std::to_string(got.dest_token) + " <- " +
                 std::to_string(got.source_page) + "," + std::to_string(got.source_token) + " x" +
                 std::to_string(got.tokens) + "), expected (" + std::to_string(want.dest_page) +
                 "," + std::to_string(want.dest_token) + " <- " + std::to_string(want.source_page) +
                 "," + std::to_string(want.source_token) + " x" + std::to_string(want.tokens) + ")");
        }
    }
    copied.alloc.relocate_copied_pages(planned.mapping, planned.deltas, planned.copy_spans,
                                       ctx.stream);
    // This cell carries no group-3 assertion of its own, and the stage between the
    // publish and the attention reads is the re-prefill of the un-relocated ranges.
    // Reading the plane on both sides of it separates a relocation whose shifts never
    // reached this cell from a re-prefill that overwrote them -- and both of those read
    // a relocated page at the source phase, which is what the equivalence below would
    // otherwise report as an unexplained mismatch against the directly prefilled cache.
    const std::vector<std::int32_t> deltas_at_publish =
        read_i32_row(copied.pool->rope_delta_row(0), ctx.stream);
    if (deltas_at_publish != expected_deltas()) {
        fail(cell_label +
             " A3: this cell's relocation published a delta plane the scenario does not describe");
    }
    append_reedited_pages(copied, curr_k, curr_v, freq, ctx.stream);
    const std::vector<std::int32_t> deltas_after_prefill =
        read_i32_row(copied.pool->rope_delta_row(0), ctx.stream);
    if (deltas_after_prefill != deltas_at_publish) {
        std::string where;
        for (std::size_t i = 0; i < deltas_after_prefill.size(); ++i) {
            if (deltas_after_prefill[i] != deltas_at_publish[i]) {
                where += " page " + std::to_string(i) + " " + std::to_string(deltas_at_publish[i]) +
                         " -> " + std::to_string(deltas_after_prefill[i]);
            }
        }
        fail(cell_label + " A3: the re-prefill changed the published delta plane (" + where +
             "), so a relocated page is read at the source phase");
    }

    const PoolCache copied_cache(variant, copied, ctx.stream);
    const PoolCache reference_cache(variant, reference, ctx.stream);
    const std::vector<__nv_bfloat16> q   = make_queries(0xa11ceu, kQueryTokens);
    const std::vector<__nv_bfloat16> q_1 = make_queries(0xfeed17u, kTailTokens);

    // A3: the decode route over the relocated cache.
    const AttentionRun got_reference =
        run_a3(reference, reference_cache, q, kQueryBase, kQueryTokens, kVisible, freq, ctx.stream);
    const AttentionRun got_copied =
        run_a3(copied, copied_cache, q, kQueryBase, kQueryTokens, kVisible, freq, ctx.stream);
    const std::vector<std::int32_t> page_deltas =
        read_i32_row(copied.pool->rope_delta_row(0), ctx.stream);
    check_stored_keys(copied, copied_cache, reference, reference_cache, curr_k, !quantized,
                      page_deltas, freq, ctx.stream, cell_label + " stored keys");
    if (!expect_conformant(measure(got_reference.out,
                                   ideal_attention(q, reference_cache, kQueryBase, kQueryTokens)),
                           cell_label + " A3 over a directly prefilled cache", criterion)) {
        ++failures;
    }
    if (!expect_conformant(measure(got_copied.out,
                                   ideal_attention_with_delta(q, copied_cache, kQueryBase,
                                                              kQueryTokens, page_deltas, freq)),
                           cell_label + " A3 over the copied cache", criterion)) {
        ++failures;
    }
    const Error equivalence = measure(got_copied.out, got_reference.out);
    if (!quantized) {
        if (!expect_equivalent(equivalence, cell_label + " A3 copied against directly prefilled")) {
            ++failures;
        }
    } else {
        // On a coarse (int4) plane the correct path's own encode error dominates
        // the comparison against another cache, which is why the anchored form
        // above is the judgement; this number is reported, not asserted.
        note(cell_label + " A3 copied against directly prefilled: observation only (relative L2 " +
             std::to_string(equivalence.relative_l2) + ")");
    }

    // A1: the append-and-attend route, over the first page after the relocated
    // region -- the route prefill itself uses.
    const std::size_t tail_elems =
        static_cast<std::size_t>(kTailTokens) * kHeadDim * kKvHeads;
    const std::size_t tail_offset =
        static_cast<std::size_t>(kTailFirst) * kHeadDim * kKvHeads;
    const std::vector<__nv_bfloat16> tail_k(curr_k.begin() + static_cast<std::ptrdiff_t>(tail_offset),
                                            curr_k.begin() +
                                                static_cast<std::ptrdiff_t>(tail_offset + tail_elems));
    const std::vector<__nv_bfloat16> tail_v(curr_v.begin() + static_cast<std::ptrdiff_t>(tail_offset),
                                            curr_v.begin() +
                                                static_cast<std::ptrdiff_t>(tail_offset + tail_elems));
    const std::uint32_t tail_visible =
        static_cast<std::uint32_t>(kTailFirst + kTailTokens);
    const AttentionRun got_tail_reference =
        run_a1(reference, reference_cache, q_1, tail_k, tail_v, kTailFirst, kTailTokens,
               tail_visible, freq, ctx.stream);
    const AttentionRun got_tail_copied =
        run_a1(copied, copied_cache, q_1, tail_k, tail_v, kTailFirst, kTailTokens, tail_visible,
               freq, ctx.stream);
    if (!expect_conformant(measure(got_tail_reference.out,
                                   ideal_attention(q_1, reference_cache, kTailFirst, kTailTokens)),
                           cell_label + " A1 over a directly prefilled cache", criterion)) {
        ++failures;
    }
    if (!expect_conformant(measure(got_tail_copied.out,
                                   ideal_attention_with_delta(q_1, copied_cache, kTailFirst,
                                                              kTailTokens, page_deltas, freq)),
                           cell_label + " A1 over the copied cache", criterion)) {
        ++failures;
    }

    // The negative control: the same cache read through a zeroed delta plane.
    // That is a relocation that published nothing, so every relocated key is
    // read at the source phase, and it has to break the same criterion.
    //
    // What it is measured against is the *correct* reading -- the delta oracle the
    // run above was judged by -- which is how every other control in this suite
    // states it (`test_scr_relocation_integration.cpp` measures its stale run
    // against the directly prefilled cache, `test_gqa_kv_page_delta_rotated.cpp`
    // against its delta reference). Measuring the stale run against an oracle
    // handed the same zeros compares the broken path with itself: the number is
    // then the kernel's self-consistency, not the damage of the skipped rotation,
    // and it is indistinguishable from the correct path's own noise.
    copied.pool->clear_rope_delta_row(0, ctx.stream);
    const AttentionRun got_stale =
        run_a3(copied, copied_cache, q, kQueryBase, kQueryTokens, kVisible, freq, ctx.stream);
    const Error control =
        measure(got_stale.out,
                ideal_attention_with_delta(q, copied_cache, kQueryBase, kQueryTokens, page_deltas,
                                           freq));
    if (quantized) {
        if (!expect_control_outside(control, cell_label + " control: a skipped rotation",
                                    kConformanceRotatedI8)) {
            ++failures;
        }
    } else if (!expect_control_separates(measure(got_copied.out,
                                                 ideal_attention_with_delta(q, copied_cache,
                                                                            kQueryBase, kQueryTokens,
                                                                            page_deltas, freq)),
                                         control, cell_label + " control: a skipped rotation")) {
        ++failures;
    }
    const AttentionRun got_tail_stale =
        run_a1(copied, copied_cache, q_1, tail_k, tail_v, kTailFirst, kTailTokens, tail_visible,
               freq, ctx.stream);
    const Error tail_control =
        measure(got_tail_stale.out,
                ideal_attention_with_delta(q_1, copied_cache, kTailFirst, kTailTokens, page_deltas,
                                           freq));
    if (quantized) {
        if (!expect_control_outside(tail_control, cell_label + " A1 control: a skipped rotation",
                                    kConformanceRotatedI8)) {
            ++failures;
        }
    } else if (!expect_control_separates(measure(got_tail_copied.out,
                                                 ideal_attention_with_delta(q_1, copied_cache,
                                                                            kTailFirst, kTailTokens,
                                                                            page_deltas, freq)),
                                         tail_control,
                                         cell_label + " A1 control: a skipped rotation")) {
        ++failures;
    }
}

// Seeds a whole run at [0, kPromptTokens) so every cache byte comes from the
// production write path. A HeadMajor plane cannot be addressed by the Ops --
// their page arithmetic is page-major -- so for that order the bytes are
// written into a scratch PageMajor pool of the same spec through the same
// writer and transplanted page by page. The page image a pool hands back is
// [head][page_offset][leading] in both orders, so the transplant is exact.
void seed_run(PoolCell& cell, const std::vector<__nv_bfloat16>& k,
              const std::vector<__nv_bfloat16>& v, const FreqTable& freq, cudaStream_t stream) {
    if (cell.order == PagedKVPlaneOrder::PageMajor) {
        append_run(cell, k, v, 0, kPromptTokens, freq, stream);
        return;
    }
    PoolCell scratch;
    build_cell(scratch, *cell.variant, PagedKVPlaneOrder::PageMajor, kPoolPages, stream);
    append_run(scratch, k, v, 0, kPromptTokens, freq, stream);
    const PlaneSnapshot images = snapshot_pages(*scratch.pool, stream);
    const std::vector<std::int32_t> scratch_ids = [&scratch] {
        const std::span<const std::int32_t> ids = scratch.alloc.page_ids();
        return std::vector<std::int32_t>(ids.begin(), ids.end());
    }();
    const std::vector<std::int32_t> target_ids = [&cell] {
        const std::span<const std::int32_t> ids = cell.alloc.page_ids();
        return std::vector<std::int32_t>(ids.begin(), ids.end());
    }();
    for (std::size_t plane = 0; plane < cell.pool->plane_count(); ++plane) {
        const std::size_t bytes = cell.pool->page_bytes(plane);
        std::vector<std::uint8_t> packed(target_ids.size() * bytes);
        for (std::size_t i = 0; i < target_ids.size(); ++i) {
            std::memcpy(packed.data() + i * bytes,
                        images[plane][static_cast<std::size_t>(scratch_ids[i])].data(), bytes);
        }
        cell.pool->copy_pages_from_host(plane, target_ids, packed.data(), stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
}

// --------------------------------------------------------- groups 1 to 4 -----

// One full cell: seed, snapshot, plan through the real planners, relocate, and
// hold the result against groups 1 through 4. `tight` sizes the pool so the
// free pages are exactly the relocation's demand, which is the configuration
// the aliasing property has to survive: a substitution that reused a source
// page as a destination would overlap the copy.
void run_cell(const Variant& variant, PagedKVPlaneOrder order, std::uint32_t page_groups,
              bool tight, const PromptPair& prompts, const std::vector<__nv_bfloat16>& k,
              const std::vector<__nv_bfloat16>& v, const FreqTable& freq, DeviceContext& ctx) {
    // The tight cell is the aliasing group: its pool offers exactly the pages
    // the relocation must take, so a substitution that handed a source page back
    // as a destination would have to overlap the copy to fill the slots at all.
    const std::string label = std::string(variant.name) + "/" +
                              (order == PagedKVPlaneOrder::PageMajor ? "page-major"
                                                                     : "head-major") +
                              (tight ? "/tight" : "");
    PoolCell cell;
    build_cell(cell, variant, order, page_groups, ctx.stream);
    seed_run(cell, k, v, freq, ctx.stream);

    const std::vector<std::int32_t> before_ids = [&cell] {
        const std::span<const std::int32_t> ids = cell.alloc.page_ids();
        return std::vector<std::int32_t>(ids.begin(), ids.end());
    }();
    const PlaneSnapshot before = snapshot_pages(*cell.pool, ctx.stream);
    const std::uint32_t free_before = cell.pool->free_pages();

    const Relocation planned = plan_relocation(cell, prompts.spans, prompts.prefix_reuse);
    const std::vector<RelocatedPageMove> moves = relocation_moves(planned);
    if (tight && cell.pool->free_pages() != expected_from_pool(planned, before_ids)) {
        fail(label + ": the tight pool offers " + std::to_string(cell.pool->free_pages()) +
             " free pages, which is not exactly the relocation's demand of " +
             std::to_string(expected_from_pool(planned, before_ids)) +
             " -- without that tension this cell cannot detect an aliasing "
             "substitution");
        return;
    }
    cell.alloc.relocate_copied_pages(planned.mapping, planned.deltas, planned.copy_spans,
                                     ctx.stream);
    const PlaneSnapshot after = snapshot_pages(*cell.pool, ctx.stream);
    const std::vector<std::int32_t> after_ids = [&cell] {
        const std::span<const std::int32_t> ids = cell.alloc.page_ids();
        return std::vector<std::int32_t>(ids.begin(), ids.end());
    }();

    check_moved_bytes(cell, before, after, before_ids, after_ids, moves, label + " bytes");
    check_nothing_else_moved(cell, before, after, before_ids, after_ids, moves,
                             label + " untouched");
    check_published(cell, planned, before_ids, label + " published", ctx.stream);
    check_pool_accounting(*cell.pool, planned, before_ids, free_before, label + " accounting");
    cell.alloc.release();
}

// ------------------------------------------------------------- group 7 -------

template <typename Fn>
void expect_refusal(Fn fn, const std::string& label, const std::string& what) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        note(label + ": refused (" + what + ")");
        return;
    } catch (const std::bad_alloc&) {
        note(label + ": refused (" + what + ")");
        return;
    } catch (const std::exception& error) {
        fail(label + ": refused with the wrong type (" + error.what() + "), expected " + what);
        return;
    }
    fail(label + ": expected a refusal (" + what + "), none was thrown");
}

// The never-half-publish property the feature's error handling rests on: every
// refusal leaves the mapping, the pool and the cache bytes exactly as they
// were, because the validation all runs before any page moves.
void check_refusals(const Variant& variant, const PromptPair& prompts,
                    const std::vector<__nv_bfloat16>& k, const std::vector<__nv_bfloat16>& v,
                    const FreqTable& freq, DeviceContext& ctx) {
    const std::string label = std::string(variant.name) + " refusals";
    PoolCell cell;
    build_cell(cell, variant, PagedKVPlaneOrder::PageMajor, kPoolPages, ctx.stream);
    seed_run(cell, k, v, freq, ctx.stream);
    const Relocation valid = plan_relocation(cell, prompts.spans, prompts.prefix_reuse);
    const std::vector<std::int32_t> before_ids = [&cell] {
        const std::span<const std::int32_t> ids = cell.alloc.page_ids();
        return std::vector<std::int32_t>(ids.begin(), ids.end());
    }();
    const std::uint32_t free_before = cell.pool->free_pages();
    const PlaneSnapshot before     = snapshot_pages(*cell.pool, ctx.stream);

    const auto unchanged = [&](const std::string& what) {
        const std::span<const std::int32_t> ids = cell.alloc.page_ids();
        if (std::vector<std::int32_t>(ids.begin(), ids.end()) != before_ids) {
            fail(label + ": " + what + " changed the published mapping");
            return false;
        }
        if (cell.pool->free_pages() != free_before) {
            fail(label + ": " + what + " changed the pool's free-page count");
            return false;
        }
        if (snapshot_pages(*cell.pool, ctx.stream) != before) {
            fail(label + ": " + what + " moved cache bytes before refusing");
            return false;
        }
        return true;
    };

    expect_refusal(
        [&] {
            std::vector<std::int32_t> short_deltas(valid.deltas.begin(),
                                                   valid.deltas.end() - 1);
            cell.alloc.relocate_copied_pages(valid.mapping, short_deltas, valid.copy_spans,
                                             ctx.stream);
        },
        label, "a delta plane not parallel to the mapping");
    unchanged("a non-parallel delta plane");

    expect_refusal(
        [&] {
            cell.alloc.relocate_copied_pages(valid.mapping, valid.deltas, {}, ctx.stream);
        },
        label, "no spans to copy");
    unchanged("no spans");

    expect_refusal(
        [&] {
            const std::vector<std::int32_t> shrunk(valid.mapping.begin(),
                                                   valid.mapping.end() - 1);
            const std::vector<std::int32_t> deltas(shrunk.size(), kRopeDeltaNone);
            cell.alloc.relocate_copied_pages(shrunk, deltas, valid.copy_spans, ctx.stream);
        },
        label, "a mapping that resizes the allocation");
    unchanged("a resized mapping");

    expect_refusal(
        [&] {
            std::vector<std::int32_t> grown = valid.mapping;
            grown.push_back(kPagedKVAquirePage);
            std::vector<std::int32_t> deltas(grown.size(), kRopeDeltaNone);
            cell.alloc.relocate_copied_pages(grown, deltas, valid.copy_spans, ctx.stream);
        },
        label, "a mapping past the page entitlement");
    unchanged("a mapping past the entitlement");

    // A pool that cannot afford the demand: a second allocation takes enough
    // free pages that the relocation's from_pool no longer fits, and the call
    // has to refuse before anything moves. The second allocation is on this
    // cell's own pool, so the shortfall is real.
    auto second = cell.pool->reserve(4);
    second.materialize_pages(2, ctx.stream);
    const std::uint32_t free_with_second = cell.pool->free_pages();
    const std::uint32_t demand           = expected_from_pool(valid, before_ids);
    if (free_with_second >= demand) {
        note(label + ": the second allocation left " + std::to_string(free_with_second) +
             " free pages against a demand of " + std::to_string(demand) +
             "; the unaffordable case is not configured, skipped");
    } else {
        expect_refusal(
            [&] {
                cell.alloc.relocate_copied_pages(valid.mapping, valid.deltas, valid.copy_spans,
                                                 ctx.stream);
            },
            label, "a pool that cannot afford the demand");
        const std::span<const std::int32_t> ids = cell.alloc.page_ids();
        if (std::vector<std::int32_t>(ids.begin(), ids.end()) != before_ids) {
            fail(label + ": an unaffordable pool changed the published mapping");
        }
        if (cell.pool->free_pages() != free_with_second) {
            fail(label + ": an unaffordable pool changed the free-page count");
        }
        if (snapshot_pages(*cell.pool, ctx.stream) != before) {
            fail(label + ": an unaffordable pool moved cache bytes before refusing");
        }
    }
    cell.alloc.release();
}

} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("no CUDA device available\n");
        return 77; // CTest SKIP_RETURN_CODE
    }
    DeviceContext ctx(0);
    const FreqTable freq = default_inv_freq();
    const PromptPair prompts = build_prompt_pair();
    if (prompts.spans.empty() || failures != 0) {
        std::printf("SCR copied-relocation fixture FAILED (%d)\n", failures);
        return 1;
    }

    const TokenContent prev_content = make_content(0xc0ffee42u);
    const TokenContent curr_content = reedited_content(prev_content, prompts.spans);
    const std::vector<double> prev_v_f = prev_content.v;
    const std::vector<double> curr_v_f = curr_content.v;
    const auto to_bf16                = [](const std::vector<double>& values) {
        std::vector<__nv_bfloat16> out(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            out[i] = __float2bfloat16(static_cast<float>(values[i]));
        }
        return out;
    };
    const std::vector<__nv_bfloat16> prev_k = rope_tokens(prev_content, freq);
    const std::vector<__nv_bfloat16> prev_v = to_bf16(prev_v_f);
    const std::vector<__nv_bfloat16> curr_k = rope_tokens(curr_content, freq);
    const std::vector<__nv_bfloat16> curr_v = to_bf16(curr_v_f);

    std::printf("SCR copied-relocation checks\n");
    std::printf("  spans:");
    for (const RelocatedSpan& span : prompts.spans) {
        std::printf(" src=%u dst=%u tok=%u", span.source_begin, span.dest_begin, span.tokens);
    }
    std::printf("  (prefix reuse %u of %d tokens)\n", prompts.prefix_reuse, kPromptTokens);

    for (const Variant& variant : kVariants) {
        for (const PagedKVPlaneOrder order : kOrders) {
            run_cell(variant, order, kPoolPages, false, prompts, prev_k, prev_v, freq, ctx);
        }
        // The attention matrix is PageMajor only: the Ops address pages
        // page-major, so the kernel side of the equivalence is exercised there
        // while the copier itself is exercised on both orders above.
        check_attention(variant, prompts, prev_k, prev_v, curr_k, curr_v, freq, ctx,
                        std::string(variant.name) + " equivalence");
        for (const PagedKVPlaneOrder order : kOrders) {
            run_cell(variant, order, kTightPoolPages, true, prompts, prev_k, prev_v, freq, ctx);
        }
        check_refusals(variant, prompts, prev_k, prev_v, freq, ctx);
    }

    if (failures != 0) {
        std::printf("SCR copied-relocation checks FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("SCR copied-relocation checks passed\n");
    return 0;
}

#include "targets/sensenova_loop_9b/impl/runtime/text_execution.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace sn = ninfer::targets::sensenova_loop_9b::detail;
using ninfer::DeviceSpan;
using ninfer::WorkspaceArena;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <typename Exception>
bool throws_invalid(std::string_view what, auto&& callable) {
    try {
        callable();
    } catch (const Exception&) {
        return true;
    } catch (...) {
    }
    std::cerr << "FAIL: " << what << " did not throw std::invalid_argument\n";
    return false;
}

// Throws std::invalid_argument whose message names the rejected input, so a test
// cannot pass on an unrelated failure.
bool throws_invalid_with(std::string_view needle, std::string_view what, auto&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()).find(needle) != std::string_view::npos) { return true; }
        ++failures;
        std::cerr << "FAIL: " << what << " threw the wrong message: " << error.what() << '\n';
        return false;
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL: " << what << " threw a non-invalid_argument: " << error.what() << '\n';
        return false;
    }
    ++failures;
    std::cerr << "FAIL: " << what << " did not throw\n";
    return false;
}

template <typename Exception>
bool throws_range(std::string_view what, auto&& callable) {
    try {
        callable();
    } catch (const Exception&) {
        return true;
    } catch (...) {
    }
    std::cerr << "FAIL: " << what << " did not throw std::out_of_range\n";
    return false;
}

constexpr std::uint32_t kKvCapacity = 64;
constexpr std::int32_t kHidden      = sn::TextConfig::hidden;

// Dummy host stand-ins: the context stores handles and computes addresses, so
// no CUDA is needed to prove the layout arithmetic and the validation.
struct Fixture {
    // One full state image for kKvCapacity tokens (15 KV domains + GDN + mHC).
    std::vector<std::byte> image{sn::StateImageLayout::total_bytes(kKvCapacity, false, false)};
    // Borrowed arena storage for the per-request workspace.
    std::array<std::byte, 4096> ws_storage{};
    sn::RuntimeModelView model{};  // zero-initialized dummy model view
    WorkspaceArena ws{DeviceSpan{ws_storage.data(), ws_storage.size()}};
    cudaStream_t stream{reinterpret_cast<cudaStream_t>(std::uintptr_t{0x1})};

    std::byte* image_base() { return image.data(); }
};

void test_valid_context() {
    Fixture f;
    sn::TextExecutionContext ctx(f.model, f.stream, f.image_base(), kKvCapacity, f.ws);

    expect(&ctx.model() == &f.model, "model accessor returns the stored view");
    expect(ctx.stream() == f.stream, "stream accessor returns the stored stream");
    expect(ctx.state_image_base() == f.image_base(), "state image base accessor");
    expect(ctx.kv_capacity() == kKvCapacity, "kv capacity accessor");
    expect(&ctx.workspace() == &f.ws, "workspace accessor returns the stored arena");
    expect(ctx.state_image_bytes() == sn::StateImageLayout::total_bytes(kKvCapacity, false, false),
           "host-computable state image size");
    expect(f.image.size() == ctx.state_image_bytes(), "test image is exactly one state image");
}

void test_state_bases() {
    Fixture f;
    const sn::TextExecutionContext ctx(f.model, f.stream, f.image_base(), kKvCapacity, f.ws);

    const std::size_t kv_bytes    = sn::StateImageLayout::total_kv_bytes(kKvCapacity, false, false);
    const std::size_t gdn_region  = std::size_t(sn::StateImageLayout::kGdnSlots) *
                                   sn::StateLayout::gdn_state_bytes_per_slot();
    const std::byte* base         = f.image_base();

    // GDN slots: unique for the revisited physical-layer pair (visits 4 and 28),
    // laid out one slot stride apart.
    const std::size_t gdn_slot4  = sn::StateImageLayout::gdn_slot_for_visit(4);
    const std::size_t gdn_slot28 = sn::StateImageLayout::gdn_slot_for_visit(28);
    const std::byte* gdn4  = ctx.gdn_state_base(4);
    const std::byte* gdn28 = ctx.gdn_state_base(28);
    expect(gdn4 != gdn28, "GDN state bases are unique for visits 4 and 28");
    expect(std::size_t(gdn4 - base) == kv_bytes + sn::StateImageLayout::gdn_offset(gdn_slot4),
           "GDN visit 4 address matches its compact slot");
    expect(std::size_t(gdn28 - base) == kv_bytes + sn::StateImageLayout::gdn_offset(gdn_slot28),
           "GDN visit 28 address matches its compact slot");
    expect(gdn_slot28 - gdn_slot4 == 18, "18 GDN visits separate layer 7's two passes");
    expect(gdn28 - gdn4 ==
               std::ptrdiff_t((gdn_slot28 - gdn_slot4) *
                              sn::StateLayout::gdn_state_bytes_per_slot()),
           "GDN visits 4 and 28 are a whole number of slot strides apart");
    expect(gdn28 + sn::StateLayout::gdn_state_bytes_per_slot() <= base + kv_bytes + gdn_region,
           "GDN visit 28 fits inside the GDN region");

    // mHC slots: unique for visits 4 and 28 as well, and the mHC region follows
    // the GDN region.
    const std::size_t mhc_slot4  = sn::StateImageLayout::mhc_slot_for_visit(4);
    const std::size_t mhc_slot28 = sn::StateImageLayout::mhc_slot_for_visit(28);
    const std::byte* mhc4  = ctx.mhc_state_base(4);
    const std::byte* mhc28 = ctx.mhc_state_base(28);
    expect(mhc4 != mhc28, "mHC state bases are unique for visits 4 and 28");
    expect(std::size_t(mhc4 - base) ==
               kv_bytes + gdn_region + sn::StateImageLayout::mhc_offset(mhc_slot4),
           "mHC visit 4 address matches its compact slot");
    expect(mhc4 > gdn28, "mHC region follows the GDN region");
    expect(mhc_slot28 - mhc_slot4 == 24, "24 mHC visits separate layer 7's two passes");
    expect(mhc28 - mhc4 ==
               std::ptrdiff_t((mhc_slot28 - mhc_slot4) *
                              sn::StateLayout::mhc_state_bytes_per_slot()),
           "mHC visits 4 and 28 are a whole number of slot strides apart");
    expect(mhc28 + sn::StateLayout::mhc_state_bytes_per_slot() <= base + f.image.size(),
           "mHC visit 28 fits inside the state image");

    // Aliasing: a GDN slot and an mHC slot of the same visit never share bytes.
    expect(gdn4 + sn::StateLayout::gdn_state_bytes_per_slot() <= base + kv_bytes + gdn_region,
           "GDN slots stay inside the GDN region");
    expect(mhc4 >= base + kv_bytes + gdn_region, "mHC slots start after the GDN region");

    // KV domains forward the compact visit mapping and never alias.
    expect(ctx.kv_domain(7) == sn::StateImageLayout::kv_domain_for_visit(7),
           "kv_domain(7) forwards StateImageLayout");
    expect(ctx.kv_domain(31) == sn::StateImageLayout::kv_domain_for_visit(31),
           "kv_domain(31) forwards StateImageLayout");
    expect(ctx.kv_domain(7) != ctx.kv_domain(31), "kv_domain(7) != kv_domain(31)");

    // Stable addresses: repeated calls and a second context agree, which is
    // what CUDA Graph replay requires.
    sn::TextExecutionContext ctx2(f.model, f.stream, f.image_base(), kKvCapacity, f.ws);
    expect(ctx2.gdn_state_base(4) == gdn4 && ctx2.mhc_state_base(28) == mhc28,
           "state bases are stable across context instances");
}

void test_invalid_construction() {
    Fixture f;

    expect(throws_invalid<std::invalid_argument>(
              "null stream", [&] { sn::TextExecutionContext ctx(f.model, nullptr, f.image_base(),
                                                                 kKvCapacity, f.ws); }),
           "null stream throws");
    expect(throws_invalid<std::invalid_argument>(
              "null state image base",
              [&] { sn::TextExecutionContext ctx(f.model, f.stream, nullptr, kKvCapacity, f.ws); }),
           "null state image base throws");
    expect(throws_invalid<std::invalid_argument>("zero kv capacity",
                                                 [&] {
                                                     sn::TextExecutionContext ctx(f.model, f.stream,
                                                                                  f.image_base(), 0,
                                                                                  f.ws);
                                                 }),
           "zero kv capacity throws");
    // A zero-capacity workspace cannot exist: a borrowed arena with empty
    // storage is rejected before the context even sees it, and the capacity
    // check in the constructor covers arenas that lose capacity after
    // construction.
    expect(throws_invalid<std::invalid_argument>("empty borrowed workspace storage",
                                                 [&] {
                                                     WorkspaceArena empty(
                                                         DeviceSpan{f.ws_storage.data(), 0});
                                                 }),
           "zero-capacity workspace storage throws");
}

void test_invalid_visit_helpers() {
    Fixture f;
    const sn::TextExecutionContext ctx(f.model, f.stream, f.image_base(), kKvCapacity, f.ws);

    // Visit 3 is the first full-attention visit, so it owns no GDN slot; layer 0
    // owns no mHC slot; and 56 is past the plan entirely.
    expect(throws_range<std::out_of_range>("GDN slot for an attention visit",
                                           [&] { (void)ctx.gdn_state_base(3); }),
           "gdn_state_base for a full-attention visit throws");
    expect(throws_range<std::out_of_range>("mHC slot before the boundary",
                                           [&] { (void)ctx.mhc_state_base(0); }),
           "mhc_state_base for a non-mHC visit throws");
    expect(throws_range<std::out_of_range>("visit out of range",
                                           [&] { (void)ctx.gdn_state_base(sn::LoopPlan::kVisits); }),
           "gdn_state_base for visit 56 throws");
    expect(throws_range<std::out_of_range>("mHC visit out of range",
                                           [&] { (void)ctx.mhc_state_base(sn::LoopPlan::kVisits); }),
           "mhc_state_base for visit 56 throws");
}

// -----------------------------------------------------------------------------
// Stage E4B: embedding gather and final head/sample.
//
// These cases cover the host-computable half of both entry points: shape,
// dtype, contiguity, binding, aliasing, and workspace contract. They are
// deliberately host-only: the dummy model view in `Fixture` carries no bound
// weights, so every path stops in validation, and the one metadata-bound head
// fixture uses an arena too small to allocate its logit column, so the branch
// that would dispatch CUDA is never reached. A numeric check of the head needs
// a registered [248320,4096] LM head (about 1 GB in W8), which is why it belongs
// to a real-artifact test rather than this host test.
// -----------------------------------------------------------------------------

struct EmbedFixture {
    std::array<std::int32_t, 4> ids{11, 12, 13, 14};
    std::array<std::uint16_t, static_cast<std::size_t>(kHidden) * 4> hidden{};

    ninfer::Tensor ids_t{ids.data(), ninfer::DType::I32, {4}};
    ninfer::Tensor hidden_t{hidden.data(), ninfer::DType::BF16, {kHidden, 4}};
};

struct HeadFixture {
    Fixture base; // dummy model view and a 4 KiB borrowed arena
    std::array<std::uint16_t, static_cast<std::size_t>(kHidden)> final_norm{};
    std::array<std::byte, 32> head_payload{};
    std::array<std::uint16_t, static_cast<std::size_t>(kHidden) * 2> hidden{};
    std::array<std::int32_t, 1> sampled_id{0};
    std::array<float, 1> logprob{0.0F};
    std::array<float, 1> margin{0.0F};

    ninfer::Tensor hidden_t{hidden.data(), ninfer::DType::BF16, {kHidden, 2}};
    ninfer::Tensor sampled_t{sampled_id.data(), ninfer::DType::I32, {1}};
    ninfer::Tensor logprob_t{logprob.data(), ninfer::DType::FP32, {1}};
    ninfer::Tensor margin_t{margin.data(), ninfer::DType::FP32, {1}};

    void bind_final_norm() {
        base.model.final_norm =
            ninfer::Tensor(final_norm.data(), ninfer::DType::BF16, {kHidden});
    }

    // Metadata only: enough for the pre-dispatch contract, never executed here.
    void bind_head_metadata() {
        bind_final_norm();
        ninfer::Weight& head = base.model.output_head;
        head.qdata           = head_payload.data();
        head.qtype           = ninfer::QType::W8G32_F16S;
        head.n               = sn::TextConfig::output_rows;
        head.k               = kHidden;
    }

    sn::TextExecutionContext context() {
        return sn::TextExecutionContext(base.model, base.stream, base.image_base(), kKvCapacity,
                                        base.ws);
    }
};

void test_embed_tokens_contract() {
    EmbedFixture e;
    Fixture f;
    const sn::TextExecutionContext ctx(f.model, f.stream, f.image_base(), kKvCapacity, f.ws);

    throws_invalid_with("token embedding is not bound", "unbound embedding table",
                        [&] { ctx.embed_tokens(e.ids_t, e.hidden_t); });

    ninfer::Tensor null_ids = e.ids_t;
    null_ids.data           = nullptr;
    throws_invalid_with("contiguous and non-null", "null ids",
                        [&] { ctx.embed_tokens(null_ids, e.hidden_t); });

    ninfer::Tensor wrong_id_dtype(e.ids.data(), ninfer::DType::FP32, {4});
    throws_invalid_with("I32 [T]", "non-I32 ids",
                        [&] { ctx.embed_tokens(wrong_id_dtype, e.hidden_t); });

    // {4,1} would be indistinguishable from [4]: trailing unit extents are the
    // same shape under the op-layer convention.
    ninfer::Tensor rank_two_ids(e.ids.data(), ninfer::DType::I32, {4, 2});
    throws_invalid_with("I32 [T]", "rank-2 ids",
                        [&] { ctx.embed_tokens(rank_two_ids, e.hidden_t); });

    ninfer::Tensor strided_ids = e.ids_t;
    strided_ids.nb[0]          = 8;
    throws_invalid_with("contiguous and non-null", "strided ids",
                        [&] { ctx.embed_tokens(strided_ids, e.hidden_t); });

    ninfer::Tensor narrow_hidden(e.hidden.data(), ninfer::DType::BF16, {kHidden - 1, 4});
    throws_invalid_with("BF16 [4096,T]", "hidden width",
                        [&] { ctx.embed_tokens(e.ids_t, narrow_hidden); });

    ninfer::Tensor short_hidden(e.hidden.data(), ninfer::DType::BF16, {kHidden, 3});
    throws_invalid_with("BF16 [4096,T]", "hidden token count",
                        [&] { ctx.embed_tokens(e.ids_t, short_hidden); });

    ninfer::Tensor wrong_hidden_dtype(e.hidden.data(), ninfer::DType::FP32, {kHidden, 4});
    throws_invalid_with("BF16 [4096,T]", "hidden dtype",
                        [&] { ctx.embed_tokens(e.ids_t, wrong_hidden_dtype); });

    ninfer::Tensor null_hidden = e.hidden_t;
    null_hidden.data           = nullptr;
    throws_invalid_with("contiguous and non-null", "null hidden",
                        [&] { ctx.embed_tokens(e.ids_t, null_hidden); });
}

void test_head_contract() {
    HeadFixture h;
    const sn::TextExecutionContext ctx = h.context();

    ninfer::Tensor narrow_hidden(h.hidden.data(), ninfer::DType::BF16, {kHidden - 1, 2});
    throws_invalid_with("hidden must be BF16 [4096,T]", "hidden width", [&] {
        ctx.project_logits_and_sample(narrow_hidden, h.sampled_t);
    });

    ninfer::Tensor strided_hidden = h.hidden_t;
    strided_hidden.nb[1]          = 2;
    throws_invalid_with("contiguous and non-null", "strided hidden", [&] {
        ctx.project_logits_and_sample(strided_hidden, h.sampled_t);
    });

    ninfer::Tensor wrong_sampled(h.sampled_id.data(), ninfer::DType::FP32, {1});
    throws_invalid_with("sampled must be I32 [1]", "non-I32 sampled", [&] {
        ctx.project_logits_and_sample(h.hidden_t, wrong_sampled);
    });

    std::array<std::int32_t, 2> two_tokens{};
    ninfer::Tensor sampled_pair(two_tokens.data(), ninfer::DType::I32, {2});
    throws_invalid_with("sampled must be I32 [1]", "two sampled tokens", [&] {
        ctx.project_logits_and_sample(h.hidden_t, sampled_pair);
    });

    // A margin without a logprob destination has no measurement to accompany.
    throws_invalid_with("margin requires a logprob destination", "margin without logprob", [&] {
        ctx.project_logits_and_sample(h.hidden_t, h.sampled_t, nullptr, &h.margin_t);
    });

    ninfer::Tensor bf16_logprob(h.logprob.data(), ninfer::DType::BF16, {1});
    throws_invalid_with("logprob must be FP32 [1]", "BF16 logprob", [&] {
        ctx.project_logits_and_sample(h.hidden_t, h.sampled_t, &bf16_logprob);
    });

    ninfer::Tensor aliased_margin = h.logprob_t;
    throws_invalid_with("logprob and margin must not alias", "aliased logprob/margin", [&] {
        ctx.project_logits_and_sample(h.hidden_t, h.sampled_t, &h.logprob_t, &aliased_margin);
    });

    // Unbound final norm, then unbound head, then a head with the wrong output
    // geometry: the model binding contract is checked before the head is staged.
    throws_invalid_with("final norm is not bound", "unbound final norm", [&] {
        ctx.project_logits_and_sample(h.hidden_t, h.sampled_t);
    });

    h.bind_final_norm();
    throws_invalid_with("output head is not bound", "unbound output head", [&] {
        ctx.project_logits_and_sample(h.hidden_t, h.sampled_t);
    });

    h.bind_head_metadata();
    h.base.model.output_head.n = sn::TextConfig::output_rows - 1;
    throws_invalid_with("output head must be [248320,4096]", "wrong head rows", [&] {
        ctx.project_logits_and_sample(h.hidden_t, h.sampled_t);
    });
    h.base.model.output_head.n = sn::TextConfig::output_rows;

    // The binding now passes, so the 4 KiB fixture arena is what rejects the
    // request: the BF16 normed column alone is 8 KiB. Nothing is dispatched.
    throws_invalid_with("workspace capacity is too small", "undersized workspace", [&] {
        ctx.project_logits_and_sample(h.hidden_t, h.sampled_t, &h.logprob_t);
    });
}

void test_head_workspace_contract() {
    const std::size_t plain = sn::TextExecutionContext::head_workspace_capacity_bytes(false, false);
    const std::size_t with_logprob =
        sn::TextExecutionContext::head_workspace_capacity_bytes(true, false);
    const std::size_t with_margin =
        sn::TextExecutionContext::head_workspace_capacity_bytes(true, true);

    const std::size_t columns =
        static_cast<std::size_t>(kHidden) * 2 +
        static_cast<std::size_t>(sn::TextConfig::output_rows) * 2;
    expect(plain >= columns, "head capacity covers the normed and logit columns");
    expect(plain % 256 == 0 && with_logprob % 256 == 0 && with_margin % 256 == 0,
           "head capacity is reported in whole allocation quanta");
    expect(with_logprob >= plain + sizeof(float),
           "a logprob without a margin destination reserves margin scratch");
    expect(with_margin == plain, "a bound margin needs no extra scratch");
}

// -----------------------------------------------------------------------------
// Stage E4C: the 56-visit loop.
//
// The injected executor proves the plan without a device: visit order, the
// physical layer whose weights each visit reads, and every state address the
// visit is bound to. The fixture's weights are empty metadata -- the recorder
// never executes them -- and the arena is host storage, which is exactly why this
// test runs where no GPU or toolchain exists.
// -----------------------------------------------------------------------------

constexpr std::uint32_t kVisitKvCapacity = 128; // two 64-token state-image KV pages

// Widths and buffer addresses are copied out of the request at call time: the
// stream buffers forward_visits allocates live on its own stack, so retaining
// their Tensor pointers would dangle once the pass returns.
struct VisitRecord {
    std::uint32_t visit              = 0;
    int physical_layer               = 0;
    const sn::TextLayerWeights* weights = nullptr;
    std::int32_t hidden_in_width  = 0;
    std::int32_t hidden_out_width = 0;
    const void* hidden_in_data    = nullptr;
    const void* hidden_out_data   = nullptr;
    const ninfer::Tensor* positions = nullptr;
    const std::byte* gdn_state       = nullptr;
    const std::byte* mhc_state       = nullptr;
    const std::byte* kv_domain       = nullptr;
    std::uint32_t kv_domain_tokens   = 0;
    const void* kv_block_table_data = nullptr;
    const void* kv_table_rows_data  = nullptr;
    const ninfer::Tensor* gdn_conv    = nullptr;
    const ninfer::Tensor* router_attn = nullptr;
    const ninfer::Tensor* router_ffn  = nullptr;
    std::uint32_t envelope_min = 0;
    std::uint32_t envelope_max = 0;
};

std::vector<VisitRecord> visit_records;

void record_visit(const sn::LayerVisitRequest& request) {
    visit_records.push_back(VisitRecord{
        .visit              = request.visit,
        .physical_layer     = sn::LoopPlan::physical_layer(request.visit),
        .weights            = request.weights,
        .hidden_in_width    = request.hidden_in != nullptr ? request.hidden_in->ne[0] : 0,
        .hidden_out_width   = request.hidden_out != nullptr ? request.hidden_out->ne[0] : 0,
        .hidden_in_data     = request.hidden_in != nullptr ? request.hidden_in->data : nullptr,
        .hidden_out_data    = request.hidden_out != nullptr ? request.hidden_out->data : nullptr,
        .positions          = request.positions,
        .gdn_state          = request.gdn_state_base,
        .mhc_state          = request.mhc_state_base,
        .kv_domain          = request.attention_kv_domain_base,
        .kv_domain_tokens   = request.attention_kv_domain_tokens,
        .kv_block_table_data = request.attention_kv_block_table != nullptr
                                   ? request.attention_kv_block_table->data
                                   : nullptr,
        .kv_table_rows_data  = request.attention_kv_table_rows != nullptr
                                   ? request.attention_kv_table_rows->data
                                   : nullptr,
        .gdn_conv           = request.gdn_convolution_weight,
        .router_attn        = request.mhc_attn_weight,
        .router_ffn         = request.mhc_ffn_weight,
        .envelope_min       = request.attention_envelope.min_visible_keys,
        .envelope_max       = request.attention_envelope.max_visible_keys,
    });
}

template <class Left, class Right>
bool same_pointer(Left* left, Right* right) {
    return static_cast<const void*>(left) == static_cast<const void*>(right);
}

struct VisitFixture {
    std::vector<std::byte> image{
        sn::StateImageLayout::total_bytes(kVisitKvCapacity, false, false)};
    std::vector<std::byte> ws_storage{sn::TextExecutionContext::forward_workspace_capacity_bytes(
        2, 2)};
    sn::RuntimeModelView model{};
    WorkspaceArena ws{DeviceSpan{ws_storage.data(), ws_storage.size()}};
    cudaStream_t stream{reinterpret_cast<cudaStream_t>(std::uintptr_t{0x1})};

    std::array<std::int32_t, 2> block_table{0, 1};
    std::array<std::int32_t, 1> table_rows{0};
    ninfer::Tensor block_table_t{block_table.data(), ninfer::DType::I32,
                                 {static_cast<std::int32_t>(block_table.size())}};
    ninfer::Tensor table_rows_t{table_rows.data(), ninfer::DType::I32, {1}};

    // Shared stub payloads: the recorder only inspects the bindings, and every
    // op-form tensor of the same kind has the same shape.
    std::array<std::uint16_t, 8192 * 4> conv_storage{};
    std::array<float, 8192 * 8> router_storage{};
    std::array<ninfer::Tensor, sn::TextConfig::layers> conv{};
    std::array<ninfer::Tensor, 2 * sn::TextConfig::layers> router{};

    VisitFixture() {
        for (auto& tensor : conv) {
            tensor = ninfer::Tensor(conv_storage.data(), ninfer::DType::BF16,
                                    {sn::TextConfig::convolution_dim, 4});
        }
        for (auto& tensor : router) {
            tensor = ninfer::Tensor(router_storage.data(), ninfer::DType::FP32,
                                    {sn::MhcConfig::input_width, sn::MhcConfig::logits});
        }
    }

    sn::TextExecutionBindings bindings() {
        return sn::TextExecutionBindings{block_table_t, table_rows_t, conv, router};
    }
};

void test_forward_visits_plan() {
    VisitFixture f;
    visit_records.clear();
    const sn::TextExecutionContext ctx(f.model, f.stream, f.image.data(), kVisitKvCapacity, f.ws,
                                      f.bindings(), &record_visit);

    std::array<std::int32_t, 2> positions_host{0, 1};
    ninfer::Tensor positions(positions_host.data(), ninfer::DType::I32, {2});
    std::array<std::uint16_t, static_cast<std::size_t>(kHidden) * 2> hidden{};
    ninfer::Tensor hidden_t(hidden.data(), ninfer::DType::BF16, {kHidden, 2});

    ctx.forward_visits(hidden_t, positions, 2);

    expect(visit_records.size() == sn::LoopPlan::kVisits,
           "forward_visits executes exactly the 56 planned visits");
    if (visit_records.size() != sn::LoopPlan::kVisits) { return; }

    for (std::size_t visit = 0; visit < sn::LoopPlan::kVisits; ++visit) {
        const VisitRecord& record = visit_records[visit];
        const int physical        = sn::LoopPlan::physical_layer(visit);
        expect(record.visit == visit, "visits run in LoopPlan order");
        expect(record.physical_layer == physical, "visit reports its physical layer");
        expect(same_pointer(record.weights, &ctx.model().layers[physical]),
               "visit reads the physical layer's shared weights");
        expect(record.positions == &positions, "visit receives the pass positions");
        expect(record.envelope_min == 2 && record.envelope_max == 2,
               "visit receives the declared visible-key envelope");

        if (sn::LoopPlan::is_full_attention(visit)) {
            const std::size_t domain = ctx.kv_domain(visit);
            expect(record.kv_domain == f.image.data() +
                                          sn::StateImageLayout::kv_offset(
                                              domain, kVisitKvCapacity, false, false),
                   "full-attention visit binds its own state-image KV domain");
            expect(record.kv_domain_tokens == kVisitKvCapacity,
                   "KV domain is bound at the declared capacity");
            expect(record.kv_block_table_data == f.block_table_t.data &&
                       record.kv_table_rows_data == f.table_rows_t.data,
                   "KV domain carries the identity map and row selector");
            expect(record.gdn_state == nullptr && record.gdn_conv == nullptr,
                   "full-attention visit takes no GDN state");
        } else {
            expect(record.gdn_state == ctx.gdn_state_base(visit),
                   "GDN visit binds its own visit state slot");
            expect(record.gdn_conv == &f.conv[static_cast<std::size_t>(physical)],
                   "GDN visit binds the physical layer's op-form convolution weight");
            expect(record.kv_domain == nullptr && record.kv_domain_tokens == 0,
                   "GDN visit takes no KV domain");
        }

        if (sn::LoopPlan::is_mhc(visit)) {
            expect(record.mhc_state == ctx.mhc_state_base(visit),
                   "mHC visit binds its own visit state slot");
            expect(record.router_attn == &f.router[2 * static_cast<std::size_t>(physical)] &&
                       record.router_ffn == &f.router[2 * static_cast<std::size_t>(physical) + 1],
                   "mHC visit binds both router branches of its physical layer");
        } else {
            expect(record.mhc_state == nullptr && record.router_attn == nullptr &&
                       record.router_ffn == nullptr,
                   "non-mHC visit takes no mHC state");
        }
    }

    // Revisited physical layers get distinct state slots, which is the whole point
    // of addressing state by visit rather than by layer.
    expect(visit_records[7].kv_domain != visit_records[31].kv_domain,
           "layer 7 pass 0 and pass 1 hold distinct KV domains");
    expect(visit_records[4].mhc_state != nullptr && visit_records[3].mhc_state == nullptr,
           "mHC state begins at the enter boundary");

    // The carried width changes exactly at the mHC boundaries and nowhere else.
    expect(visit_records[4].hidden_in_width == kHidden &&
               visit_records[4].hidden_out_width == sn::MhcConfig::input_width,
           "mHC enter widens 4096 to 8192");
    expect(visit_records[5].hidden_in_width == sn::MhcConfig::input_width &&
               visit_records[5].hidden_out_width == sn::MhcConfig::input_width,
           "an interior mHC visit carries 8192");
    expect(visit_records[5].hidden_in_data != visit_records[5].hidden_out_data,
           "interior mHC visits ping-pong between two stream buffers");
    expect(visit_records[26].hidden_out_data != visit_records[27].hidden_out_data,
           "consecutive visits alternate their destination");
    expect(visit_records[27].hidden_in_width == sn::MhcConfig::input_width &&
               visit_records[27].hidden_out_width == kHidden,
           "mHC leave narrows 8192 to 4096");
    // Visit 28 is the enter boundary again (layer 4, pass 1); the post-mHC run
    // starts at visit 52 (layer 28).
    expect(visit_records[52].hidden_in_width == kHidden &&
               visit_records[52].hidden_out_width == kHidden,
           "post-mHC visits carry 4096 again");
}

void test_forward_visits_requires_bindings() {
    VisitFixture f;
    const sn::TextExecutionContext ctx(f.model, f.stream, f.image.data(), kVisitKvCapacity, f.ws);

    std::array<std::int32_t, 2> positions_host{0, 1};
    ninfer::Tensor positions(positions_host.data(), ninfer::DType::I32, {2});
    std::array<std::uint16_t, static_cast<std::size_t>(kHidden) * 2> hidden{};
    ninfer::Tensor hidden_t(hidden.data(), ninfer::DType::BF16, {kHidden, 2});

    throws_invalid_with("identity KV block table", "unbound KV assets",
                        [&] { ctx.forward_visits(hidden_t, positions, 2); });
}

} // namespace

int main() {
    test_valid_context();
    test_state_bases();
    test_invalid_construction();
    test_invalid_visit_helpers();
    test_embed_tokens_contract();
    test_head_contract();
    test_head_workspace_contract();
    test_forward_visits_plan();
    test_forward_visits_requires_bindings();
    if (failures != 0) {
        std::cerr << "FAIL SenseNova text execution: " << failures << '\n';
        return 1;
    }
    std::cout << "OK SenseNova text execution\n";
    return 0;
}
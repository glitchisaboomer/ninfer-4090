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
    const std::byte* gdn4 = ctx.gdn_state_base(4);
    const std::byte* gdn28 = ctx.gdn_state_base(28);
    expect(gdn4 != gdn28, "GDN state bases are unique for visits 4 and 28");
    expect(std::size_t(gdn4 - base) == kv_bytes + sn::StateImageLayout::gdn_offset(4),
           "GDN slot 4 address matches StateImageLayout");
    expect(std::size_t(gdn28 - base) == kv_bytes + sn::StateImageLayout::gdn_offset(28),
           "GDN slot 28 address matches StateImageLayout");
    expect(gdn28 - gdn4 ==
               std::ptrdiff_t(24 * sn::StateLayout::gdn_state_bytes_per_slot()),
           "GDN slot 28 is 24 slot strides after slot 4");
    expect(gdn28 + sn::StateLayout::gdn_state_bytes_per_slot() <= base + kv_bytes + gdn_region,
           "GDN slot 28 fits inside the GDN region");

    // mHC slots: unique for visits 4 and 28 as well, and the mHC region follows
    // the GDN region.
    const std::byte* mhc4 = ctx.mhc_state_base(4);
    const std::byte* mhc28 = ctx.mhc_state_base(28);
    expect(mhc4 != mhc28, "mHC state bases are unique for visits 4 and 28");
    expect(std::size_t(mhc4 - base) == kv_bytes + gdn_region + sn::StateImageLayout::mhc_offset(4),
           "mHC slot 4 address matches StateImageLayout");
    expect(mhc4 > gdn28, "mHC region follows the GDN region");
    expect(mhc28 - mhc4 ==
               std::ptrdiff_t(24 * sn::StateLayout::mhc_state_bytes_per_slot()),
           "mHC slot 28 is 24 slot strides after slot 4");
    expect(mhc28 + sn::StateLayout::mhc_state_bytes_per_slot() <= base + f.image.size(),
           "mHC slot 28 fits inside the state image");

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

    expect(throws_range<std::out_of_range>(
              "GDN slot past the region",
              [&] { (void)ctx.gdn_state_base(sn::StateImageLayout::kGdnSlots); }),
           "gdn_state_base at the region end throws");
    expect(throws_range<std::out_of_range>("mHC slot past the region",
                                           [&] { (void)ctx.mhc_state_base(sn::StateImageLayout::kMhcSlots); }),
           "mhc_state_base at the region end throws");
    expect(throws_range<std::out_of_range>("visit out of range",
                                           [&] { (void)ctx.gdn_state_base(sn::LoopPlan::kVisits); }),
           "gdn_state_base for visit 56 throws");
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

} // namespace

int main() {
    test_valid_context();
    test_state_bases();
    test_invalid_construction();
    test_invalid_visit_helpers();
    test_embed_tokens_contract();
    test_head_contract();
    test_head_workspace_contract();
    if (failures != 0) {
        std::cerr << "FAIL SenseNova text execution: " << failures << '\n';
        return 1;
    }
    std::cout << "OK SenseNova text execution\n";
    return 0;
}
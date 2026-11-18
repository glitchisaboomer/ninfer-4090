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

} // namespace

int main() {
    test_valid_context();
    test_state_bases();
    test_invalid_construction();
    test_invalid_visit_helpers();
    if (failures != 0) {
        std::cerr << "FAIL SenseNova text execution: " << failures << '\n';
        return 1;
    }
    std::cout << "OK SenseNova text execution\n";
    return 0;
}
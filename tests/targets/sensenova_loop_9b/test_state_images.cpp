#include "targets/sensenova_loop_9b/impl/loop_plan.h"
#include "targets/sensenova_loop_9b/impl/runtime/state_images.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <set>
#include <string_view>
#include <vector>

namespace {

namespace sn = ninfer::targets::sensenova_loop_9b::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_domains() {
    static_assert(sn::StateImageLayout::kKvDomains == 15);
    static_assert(sn::StateImageLayout::kGdnSlots == 42);
    static_assert(sn::StateImageLayout::kMhcSlots == 48);

    constexpr std::size_t tokens = 4096;
    constexpr std::size_t per_domain =
        sn::StateImageLayout::kv_bytes_per_domain(tokens, false, false);
    expect(per_domain == tokens * 2 * 4 * 256 * 2, "BF16 KV bytes in one domain");
    expect(sn::StateImageLayout::total_kv_bytes(tokens, false, false) == 15 * per_domain,
           "14 text visits plus one MTP KV domain");
    expect(sn::StateImageLayout::kv_offset(0, tokens, false, false) == 0, "first KV domain");
    expect(sn::StateImageLayout::kv_offset(14, tokens, false, false) == 14 * per_domain,
           "MTP KV domain is last");
}

void test_stable_visit_mapping() {
    std::set<std::size_t> kv;
    std::set<std::size_t> gdn;
    std::set<std::size_t> mhc;
    for (std::size_t visit = 0; visit < sn::LoopPlan::kVisits; ++visit) {
        if (sn::LoopPlan::is_full_attention(visit)) {
            kv.insert(sn::StateImageLayout::kv_domain_for_visit(visit));
        }
        if (sn::LoopPlan::is_gdn(visit)) {
            gdn.insert(sn::StateImageLayout::gdn_slot_for_visit(visit));
        }
        if (sn::LoopPlan::is_mhc(visit)) {
            mhc.insert(sn::StateImageLayout::mhc_slot_for_visit(visit));
        }
    }
    expect(kv.size() == sn::StateLayout::full_attention_state_slots, "unique full KV domains");
    expect(gdn.size() == sn::StateLayout::gdn_state_slots, "unique GDN slots");
    expect(mhc.size() == sn::StateLayout::mhc_state_slots, "unique mHC slots");
    expect(sn::StateImageLayout::kv_domain_for_visit(0) !=
               sn::StateImageLayout::kv_domain_for_visit(14),
           "distinct KV visits do not alias");
    expect(sn::StateImageLayout::gdn_slot_for_visit(4) !=
               sn::StateImageLayout::gdn_slot_for_visit(28),
           "revisited physical layers use different GDN slots");
    expect(sn::StateImageLayout::mhc_slot_for_visit(4) !=
               sn::StateImageLayout::mhc_slot_for_visit(28),
           "revisited physical layers use different mHC slots");
}

void test_offsets() {
    constexpr std::size_t tokens = 1024;
    constexpr bool i8 = true;
    constexpr bool scales = true;
    const std::size_t kv_total = sn::StateImageLayout::total_kv_bytes(tokens, i8, scales);
    const std::size_t gdn0 = kv_total + sn::StateImageLayout::gdn_offset(0);
    const std::size_t gdn1 = kv_total + sn::StateImageLayout::gdn_offset(1);
    const std::size_t mhc0 = kv_total + sn::StateImageLayout::fixed_bytes() - 0;
    expect(gdn1 - gdn0 == sn::StateLayout::gdn_state_bytes_per_slot(), "GDN slot stride");
    expect(sn::StateImageLayout::mhc_offset(1) - sn::StateImageLayout::mhc_offset(0) ==
               sn::StateLayout::mhc_state_bytes_per_slot(),
           "mHC slot stride");
    expect(mhc0 >= gdn1, "mHC region follows GDN region");
    expect(sn::StateImageLayout::total_bytes(tokens, i8, scales) ==
               kv_total + sn::StateImageLayout::fixed_bytes(),
           "total state bytes");
}

} // namespace

int main() {
    test_domains();
    test_stable_visit_mapping();
    test_offsets();
    if (failures != 0) {
        std::cerr << "FAIL senseNova state images: " << failures << '\n';
        return 1;
    }
    std::cout << "OK senseNova state images\n";
    return 0;
}

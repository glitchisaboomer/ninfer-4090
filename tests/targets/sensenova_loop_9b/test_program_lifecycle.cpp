#include "targets/sensenova_loop_9b/impl/runtime_stub.h"
#include "targets/sensenova_loop_9b/impl/runtime/state_images.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/sensenova_loop_9b/package.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
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

void test_lifecycle() {
    auto program = sn::Program::create(2, 32, 4096, 1U << 30U);
    const auto capacity = program->admission_capacity();
    expect(capacity.active_lanes == 2 && capacity.main_kv_pages == 32, "admission capacity");
    expect(program->config_signature_slug().find("sensenova_loop_9b") != std::string::npos,
           "config signature");

    auto plan = sn::RequestPlanForTest::make();

    expect(program->can_admit_lane(0, plan), "empty lane admits");
    expect(program->can_admit_lane(2, plan) == false, "out-of-range lane rejects");

    auto active = sn::Program::inspect_lane_for_test(*program, 0);
    expect(!active.occupied && !active.retained, "lane starts empty");
}

void test_state_images() {
    auto program = sn::Program::create(1, 16, 2048, 1U << 20U);
    const auto images = sn::Program::inspect_state_images_for_test(*program);
    expect(images.base != nullptr, "state images are device-backed");
    expect(images.bytes == images.kv_bytes + images.gdn_bytes + images.mhc_bytes,
           "state image regions cover the allocation");
    expect(images.kv_bytes == sn::StateImageLayout::total_kv_bytes(16 * 128, false, false),
           "KV region uses declared KV capacity");
    expect(images.gdn_bytes == 42 * sn::StateLayout::gdn_state_bytes_per_slot(),
           "GDN region covers 42 visit slots");
    expect(images.mhc_bytes == 48 * sn::StateLayout::mhc_state_bytes_per_slot(),
           "mHC region covers 48 visit slots");
    expect(images.gdn_region != images.mhc_region && images.gdn_region != images.mtp_kv_region,
           "state domains do not alias");

    sn::Program::write_state_image_markers_for_test(*program, 0xa5);
    expect(sn::Program::verify_state_image_markers_for_test(*program, 0xa5),
           "state image marker write/verify");
    expect(!sn::Program::verify_state_image_markers_for_test(*program, 0x5a),
           "state image marker mismatch is observed");
    sn::Program::write_state_image_markers_for_test(*program, 0);

    const auto again = sn::Program::inspect_state_images_for_test(*program);
    expect(again.base == images.base && again.gdn_region == images.gdn_region &&
               again.mhc_region == images.mhc_region,
           "state addresses are stable across access");
}

void test_memory() {
    auto program   = sn::Program::create(1, 16, 2048, 987654321ULL);
    const auto mem = program->memory_summary();
    expect(mem.kv_capacity == 16 * 128, "memory summary KV capacity");
    expect(mem.runtime_reservation_bytes == 987654321ULL, "memory summary reservation");
    program->reset_memory_peaks();
}

} // namespace

int main() {
    test_lifecycle();
    test_state_images();
    test_memory();
    if (failures != 0) {
        std::cerr << "FAIL senseNova program lifecycle: " << failures << '\n';
        return 1;
    }
    std::cout << "OK senseNova program lifecycle\n";
    return 0;
}

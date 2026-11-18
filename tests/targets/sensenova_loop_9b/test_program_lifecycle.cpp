#include "targets/sensenova_loop_9b/impl/runtime_stub.h"

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
    test_memory();
    if (failures != 0) {
        std::cerr << "FAIL senseNova program lifecycle: " << failures << '\n';
        return 1;
    }
    std::cout << "OK senseNova program lifecycle\n";
    return 0;
}

#include "targets/sensenova_loop_9b/impl/runtime/layouts.h"
#include "targets/sensenova_loop_9b/impl/state_layout.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

namespace sn = ninfer::targets::sensenova_loop_9b::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

sn::SequencePlanningInputs inputs() {
    return sn::SequencePlanningInputs{
        .capacity        = 4096,
        .max_concurrency = 2,
        .prefill_chunk   = 1024,
        .draft_window    = 5,
        .kv_cache        = ninfer::KvCacheStorage::BFloat16,
        .use_cuda_graph  = true,
        .device          = 0,
    };
}

void test_curve() {
    const auto in = inputs();
    const auto curve = sn::make_sequence_capacity_curve(in);
    expect(curve.main_page_tokens == 128, "page token count");
    expect(curve.minimum_main_page_groups == 32, "minimum page groups");
    expect(curve.maximum_main_page_groups == 64, "maximum page groups");
    expect(curve.minimum_device_reservation_bytes > 0, "minimum reservation");
    expect(curve.bytes_per_additional_main_page_group ==
               sn::StateLayout::total_kv_bytes_per_token(false, false, true) * 128,
           "per-page-group stride uses 14 full KV visits plus MTP");
}

void test_layout() {
    const auto in     = inputs();
    const auto layout = sn::finalize_sequence_layout(in, 32);
    expect(layout.capacity == 4096, "capacity");
    expect(layout.kv_capacity == 4096, "resolved KV capacity");
    expect(layout.main_page_groups == 32, "page groups");
    expect(layout.draft_window == 5, "draft window");
    expect(layout.persistent_bytes ==
               sn::StateLayout::total_fixed_state_bytes() +
                   sn::StateLayout::total_kv_bytes_per_token(false, false, true) * 4096,
           "persistent bytes include all logical state");
    expect(layout.device_reservation_bytes ==
               layout.persistent_bytes + layout.workspace_bytes + layout.graph_allowance_bytes +
                   layout.request_transient_bytes,
           "reservation equals declared regions");

    auto int8 = inputs();
    int8.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    const auto int8_layout = sn::finalize_sequence_layout(int8, 32);
    expect(int8_layout.persistent_bytes == sn::StateLayout::total_fixed_state_bytes() +
                                              sn::StateLayout::total_kv_bytes_per_token(
                                                  true, true, true) * 4096,
           "INT8 scale planes remain accounted");

    auto no_graph = inputs();
    no_graph.use_cuda_graph = false;
    const auto eager = sn::finalize_sequence_layout(no_graph, 32);
    expect(eager.graph_allowance_bytes == 0 && eager.device_reservation_bytes <
                                                   layout.device_reservation_bytes,
           "graph allowance is explicit");
}

void test_rejection() {
    auto invalid = inputs();
    invalid.draft_window = 16;
    bool rejected = false;
    try {
        (void)sn::make_sequence_capacity_curve(invalid);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "draft window over 15 is rejected");

    invalid           = inputs();
    invalid.kv_cache  = ninfer::KvCacheStorage::RK4V4E8;
    rejected          = false;
    try {
        (void)sn::make_sequence_capacity_curve(invalid);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "unsupported KV mode is rejected");

    invalid              = inputs();
    invalid.max_concurrency = 9;
    rejected             = false;
    try {
        (void)sn::make_sequence_capacity_curve(invalid);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "invalid concurrency is rejected");
}

} // namespace

int main() {
    test_curve();
    test_layout();
    test_rejection();
    if (failures != 0) {
        std::cerr << "FAIL senseNova sequence plan: " << failures << '\n';
        return 1;
    }
    std::cout << "OK senseNova sequence plan\n";
    return 0;
}

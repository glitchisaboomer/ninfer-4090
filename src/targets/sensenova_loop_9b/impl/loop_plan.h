#pragma once

#include "targets/sensenova_loop_9b/impl/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::sensenova_loop_9b::detail {

namespace loop_plan_detail {

// clang-format off
inline constexpr std::array<int, 56> kPhysicalLayers{
    0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31};
// clang-format on

constexpr std::array<int, 56> make_passes() {
    std::array<int, 56> result{};
    for (std::size_t visit = 0; visit < result.size(); ++visit) {
        const int layer = kPhysicalLayers[visit];
        result[visit]   = (layer >= 4 && layer < 28 && visit >= 28) ? 1 : 0;
    }
    return result;
}

inline constexpr std::array<int, 56> kPassesByVisit = make_passes();

} // namespace loop_plan_detail

// The only supported execution plan. The official `execution_plan()` requires
// `counts == [eval_count]`; SenseNova-Loop-9B publishes `eval_count=2`, `start=4`, `end=28`.
// There is deliberately no runtime-settable loop count or alternate plan constructor.
class LoopPlan {
public:
    static constexpr std::size_t kVisits = 56;
    static constexpr std::size_t kPasses = 2;

    using Table = std::array<int, kVisits>;
    using Pass  = std::array<int, kVisits>;

    static constexpr const Table& kPhysicalLayers = loop_plan_detail::kPhysicalLayers;
    static constexpr const Pass& kPassesByVisit   = loop_plan_detail::kPassesByVisit;

    [[nodiscard]] static constexpr std::size_t size() noexcept { return kVisits; }

    [[nodiscard]] static constexpr std::size_t pass_count() noexcept { return kPasses; }

    [[nodiscard]] static constexpr int physical_layer(std::size_t visit) {
        if (visit >= kVisits) { throw std::out_of_range("SenseNova loop visit is out of range"); }
        return kPhysicalLayers[visit];
    }

    [[nodiscard]] static constexpr int pass(std::size_t visit) {
        if (visit >= kVisits) { throw std::out_of_range("SenseNova loop visit is out of range"); }
        return kPassesByVisit[visit];
    }

    [[nodiscard]] static constexpr bool is_full_attention(std::size_t visit) {
        return TextConfig::is_full_attention(physical_layer(visit));
    }

    [[nodiscard]] static constexpr bool is_gdn(std::size_t visit) {
        return !is_full_attention(visit);
    }

    [[nodiscard]] static constexpr bool is_mhc(std::size_t visit) {
        return TextConfig::is_mhc_layer(physical_layer(visit));
    }

    [[nodiscard]] static constexpr std::size_t state_slot(std::size_t visit) noexcept {
        return visit;
    }

    [[nodiscard]] static constexpr std::size_t full_attention_visits() noexcept { return 14; }

    [[nodiscard]] static constexpr std::size_t gdn_visits() noexcept { return 42; }

    [[nodiscard]] static constexpr std::size_t mhc_visits() noexcept { return 48; }
};

static_assert(LoopPlan::kPhysicalLayers.size() == 56);
static_assert(LoopPlan::full_attention_visits() + LoopPlan::gdn_visits() == LoopPlan::kVisits);

} // namespace ninfer::targets::sensenova_loop_9b::detail

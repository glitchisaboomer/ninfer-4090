#pragma once

// The Q5 LinearAdd schedule catalog: which schedule a given exact problem resolves to, and which of
// those schedules may write an output element from more than one CTA.
//
// This header is deliberately free of any CUDA dependency so that the routing policy can be
// unit-tested without a CUDA toolchain, the same way `targets/qwen3_6/impl/runtime/scr_quality.h`
// is. `q5_linear_add_plan.h` owns the CUDA-typed execution surface and includes this, so nothing
// downstream of it changes.
//
// Two parts, and the second is the reason this file exists:
//
//   1. the closed, contiguous route tables below, and
//   2. `q5_linear_add_schedule_owns_output_element`, the single declaration of which schedules write
//      each output element from exactly one place.
//
// (2) is a determinism contract, not a performance note. `Split2ExactResidual` splits K across two
// CTAs that both `atomicAdd` their partial into the same bf16 output element
// (`q5_small_t_mma.cuh:44`). Nothing fixes the order of those two additions and bf16 addition is not
// associative, so two otherwise identical prefills of the same prompt can leave residuals differing
// in their low bits. Every other schedule either owns its output tile outright or merges its splits
// in a fixed order, so it cannot. Claiming that in code, next to the table that decides reachability,
// is what makes the reachable nondeterminism auditable instead of folklore -- see
// tests/ops/linear_add/test_q5_residual_band.cpp, which fails if the band widens.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

enum class Q5LinearAddScheduleId {
    GemvResidual,
    Split2ExactResidual,
    MmaResidualR64C16,
    MmaResidualR64C24,
    MmaResidualR64C32,
    MmaResidualR64C64,
    MmaResidualR64C128,
};

struct Q5LinearAddProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q5LinearAddPlan {
    Q5LinearAddScheduleId schedule;
    std::size_t workspace_bytes;
};

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
};

struct RouteSpec {
    ColsSet cols;
    Q5LinearAddScheduleId schedule;
};

// The two shapes this op admits: attention output (k=6144) and MLP down (k=17408), both rows=5120.
constexpr std::array<SupportSpec, 2> kQ5LinearAddSupports{{
    {5120, 6144, 6144},
    {5120, 17408, 17408},
}};

constexpr std::array<RouteSpec, 6> kQ5LinearAddK6144Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 13}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{14, 16}, Q5LinearAddScheduleId::MmaResidualR64C16},
    {{17, 64}, Q5LinearAddScheduleId::MmaResidualR64C32},
    {{65, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128},
}};

constexpr std::array<RouteSpec, 5> kQ5LinearAddK17408Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 16}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{17, 64}, Q5LinearAddScheduleId::MmaResidualR64C32},
    {{65, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128},
}};

// True when the table covers every column count from 1 upward with no gap and no overlap.
template <std::size_t N>
consteval bool q5_linear_add_routes_are_closed(const std::array<RouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(q5_linear_add_routes_are_closed(kQ5LinearAddK6144Routes) &&
                  q5_linear_add_routes_are_closed(kQ5LinearAddK17408Routes),
              "Q5 LinearAdd routes must be exact, contiguous, and closed");

constexpr const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q5LinearAddScheduleId::GemvResidual:
        return "linear_add.q5.gemv.residual";
    case Q5LinearAddScheduleId::Split2ExactResidual:
        return "linear_add.q5.simt.split2.exact.residual";
    case Q5LinearAddScheduleId::MmaResidualR64C16:
        return "linear_add.q5.mma.r64.c16.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C24:
        return "linear_add.q5.mma.r64.c24.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C32:
        return "linear_add.q5.mma.r64.c32.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        return "linear_add.q5.mma.r64.c64.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        return "linear_add.q5.mma.r64.c128.cta_collective_residual";
    }
    return "linear_add.q5.unknown";
}

// Whether a schedule writes every one of its output elements from exactly one place, in a fixed
// order. False for one schedule only: the two-K-split residual epilogue, whose two CTAs both add into
// the same bf16 element in an order the programming model does not fix. False means "two runs of
// identical inputs may differ in the low bits", nothing weaker.
constexpr bool q5_linear_add_schedule_owns_output_element(Q5LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q5LinearAddScheduleId::GemvResidual:
    case Q5LinearAddScheduleId::MmaResidualR64C16:
    case Q5LinearAddScheduleId::MmaResidualR64C24:
    case Q5LinearAddScheduleId::MmaResidualR64C32:
    case Q5LinearAddScheduleId::MmaResidualR64C64:
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        return true;
    case Q5LinearAddScheduleId::Split2ExactResidual:
        return false;
    }
    return false;
}

constexpr bool q5_linear_add_supports_shape(const Q5LinearAddProblem& problem) noexcept {
    for (const SupportSpec& support : kQ5LinearAddSupports) {
        if (problem.rows == support.rows && problem.k == support.k &&
            problem.padded_k == support.padded_k) {
            return true;
        }
    }
    return false;
}

constexpr bool q5_linear_add_admits_shape(const Q5LinearAddProblem& problem) noexcept {
    return q5_linear_add_supports_shape(problem) && problem.cols >= 1;
}

// The route the exact problem takes, or nullptr when no route covers it. `constexpr`, not
// `consteval`, so a caller can sweep the table at runtime as well as pin it in a static_assert. Null is unreachable for an
// admitted problem -- the tables are closed by the static_assert above and `admits_shape` guarantees
// cols >= 1 -- which is exactly why finding one is worth checking rather than assuming: it is the
// invariant that makes routing total.
constexpr const RouteSpec* q5_linear_add_find_route(const Q5LinearAddProblem& problem) noexcept {
    if (problem.k == 6144) {
        for (const RouteSpec& route : kQ5LinearAddK6144Routes) {
            if (route.cols.contains(problem.cols)) { return &route; }
        }
        return nullptr;
    }
    for (const RouteSpec& route : kQ5LinearAddK17408Routes) {
        if (route.cols.contains(problem.cols)) { return &route; }
    }
    return nullptr;
}

constexpr Q5LinearAddPlan q5_linear_add_resolve_catalog(const Q5LinearAddProblem& problem) noexcept {
    const RouteSpec* route = q5_linear_add_find_route(problem);
    return route == nullptr ? Q5LinearAddPlan{Q5LinearAddScheduleId::GemvResidual, 0}
                            : Q5LinearAddPlan{route->schedule, 0};
}

} // namespace ninfer::ops::detail

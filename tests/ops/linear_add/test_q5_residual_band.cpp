// Test for the Q5 LinearAdd schedule catalog: which schedule an exact problem resolves to, and how
// much of that routing writes an output element from more than one place.
//
// Why this test exists. Two prefills of one identical prompt disagreed in the first-token logprob by
// 1.7e-4..5.1e-3 nats, and the cause was the two-K-split residual epilogue:
// `Q5SmallTMmaResidualAtomicEpilogue` (`src/ops/linear/q5/q5_small_t_mma.cuh:44`) has two CTAs,
// each owning half of K, both doing `atomicAdd` into the same bf16 output element. The programming
// model fixes no order between them and bf16 addition is not associative, so the residual's low bits
// are not reproducible. See docs/SCR.md.
//
// The band is reachable at every prompt length, which is the part that made the bug hard to see: the
// last prefill chunk is a short tail, so a 660-token prompt and a 67-token prompt both land in it.
//
// What this test guards. The catalog is compile-time policy about *reachability*, so it is the one
// place a person can widen the nondeterministic band by adding a row to a table without touching a
// kernel. The assertions below therefore pin the band explicitly:
//
//   * the tables stay exact, contiguous and closed;
//   * only `Split2ExactResidual` may reach a schedule that does not own its output element;
//   * the reachable band is exactly cols 2..13 for k=6144 and 2..16 for k=17408;
//   * every declared schedule id is either reachable or listed as deliberately unreachable.
//
// Widening the band should require editing this file, not silently changing the schedule table. Note
// what this test cannot do: it proves which schedules are reachable and states the determinism of
// each, but the *write itself* is a kernel and only a device run (or compute-sanitizer) can show the
// two orders actually differing. Part 3 below sizes that on the host so the claim is not folklore.
//
// This test is host-only on purpose. The catalog header includes no CUDA header, matching how
// `scr_quality.h` is built and tested; it runs in an environment with no nvcc and no GPU.

#include "ops/linear_add/q5/q5_linear_add_catalog.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

using ninfer::ops::detail::ColsSet;
using ninfer::ops::detail::Q5LinearAddProblem;
using ninfer::ops::detail::Q5LinearAddScheduleId;

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("  FAIL: %s\n", what);
        ++g_failures;
    }
}

// The two shapes the op admits, in the same order the support inventory lists them.
constexpr Q5LinearAddProblem kShapes[] = {
    {5120, 6144, 6144, 1},
    {5120, 17408, 17408, 1},
};

// ---------------------------------------------------------------------------------------------
// Part 1: the catalog is total.
//
// A gap in a route table is a shape that no schedule serves, and the failure mode downstream is a
// silent fallthrough rather than an error, so the sweep below asks for a route at every column count
// in the band that matters rather than trusting the closedness static_assert alone.
// ---------------------------------------------------------------------------------------------
void test_catalog_is_total() {
    std::printf("part 1: catalog is total over cols 1..64\n");
    for (const Q5LinearAddProblem& shape : kShapes) {
        for (std::int32_t cols = 1; cols <= 64; ++cols) {
            const Q5LinearAddProblem problem{shape.rows, shape.k, shape.padded_k, cols};
            check(ninfer::ops::detail::q5_linear_add_admits_shape(problem),
                  "admitted shape must be admitted");
            check(ninfer::ops::detail::q5_linear_add_find_route(problem) != nullptr,
                  "every admitted (rows,k,cols) must have a covering route");
        }
    }

    // A column count of 0 and an unsupported shape must be rejected, not routed.
    check(!ninfer::ops::detail::q5_linear_add_admits_shape({5120, 6144, 6144, 0}),
          "cols=0 must not be admitted");
    check(!ninfer::ops::detail::q5_linear_add_admits_shape({5120, 6143, 6143, 1}),
          "unsupported k must not be admitted");
    check(!ninfer::ops::detail::q5_linear_add_admits_shape({4096, 6144, 6144, 1}),
          "unsupported rows must not be admitted");
}

// ---------------------------------------------------------------------------------------------
// Part 2: the routing table is pinned, and so is the nondeterministic band.
//
// The expected schedule is spelled out per shape and column count. This is the assertion that turns
// "the table is closed" into "the table is *this*": a reroute, a new band, or a schedule id removed
// from a band all fail here rather than shipping.
// ---------------------------------------------------------------------------------------------
Q5LinearAddScheduleId expected_schedule(std::int32_t k, std::int32_t cols) {
    if (k == 6144) {
        if (cols == 1) { return Q5LinearAddScheduleId::GemvResidual; }
        if (cols <= 13) { return Q5LinearAddScheduleId::Split2ExactResidual; }
        if (cols <= 16) { return Q5LinearAddScheduleId::MmaResidualR64C16; }
        if (cols <= 64) { return Q5LinearAddScheduleId::MmaResidualR64C32; }
        if (cols <= 128) { return Q5LinearAddScheduleId::MmaResidualR64C64; }
        return Q5LinearAddScheduleId::MmaResidualR64C128;
    }
    if (k == 17408) {
        if (cols == 1) { return Q5LinearAddScheduleId::GemvResidual; }
        if (cols <= 16) { return Q5LinearAddScheduleId::Split2ExactResidual; }
        if (cols <= 64) { return Q5LinearAddScheduleId::MmaResidualR64C32; }
        if (cols <= 128) { return Q5LinearAddScheduleId::MmaResidualR64C64; }
        return Q5LinearAddScheduleId::MmaResidualR64C128;
    }
    return Q5LinearAddScheduleId::GemvResidual; // not reached
}

void test_routes_are_pinned() {
    std::printf("part 2: routes are pinned, and the order-dependent band is exactly what it is\n");

    // Every column count in the swept band resolves to the schedule named above.
    for (const Q5LinearAddProblem& shape : kShapes) {
        for (std::int32_t cols = 1; cols <= 200; ++cols) {
            const Q5LinearAddProblem problem{shape.rows, shape.k, shape.padded_k, cols};
            const auto plan = ninfer::ops::detail::q5_linear_add_resolve_catalog(problem);
            const Q5LinearAddScheduleId expected = expected_schedule(shape.k, cols);
            if (plan.schedule != expected) {
                std::printf("  FAIL: k=%d cols=%d resolved to %s, expected %s\n", shape.k, cols,
                            ninfer::ops::detail::q5_linear_add_schedule_name(plan.schedule),
                            ninfer::ops::detail::q5_linear_add_schedule_name(expected));
                ++g_failures;
            }
            check(plan.workspace_bytes == 0, "the catalog assigns no workspace");
        }
    }

    // The determinism contract: a problem is order-dependent exactly when its schedule does not own
    // its output element. Anything that is not `Split2ExactResidual` must own it.
    for (const Q5LinearAddProblem& shape : kShapes) {
        for (std::int32_t cols = 1; cols <= 200; ++cols) {
            const Q5LinearAddProblem problem{shape.rows, shape.k, shape.padded_k, cols};
            const auto plan = ninfer::ops::detail::q5_linear_add_resolve_catalog(problem);
            const bool owns =
                ninfer::ops::detail::q5_linear_add_schedule_owns_output_element(plan.schedule);
            const bool is_split2 = plan.schedule == Q5LinearAddScheduleId::Split2ExactResidual;
            check(owns != is_split2,
                  "a schedule is order-dependent exactly when it does not own its output");
        }
    }

    // The band itself, stated as the number it is. If the reachable band is ever widened, this is the
    // line that has to change, and it should not change without a benchmark beside it.
    const auto first_bad = [](std::int32_t k) {
        for (std::int32_t cols = 1; cols <= 512; ++cols) {
            const Q5LinearAddProblem problem{5120, k, k, cols};
            if (!ninfer::ops::detail::q5_linear_add_schedule_owns_output_element(
                    ninfer::ops::detail::q5_linear_add_resolve_catalog(problem).schedule)) {
                return cols;
            }
        }
        return -1;
    };
    const auto last_bad = [](std::int32_t k) {
        for (std::int32_t cols = 512; cols >= 1; --cols) {
            const Q5LinearAddProblem problem{5120, k, k, cols};
            if (!ninfer::ops::detail::q5_linear_add_schedule_owns_output_element(
                    ninfer::ops::detail::q5_linear_add_resolve_catalog(problem).schedule)) {
                return cols;
            }
        }
        return -1;
    };
    check(first_bad(6144) == 2, "k=6144 order-dependent band starts at cols=2");
    check(last_bad(6144) == 13, "k=6144 order-dependent band ends at cols=13");
    check(first_bad(17408) == 2, "k=17408 order-dependent band starts at cols=2");
    check(last_bad(17408) == 16, "k=17408 order-dependent band ends at cols=16");

    // The band has to include the tail the trace actually observed: the last prefill chunk is short,
    // and a 2-token tail must land inside it. If a future change makes col 2 deterministic, this
    // documents that the observed divergence no longer has a reachable cause on this path.
    for (std::int32_t cols = 1; cols <= 16; ++cols) {
        const Q5LinearAddProblem problem{5120, 6144, 6144, cols};
        const auto plan = ninfer::ops::detail::q5_linear_add_resolve_catalog(problem);
        if (cols >= 2 && cols <= 13) {
            check(plan.schedule == Q5LinearAddScheduleId::Split2ExactResidual,
                  "the 2-token tail must still route to the split-K residual schedule, or the "
                  "divergence this test was written for has become unreachable");
        } else {
            check(plan.schedule != Q5LinearAddScheduleId::Split2ExactResidual,
                  "outside the pinned band nothing may reach the split-K residual schedule");
        }
    }

    // Every declared id is either reachable or deliberately not. `MmaResidualR64C24` exists in the
    // enum and in the name table but in no route table; it is dead today and must stay knowingly so,
    // because a schedule that is declared and unreachable is indistinguishable from one that was
    // forgotten.
    const auto reachable = [](Q5LinearAddScheduleId id) {
        for (const Q5LinearAddProblem& shape : kShapes) {
            for (std::int32_t cols = 1; cols <= 200; ++cols) {
                const Q5LinearAddProblem problem{shape.rows, shape.k, shape.padded_k, cols};
                if (ninfer::ops::detail::q5_linear_add_resolve_catalog(problem).schedule == id) {
                    return true;
                }
            }
        }
        return false;
    };
    check(reachable(Q5LinearAddScheduleId::GemvResidual), "GemvResidual must be reachable");
    check(reachable(Q5LinearAddScheduleId::Split2ExactResidual), "Split2ExactResidual reachable");
    check(reachable(Q5LinearAddScheduleId::MmaResidualR64C16), "MmaResidualR64C16 must be reachable");
    check(reachable(Q5LinearAddScheduleId::MmaResidualR64C32), "MmaResidualR64C32 must be reachable");
    check(reachable(Q5LinearAddScheduleId::MmaResidualR64C64), "MmaResidualR64C64 must be reachable");
    check(reachable(Q5LinearAddScheduleId::MmaResidualR64C128),
          "MmaResidualR64C128 must be reachable");
    check(!reachable(Q5LinearAddScheduleId::MmaResidualR64C24),
          "MmaResidualR64C24 is declared but unreachable and must stay so");
}

// ---------------------------------------------------------------------------------------------
// Part 3: size the write this band performs, so "two CTAs add into one bf16 element" is a number
// rather than a sentence.
//
// This is a host model of the kernel epilogue, not the kernel. It establishes that the *arithmetic
// class* is real and how large it is in nats; it cannot show that the two CTAs actually arrive in a
// different order on the device, because that is a scheduling property of the hardware.
//
// The kernel accumulates each CTA's half of K in FP32 and converts once in the epilogue, so:
//
//   two writers, A first:  bf16( bf16(R + bf16(A)) + bf16(B) )
//   two writers, B first:  bf16( bf16(R + bf16(B)) + bf16(A) )
//   one writer:           bf16( R + (A + B) )                 // single FP32 accumulate, one rounding
//
// `R` is the residual stream the epilogue adds onto, `A` and `B` the two K-halves.
// ---------------------------------------------------------------------------------------------
float round_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round to nearest, ties to even, on the low 16 bits -- the same rounding
    // `__float2bfloat16_rn` performs on the device path the epilogue calls. Valid for finite
    // normal operands, which is all this exhibit uses.
    const std::uint32_t lsb = (bits >> 16) & 1u;
    const std::uint32_t rounded = bits + 0x7fffu + lsb;
    const std::uint32_t narrowed = (rounded >> 16) << 16;
    float result = 0.0f;
    std::memcpy(&result, &narrowed, sizeof(result));
    return result;
}

// bf16 addition as the atomic performs it: both operands rounded to bf16, summed, rounded once.
float bf16_add(float lhs, float rhs) { return round_to_bf16(round_to_bf16(lhs) + round_to_bf16(rhs)); }

// One bf16 step at `value`: the distance to the next representable bf16. Used to bound the damage a
// reordering can do, because a bf16 addition can move a result by at most one step of the operand
// that carries it.
float bf16_step(float value) {
    const float rounded = round_to_bf16(value);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &rounded, sizeof(bits));
    const std::uint16_t pattern = static_cast<std::uint16_t>(bits >> 16);
    const std::uint16_t stepped  = static_cast<std::uint16_t>(pattern + 1u);
    const std::uint32_t stepped_bits = static_cast<std::uint32_t>(stepped) << 16;
    float stepped_value = 0.0f;
    std::memcpy(&stepped_value, &stepped_bits, sizeof(stepped_value));
    return std::fabs(stepped_value - rounded);
}

void test_two_writers_can_differ() {
    std::printf("part 3: the two-CTAs/one-element write, sized\n");

    // Deterministic inputs in two bands, because the answer depends on the magnitude of the residual
    // stream: a bf16 step at 2 is 0.016 and at 8 it is 0.0625, so a reordering that is negligible in
    // one band exceeds the roadmap's 0.05-nat bound in the other. Both bands occur in a real forward
    // pass, which is the point of showing them side by side.
    struct Band {
        const char* name;
        float residual_lo;
        float residual_hi;
    };
    constexpr Band kBands[] = {
        {"residual ~1..4", 1.0f, 4.0f},
        {"residual ~4..16", 4.0f, 16.0f},
    };

    for (const Band& band : kBands) {
        std::uint32_t state = 0x2545f491u; // xorshift32, no library dependency
        auto next_uniform = [&state]() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return static_cast<float>(state & 0xffffffu) / static_cast<float>(0xffffffu);
        };

        constexpr int kSamples = 20000;
        int disagreements = 0;
        float worst_nats = 0.0f;
        float worst_one_writer_nats = 0.0f;
        float worst_step = 0.0f;

        for (int sample = 0; sample < kSamples; ++sample) {
            // K-half contributions an order of magnitude smaller than the residual, signs random.
            const float residual = band.residual_lo +
                                   (band.residual_hi - band.residual_lo) * next_uniform();
            const float magnitude = 1e-3f + 5e-2f * next_uniform();
            const float a = (next_uniform() < 0.5f ? -1.0f : 1.0f) * magnitude;
            const float b = (next_uniform() < 0.5f ? -1.0f : 1.0f) * magnitude;

            // Two writers: each CTA rounds its own half to bf16 and adds onto whatever is there.
            const float a_first = bf16_add(bf16_add(residual, a), b);
            const float b_first = bf16_add(bf16_add(residual, b), a);
            // One writer: the single CTA accumulates both halves in FP32 and rounds once at the end.
            const float one_writer = bf16_add(residual, a + b);

            if (a_first != b_first) { ++disagreements; }
            worst_nats = std::max(worst_nats, std::fabs(a_first - b_first));
            worst_one_writer_nats = std::max(worst_one_writer_nats,
                                             std::min(std::fabs(a_first - one_writer),
                                                      std::fabs(b_first - one_writer)));
            worst_step = std::max(worst_step, bf16_step(a_first));
        }

        std::printf("  %-16s samples=%d disagreements=%d (%.2f%%)  worst=%.3e nats  bf16 step=%.3e",
                    band.name, kSamples, disagreements,
                    100.0 * static_cast<double>(disagreements) / static_cast<double>(kSamples),
                    worst_nats, worst_step);
        std::printf("  %s the 0.05-nat roadmap bound\n",
                    worst_nats > 0.05 ? "ABOVE" : "below");
        std::printf("                   one-writer result within %.3e nats of a two-writer answer\n",
                    worst_one_writer_nats);

        // The two orderings must be able to differ somewhere. If they never did, the band would be
        // harmless and this test would be asserting a bug that does not exist.
        check(disagreements > 0, "the two atomic orderings must be able to differ, or the band is "
                                 "not a bug");

        // The damage must be exactly the one thing bf16 addition can produce -- a single step. This
        // is the real bound, and it is what makes the numbers above interpretable rather than a
        // coincidence of the sample.
        check(worst_nats > 0.0f, "the disagreement must be nonzero");
        check(worst_nats <= worst_step,
              "a reordered bf16 atomic can move the result by at most one bf16 step");

        // The single-split override must land within a step or two of the two-writer answer: it is
        // the same math with a fixed association, not a different computation, so a fix that moved
        // the value anywhere else would mean the association changed the value materially.
        check(worst_one_writer_nats <= 2.0f * worst_step,
              "the one-writer result must stay within a couple of bf16 steps of the two-writer one");
    }
}

} // namespace

int main() {
    std::printf("q5 linear_add schedule catalog and residual band\n");

    // A compile-time sample of the catalog, so the pinning is checked before any code runs.
    static_assert(ninfer::ops::detail::q5_linear_add_resolve_catalog({5120, 6144, 6144, 1}).schedule ==
                  Q5LinearAddScheduleId::GemvResidual);
    static_assert(ninfer::ops::detail::q5_linear_add_resolve_catalog({5120, 6144, 6144, 2}).schedule ==
                  Q5LinearAddScheduleId::Split2ExactResidual);
    static_assert(ninfer::ops::detail::q5_linear_add_resolve_catalog({5120, 17408, 17408, 17})
                      .schedule == Q5LinearAddScheduleId::MmaResidualR64C32);
    static_assert(
        !ninfer::ops::detail::q5_linear_add_schedule_owns_output_element(
            ninfer::ops::detail::q5_linear_add_resolve_catalog({5120, 6144, 6144, 2}).schedule));
    static_assert(ninfer::ops::detail::q5_linear_add_schedule_owns_output_element(
        ninfer::ops::detail::q5_linear_add_resolve_catalog({5120, 6144, 6144, 17}).schedule));

    test_catalog_is_total();
    test_routes_are_pinned();
    test_two_writers_can_differ();

    if (g_failures != 0) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

#pragma once

#include "ops/linear_add/q5/q5_linear_add_catalog.h"

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Routing policy lives in `q5_linear_add_catalog.h`, which is CUDA-free on purpose so it can be
// tested without a toolchain. What remains here is the CUDA-typed execution surface. The schedule
// names, the schedule ids and the routing table all come from the catalog.

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept;
Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem);

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols);

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream);
void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail

// Test-only hooks onto ninfer::ops::detail attention launch paths.
//
// This header is included only by tests. It forward-declares the detail:: launcher entry points the
// oracle test needs so it can pass a real delta plane and a real RerotateFreq directly to the kernels
// that Phase 4a already accepts them from, without going through the public API's hardcoded nullptr
// path. The production path still passes nullptr until Phase 4c.
//
// DO NOT include this from production code.
#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/gqa_kv_rerotate.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Decode-side small-T attention, called directly by the oracle test with a real view that carries the
// delta plane and the RoPE frequency table. This is the same path the public gqa_attention_cached
// route reaches through gqa_attention_cached_small_t_launch, but reached directly so the test can
// attach the delta plane without changing the public API.
void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream);

// Prefill-side prompt attention used by the oracle test's prefill path. This is the detail:: wrapper
// around the prompt fill + attention launch, called directly with a real view whose block table and
// delta plane are already staged.
void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail


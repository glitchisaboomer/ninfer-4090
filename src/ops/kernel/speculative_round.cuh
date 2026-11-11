#pragma once

// Implements: include/ninfer/ops/speculative_round.h
// Match: contiguous request-major state and BF16 verification logits.
// Algorithm assumptions: small vocabularies use one cooperative block; the
// registered full-vocabulary stochastic route uses the sampling partial/group
// pipeline and caller-owned workspace, while greedy commit remains one thread.

#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

__device__ inline std::int32_t speculative_masked_argmax(const __nv_bfloat16* logits,
                                                         std::int32_t token_domain,
                                                         const SamplingConfig& cfg,
                                                         float* values, int* indices) {
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int v = threadIdx.x; v < token_domain; v += blockDim.x) {
        const float value = sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg);
        if (sampling_better(value, v, best, best_i)) {
            best = value;
            best_i = v;
        }
    }
    values[threadIdx.x] = best;
    indices[threadIdx.x] = best_i;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            sampling_better(values[threadIdx.x + stride], indices[threadIdx.x + stride],
                            values[threadIdx.x], indices[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + stride];
            indices[threadIdx.x] = indices[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return indices[0];
}

__global__ void speculative_prepare_verify_inputs_kernel(const std::int32_t* anchors,
                                                         const std::int32_t* drafts,
                                                         const std::int32_t* base_positions,
                                                         const std::int32_t* current_extents,
                                                         std::int32_t* verify_ids,
                                                         std::int32_t* positions, std::int32_t k) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int extent    = current_extents[row];
    extent        = extent < 0 ? 0 : (extent > k ? k : extent);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        const int off = row * T + j;
        verify_ids[off] =
            j == 0 ? anchors[row] : (j <= extent ? drafts[row * k + j - 1] : anchors[row]);
        if (positions != nullptr) {
            positions[off] = base_positions[row] + (j <= extent ? j : extent);
        }
    }
}

template <typename T>
__device__ inline T* speculative_workspace_offset(T* ptr, std::size_t byte_offset) {
    return ptr == nullptr
               ? nullptr
               : reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(ptr) + byte_offset);
}

__device__ inline SamplingWorkspace
speculative_workspace_row(SamplingWorkspace workspace, std::size_t row_stride, std::int32_t row) {
    const std::size_t offset = static_cast<std::size_t>(row) * row_stride;
    workspace.partial_keys   = speculative_workspace_offset(workspace.partial_keys, offset);
    workspace.dist_idx       = speculative_workspace_offset(workspace.dist_idx, offset);
    workspace.dist_prob      = speculative_workspace_offset(workspace.dist_prob, offset);
    workspace.dist_support   = speculative_workspace_offset(workspace.dist_support, offset);
    workspace.group_done     = speculative_workspace_offset(workspace.group_done, offset);
    workspace.speculative_finalize_count =
        speculative_workspace_offset(workspace.speculative_finalize_count, offset);
    return workspace;
}

// Commits the round's accepted tokens plus one correction/bonus token, then
// advances the target length. The greedy branch
// (config temperature <= 0) is bit-identical to the original argmax accept: keep
// the longest draft prefix whose target argmax matches, then take the target
// argmax at the divergence column. The sampling branch (temperature > 0) runs
// distribution-correct speculative rejection sampling over the verify logits with
// a one-hot (greedy) draft: accept drafts[i] with probability p_i(drafts[i]) under
// the truncated target distribution, resample from the masked residual on the
// first rejection, and draw a bonus from the last column when every draft accepts.
// The draft-proposal path stays greedy, so q is one-hot and the accept test
// collapses to `u < p_i(drafts[i])`. Launch with a single block of kSamplerBlock
// threads; only thread 0 performs the sequential accept/commit while the whole
// block cooperates on the per-column truncated-distribution build.
// Probe rows: six column statistics per draft position, then two per-row counters.
// `probe` is null on the production path, which makes every write below a predicate the compiler
// removes rather than a branch it has to keep. The layout is documented in
// include/ninfer/ops/speculative_round.h.
// `probe` is contiguous [6*K+2, batch], so the row axis is the batch width and a stat cell for
// (row, column) sits at (6*column)*batch + row. Callers pass `batch` as `stride`.
__device__ __forceinline__ float* speculative_probe_column(float* probe,
                                                                std::int32_t row,
                                                                std::int32_t stride,
                                                                std::int32_t column) {
    return probe + static_cast<std::int64_t>(6 * column) * stride + row;
}

__device__ __forceinline__ void speculative_probe_init_row(float* probe, std::int32_t row,
                                                           std::int32_t k, std::int32_t stride,
                                                           std::int32_t tid,
                                                           std::int32_t block) {
    const std::int32_t rows = 6 * k + 2;
    for (std::int32_t r = tid; r < rows; r += block) { probe[r * stride + row] = 0.0f; }
}

// Fills cells 0..3 of one reached column. `support_idx`/`support_prob` are the normalized truncated
// support this column will be accepted against; pass null to report a greedy column, which has no
// truncated support to summarize. `finite` is false when the column's logits contain a non-finite
// value, in which case the collision probability and draft probability are reported as -1.
// Cells 4 and 5 are written later by speculative_probe_commit_columns, because only the commit step
// knows the round's accepted count.
__device__ __forceinline__ void speculative_probe_write_column(float* probe,
                                                               std::int32_t row, std::int32_t k,
                                                               std::int32_t stride,
                                                               std::int32_t column,
                                                               const int* support_idx,
                                                               const float* support_prob,
                                                               std::int32_t n_support,
                                                               std::int32_t draft, bool finite) {
    // Cell 0 (the reached marker) is written by speculative_probe_commit_columns, not here: the
    // multiblock route builds every offered column's support, so a writer that marked reachability
    // would claim columns the accept loop never verified.
    //
    // The accept loop runs columns 0..extent, so column `extent` is offered whether or not it is a
    // draft. It has no draft identity -- rank and p(draft) are meaningless -- and no cell slot
    // either: the k column slots occupy rows 0..6K-1 while the row summary sits at rows 6K and
    // 6K+1, and the column's remaining cells would run past the end of a [6*K+2, batch] matrix.
    // Return before forming the cell pointer so nothing at all is written for it.
    if (column >= k) { return; }
    float* cell       = speculative_probe_column(probe, row, stride, column);
    if (support_idx == nullptr) {
        cell[1 * stride] = -1.0f;
        cell[2 * stride] = -1.0f;
        cell[3 * stride] = -1.0f;
        return;
    }
    float sum_sq      = -1.0f;
    std::int32_t rank = -1;
    float pd          = 0.0f;
    if (finite) {
        sum_sq = 0.0f;
        for (std::int32_t j = 0; j < n_support; ++j) {
            const float p = support_prob[j];
            sum_sq += p * p;
            if (rank < 0 && support_idx[j] == draft) { rank = j; }
        }
        if (rank >= 0) { pd = support_prob[rank]; }
    }
    cell[1 * stride] = sum_sq;
    cell[2 * stride] = static_cast<float>(rank);
    cell[3 * stride] = pd;
}

// Writes cells 4 and 5 for the columns this round actually reached. The accept loop breaks at the
// first rejection, so a column beyond it was offered as a draft but never verified; marking it
// drafted would put an unverified column in the acceptance denominator and understate q, which is
// the exact confusion this probe exists to remove. `accepted` is the round's accepted-draft count,
// and the reached draft columns are 0..accepted inclusive: the rejection column itself is reached
// and rejected, while accepted == extent means every draft verified.
__device__ __forceinline__ void speculative_probe_commit_columns(float* probe,
                                                                std::int32_t row, std::int32_t stride,
                                                                std::int32_t extent,
                                                                std::int32_t accepted) {
    const std::int32_t reached = accepted + 1;
    const std::int32_t limit   = reached < extent ? reached : extent;
    for (std::int32_t i = 0; i < limit; ++i) {
        float* cell      = speculative_probe_column(probe, row, stride, i);
        cell[0 * stride] = 1.0f;
        cell[4 * stride] = 1.0f;
        cell[5 * stride] = (i < accepted) ? 1.0f : 0.0f;
    }
    // The multiblock column writer fills a column's statistics as soon as that column's support is
    // built, which happens for every offered column, not only the verified ones. Clear the
    // statistics of the offered-but-unverified tail so an unreached column contributes nothing to
    // the conditioned means and cannot be mistaken for a rejected draft.
    for (std::int32_t i = limit; i < extent; ++i) {
        float* cell = speculative_probe_column(probe, row, stride, i);
        for (std::int32_t stat = 0; stat < 6; ++stat) { cell[stat * stride] = 0.0f; }
    }
}

// Writes the two trailing row counters: drafts offered this round, then drafts accepted.
__device__ __forceinline__ void speculative_probe_write_row_summary(float* probe,
                                                                    std::int32_t row,
                                                                    std::int32_t k,
                                                                    std::int32_t stride,
                                                                    std::int32_t offered,
                                                                    std::int32_t accepted) {
    probe[(6 * k + 0) * stride + row] = static_cast<float>(offered);
    probe[(6 * k + 1) * stride + row] = static_cast<float>(accepted);
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_accept_greedy_drafts_kernel(
    const std::int32_t* target_tokens, const __nv_bfloat16* logits, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t k, float* probe) {
    const int tid                   = threadIdx.x;
    const int row                   = static_cast<int>(blockIdx.x);
    const int cols                  = k + 1;
    // The probe's row axis is the batch width; this kernel launches one block per batch row.
    const std::int32_t batch_stride = static_cast<std::int32_t>(gridDim.x);
    int extent                      = current_extents[row];
    extent                          = extent < 0 ? 0 : (extent > k ? k : extent);
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    const __nv_bfloat16* row_logits =
        logits + static_cast<std::int64_t>(row) * cols * physical_rows;

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];

    if (probe != nullptr) {
        speculative_probe_init_row(probe, row, k, static_cast<std::int32_t>(gridDim.x), tid,
                                   blockDim.x);
        __syncthreads();
    }

    if (!(cfg.temperature > 0.0f)) {
        int first_target = row_targets[0];
        if (cfg.token_mask != nullptr) {
            first_target = speculative_masked_argmax(row_logits, token_domain, cfg, red_val, red_idx);
        }
        if (tid == 0) {
            int a = 0;
            while (a < extent && (a == 0 ? first_target : row_targets[a]) == row_drafts[a]) { ++a; }
            const int t_star = a == 0 ? first_target : row_targets[a];

            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a] = t_star;

            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = t_star;
            lengths[row] += produced;
            if (probe != nullptr) {
                for (int i = 0; i < extent; ++i) {
                    speculative_probe_write_column(probe, row, k, batch_stride, i, nullptr,
                                                   nullptr, 0, row_drafts[i], true);
                }
                speculative_probe_commit_columns(probe, row, batch_stride, extent, a);
                speculative_probe_write_row_summary(probe, row, k, batch_stride, extent, a);
            }
        }
        return;
    }

    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int n_support;
    __shared__ int a_sh;
    __shared__ int done_sh;
    __shared__ int tstar_sh;
    __shared__ int L_sh;

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape.
    if (sampler_multiblock_ok(token_domain, cols, partial_blocks, group_count)) { return; }

    if (tid == 0) {
        a_sh     = 0;
        done_sh  = 0;
        tstar_sh = 0;
        L_sh     = lengths[row];
    }
    __syncthreads();

    for (int i = 0; i <= extent; ++i) {
        // Column i is only reached when drafts[0..i-1] were all accepted, so the
        // round-local penalty overlay for this column is exactly those i drafts.
        SamplingConfig column_cfg = cfg;
        if (i != 0) { column_cfg.token_mask = nullptr; }
        const std::int64_t base = static_cast<std::int64_t>(i) * physical_rows;
        if (token_domain <= kSamplerTileItems) {
            sampling_build_truncated_small(row_logits, base, token_domain, column_cfg, red_val,
                                           red_idx, cand_val, cand_idx, prob, &n_support,
                                           row_drafts, i);
        } else {
            sampling_build_truncated_block_fast(row_logits, base, token_domain, column_cfg,
                                                merge_val, merge_idx, cand_val, cand_idx, prob,
                                                &n_support, row_drafts, i);
        }
        // The probe reads the same normalized `prob` support the acceptance test consumes, so the
        // collision probability it reports describes the distribution that was actually sampled.
        // It is evaluated before the accept decision and writes only the probe matrix, so the
        // decision and every sampled token stay bit-identical to the probe-free path.
        //
        // The finiteness scan is split into its own uniform region because speculative_round.h
        // requires -1 rather than a NaN or Inf for a non-finite logit column, and only this kernel
        // can tell that apart from a legitimately large collision probability. The all-finite case
        // is the shipped one and compiles down to one __syncthreads() saving nothing it uses.
        if (probe != nullptr) {
            int nonfinite = 0;
            for (int v = tid; v < token_domain; v += blockDim.x) {
                if (!isfinite(__bfloat162float(row_logits[base + v]))) {
                    nonfinite = 1;
                    break;
                }
            }
            // Every thread scans a strided slice, so the flag has to be reduced across the block:
            // reading this thread's own local reports the column finite whenever the non-finite
            // element happens to fall outside thread 0's slice.
            const int any_nonfinite = __syncthreads_or(nonfinite);
            if (tid == 0 && i < extent) {
                speculative_probe_write_column(probe, row, k, batch_stride, i, cand_idx, prob,
                                                n_support, row_drafts[i], any_nonfinite == 0);
            } else if (tid == 0) {
                speculative_probe_write_column(probe, row, k, batch_stride, i, cand_idx, prob,
                                                n_support, -1, any_nonfinite == 0);
            }
            __syncthreads();
        }
        if (tid == 0 && done_sh == 0) {
            const int L = L_sh;
            if (i < extent) {
                const int d = row_drafts[i];
                float pd    = 0.0f;
                for (int j = 0; j < n_support; ++j) {
                    if (cand_idx[j] == d) {
                        pd = prob[j];
                        break;
                    }
                }
                const float u =
                    sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                if (u < pd) {
                    a_sh = i + 1; // accept drafts[i], keep verifying
                } else {
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar_sh       = sampling_pick_from_support(cand_idx, prob, n_support, d, ur);
                    done_sh        = 1;
                }
            } else {
                // Every draft accepted: bonus token from the last verify column.
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar_sh = sampling_pick_from_support(cand_idx, prob, n_support, -1, u);
                done_sh  = 1;
            }
        }
        __syncthreads();
        if (done_sh) { break; }
    }

    if (tid == 0) {
        const int a     = a_sh;
        const int tstar = tstar_sh;
        const int L     = L_sh;

        for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
        for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
        row_tokens[a] = tstar;

        const int produced   = a + 1;
        licensed_counts[row] = produced;
        accepted[row]        = a;
        anchors[row]         = tstar;
        lengths[row]         = L + produced;
        if (probe != nullptr) {
            speculative_probe_commit_columns(probe, row, batch_stride, extent, a_sh);
            speculative_probe_write_row_summary(probe, row, k, batch_stride, extent, a_sh);
        }
        if (cfg.token_counts != nullptr) {
            for (int i = 0; i < produced; ++i) { atomicAdd(&cfg.token_counts[row_tokens[i]], 1); }
        }
    }
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, SamplingWorkspace workspace,
    std::size_t workspace_row_stride) {
    const int row     = static_cast<int>(blockIdx.z);
    const int col     = static_cast<int>(blockIdx.y);
    const int partial = static_cast<int>(blockIdx.x);
    int extent        = current_extents[row];
    extent            = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    SamplingConfig cfg = configs[row];
    if (col != 0) { cfg.token_mask = nullptr; }
    if (!(cfg.temperature > 0.0f) || token_domain <= kSamplerTileItems) { return; }
    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
    if (partial == 0 && threadIdx.x == 0) {
        workspace.group_done[col] = 0;
        if (col == 0) { *workspace.speculative_finalize_count = 0; }
    }

    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    unsigned long long keys[kSamplerItemsPerThread];

    const int cap                  = sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base        = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
    const std::int32_t* row_drafts = drafts + row * k;
    const int tile_start           = partial * kSamplerPartialTileItems;
    // Column col's penalty overlay is the first `col` drafts (see accept loop);
    // applying it before top-k selection lets it change the candidate set, not
    // just the post-truncation probabilities.
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg,
                                                    row_drafts, col);
            keys[item]    = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off               = sampling_partial_offset(workspace, col, partial, rank);
            workspace.partial_keys[off] = keys[item];
        }
    }
}

__launch_bounds__(kSamplerGroupBlock) __global__ void speculative_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const __nv_bfloat16* logits, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols,
    std::int32_t partial_blocks, std::int32_t group_count, SamplingWorkspace workspace,
    std::size_t workspace_row_stride, float* probe) {
    const int row   = static_cast<int>(blockIdx.z);
    const int group = static_cast<int>(blockIdx.x);
    const int col   = static_cast<int>(blockIdx.y);
    const int tid   = threadIdx.x;
    const int k     = cols - 1;
    // The probe's row axis is the batch width, which this launch carries on gridDim.z.
    const std::int32_t batch_stride = static_cast<std::int32_t>(gridDim.z);
    int extent      = current_extents[row];
    extent          = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    const __nv_bfloat16* row_logits =
        logits + static_cast<std::int64_t>(row) * cols * physical_rows;
    __shared__ float greedy_val[kSamplerGroupBlock];
    __shared__ int greedy_idx[kSamplerGroupBlock];
    if (token_domain <= kSamplerTileItems) { return; }

    if (!(cfg.temperature > 0.0f)) {
        int first_target = row_targets[0];
        if (col == 0 && group == 0 && cfg.token_mask != nullptr) {
            first_target = speculative_masked_argmax(
                row_logits, token_domain, cfg, greedy_val, greedy_idx);
        }
        if (tid == 0 && col == 0 && group == 0) {
            int a = 0;
            while (a < extent && (a == 0 ? first_target : row_targets[a]) == row_drafts[a]) { ++a; }
            const int t_star = a == 0 ? first_target : row_targets[a];
            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a]        = t_star;
            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = t_star;
            lengths[row] += produced;
            if (probe != nullptr) {
                for (int i = 0; i < extent; ++i) {
                    speculative_probe_write_column(probe, row, k, batch_stride, i, nullptr, nullptr,
                                                    0, row_drafts[i], true);
                }
                speculative_probe_commit_columns(probe, row, batch_stride, extent, a);
                speculative_probe_write_row_summary(probe, row, k, batch_stride, extent, a);
            }
        }
        return;
    }

    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);

    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int cap = sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes all caller-owned counters. CUDA
    // stream ordering makes those writes visible before this launch begins.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }
    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off =
                sampling_partial_offset(workspace, col, partial_blocks + group, rank);
            workspace.partial_keys[out_off] = keys[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last_group  = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last_group) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);

    // The probe reads the same normalized `prob` support the acceptance test consumes, so the
    // collision probability it reports describes the distribution that was actually sampled.
    // It is evaluated before the accept decision and writes only the probe matrix, so the
    // decision and every sampled token stay bit-identical to the probe-free path.
    if (probe != nullptr) {
        // Cooperative, for the same reason as the block route: a per-thread local would only ever
        // see that thread's own strided slice. Scanning the whole domain from one thread instead
        // would read every logit serially -- 248,077 loads per column per round at the packed
        // vocabulary. Each thread of the column's groups reaches this point -- `col` is
        // block-uniform and the early exits above are too -- so the barrier is safe.
        const __nv_bfloat16* col_logits =
            row_logits + static_cast<std::int64_t>(col) * physical_rows;
        int nonfinite = 0;
        for (int v = tid; v < token_domain; v += blockDim.x) {
            if (!isfinite(__bfloat162float(col_logits[v]))) {
                nonfinite = 1;
                break;
            }
        }
        const int any_nonfinite = __syncthreads_or(nonfinite);
        // Only the column's own last group holds the final support, which is orthogonal to the scan
        // above -- that one reads the raw logits rather than the truncated support.
        if (is_last_group == 1 && tid == 0) {
            speculative_probe_write_column(probe, row, k, batch_stride, col, cand_idx, prob,
                                            n_support, col < extent ? row_drafts[col] : -1,
                                            any_nonfinite == 0);
        }
    }

    if (tid == 0) {
        workspace.dist_support[col] = n_support;
        for (int j = 0; j < n_support; ++j) {
            const int off            = sampling_dist_offset(col, j);
            workspace.dist_idx[off]  = cand_idx[j];
            workspace.dist_prob[off] = prob[j];
        }
        workspace.group_done[col] = 0;
        __threadfence();
        const int done_cols = atomicAdd(workspace.speculative_finalize_count, 1) + 1;
        if (done_cols == extent + 1) {
            const int L = lengths[row];
            int a       = 0;
            int tstar   = 0;
            for (int i = 0; i <= extent; ++i) {
                const int n            = workspace.dist_support[i];
                const int* dist_idx    = workspace.dist_idx + sampling_dist_offset(i, 0);
                const float* dist_prob = workspace.dist_prob + sampling_dist_offset(i, 0);
                if (i < extent) {
                    const int d = row_drafts[i];
                    float pd    = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        if (dist_idx[j] == d) {
                            pd = dist_prob[j];
                            break;
                        }
                    }
                    const float u =
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                    if (u < pd) {
                        a = i + 1;
                        continue;
                    }
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar          = sampling_pick_from_support(dist_idx, dist_prob, n, d, ur);
                    break;
                }
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
            }
            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a]        = tstar;
            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = tstar;
            lengths[row]         = L + produced;
            if (probe != nullptr) {
                speculative_probe_commit_columns(probe, row, batch_stride, extent, a);
                speculative_probe_write_row_summary(probe, row, k, batch_stride, extent, a);
            }
            if (cfg.token_counts != nullptr) {
                for (int i = 0; i < produced; ++i) {
                    atomicAdd(&cfg.token_counts[row_tokens[i]], 1);
                }
            }
            *workspace.speculative_finalize_count = 0;
        }
    }
}

__global__ void speculative_select_accepted_hidden_kernel(const __nv_bfloat16* hidden,
                                                          const std::int32_t* selectors,
                                                          __nv_bfloat16* out, std::int32_t rows,
                                                          std::int32_t cols) {
    const int batch = static_cast<int>(blockIdx.y);
    const int row   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = selectors[batch];
    if (col < 0 || col >= cols) { return; }
    out[static_cast<std::int64_t>(batch) * rows + row] =
        hidden[(static_cast<std::int64_t>(batch) * cols + col) * rows + row];
}

__global__ void proposal_remap_token_ids_kernel(std::int32_t* proposal_tokens,
                                                std::int32_t proposal_count,
                                                const std::int32_t* id_map, std::int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= proposal_count) { return; }
    const int idx = proposal_tokens[i];
    if (idx >= 0 && idx < n) { proposal_tokens[i] = id_map[idx]; }
}

} // namespace ninfer::ops

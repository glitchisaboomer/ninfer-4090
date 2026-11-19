#pragma once

// Stage F1: device-backed one-layer MTP stem.
//
// This module owns no memory. It composes the official SenseNova/Qwen3.5 MTP
// stem over caller-owned tensors and the Program's stable state image:
//
//   e = RMSNorm(embed(next_token), mtp.embedding_norm)
//   h = RMSNorm(target_hidden,    mtp.hidden_norm)
//   x = fc([e;h])                                  W8 [4096,8192]
//   x = RMSNorm(x, mtp.input_norm)
//   one 16/4 full-attention layer with its own KV domain
//   x = residual + attention; post norm; SwiGLU; down; residual
//   d = RMSNorm(x, mtp.final_norm)
//   proposal = argmax(draft_head * d) remapped through draft_head_token_ids
//
// The target hidden must be the hidden state paired with the proposed token
// (`h_t` with `x_(t+1)`), which is the official MTP input contract. The target
// model, not this drafter, remains the only output authority.

#include "targets/sensenova_loop_9b/impl/config.h"
#include "targets/sensenova_loop_9b/impl/loop_plan.h"
#include "targets/sensenova_loop_9b/impl/runtime/layer_execution.h"
#include "targets/sensenova_loop_9b/impl/runtime/state_images.h"
#include "targets/sensenova_loop_9b/impl/runtime/text_execution.h"

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::sensenova_loop_9b::detail {

struct MtpStemBindings {
    Tensor kv_block_table;
    Tensor kv_table_rows;
};

// Caller transient capacity for one MTP proposal step of `tokens` columns. The
// step is independent of the target pass and uses the same W8 A16 route as the
// text model, so all scratch is returned to the caller's arena before return.
[[nodiscard]] std::size_t mtp_stem_workspace_capacity_bytes(std::uint32_t tokens,
                                                            std::uint32_t visible_keys);

// Appends one token's K/V to the MTP KV domain and writes the shortlisted draft
// id for every supplied column.
//
// `token_ids`, `target_hidden`, and `positions` are contiguous device tensors
// [T], [4096,T], and [T]. `target_hidden` is the hidden paired with token_ids.
// `draft_ids` is I32 [T]; `draft_logits` is optional BF16 [131072,T] shortlist
// logits. `mtp_state_base` is the stable base of one lane's complete state image.
void mtp_propose_tokens(const RuntimeModelView& model, cudaStream_t stream,
                        std::byte* mtp_state_base, std::uint32_t kv_capacity,
                        WorkspaceArena& workspace, const MtpStemBindings& bindings,
                        const Tensor& token_ids, const Tensor& target_hidden,
                        const Tensor& positions, std::uint32_t visible_keys,
                        Tensor& draft_ids, Tensor* draft_logits = nullptr);

} // namespace ninfer::targets::sensenova_loop_9b::detail

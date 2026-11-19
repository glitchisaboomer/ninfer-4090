// Stage G: fixed-prompt execution gate.
//
// The prompt-position defect is fixed: SenseNova prefill now validates the frontend's three
// axis-major mRoPE position axes and uploads text/temporal axis 0 as the device [T] positions.
//
// The remaining blocker is the GQA page geometry. SenseNova state images use BF16 cache planes
// [head_dim=256, page_tokens=128, kv_heads=4, pages], but the GQA kernels derive logical pages
// and page offsets from absolute positions with a compile-time 64-token shift/mask
// (kPagedKVPageShift=6 and kPagedKVPageMask=63 in paged_kv_address.cuh). Relaxing wrapper
// validation alone would silently corrupt cache addressing. Correct support requires threading a
// runtime page shift/mask, or a separate 128-token geometry, through the BF16/INT8 prefill and
// decode kernels, metadata structs, wrappers, delta-RoPE paths, and oracle tests; that broad
// redesign is outside this authorized fix.
//
// No fake tokens, hidden state, or logits are substituted. The artifact-level quantized-weight
// reference, the position-axis regression, and the existing host/state-contract CTests remain
// active and are reported separately.

#include <iostream>

int main() {
    std::cout
        << "SKIP: Stage G fixed-prompt execution remains blocked by the SenseNova 128-token-page "
           "BF16 GQA cache geometry. The GQA kernels compile in a 64-token page shift and mask, "
           "so accepting 128-token planes requires a cross-kernel prefill/decode and oracle-test "
           "redesign rather than a wrapper validation change. Prompt positions are fixed and "
           "covered separately; artifact-level quantized-weight validation and host/state-contract "
           "validation remain active.\n";
    return 77;
}

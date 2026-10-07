// ninfer::ops - logprob_of_choice wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/logprob_of_choice.h"

#include "ops/launcher/logprob_of_choice.h" // detail::logprob_of_choice_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

// The chosen ids are read on the device, so an out-of-domain value cannot be caught here. The
// contract states the caller guarantees it; this only rejects a structurally impossible request.
void require_rank2_logits(const Tensor& logits, const char* op) {
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": logits must be rank-2 [physical_rows,T]");
    }
}

void require_rank1(const Tensor& tensor, std::int32_t expected, const char* op, const char* name) {
    if (tensor.ne[0] != expected || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be rank-1 [" +
                                    std::to_string(expected) + "]");
    }
}

} // namespace

void logprob_of_choice(const Tensor& logits, const Tensor& chosen, std::int32_t valid_rows,
                       Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "logprob_of_choice";

    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": logits must be BF16");
    }
    if (chosen.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": chosen must be I32");
    }
    if (out.dtype != DType::FP32) {
        throw std::invalid_argument(std::string(op) + ": out must be FP32");
    }
    if (logits.ne[1] <= 0) {
        throw std::invalid_argument(std::string(op) + ": T must be positive");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument(std::string(op) + ": physical rows must be positive");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument(std::string(op) +
                                    ": valid_rows must be in [1, logits.ne[0]]");
    }
    require_rank2_logits(logits, op);
    require_rank1(chosen, logits.ne[1], op, "chosen");
    require_rank1(out, logits.ne[1], op, "out");

    if (!logits.is_contiguous() || !chosen.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": all tensors must be contiguous");
    }
    if (logits.data == nullptr || chosen.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": all tensors must be non-null");
    }

    detail::logprob_of_choice_launch(logits, chosen, valid_rows, out, stream);
}

} // namespace ninfer::ops
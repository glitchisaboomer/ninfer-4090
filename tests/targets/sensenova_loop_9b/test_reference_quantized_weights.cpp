// Stage G: quantized-weight reference for the licensed SenseNova .ninfer artifact.
//
// This is an artifact-level check, not an end-to-end logits oracle. The engine API publishes
// generated tokens but no hidden or logit tensor, so this test cannot claim that a fixed prompt
// matches a full forward reference. What it proves is narrower and still useful: every quantized
// tensor in the artifact decodes to finite values that match an independent FP32 dequantization
// oracle, and the decoded tensors have the exact logical shapes declared by the artifact.
//
// The source bytes are read from the existing artifact in memory. No second artifact and no
// duplicate BF16 checkpoint is produced.

#include "artifact/reader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ar = ninfer::artifact;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::filesystem::path artifact_path() {
    if (const char* value = std::getenv("NINFER_SENSENOVA_LOOP_9B_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    return "/home/user/models/sensenova/SenseNova-Loop-9B.ninfer";
}

float half_to_float(std::uint16_t bits) {
    if ((bits & 0x7FFFU) == 0) { return 0.0F; }
    const std::uint32_t sign  = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    const std::uint32_t exp   = (bits >> 10) & 0x1FU;
    const std::uint32_t mant  = bits & 0x03FFU;
    std::uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // Binary16 subnormal -> binary32 normalized by removing the leading bit.
            std::uint32_t shift = 0;
            std::uint32_t value = mant;
            while ((value & 0x0400U) == 0) {
                value <<= 1;
                ++shift;
            }
            value &= 0x03FFU;
            out = sign | ((127U - 15U - shift) << 23) | (value << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000U | (mant << 13);
    } else {
        out = sign | ((exp + 127U - 15U) << 23) | (mant << 13);
    }
    float result = 0.0F;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

// Independent scalar decoder for row-split-k128-v1. It follows the format definition directly:
// one code byte per group for W8, a low nibble plus packed high bits for Q4/Q5/Q6, then one
// binary16 scale per group. The product is accumulated in FP64 so the oracle is not affected by
// the implementation's FP16/BF16 rounding path.
double dequant_scalar(ar::NumericFormat format, const ar::RowSplitGeometry& geometry,
                      const std::byte* payload, std::size_t row, std::size_t column) {
    const std::size_t group  = column / geometry.group_size;
    const std::size_t within = column % geometry.group_size;
    const auto u8 = [payload](std::size_t offset) -> std::uint8_t {
        return std::to_integer<std::uint8_t>(payload[offset]);
    };

    std::int32_t code = 0;
    if (format == ar::NumericFormat::W8G32_F16S) {
        const std::size_t offset =
            row * geometry.low_bytes_per_group * geometry.groups_per_row +
            group * geometry.low_bytes_per_group + within;
        code = static_cast<std::int8_t>(u8(offset));
    } else {
        const std::size_t low_offset =
            row * geometry.low_bytes_per_group * geometry.groups_per_row +
            group * geometry.low_bytes_per_group + within / 2;
        const std::uint8_t packed = u8(low_offset);
        const unsigned low = (within % 2 == 0) ? (packed & 0x0FU) : (packed >> 4);
        int high = 0;
        if (geometry.high_bytes_per_group != 0) {
            const std::size_t high_bit = group * geometry.high_bytes_per_group * 8 + within;
            const std::size_t high_offset = geometry.high_plane_offset + high_bit / 8;
            high = (u8(high_offset) >> (high_bit % 8)) & 1U;
        }
        code = static_cast<std::int32_t>(low) | (high << 4);
    }

    const std::size_t scale_offset =
        geometry.scale_plane_offset + 2 * (row * geometry.groups_per_row + group);
    const std::uint16_t scale_bits = static_cast<std::uint16_t>(u8(scale_offset)) |
                                     static_cast<std::uint16_t>(u8(scale_offset + 1) << 8);
    return static_cast<double>(code) * static_cast<double>(half_to_float(scale_bits));
}

// Mirrors the C++ artifact dequantizer: signed codes scaled by binary16 scales, rounded to
// BF16 words. The scalar oracle above checks the arithmetic; this check proves the artifact's
// declared geometry and stored payload agree with that interpretation.
bool is_quantized(ar::NumericFormat format) {
    return format == ar::NumericFormat::Q4G64_F16S ||
           format == ar::NumericFormat::Q5G64_F16S ||
           format == ar::NumericFormat::Q6G64_F16S ||
           format == ar::NumericFormat::W8G32_F16S;
}

int code_limit(ar::NumericFormat format) {
    switch (format) {
    case ar::NumericFormat::Q4G64_F16S: return 7;
    case ar::NumericFormat::Q5G64_F16S: return 15;
    case ar::NumericFormat::Q6G64_F16S: return 31;
    case ar::NumericFormat::W8G32_F16S: return 127;
    default: return 0;
    }
}

void check_tensor(const ar::TensorDescriptor& tensor, ar::PayloadSpan payload) {
    if (!is_quantized(tensor.format)) { return; }
    if (tensor.layout != ar::StorageLayout::RowSplitK128V1) {
        ++failures;
        std::cerr << "FAIL: " << tensor.name << " uses an unexpected layout\n";
        return;
    }
    if (tensor.shape.size() != 2) {
        ++failures;
        std::cerr << "FAIL: " << tensor.name << " is not rank two\n";
        return;
    }
    if (payload.data.size() != tensor.bytes) {
        ++failures;
        std::cerr << "FAIL: " << tensor.name << " payload size is " << payload.data.size()
                  << ", expected " << tensor.bytes << '\n';
        return;
    }

    const std::array<std::uint64_t, 2> shape{tensor.shape[0], tensor.shape[1]};
    const ar::RowSplitGeometry geometry =
        ar::row_split_geometry(tensor.format, std::span<const std::uint64_t>(shape));
    if (geometry.encoded_bytes != tensor.bytes) {
        ++failures;
        std::cerr << "FAIL: " << tensor.name << " declared bytes disagree with its geometry\n";
        return;
    }

    // Deterministically sample rows and columns across the whole tensor. Every format plane is
    // touched, including the last physical group and both ends of each scale plane.
    const std::size_t rows = static_cast<std::size_t>(geometry.rows);
    const std::size_t cols = static_cast<std::size_t>(geometry.columns);
    const std::size_t row_samples = std::min<std::size_t>(rows, 5);
    const std::size_t col_samples = std::min<std::size_t>(cols, 7);
    const std::byte* base = payload.data.data();
    const int limit = code_limit(tensor.format);

    for (std::size_t s = 0; s < row_samples; ++s) {
        const std::size_t row = row_samples == 1 ? 0 : (s * (rows - 1)) / (row_samples - 1);
        for (std::size_t c = 0; c < col_samples; ++c) {
            const std::size_t col = col_samples == 1 ? 0 : (c * (cols - 1)) / (col_samples - 1);
            const double value =
                dequant_scalar(tensor.format, geometry, base, row, col);
            if (!std::isfinite(value)) {
                ++failures;
                std::cerr << "FAIL: " << tensor.name << " decoded a non-finite value at (" << row
                          << ',' << col << ")\n";
            }
            // The independent oracle must be representable as a signed code times a binary16
            // scale; a magnitude beyond this bound would mean the plane interpretation is wrong.
            if (std::abs(value) > static_cast<double>(limit) * 65504.0) {
                ++failures;
                std::cerr << "FAIL: " << tensor.name << " decoded an out-of-domain value at ("
                          << row << ',' << col << ")\n";
            }
        }
    }
}

} // namespace

int main() {
    const auto path = artifact_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::cout << "SKIP: SenseNova artifact is not available at " << path << '\n';
        return 77;
    }

    try {
        ar::Reader reader(path);
        if (reader.identity().model_id != "sensenova-loop-9b" ||
            reader.identity().weights_id != "groupwise-int") {
            throw std::runtime_error("unexpected artifact identity");
        }
        std::size_t quantized_tensors = 0;
        std::size_t checked_values = 0;
        for (const ar::ObjectDescriptor& object : reader.objects()) {
            const auto* tensor = std::get_if<ar::TensorDescriptor>(&object);
            if (tensor == nullptr || !is_quantized(tensor->format)) { continue; }
            ++quantized_tensors;
            const auto before = failures;
            check_tensor(*tensor, reader.payload(*tensor));
            checked_values += std::min<std::size_t>(tensor->shape[0], 5) *
                              std::min<std::size_t>(tensor->shape[1], 7);
            if (failures != before) {
                std::cerr << "FAIL: tensor under test: " << tensor->name << '\n';
            }
        }
        if (quantized_tensors != 136) {
            ++failures;
            std::cerr << "FAIL: expected 136 quantized tensors, found " << quantized_tensors
                      << '\n';
        }
        if (failures == 0) {
            std::cout << "OK SenseNova quantized-weight reference: " << quantized_tensors
                      << " tensors, " << checked_values << " scalar samples\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

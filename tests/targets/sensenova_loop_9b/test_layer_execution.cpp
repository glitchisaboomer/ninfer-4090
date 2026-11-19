// Stage E3: one-visit layer execution for the fixed 56-visit SenseNova-Loop-9B plan.
//
// Host tests (always run): the visit->profile mapping, mHC boundary classification and input/
// output widths, `execute_layer_visit` validation failures, workspace capacity sanity, and the
// GDN state-slot views.
//
// Device tests (skip with exit 77 when no CUDA device is present): one GDN mHC-enter visit, a
// revisit pair on the same physical full-attention layer (two KV domains, two mHC slots), one
// full-attention A3 cached visit, and one mHC-leave visit. Each asserts that execution mutates
// exactly the visit state it addresses -- the GDN recurrent/conv slot, the mHC slot, and the
// bound KV domain -- and nothing else.

#include "targets/sensenova_loop_9b/impl/runtime/layer_execution.h"
#include "targets/sensenova_loop_9b/impl/runtime/state_images.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/direct_bf16_weight.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace ninfer;
namespace le = ninfer::targets::sensenova_loop_9b::detail;
namespace qt = ninfer::test;

namespace {

int failures = 0;

void expect(bool ok, const char* label) {
    if (!ok) {
        std::cerr << "FAIL: " << label << "\n";
        ++failures;
    }
}

template <class F>
bool throws_layer_visit_error(F&& f) {
    try {
        f();
    } catch (const le::LayerVisitError&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

std::vector<std::uint8_t> read_bytes(const void* device, std::size_t bytes) {
    return qt::from_device<std::uint8_t>(device, bytes);
}

// ---------------- host: plan mapping ----------------------------------------------

void test_plan_mapping() {
    std::set<std::size_t> slots;
    std::set<std::size_t> domains;
    std::size_t full = 0, gdn = 0, mhc = 0;
    for (std::uint32_t v = 0; v < le::LoopPlan::kVisits; ++v) {
        const le::VisitExecutionProfile p = le::plan_visit(v);
        expect(p.state_slot == v, "state_slot equals the visit index");
        expect(p.physical_layer == le::LoopPlan::physical_layer(v), "physical layer mapping");
        expect(p.pass == le::LoopPlan::pass(v), "pass mapping");
        slots.insert(p.state_slot);
        if (p.kind == le::VisitKind::FullAttention) {
            ++full;
            domains.insert(le::full_attention_domain_index(v));
        } else {
            ++gdn;
        }
        if (p.is_mhc) {
            ++mhc;
        }
    }
    expect(slots.size() == le::LoopPlan::kVisits, "every visit owns a distinct state slot");
    expect(full == le::LoopPlan::full_attention_visits(), "full-attention visit count");
    expect(gdn == le::LoopPlan::gdn_visits(), "GDN visit count");
    expect(mhc == le::LoopPlan::mhc_visits(), "mHC visit count");
    expect(domains.size() == le::StateLayout::full_attention_state_slots,
           "full-attention KV domains are compact");
    for (std::size_t d = 0; d < le::StateLayout::full_attention_state_slots; ++d) {
        expect(domains.count(d) == 1, "every full-attention KV domain is used exactly once");
    }
    // The MTP domain is deliberately last and never aliased with a text visit.
    expect(le::StateImageLayout::kKvDomains == le::StateLayout::full_attention_state_slots +
                                                   le::StateLayout::mtp_kv_domains,
           "MTP KV domain is the last domain");
    // Revisit: physical layer 7 runs at visit 7 (pass 0) and visit 31 (pass 1); the two visits
    // must hit distinct state identities even though the physical weights are shared.
    expect(le::plan_visit(7).state_slot != le::plan_visit(31).state_slot,
           "revisit uses a distinct state slot");
    expect(le::full_attention_domain_index(7) != le::full_attention_domain_index(31),
           "revisit uses a distinct KV domain");
}

void test_mhc_boundaries() {
    using B = le::MhcBoundary;
    expect(le::plan_visit(4).mhc_boundary == B::Enter, "visit 4 (layer 4 pass 0) enters mHC");
    expect(le::plan_visit(28).mhc_boundary == B::Enter, "visit 28 (layer 4 pass 1) enters mHC");
    expect(le::plan_visit(27).mhc_boundary == B::Leave, "visit 27 (layer 27 pass 0) leaves mHC");
    expect(le::plan_visit(51).mhc_boundary == B::Leave, "visit 51 (layer 27 pass 1) leaves mHC");
    expect(le::plan_visit(5).mhc_boundary == B::None, "interior visit has no boundary");
    expect(le::plan_visit(3).mhc_boundary == B::None, "layer 3 is outside the mHC range");
    expect(le::plan_visit(31).mhc_boundary == B::None, "layer 31 is outside the mHC range");

    expect(le::plan_visit(4).input_width == 4096, "enter visit reads a scalar hidden");
    expect(le::plan_visit(4).output_width == 8192, "enter visit publishes two flattened streams");
    expect(le::plan_visit(5).input_width == 8192, "interior visit reads flattened streams");
    expect(le::plan_visit(5).output_width == 8192, "interior visit publishes flattened streams");
    expect(le::plan_visit(27).input_width == 8192, "leave visit reads flattened streams");
    expect(le::plan_visit(27).output_width == 4096, "leave visit reduces to a scalar hidden");
    expect(le::plan_visit(51).input_width == 8192, "pass-1 leave visit reads flattened streams");
    expect(le::plan_visit(51).output_width == 4096, "pass-1 leave visit reduces to a scalar");
    expect(le::plan_visit(3).input_width == 4096 && le::plan_visit(3).output_width == 4096,
           "non-mHC visit preserves the scalar width");
}

// ---------------- host: validation ------------------------------------------------

// Profile-valid dummy weights over host storage: shape/qtype/group checks of
// `validate_request` pass without a device; execution is never attempted on these.
struct DummyStorage {
    std::vector<std::byte> bytes;

    std::byte* take(std::size_t n) {
        std::byte* p = bytes.data() + bytes.size();
        bytes.resize(bytes.size() + n);
        return p;
    }
    [[nodiscard]] Tensor bf16(int rows) { return Tensor(take(std::size_t(rows) * 2), DType::BF16, {rows}); }
    [[nodiscard]] Tensor bf16_2d(int rows, int cols) {
        return Tensor(take(std::size_t(rows) * cols * 2), DType::BF16, {rows, cols});
    }
    [[nodiscard]] Tensor fp32(int rows) { return Tensor(take(std::size_t(rows) * 4), DType::FP32, {rows}); }
    [[nodiscard]] Tensor fp32_2d(int rows, int cols) {
        return Tensor(take(std::size_t(rows) * cols * 4), DType::FP32, {rows, cols});
    }
    [[nodiscard]] Tensor i32(int rows) { return Tensor(take(std::size_t(rows) * 4), DType::I32, {rows}); }
    [[nodiscard]] Weight w8(int n, int k) {
        Weight w{};
        w.payload     = take(std::size_t(n) * k);
        w.qtype       = QType::W8G32_F16S;
        w.layout      = QuantLayout::RowSplit;
        w.ndim        = 2;
        w.n           = n;
        w.k           = k;
        w.group_size  = 32;
        return w;
    }
    [[nodiscard]] Weight bf16w(int n, int k) {
        Weight w{};
        w.payload   = take(std::size_t(n) * k * 2);
        w.qtype     = QType::BF16_CTRL;
        w.layout    = QuantLayout::Contiguous;
        w.ndim      = 2;
        w.n         = n;
        w.k         = k;
        return w;
    }
};

struct DummyLayer {
    DummyStorage storage;
    le::TextLayerWeights weights;

    explicit DummyLayer(bool full_attention, bool with_mhc) {
        const int H = le::TextConfig::hidden;
        weights.input_norm           = storage.bf16(H);
        weights.post_attention_norm  = storage.bf16(H);
        weights.mlp_gate_up          = storage.w8(2 * 12288, H);
        weights.mlp_down             = storage.w8(H, 12288);
        weights.is_full_attention    = full_attention;
        weights.has_mhc              = with_mhc;
        if (full_attention) {
            weights.attention.query_key_gate_value = storage.w8(10240, H);
            weights.attention.output               = storage.w8(H, H);
            weights.attention.query_norm           = storage.bf16(256);
            weights.attention.key_norm             = storage.bf16(256);
        } else {
            weights.gdn.query_key_value_z = storage.w8(12288, H);
            weights.gdn.a_b_projection    = storage.bf16w(64, H);
            weights.gdn.output            = storage.w8(H, H);
            weights.gdn.a_log             = storage.fp32(32);
            weights.gdn.dt_bias           = storage.fp32(32);
            weights.gdn.norm              = storage.bf16(128);
        }
    }
};

void test_validation() {
    expect(throws_layer_visit_error([&] { le::plan_visit(56); }), "visit 56 is out of plan");
    expect(throws_layer_visit_error([&] { le::full_attention_domain_index(4); }),
           "domain index of a GDN visit throws");
    expect(throws_layer_visit_error([&] { le::layer_visit_workspace_capacity_bytes(4, 4, 2); }),
           "min > max token interval throws");
    expect(throws_layer_visit_error([&] { le::layer_visit_workspace_capacity_bytes(4, 0, 1); }),
           "zero min token interval throws");
    expect(throws_layer_visit_error([&] { le::gdn_state_views(nullptr); }), "null GDN slot throws");
    expect(le::layer_visit_workspace_capacity_bytes(4, 1, 1) > 0, "GDN capacity is positive");
    expect(le::layer_visit_workspace_capacity_bytes(7, 1, 1) > 0,
           "full-attention capacity is positive");
    expect(le::layer_visit_workspace_capacity_bytes(7, 1, 64) >=
               le::layer_visit_workspace_capacity_bytes(7, 1, 1),
           "capacity is monotone in the token interval");

    {
        // Full-attention visit with no GQA form bound: throws before any device access.
        DummyStorage s;
        DummyLayer layer(true, false);
        const Tensor in = s.bf16(4096);
        Tensor out = s.bf16(4096);
        const Tensor pos = s.i32(1);
        const Tensor kvk = s.bf16(1024);
        const Tensor kvv = s.bf16(1024);
        const Tensor rows = s.i32(1);
        le::LayerVisitRequest req;
        req.visit      = 3;
        req.hidden_in  = &in;
        req.hidden_out = &out;
        req.weights    = &layer.weights;
        req.positions  = &pos;
        req.stream     = reinterpret_cast<cudaStream_t>(0x1);  // never reached on host
        req.workspace  = reinterpret_cast<WorkspaceArena*>(0x1);
        req.attention_envelope = {1, 1024};
        expect(throws_layer_visit_error([&] { le::execute_layer_visit(req); }),
               "full-attention visit without a GQA form throws");

        // Append k/v but no bound batch view: throws.
        req.attention_k           = &kvk;
        req.attention_v           = &kvv;
        req.attention_kv_table_rows = &rows;
        expect(throws_layer_visit_error([&] { le::execute_layer_visit(req); }),
               "append without a bound batch KV view throws");

        // Both GQA forms bound: throws.
        const Tensor ck = s.bf16(256 * 64 * 4 * 2);
        const Tensor cv = s.bf16(256 * 64 * 4 * 2);
        const Tensor ct = s.i32(2);
        req.attention_kv_cached.k_pages     = ck;
        req.attention_kv_cached.v_pages     = cv;
        req.attention_kv_cached.block_table = ct;
        req.attention_kv_cached.num_kv_heads = 4;
        req.attention_kv_cached.head_dim     = 256;
        expect(throws_layer_visit_error([&] { le::execute_layer_visit(req); }),
               "binding both GQA forms throws");
    }

    {
        // GDN visit without the visit state slot: throws.
        DummyStorage s;
        DummyLayer layer(false, false);
        const Tensor in = s.bf16(4096);
        Tensor out = s.bf16(4096);
        const Tensor conv = s.bf16_2d(8192, 4);
        le::LayerVisitRequest req;
        req.visit      = 1;
        req.hidden_in  = &in;
        req.hidden_out = &out;
        req.weights    = &layer.weights;
        req.gdn_state_base         = nullptr;
        req.gdn_convolution_weight = &conv;
        req.stream     = reinterpret_cast<cudaStream_t>(0x1);
        req.workspace  = reinterpret_cast<WorkspaceArena*>(0x1);
        expect(throws_layer_visit_error([&] { le::execute_layer_visit(req); }),
               "GDN visit without a state slot throws");

        // GDN visit whose convolution weight is in the stored [4,C] orientation: throws.
        const Tensor conv_stored = s.bf16_2d(4, 8192);
        req.gdn_state_base         = s.take(4);
        req.gdn_convolution_weight = &conv_stored;
        expect(throws_layer_visit_error([&] { le::execute_layer_visit(req); }),
               "stored-form convolution weight throws");
    }

    {
        // mHC visit without the mHC state slot: throws.
        DummyStorage s;
        DummyLayer layer(false, true);
        const Tensor in4 = s.bf16(4096);   // enter visit reads a scalar hidden
        Tensor out4 = s.bf16(8192);
        const Tensor conv = s.bf16_2d(8192, 4);
        const Tensor mw0 = s.fp32_2d(8192, 8);
        const Tensor mb0 = s.fp32(8);
        const Tensor mw1 = s.fp32_2d(8192, 8);
        const Tensor mb1 = s.fp32(8);
        le::LayerVisitRequest req;
        req.visit      = 4;
        req.hidden_in  = &in4;
        req.hidden_out = &out4;
        req.weights    = &layer.weights;
        req.gdn_state_base         = s.take(4);
        req.gdn_convolution_weight = &conv;
        req.mhc_attn_weight = &mw0;
        req.mhc_attn_bias   = &mb0;
        req.mhc_ffn_weight  = &mw1;
        req.mhc_ffn_bias    = &mb1;
        req.mhc_state_base  = nullptr;
        req.stream          = reinterpret_cast<cudaStream_t>(0x1);
        req.workspace       = reinterpret_cast<WorkspaceArena*>(0x1);
        expect(throws_layer_visit_error([&] { le::execute_layer_visit(req); }),
               "mHC visit without a state slot throws");

        // Width mismatch at the enter boundary: [8192] into a [4096] visit throws.
        const Tensor in8 = s.bf16(8192);
        req.hidden_in = &in8;
        expect(throws_layer_visit_error([&] { le::execute_layer_visit(req); }),
               "enter visit with a stream-width hidden throws");

        // mHC state on a non-mHC visit: membership mismatch throws.
        DummyLayer plain(false, false);
        const Tensor in1 = s.bf16(4096);
        Tensor out1 = s.bf16(4096);
        const Tensor conv2 = s.bf16_2d(8192, 4);
        le::LayerVisitRequest plain_req;
        plain_req.visit      = 1;
        plain_req.hidden_in  = &in1;
        plain_req.hidden_out = &out1;
        plain_req.weights    = &plain.weights;
        plain_req.gdn_state_base         = s.take(4);
        plain_req.gdn_convolution_weight = &conv2;
        plain_req.mhc_state_base         = s.take(4);
        plain_req.stream                 = reinterpret_cast<cudaStream_t>(0x1);
        plain_req.workspace              = reinterpret_cast<WorkspaceArena*>(0x1);
        expect(throws_layer_visit_error([&] { le::execute_layer_visit(plain_req); }),
               "mHC state on a non-mHC visit throws");
    }
}

void test_gdn_state_views() {
    const std::size_t bytes = le::StateLayout::gdn_state_bytes_per_slot();
    std::vector<char> slot(bytes, 0);
    const le::GdnVisitStateViews views = le::gdn_state_views(reinterpret_cast<const std::byte*>(slot.data()));
    const std::size_t recurrent_bytes = le::StateLayout::gdn_recurrent_bytes_per_slot();
    expect(views.recurrent.dtype == DType::FP32, "recurrent state is FP32");
    expect(views.recurrent.ne[0] == static_cast<int>(le::StateLayout::gdn_qk_head_dim) &&
               views.recurrent.ne[1] == static_cast<int>(le::StateLayout::gdn_qk_head_dim) &&
               views.recurrent.ne[2] == static_cast<int>(le::StateLayout::gdn_value_heads),
           "recurrent state is [128,128,32]");
    expect(views.conv.dtype == DType::BF16, "conv state is BF16");
    expect(views.conv.ne[0] == static_cast<int>(le::StateLayout::gdn_conv_channels) &&
               views.conv.ne[1] == static_cast<int>(le::StateLayout::gdn_conv_state_width),
           "conv state is [8192,3]");
    expect(views.conv.data == slot.data() + recurrent_bytes, "conv state follows the recurrent state");
    expect(recurrent_bytes + le::StateLayout::gdn_conv_bytes_per_slot() == bytes,
           "slot byte accounting covers both views");
}

// ---------------- device: one-visit execution ------------------------------------

struct DeviceFixture {
    std::vector<DeviceBuffer> buffers;

    Weight w8(const qt::quantized_weight::PackedWeight& host, std::int32_t n, std::int32_t k) {
        DeviceBuffer buf(host.payload.size());
        buf.copy_from_host(host.payload.data(), host.payload.size());
        buffers.push_back(std::move(buf));
        Weight w = host.device_weight(buffers.back().p);
        w.n = n;
        w.k = k;
        return w;
    }
    Weight bf16_weight(const qt::direct_bf16_weight::HostWeight& host) {
        DeviceBuffer buf(host.bits.size() * 2);
        buf.copy_from_host(host.bits.data(), host.bits.size() * 2);
        buffers.push_back(std::move(buf));
        return host.device_weight(buffers.back().p);
    }
    Tensor tensor_bf16(const std::vector<float>& values, std::initializer_list<int> shape) {
        std::vector<std::uint16_t> bits(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            bits[i] = qt::f32_to_bf16(values[i]);
        }
        DeviceBuffer buf = qt::to_device(bits);
        buffers.push_back(std::move(buf));
        return Tensor(buffers.back().p, DType::BF16, shape);
    }
    Tensor tensor_f32(const std::vector<float>& values, std::initializer_list<int> shape) {
        DeviceBuffer buf = qt::to_device(values);
        buffers.push_back(std::move(buf));
        return Tensor(buffers.back().p, DType::FP32, shape);
    }
    Tensor tensor_i32(const std::vector<int>& values, std::initializer_list<int> shape) {
        DeviceBuffer buf = qt::to_device(values);
        buffers.push_back(std::move(buf));
        return Tensor(buffers.back().p, DType::I32, shape);
    }

    le::TextLayerWeights make_full_attention_weights(bool with_mhc) {
        const int H = le::TextConfig::hidden;
        le::TextLayerWeights w;
        std::vector<float> nrm(H, 1.0F);
        std::vector<float> qn(256, 1.0F);
        w.input_norm           = tensor_bf16(nrm, {H});
        w.post_attention_norm  = tensor_bf16(nrm, {H});
        w.mlp_gate_up          = w8(qt::linear::make_w8g32_f16s_weight(24576, H, 13), 24576, H);
        w.mlp_down             = w8(qt::linear::make_w8g32_f16s_weight(H, 12288, 14), H, 12288);
        w.attention.query_key_gate_value =
            w8(qt::linear::make_w8g32_f16s_weight(10240, H, 15), 10240, H);
        w.attention.output = w8(qt::linear::make_w8g32_f16s_weight(H, H, 16), H, H);
        w.attention.query_norm = tensor_bf16(qn, {256});
        w.attention.key_norm   = tensor_bf16(qn, {256});
        w.is_full_attention    = true;
        w.has_mhc              = with_mhc;
        return w;
    }

    le::TextLayerWeights make_gdn_weights(bool with_mhc) {
        const int H = le::TextConfig::hidden;
        le::TextLayerWeights w;
        std::vector<float> nrm(H, 1.0F);
        w.input_norm           = tensor_bf16(nrm, {H});
        w.post_attention_norm  = tensor_bf16(nrm, {H});
        w.mlp_gate_up          = w8(qt::linear::make_w8g32_f16s_weight(24576, H, 23), 24576, H);
        w.mlp_down             = w8(qt::linear::make_w8g32_f16s_weight(H, 12288, 24), H, 12288);
        w.gdn.query_key_value_z =
            w8(qt::linear::make_w8g32_f16s_weight(12288, H, 25), 12288, H);
        w.gdn.a_b_projection   = bf16_weight(qt::direct_bf16_weight::make_patterned(64, H, 26));
        w.gdn.output           = w8(qt::linear::make_w8g32_f16s_weight(H, H, 27), H, H);
        std::vector<float> a_log(32), dt_bias(32);
        qt::fill_uniform(a_log, 28, -1.0F, 1.0F);
        qt::fill_uniform(dt_bias, 29, -1.0F, 1.0F);
        w.gdn.a_log    = tensor_f32(a_log, {32});
        w.gdn.dt_bias  = tensor_f32(dt_bias, {32});
        std::vector<float> gnorm(128, 1.0F);
        w.gdn.norm     = tensor_bf16(gnorm, {128});
        w.is_full_attention = false;
        w.has_mhc           = with_mhc;
        return w;
    }
};

int run_device_tests(cudaStream_t stream) {
    DeviceFixture f;

    // mHC router pairs in op form, prepared from the stored [8,8192] rows through the
    // setup-time transpose helper.
    auto make_routers = [&] {
        std::vector<float> w0(8 * 8192), w1(8 * 8192);
        qt::fill_uniform(w0, 41, -0.1F, 0.1F);
        qt::fill_uniform(w1, 42, -0.1F, 0.1F);
        Tensor stored0 = f.tensor_f32(w0, {8, 8192});
        Tensor stored1 = f.tensor_f32(w1, {8, 8192});
        Tensor op0 = f.tensor_f32(std::vector<float>(8192 * 8), {8192, 8});
        Tensor op1 = f.tensor_f32(std::vector<float>(8192 * 8), {8192, 8});
        le::transpose_mhc_router_weight(stored0, op0, stream);
        le::transpose_mhc_router_weight(stored1, op1, stream);
        std::vector<float> b0(8), b1(8);
        qt::fill_uniform(b0, 43, -0.05F, 0.05F);
        qt::fill_uniform(b1, 44, -0.05F, 0.05F);
        return std::make_tuple(f.tensor_f32(b0, {8}), f.tensor_f32(b1, {8}), op0, op1);
    };
    auto [bias0, bias1, router0, router1] = make_routers();

    // GDN convolution in op form. The artifact stores kernel-major bytes; the op reinterprets the
    // same bytes as [8192,4].
    std::vector<float> conv_host(8192 * 4);
    qt::fill_uniform(conv_host, 51, -0.2F, 0.2F);
    Tensor conv_op = f.tensor_bf16(conv_host, {8192, 4});

    // GDN visit state image: two adjacent slots; only the bound slot may be mutated.
    const std::size_t gdn_bytes = le::StateLayout::gdn_state_bytes_per_slot();
    DeviceBuffer gdn_image(2 * gdn_bytes);
    {
        std::vector<std::uint8_t> image(2 * gdn_bytes);
        for (std::size_t i = 0; i < gdn_bytes; ++i) {
            image[i]             = 0xA5;
            image[gdn_bytes + i] = 0x5A;
        }
        gdn_image.copy_from_host(image.data(), image.size());
    }
    f.buffers.push_back(std::move(gdn_image));
    std::byte* const gdn_slot0      = static_cast<std::byte*>(f.buffers.back().p);
    std::byte* const gdn_slot1      = gdn_slot0 + gdn_bytes;
    auto all_bytes_equal = [](const std::vector<std::uint8_t>& bytes, std::uint8_t value) {
        return std::all_of(bytes.begin(), bytes.end(), [value](std::uint8_t b) { return b == value; });
    };

    auto make_mhc_slot = [&] {
        DeviceBuffer slot(le::StateLayout::mhc_state_bytes_per_slot());
        slot.fill(0x00);
        f.buffers.push_back(std::move(slot));
        return std::pair<std::byte*, std::size_t>{static_cast<std::byte*>(f.buffers.back().p),
                                                  f.buffers.back().bytes};
    };

    // One full-attention KV domain: two physical pages of 64 tokens, marker-filled.
    struct KvDomain {
        std::size_t bytes = 0;
        std::byte* k      = nullptr;
        std::byte* v      = nullptr;
        std::byte* table  = nullptr;
    };
    auto make_kv_domain = [&] {
        KvDomain d;
        d.bytes = std::size_t(le::TextConfig::head_dim) * kPagedKVPageSize *
                  le::TextConfig::kv_heads * 2 /* pages */ * 2 /* bf16 */;
        DeviceBuffer k(d.bytes);
        k.fill(0xA5);
        DeviceBuffer v(d.bytes);
        v.fill(0xA5);
        DeviceBuffer t = qt::to_device(std::vector<int>{0, 1});
        f.buffers.push_back(std::move(k));
        f.buffers.push_back(std::move(v));
        f.buffers.push_back(std::move(t));
        d.k = static_cast<std::byte*>(f.buffers[f.buffers.size() - 3].p);
        d.v = static_cast<std::byte*>(f.buffers[f.buffers.size() - 2].p);
        d.table = static_cast<std::byte*>(f.buffers.back().p);
        return d;
    };
    auto batch_view = [](const KvDomain& d) {
        PagedKVBatchLayerView view;
        view.k_pages       = Tensor(d.k, DType::BF16, {256, 64, 4, 2});
        view.v_pages       = Tensor(d.v, DType::BF16, {256, 64, 4, 2});
        view.block_tables  = Tensor(d.table, DType::I32, {2, 1});
        view.head_dim      = 256;
        view.num_kv_heads  = 4;
        view.dtype         = DType::BF16;
        return view;
    };
    auto direct_view = [](const KvDomain& d) {
        PagedKVLayerView view;
        view.k_pages      = Tensor(d.k, DType::BF16, {256, 64, 4, 2});
        view.v_pages      = Tensor(d.v, DType::BF16, {256, 64, 4, 2});
        view.block_table  = Tensor(d.table, DType::I32, {2});
        view.head_dim     = 256;
        view.num_kv_heads = 4;
        view.dtype        = DType::BF16;
        return view;
    };

    auto run_visit = [&](const le::LayerVisitRequest& req) {
        le::execute_layer_visit(req);
        qt::cuda_synchronize();
    };
    auto make_hidden = [&](std::uint32_t seed, int width, int tokens) {
        std::vector<float> values(std::size_t(width) * tokens);
        qt::fill_uniform(values, seed, -1.0F, 1.0F);
        qt::round_to_bf16(values);
        return f.tensor_bf16(values, {width, tokens});
    };
    auto make_hidden_out = [&](int width, int tokens) {
        return f.tensor_bf16(std::vector<float>(std::size_t(width) * tokens, 0.0F),
                             {width, tokens});
    };

    // --------------------------------------------- visit 4: GDN + mHC enter, T = 1
    {
        const std::size_t cap = le::layer_visit_workspace_capacity_bytes(4, 1, 1);
        DeviceBuffer ws(cap);
        WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
        f.buffers.push_back(std::move(ws));

        const Tensor hidden_in  = make_hidden(61, 4096, 1);
        Tensor hidden_out = make_hidden_out(8192, 1);
        auto [mhc_slot, mhc_slot_bytes] = make_mhc_slot();
        const le::TextLayerWeights layer4 = f.make_gdn_weights(true);

        le::LayerVisitRequest req;
        req.visit                  = 4;
        req.hidden_in              = &hidden_in;
        req.hidden_out             = &hidden_out;
        req.weights                = &layer4;
        req.gdn_state_base         = gdn_slot0;
        req.gdn_convolution_weight = &conv_op;
        req.mhc_attn_weight        = &router0;
        req.mhc_attn_bias          = &bias0;
        req.mhc_ffn_weight         = &router1;
        req.mhc_ffn_bias           = &bias1;
        req.mhc_state_base         = mhc_slot;
        req.stream                 = stream;
        req.workspace              = &arena;
        run_visit(req);

        expect(!all_bytes_equal(read_bytes(gdn_slot0, gdn_bytes), 0xA5),
               "GDN visit slot is mutated");
        expect(all_bytes_equal(read_bytes(gdn_slot1, gdn_bytes), 0x5A),
               "GDN neighbor slot is untouched");
        const std::vector<std::uint8_t> mhc_after =
            read_bytes(mhc_slot, mhc_slot_bytes);
        const std::vector<std::uint8_t> out_bytes =
            read_bytes(hidden_out.data, hidden_out.bytes());
        expect(mhc_after == out_bytes, "mHC slot is the published flattened stream state");
        expect(std::any_of(mhc_after.begin(), mhc_after.end(),
                           [](std::uint8_t b) { return b != 0; }),
               "mHC slot is not all zero after an enter visit");
    }

    // ---------------------- visits 7 / 31: revisit pair, full-attention + mHC, T = 2
    {
        const std::size_t cap = le::layer_visit_workspace_capacity_bytes(7, 2, 2);
        DeviceBuffer ws(cap);
        WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
        f.buffers.push_back(std::move(ws));
        const le::TextLayerWeights layer7 = f.make_full_attention_weights(true);

        KvDomain kv_a = make_kv_domain();
        KvDomain kv_b = make_kv_domain();
        auto [mhc_a, mhc_a_bytes] = make_mhc_slot();
        auto [mhc_b, mhc_b_bytes] = make_mhc_slot();

        const std::size_t token_state = std::size_t(8192) * 2;
        const Tensor k   = f.tensor_bf16(std::vector<float>(1024 * 2, 0.5F), {1024, 2});
        const Tensor v   = f.tensor_bf16(std::vector<float>(1024 * 2, 0.25F), {1024, 2});
        const Tensor positions   = f.tensor_i32({0, 1}, {2, 1, 1, 1});
        const Tensor table_rows  = f.tensor_i32({0}, {1, 1, 1, 1});

        auto run_visit_one = [&](std::uint32_t visit, const KvDomain& kv, std::byte* slot,
                                 const Tensor& in, Tensor& out) {
            le::LayerVisitRequest req;
            req.visit                  = visit;
            req.hidden_in              = &in;
            req.hidden_out             = &out;
            req.weights                = &layer7;
            req.positions              = &positions;
            req.attention_k            = &k;
            req.attention_v            = &v;
            req.attention_kv_table_rows = &table_rows;
            req.attention_kv_append    = batch_view(kv);
            req.attention_envelope     = {1, 128};
            req.mhc_attn_weight        = &router0;
            req.mhc_attn_bias          = &bias0;
            req.mhc_ffn_weight         = &router1;
            req.mhc_ffn_bias           = &bias1;
            req.mhc_state_base         = slot;
            req.stream                 = stream;
            req.workspace              = &arena;
            run_visit(req);
        };

        const Tensor in_a  = make_hidden(71, 8192, 2);
        Tensor out_a = make_hidden_out(8192, 2);
        const std::size_t out_a_bytes = std::size_t(8192) * 2 * 2;
        run_visit_one(7, kv_a, mhc_a, in_a, out_a);

        const std::vector<std::uint8_t> slot_a = read_bytes(mhc_a, mhc_a_bytes);
        const std::vector<std::uint8_t> out_a_image  = read_bytes(out_a.data, out_a_bytes);

        expect(!all_bytes_equal(read_bytes(kv_a.k, kv_a.bytes), 0xA5),
               "visit 7 appended its K page");
        expect(!all_bytes_equal(read_bytes(kv_a.v, kv_a.bytes), 0xA5),
               "visit 7 appended its V page");
        expect(slot_a ==
                   std::vector<std::uint8_t>(out_a_image.begin() +
                                                static_cast<std::ptrdiff_t>(out_a_image.size() -
                                                                            token_state),
                                             out_a_image.end()),
               "visit 7 mHC slot is its final flattened token");

        // Pass 1 of the same physical layer: distinct KV domain and mHC slot, same weights.
        const Tensor in_b  = make_hidden(72, 8192, 2);
        Tensor out_b = make_hidden_out(8192, 2);
        run_visit_one(31, kv_b, mhc_b, in_b, out_b);

        expect(read_bytes(mhc_a, mhc_a_bytes) == slot_a,
               "visit 31 left visit 7's mHC slot untouched");
        expect(!all_bytes_equal(read_bytes(kv_b.k, kv_b.bytes), 0xA5),
               "visit 31 appended into its own KV domain");
        const std::vector<std::uint8_t> slot_b = read_bytes(mhc_b, mhc_b_bytes);
        const std::vector<std::uint8_t> out_b_image = read_bytes(out_b.data, out_a_bytes);
        expect(slot_b ==
                   std::vector<std::uint8_t>(out_b_image.begin() +
                                                static_cast<std::ptrdiff_t>(out_b_image.size() -
                                                                            token_state),
                                             out_b_image.end()),
               "visit 31 mHC slot is its own final flattened token");
    }

    // ------------------------- visit 3: full-attention A3 cached visit, non-mHC, T = 2
    {
        const std::size_t cap = le::layer_visit_workspace_capacity_bytes(3, 2, 2);
        DeviceBuffer ws(cap);
        WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
        f.buffers.push_back(std::move(ws));
        const le::TextLayerWeights layer3 = f.make_full_attention_weights(false);

        KvDomain kv = make_kv_domain();
        const Tensor hidden_in  = make_hidden(81, 4096, 2);
        Tensor hidden_out = make_hidden_out(4096, 2);
        const Tensor positions   = f.tensor_i32({0, 1}, {2, 1, 1, 1});

        le::LayerVisitRequest req;
        req.visit                = 3;
        req.hidden_in            = &hidden_in;
        req.hidden_out           = &hidden_out;
        req.weights              = &layer3;
        req.positions            = &positions;
        req.attention_kv_cached  = direct_view(kv);
        req.attention_envelope   = {1, 128};
        req.stream               = stream;
        req.workspace            = &arena;
        run_visit(req);

        expect(all_bytes_equal(read_bytes(kv.k, kv.bytes), 0xA5),
               "A3 cached visit reads but does not write K");
        expect(all_bytes_equal(read_bytes(kv.v, kv.bytes), 0xA5),
               "A3 cached visit reads but does not write V");
        const std::vector<std::uint8_t> out = read_bytes(hidden_out.data, hidden_out.bytes());
        expect(std::any_of(out.begin(), out.end(), [](std::uint8_t b) { return b != 0; }),
               "A3 cached visit produces a hidden output");
    }

    // ------------------------------- visit 27: full-attention + mHC leave, T = 2
    {
        const std::size_t cap = le::layer_visit_workspace_capacity_bytes(27, 2, 2);
        DeviceBuffer ws(cap);
        WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
        f.buffers.push_back(std::move(ws));
        const le::TextLayerWeights layer27 = f.make_full_attention_weights(true);

        KvDomain kv = make_kv_domain();
        const Tensor hidden_in  = make_hidden(91, 8192, 2);
        Tensor hidden_out = make_hidden_out(4096, 2);
        auto [mhc_slot, mhc_slot_bytes] = make_mhc_slot();
        const Tensor positions   = f.tensor_i32({0, 1}, {2, 1, 1, 1});
        const Tensor table_rows  = f.tensor_i32({0}, {1, 1, 1, 1});
        const Tensor k           = f.tensor_bf16(std::vector<float>(1024 * 2, 0.5F), {1024, 2});
        const Tensor v           = f.tensor_bf16(std::vector<float>(1024 * 2, 0.25F), {1024, 2});

        le::LayerVisitRequest req;
        req.visit                = 27;
        req.hidden_in            = &hidden_in;
        req.hidden_out           = &hidden_out;
        req.weights              = &layer27;
        req.positions            = &positions;
        req.attention_k          = &k;
        req.attention_v          = &v;
        req.attention_kv_table_rows = &table_rows;
        req.attention_kv_append  = batch_view(kv);
        req.attention_envelope   = {1, 128};
        req.mhc_attn_weight      = &router0;
        req.mhc_attn_bias        = &bias0;
        req.mhc_ffn_weight       = &router1;
        req.mhc_ffn_bias         = &bias1;
        req.mhc_state_base       = mhc_slot;
        req.stream               = stream;
        req.workspace            = &arena;
        run_visit(req);

        // Check the leave reduction directly. A full visit modifies the residual twice, so the
        // original input streams are not the streams that enter the final reduction.
        const std::size_t d = 4096;
        const int tokens    = 2;
        std::vector<float> merged_values(d * 2 * tokens);
        qt::fill_uniform(merged_values, 97, -1.0F, 1.0F);
        qt::round_to_bf16(merged_values);
        const Tensor merged =
            f.tensor_bf16(merged_values, {static_cast<int>(d), 2, tokens});
        Tensor reduced = le::mhc_reduce_streams(merged, nullptr, arena, stream);
        qt::cuda_synchronize();

        const std::vector<uint8_t> merged_bytes = read_bytes(merged.data, merged.bytes());
        const std::vector<uint8_t> out = read_bytes(reduced.data, reduced.bytes());
        float max_error = 0.0F;
        bool finite = true;
        for (int t = 0; t < tokens; ++t) {
            for (std::size_t i = 0; i < d; ++i) {
                const std::size_t s0_index = (std::size_t(t) * 2 * d + i) * 2;
                const std::size_t s1_index = (std::size_t(t) * 2 * d + d + i) * 2;
                const float s0 = qt::bf16_to_f32(std::uint16_t(merged_bytes[s0_index] |
                                                               (merged_bytes[s0_index + 1] << 8)));
                const float s1 = qt::bf16_to_f32(std::uint16_t(merged_bytes[s1_index] |
                                                               (merged_bytes[s1_index + 1] << 8)));
                const std::size_t out_index = (std::size_t(t) * d + i) * 2;
                const float o = qt::bf16_to_f32(std::uint16_t(out[out_index] |
                                                              (out[out_index + 1] << 8)));
                const float ref = 0.5F * s0 + 0.5F * s1;
                if (!std::isfinite(o) || !std::isfinite(ref)) { finite = false; }
                const float denom = std::max(1.0F, std::abs(ref));
                max_error = std::max(max_error, std::abs(o - ref) / denom);
            }
        }
        expect(finite, "leave visit outputs are finite");
        expect(max_error <= 0.02F, "leave visit reduces the streams by their mean");
    }

    return failures;
}

} // namespace

int main() {
    test_plan_mapping();
    test_mhc_boundaries();
    test_validation();
    test_gdn_state_views();

    if (qt::cuda_unavailable()) {
        std::cerr << "SKIP: no CUDA device available\n";
        return 77;
    }
    cudaSetDevice(0);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::cerr << "SKIP: cannot create a CUDA stream\n";
        return 77;
    }
    const int device_result = run_device_tests(stream);
    cudaStreamDestroy(stream);
    if (device_result == 77) {
        return 77;
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
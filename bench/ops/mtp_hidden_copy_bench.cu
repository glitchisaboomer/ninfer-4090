// CUDA-Graph microbenchmark for the MTP hidden-state copy-back candidate.
//
// It models the production loop's buffer dependency and graph-node structure, not the MTP layer
// math: each synthetic MTP step transforms one BF16-sized hidden buffer, then the reference path
// copies the result back while the candidate path ping-pongs the two buffers. The paired timing
// isolates the removable D2D-copy-node cost. It must not be read as an end-to-end MTP speedup.
//
// Typical Qwen3.8-27B invocation:
//   ninfer_mtp_hidden_copy_bench --hidden 5120 --batch 1 --ks 1,3,5,7,15

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    int device = 0;
    int hidden = 5120;
    int batch = 1;
    int warmup = 100;
    int repeats = 500;
    int samples = 9;
    std::vector<int> ks{1, 2, 3, 5, 7, 15};
};

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

void cuda_check(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " failed at line " +
                                 std::to_string(line) + ": " + cudaGetErrorString(error));
    }
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr, __LINE__)

int parse_positive(const char* flag, const char* value, int minimum, int maximum) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        fail(std::string("invalid value for ") + flag + ": " + value);
    }
    return static_cast<int>(parsed);
}

std::vector<int> parse_ks(const char* value) {
    std::vector<int> result;
    const char* cursor = value;
    while (*cursor != '\0') {
        errno = 0;
        char* end = nullptr;
        const long parsed = std::strtol(cursor, &end, 10);
        if (errno != 0 || end == cursor || parsed < 1 || parsed > 15) {
            fail("--ks values must be in [1,15]");
        }
        result.push_back(static_cast<int>(parsed));
        if (*end == '\0') { break; }
        if (*end != ',') { fail("--ks must be a comma-separated list"); }
        cursor = end + 1;
    }
    if (result.empty()) { fail("--ks must not be empty"); }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto take = [&](const char* flag) -> const char* {
            if (++i >= argc) { fail(std::string("missing value for ") + flag); }
            return argv[i];
        };
        if (arg == "--device") {
            options.device = parse_positive("--device", take("--device"), 0, 1024);
        } else if (arg == "--hidden") {
            options.hidden = parse_positive("--hidden", take("--hidden"), 1, 1 << 20);
        } else if (arg == "--batch") {
            options.batch = parse_positive("--batch", take("--batch"), 1, 8);
        } else if (arg == "--warmup") {
            options.warmup = parse_positive("--warmup", take("--warmup"), 0, 1000000);
        } else if (arg == "--repeats") {
            options.repeats = parse_positive("--repeats", take("--repeats"), 1, 1000000);
        } else if (arg == "--samples") {
            options.samples = parse_positive("--samples", take("--samples"), 1, 101);
        } else if (arg == "--ks") {
            options.ks = parse_ks(take("--ks"));
        } else if (arg == "-h" || arg == "--help") {
            std::printf("usage: %s [--device N] [--hidden N] [--batch 1..8] "
                        "[--ks 1,3,5,7,15] [--warmup N] [--repeats N] [--samples N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            fail("unknown argument: " + std::string(arg));
        }
    }
    return options;
}

// Deterministic, bit-exact stand-in for the MTP forward's hidden input/output dependency.
// Treats BF16 storage as opaque 16-bit words; incrementing is only a dataflow sentinel, not math.
__global__ void synthetic_mtp_step(const std::uint16_t* input, std::uint16_t* output,
                                   std::size_t elements) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) { output[i] = static_cast<std::uint16_t>(input[i] + 1U); }
}

enum class Variant { CopyBack, PingPong };

struct CapturedGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::size_t node_count = 0;
    std::size_t memcpy_nodes = 0;
    std::uint16_t* final_buffer = nullptr;
};

CapturedGraph capture_graph(cudaStream_t stream, std::uint16_t* buffer_a, std::uint16_t* buffer_b,
                            std::size_t elements, int steps, Variant variant) {
    CapturedGraph result;
    std::uint16_t* current = buffer_a;
    std::uint16_t* output = buffer_b;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);

    if (steps == 0) {
        CUDA_CHECK(cudaGraphCreate(&result.graph, 0));
    } else {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int step = 0; step < steps; ++step) {
            synthetic_mtp_step<<<blocks, threads, 0, stream>>>(current, output, elements);
            CUDA_CHECK(cudaGetLastError());
            if (variant == Variant::CopyBack) {
                CUDA_CHECK(cudaMemcpyAsync(current, output, elements * sizeof(std::uint16_t),
                                           cudaMemcpyDeviceToDevice, stream));
            } else {
                std::swap(current, output);
            }
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &result.graph));
    }
    CUDA_CHECK(cudaGraphInstantiate(&result.executable, result.graph, 0));
    CUDA_CHECK(cudaGraphGetNodes(result.graph, nullptr, &result.node_count));
    std::vector<cudaGraphNode_t> nodes(result.node_count);
    CUDA_CHECK(cudaGraphGetNodes(result.graph, nodes.data(), &result.node_count));
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeMemcpy) { ++result.memcpy_nodes; }
    }
    result.final_buffer = current;
    return result;
}

void destroy_graph(CapturedGraph& graph) {
    if (graph.executable != nullptr) { CUDA_CHECK(cudaGraphExecDestroy(graph.executable)); }
    if (graph.graph != nullptr) { CUDA_CHECK(cudaGraphDestroy(graph.graph)); }
}

double percentile(const std::vector<double>& sorted, double p) {
    const std::size_t index = static_cast<std::size_t>(p * (sorted.size() - 1));
    return sorted[index];
}

struct Measurement {
    double median_us = 0.0;
    double p10_us = 0.0;
    double p90_us = 0.0;
    std::size_t nodes = 0;
    std::size_t memcpy_nodes = 0;
};

Measurement measure(CapturedGraph& graph, std::uint16_t* a, std::uint16_t* b,
                    std::size_t elements, int warmup, int repeats, int samples,
                    cudaStream_t stream) {
    const std::size_t bytes = elements * sizeof(std::uint16_t);
    std::vector<double> elapsed_us;
    elapsed_us.reserve(static_cast<std::size_t>(samples));
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Initialize outside the measured/captured region; correctness is checked separately.
    std::vector<std::uint16_t> initial(elements, 37U);
    CUDA_CHECK(cudaMemcpyAsync(a, initial.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(b, 0, bytes, stream));
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < repeats; ++i) {
            CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        elapsed_us.push_back(static_cast<double>(milliseconds) * 1000.0 / repeats);
    }
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));

    std::sort(elapsed_us.begin(), elapsed_us.end());
    return Measurement{.median_us = percentile(elapsed_us, 0.50),
                       .p10_us = percentile(elapsed_us, 0.10),
                       .p90_us = percentile(elapsed_us, 0.90),
                       .nodes = graph.node_count,
                       .memcpy_nodes = graph.memcpy_nodes};
}

void validate_result(const CapturedGraph& graph, std::uint16_t* a, std::uint16_t* b,
                     std::size_t elements, int steps, cudaStream_t stream) {
    const std::size_t bytes = elements * sizeof(std::uint16_t);
    std::vector<std::uint16_t> initial(elements, 37U);
    std::vector<std::uint16_t> observed(elements);
    CUDA_CHECK(cudaMemcpyAsync(a, initial.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(b, 0, bytes, stream));
    CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
    CUDA_CHECK(cudaMemcpyAsync(observed.data(), graph.final_buffer, bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto expected = static_cast<std::uint16_t>(37U + steps);
    if (!std::all_of(observed.begin(), observed.end(),
                     [expected](std::uint16_t value) { return value == expected; })) {
        throw std::runtime_error("copy-back and ping-pong dataflow sentinel validation failed");
    }
}

int run(const Options& options) {
    CUDA_CHECK(cudaSetDevice(options.device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, options.device));

    const std::size_t elements = static_cast<std::size_t>(options.hidden) * options.batch;
    const std::size_t bytes = elements * sizeof(std::uint16_t);
    std::uint16_t* a = nullptr;
    std::uint16_t* b = nullptr;
    cudaStream_t stream{};
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::printf("# synthetic CUDA-Graph proxy; not an MTP end-to-end timing\n");
    std::printf("# device=%s hidden=%d batch=%d hidden_bytes=%zu warmup=%d repeats=%d samples=%d\n",
                props.name, options.hidden, options.batch, bytes, options.warmup,
                options.repeats, options.samples);
    std::printf("K,steps,variant,graph_nodes,memcpy_nodes,median_us,p10_us,p90_us\n");

    for (const int k : options.ks) {
        const int steps = k - 1; // MTP loop has k-1 forward/copy iterations after draft 0.
        CapturedGraph copy_graph = capture_graph(stream, a, b, elements, steps, Variant::CopyBack);
        CapturedGraph ping_graph = capture_graph(stream, a, b, elements, steps, Variant::PingPong);
        validate_result(copy_graph, a, b, elements, steps, stream);
        validate_result(ping_graph, a, b, elements, steps, stream);
        const Measurement copy = measure(copy_graph, a, b, elements, options.warmup,
                                         options.repeats, options.samples, stream);
        const Measurement ping = measure(ping_graph, a, b, elements, options.warmup,
                                         options.repeats, options.samples, stream);
        std::printf("%d,%d,copy_back,%zu,%zu,%.4f,%.4f,%.4f\n", k, steps, copy.nodes,
                    copy.memcpy_nodes, copy.median_us, copy.p10_us, copy.p90_us);
        std::printf("%d,%d,ping_pong,%zu,%zu,%.4f,%.4f,%.4f\n", k, steps, ping.nodes,
                    ping.memcpy_nodes, ping.median_us, ping.p10_us, ping.p90_us);
        std::printf("# K=%d paired median delta (copy_back - ping_pong): %.4f us/graph\n",
                    k, copy.median_us - ping.median_us);
        destroy_graph(ping_graph);
        destroy_graph(copy_graph);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_mtp_hidden_copy_bench: %s\n", error.what());
        return 1;
    }
}

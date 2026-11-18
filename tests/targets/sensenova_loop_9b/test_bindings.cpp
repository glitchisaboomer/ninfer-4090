#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/sensenova_loop_9b/impl/config.h"
#include "targets/sensenova_loop_9b/impl/load/bindings.h"

#include <ninfer/targets/sensenova_loop_9b/package.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path artifact_path() {
    if (const char* value = std::getenv("NINFER_SENSENOVA_LOOP_9B_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    return "/home/user/models/sensenova/SenseNova-Loop-9B.ninfer";
}

int verify_identity() {
        const ninfer::targets::sensenova_loop_9b::Package::WeightsProfile profile =
        ninfer::targets::sensenova_loop_9b::Package::resolve_weights({"sensenova-loop-9b", "groupwise-int"});
    if (profile != ninfer::targets::sensenova_loop_9b::detail::WeightsProfile::GroupwiseInt) { return 1; }
    try {
        (void)ninfer::targets::sensenova_loop_9b::Package::resolve_weights({"sensenova-loop-9b", "fp16"});
        return 1;
    } catch (const std::runtime_error&) {}
    return 0;
}

} // namespace

int main() {
    if (verify_identity() != 0) {
        std::cerr << "FAIL: SenseNova identity resolution\n";
        return 1;
    }

    const auto path = artifact_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: SenseNova artifact is not available at " << path << '\n';
        return 77;
    }

    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const auto plan = ninfer::targets::sensenova_loop_9b::detail::bind_artifact(
        binder, ninfer::targets::sensenova_loop_9b::detail::WeightsProfile::GroupwiseInt);
    if (plan.materialization.object_count != 447 || plan.materialization.device_objects.size() != 441 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "FAIL: unexpected SenseNova materialization plan\n";
        return 1;
    }
    std::cout << "OK senseNova bindings\n";
    return 0;
}

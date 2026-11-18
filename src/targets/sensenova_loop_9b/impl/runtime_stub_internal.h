#pragma once

#include "targets/sensenova_loop_9b/impl/runtime_stub.h"

namespace ninfer::targets::sensenova_loop_9b::detail {

runtime::SequenceCapacityCurve make_curve(const EngineOptions& options);

} // namespace ninfer::targets::sensenova_loop_9b::detail

target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime_stub.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/layer_execution.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/layouts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/text_execution.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state_layout.cpp)

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/export)

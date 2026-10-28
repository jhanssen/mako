add_executable(io_bench io_bench.cpp)

add_executable(mm_bench mm_bench.cpp)
target_link_libraries(mm_bench PRIVATE mako_mtl)

add_executable(mem_watchdog mem_watchdog.cpp)

add_executable(mm_types_bench mm_types_bench.cpp)
target_link_libraries(mm_types_bench PRIVATE mako_mtl)

add_executable(gemm_bench gemm_bench.cpp)
target_link_libraries(gemm_bench PRIVATE mako_mtl)

add_executable(gemv_bench gemv_bench.cpp)
target_link_libraries(gemv_bench PRIVATE mako_mtl mako_gguf)

add_executable(chain_bench chain_bench.cpp)
target_link_libraries(chain_bench PRIVATE mako_mtl)

add_executable(perf_check perf_check.cpp)
target_link_libraries(perf_check PRIVATE mako_glm mako_dsv41 mako_text mako_app glaze_headers)
target_compile_definitions(perf_check PRIVATE MAKO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

add_executable(io_bench io_bench.cpp)

add_executable(sync_bench sync_bench.cpp)
target_link_libraries(sync_bench PRIVATE mako_mtl)

add_executable(stream_bench stream_bench.cpp)
target_link_libraries(stream_bench PRIVATE mako_mtl)

add_executable(mm_bench mm_bench.cpp)
target_link_libraries(mm_bench PRIVATE mako_mtl)

add_executable(qformat_bench qformat_bench.cpp)
target_link_libraries(qformat_bench PRIVATE mako_mtl)

add_executable(lend_bench lend_bench.cpp)
target_link_libraries(lend_bench PRIVATE mako_mtl)

add_executable(mem_watchdog mem_watchdog.cpp)

add_executable(mm_types_bench mm_types_bench.cpp)
target_link_libraries(mm_types_bench PRIVATE mako_mtl)

add_executable(gemm_bench gemm_bench.cpp)
target_link_libraries(gemm_bench PRIVATE mako_mtl)

add_executable(gemv_bench gemv_bench.cpp)
target_link_libraries(gemv_bench PRIVATE mako_mtl mako_gguf)

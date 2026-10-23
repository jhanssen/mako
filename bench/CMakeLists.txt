add_executable(io_bench io_bench.cpp)

add_executable(sync_bench sync_bench.cpp)
target_link_libraries(sync_bench PRIVATE mako_mtl)

add_executable(stream_bench stream_bench.cpp)
target_link_libraries(stream_bench PRIVATE mako_mtl)

add_executable(mm_bench mm_bench.cpp)
target_link_libraries(mm_bench PRIVATE mako_mtl)

include(llama_metal.cmake)
add_executable(qformat_bench qformat_bench.cpp)
target_link_libraries(qformat_bench PRIVATE mako_mtl)
target_compile_definitions(qformat_bench PRIVATE LLAMA_MUL_MV_METALLIB="${LLAMA_MUL_MV_METALLIB}")
add_dependencies(qformat_bench llama_mul_mv_metallib)

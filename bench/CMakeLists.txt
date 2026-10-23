add_executable(io_bench io_bench.cpp)

add_executable(sync_bench sync_bench.cpp)
target_link_libraries(sync_bench PRIVATE mako_mtl)

add_executable(stream_bench stream_bench.cpp)
target_link_libraries(stream_bench PRIVATE mako_mtl)

add_executable(mm_bench mm_bench.cpp)
target_link_libraries(mm_bench PRIVATE mako_mtl)

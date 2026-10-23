add_library(mako_mtl STATIC mtl/metal_impl.cpp mtl/context.cpp)
target_include_directories(mako_mtl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_mtl PUBLIC metal_cpp)

add_library(mako_gguf STATIC gguf/gguf.cpp)
target_include_directories(mako_gguf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

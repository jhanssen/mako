add_library(mako_mtl STATIC mtl/metal_impl.cpp mtl/context.cpp mtl/gemv.cpp)
target_include_directories(mako_mtl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_mtl PUBLIC metal_cpp mako_gguf)

add_library(mako_gguf STATIC gguf/gguf.cpp gguf/model_files.cpp quant/dequant.cpp)
target_include_directories(mako_gguf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(MAKO_METALLIB ${CMAKE_BINARY_DIR}/mako.metallib)
set(MAKO_METAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/glm.metal)
set(_airs)
foreach(src ${MAKO_METAL_SOURCES})
    get_filename_component(stem ${src} NAME_WE)
    set(air ${CMAKE_BINARY_DIR}/kernels/${stem}.air)
    add_custom_command(OUTPUT ${air}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/kernels
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -Wall -c ${src} -o ${air}
        DEPENDS ${src}
        COMMENT "Compiling ${stem}.metal")
    list(APPEND _airs ${air})
endforeach()
add_custom_command(OUTPUT ${MAKO_METALLIB}
    COMMAND xcrun -sdk macosx metallib ${_airs} -o ${MAKO_METALLIB}
    DEPENDS ${_airs}
    COMMENT "Linking mako.metallib")
add_custom_target(mako_metallib DEPENDS ${MAKO_METALLIB})

add_library(mako_glm STATIC glm/config.cpp glm/engine.cpp)
target_link_libraries(mako_glm PUBLIC mako_mtl mako_gguf)
add_dependencies(mako_glm mako_metallib gemv_metallib)
target_compile_definitions(mako_glm PUBLIC MAKO_DEFAULT_METALLIB="${MAKO_METALLIB}" MAKO_DEFAULT_GEMV_METALLIB="${GEMV_METALLIB}")

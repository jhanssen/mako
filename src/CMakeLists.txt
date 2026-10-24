add_library(mako_mtl STATIC mtl/metal_impl.cpp mtl/context.cpp mtl/gemv.cpp mtl/gemm.cpp mtl/embedded.cpp)
target_include_directories(mako_mtl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_mtl PUBLIC metal_cpp mako_gguf)

add_library(mako_gguf STATIC gguf/gguf.cpp gguf/model_files.cpp quant/dequant.cpp)
target_include_directories(mako_gguf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(MAKO_METALLIB ${CMAKE_BINARY_DIR}/mako.metallib)
set(MAKO_METAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/glm.metal ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gemv.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gemm.metal)
set(_airs)
foreach(src ${MAKO_METAL_SOURCES})
    get_filename_component(stem ${src} NAME_WE)
    set(air ${CMAKE_BINARY_DIR}/kernels/${stem}.air)
    add_custom_command(OUTPUT ${air}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/kernels
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -Wall -I ${GEMV_KERNELS_DIR} -c ${src} -o ${air}
        DEPENDS ${src} ${GEMV_KERNELS_DIR}/ggml-common.h
        COMMENT "Compiling ${stem}.metal")
    list(APPEND _airs ${air})
endforeach()
add_custom_command(OUTPUT ${MAKO_METALLIB}
    COMMAND xcrun -sdk macosx metallib ${_airs} -o ${MAKO_METALLIB}
    DEPENDS ${_airs}
    COMMENT "Linking mako.metallib")
add_custom_target(mako_metallib DEPENDS ${MAKO_METALLIB})

add_custom_target(mako_version
    COMMAND ${CMAKE_COMMAND} -DSRC=${PROJECT_SOURCE_DIR} -DOUT=${CMAKE_BINARY_DIR}/generated/mako_version.hpp
            -P ${PROJECT_SOURCE_DIR}/cmake/version.cmake
    BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/mako_version.hpp)

add_library(mako_chat STATIC chat/template.cpp)
target_include_directories(mako_chat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_chat PRIVATE jinja_engine)

add_library(mako_gen STATIC gen/sampler.cpp)
target_include_directories(mako_gen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(mako_glm STATIC glm/arena.cpp glm/config.cpp glm/engine.cpp glm/profile.cpp glm/session.cpp glm/telemetry.cpp)
target_link_libraries(mako_glm PUBLIC mako_mtl mako_gguf mako_gen)
target_include_directories(mako_glm PRIVATE ${CMAKE_BINARY_DIR}/generated)
add_dependencies(mako_glm mako_version)

set_source_files_properties(mtl/embedded.cpp PROPERTIES
    COMPILE_OPTIONS "--embed-dir=${CMAKE_BINARY_DIR};-Wno-c23-extensions"
    OBJECT_DEPENDS "${MAKO_METALLIB};${GEMV_METALLIB}")
add_dependencies(mako_mtl mako_metallib gemv_metallib)

add_library(mako_text STATIC text/stream.cpp text/tokenizer.cpp)
target_link_libraries(mako_text PUBLIC mako_gguf)

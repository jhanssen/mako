add_library(mako_mtl STATIC mtl/ablation.cpp mtl/metal_impl.cpp mtl/context.cpp mtl/gemv.cpp mtl/gemm.cpp mtl/embedded.cpp)
target_include_directories(mako_mtl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_mtl PUBLIC metal_cpp mako_gguf)

add_library(mako_gguf STATIC gguf/gguf.cpp gguf/model_files.cpp gguf/read_pool.cpp quant/dequant.cpp quant/quantize.cpp)
target_include_directories(mako_gguf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(MAKO_METALLIB ${CMAKE_BINARY_DIR}/mako.metallib)
set(MAKO_METAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/glm.metal ${CMAKE_CURRENT_SOURCE_DIR}/kernels/dsv41.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gemv.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gemm.metal ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vision.metal)
set(_airs)
foreach(src ${MAKO_METAL_SOURCES})
    get_filename_component(stem ${src} NAME_WE)
    set(air ${CMAKE_BINARY_DIR}/kernels/${stem}.air)
    add_custom_command(OUTPUT ${air}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/kernels
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -Wall -Werror -I ${GEMV_KERNELS_DIR} -c ${src} -o ${air}
        DEPENDS ${src} ${GEMV_KERNELS_DIR}/ggml-common.h ${CMAKE_CURRENT_SOURCE_DIR}/kernels/moe_route.h
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
    BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/mako_version.hpp
    COMMENT "Checking the commit")

add_library(mako_chat STATIC chat/template.cpp)
target_include_directories(mako_chat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_chat PRIVATE jinja_engine)

add_library(mako_server STATIC server/anthropic.cpp server/openai.cpp server/server.cpp)
target_link_libraries(mako_server PUBLIC mako_glm mako_text mako_chat mako_vision PRIVATE glaze_headers httplib_headers)

add_library(mako_dsv41 STATIC dsv41/config.cpp dsv41/engine.cpp dsv41/engram.cpp)
target_include_directories(mako_dsv41 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_dsv41 PUBLIC mako_core mako_mtl mako_gguf PRIVATE glaze_headers)

add_library(mako_app STATIC app/presets.cpp app/cli.cpp)
target_include_directories(mako_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_app PRIVATE glaze_headers)

add_library(mako_gen STATIC gen/loop_guard.cpp gen/pool.cpp gen/sampler.cpp gen/session.cpp)
target_include_directories(mako_gen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(mako_core STATIC core/arena.cpp core/engine_base.cpp core/profile.cpp core/stager.cpp core/telemetry.cpp)
target_include_directories(mako_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_core PUBLIC mako_mtl mako_gguf)
target_include_directories(mako_core PRIVATE ${CMAKE_BINARY_DIR}/generated)
add_dependencies(mako_core mako_version)

add_library(mako_glm STATIC glm/config.cpp glm/engine.cpp)
target_link_libraries(mako_glm PUBLIC mako_core mako_mtl mako_gguf mako_gen)

set_source_files_properties(mtl/embedded.cpp PROPERTIES
    COMPILE_OPTIONS "--embed-dir=${CMAKE_BINARY_DIR};-Wno-c23-extensions"
    OBJECT_DEPENDS "${MAKO_METALLIB};${GEMV_METALLIB}")
add_dependencies(mako_mtl mako_metallib gemv_metallib)

add_library(mako_vision STATIC vision/encoder.cpp vision/image.cpp vision/mmproj.cpp vision/preprocess.cpp
            vision/prompt.cpp)
target_include_directories(mako_vision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_vision PUBLIC mako_gguf mako_mtl PRIVATE "-framework ImageIO" "-framework CoreGraphics"
                      "-framework CoreFoundation")

add_library(mako_text STATIC text/stream.cpp text/tokenizer.cpp)
target_link_libraries(mako_text PUBLIC mako_gguf)

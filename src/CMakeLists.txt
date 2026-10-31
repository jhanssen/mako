# The program's kernels: mako-common's and the models' (linked to mako_mtl).
mako_add_kernels(mako_kernels SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/glm.metal
                 ${CMAKE_CURRENT_SOURCE_DIR}/kernels/dsv41.metal ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vision.metal)

add_custom_target(mako_version
    COMMAND ${CMAKE_COMMAND} -DSRC=${PROJECT_SOURCE_DIR} -DOUT=${CMAKE_BINARY_DIR}/generated/mako_version.hpp
            -P ${PROJECT_SOURCE_DIR}/cmake/version.cmake
    BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/mako_version.hpp
    COMMENT "Checking the commit")

add_library(mako_chat STATIC chat/template.cpp)
target_include_directories(mako_chat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_chat PRIVATE jinja_engine)

# Detokenizing a stream into whole characters.
add_library(mako_stream STATIC text/stream.cpp)
target_include_directories(mako_stream PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_stream PUBLIC mako_text)

add_library(mako_server STATIC server/anthropic.cpp server/models.cpp server/openai.cpp server/server.cpp
            server/status.cpp)
target_link_libraries(mako_server PUBLIC mako_glm mako_text mako_stream mako_chat mako_vision mako_http mako_memory
                      mako_log PRIVATE glaze_headers httplib_headers)

add_library(mako_dsv41 STATIC dsv41/config.cpp dsv41/engine.cpp dsv41/engram.cpp)
target_include_directories(mako_dsv41 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_dsv41 PUBLIC mako_core mako_mtl mako_gguf PRIVATE glaze_headers)

add_library(mako_gen STATIC gen/loop_guard.cpp gen/pool.cpp gen/sampler.cpp gen/session.cpp)
target_include_directories(mako_gen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_gen PUBLIC mako_log)

add_library(mako_core STATIC core/arena.cpp core/engine_base.cpp core/profile.cpp core/stager.cpp core/telemetry.cpp
            mtl/ablation.cpp)
target_include_directories(mako_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_core PUBLIC mako_gpu mako_mtl mako_gguf mako_log)
target_include_directories(mako_core PRIVATE ${CMAKE_BINARY_DIR}/generated)
add_dependencies(mako_core mako_version)

add_library(mako_glm STATIC glm/config.cpp glm/engine.cpp)
target_link_libraries(mako_glm PUBLIC mako_core mako_mtl mako_gguf mako_gen)

add_library(mako_vision STATIC vision/encoder.cpp vision/image.cpp vision/mmproj.cpp vision/preprocess.cpp
            vision/prompt.cpp)
target_include_directories(mako_vision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mako_vision PUBLIC mako_gguf mako_mtl mako_log PRIVATE "-framework ImageIO"
                      "-framework CoreGraphics" "-framework CoreFoundation")

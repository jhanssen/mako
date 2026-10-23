add_library(mako_mtl STATIC mtl/metal_impl.cpp mtl/context.cpp mtl/gemv.cpp mtl/embedded.cpp)
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

set_source_files_properties(mtl/embedded.cpp PROPERTIES
    COMPILE_OPTIONS "--embed-dir=${CMAKE_BINARY_DIR};-Wno-c23-extensions"
    OBJECT_DEPENDS "${MAKO_METALLIB};${GEMV_METALLIB}")
add_dependencies(mako_mtl mako_metallib gemv_metallib)

add_library(mako_text STATIC text/tokenizer.cpp)
target_link_libraries(mako_text PUBLIC mako_gguf)

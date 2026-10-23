# Optional: needs a source tree and build of the reference implementation.
#   -DMAKO_REF_SRC=<source dir> -DMAKO_REF_BUILD=<build dir>
add_executable(ref_dump ref_dump.cpp)
target_include_directories(ref_dump PRIVATE ${MAKO_REF_SRC}/include ${MAKO_REF_SRC}/ggml/include)
target_link_directories(ref_dump PRIVATE ${MAKO_REF_BUILD}/bin)
target_link_libraries(ref_dump PRIVATE llama ggml ggml-base)
set_target_properties(ref_dump PROPERTIES BUILD_RPATH ${MAKO_REF_BUILD}/bin)

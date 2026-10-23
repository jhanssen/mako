add_executable(gguf_dump gguf_dump.cpp)
target_link_libraries(gguf_dump PRIVATE mako_gguf)

add_executable(mako_run mako_run.cpp)
target_link_libraries(mako_run PRIVATE mako_glm)

if(MAKO_REF_SRC AND MAKO_REF_BUILD)
    add_subdirectory(ref_dump)
endif()

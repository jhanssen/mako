add_executable(gguf_dump gguf_dump.cpp)
target_link_libraries(gguf_dump PRIVATE mako_gguf)

add_executable(tokenize tokenize.cpp)
target_link_libraries(tokenize PRIVATE mako_text)

add_executable(mako_run mako_run.cpp)
target_link_libraries(mako_run PRIVATE mako_glm mako_text)

add_executable(ffn_check ffn_check.cpp)
target_link_libraries(ffn_check PRIVATE mako_glm)

if(MAKO_REF_SRC AND MAKO_REF_BUILD)
    add_subdirectory(ref_dump)
endif()

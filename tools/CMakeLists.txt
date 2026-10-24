add_executable(gguf_dump gguf_dump.cpp)
target_link_libraries(gguf_dump PRIVATE mako_gguf)

add_executable(tokenize tokenize.cpp)
target_link_libraries(tokenize PRIVATE mako_text)

add_executable(mako_run mako_run.cpp)
target_link_libraries(mako_run PRIVATE mako_glm mako_text)

add_executable(mako_calibrate mako_calibrate.cpp)
target_link_libraries(mako_calibrate PRIVATE mako_glm mako_text)
set_source_files_properties(mako_calibrate.cpp PROPERTIES
    COMPILE_OPTIONS "--embed-dir=${PROJECT_SOURCE_DIR}/data;-Wno-c23-extensions"
    OBJECT_DEPENDS "${PROJECT_SOURCE_DIR}/data/calibration.txt")

add_executable(logit_compare logit_compare.cpp)

add_executable(ffn_check ffn_check.cpp)
target_link_libraries(ffn_check PRIVATE mako_glm)

if(MAKO_REF_SRC AND MAKO_REF_BUILD)
    add_subdirectory(ref_dump)
endif()

add_executable(mako_serve mako_serve.cpp)
target_link_libraries(mako_serve PRIVATE mako_server)

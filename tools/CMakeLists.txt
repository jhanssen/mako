add_executable(gguf_dump gguf_dump.cpp)
target_link_libraries(gguf_dump PRIVATE mako_gguf)

add_executable(tokenize tokenize.cpp)
target_link_libraries(tokenize PRIVATE mako_text)

add_executable(mako_run mako_run.cpp)
target_link_libraries(mako_run PRIVATE mako_glm mako_text mako_app)

add_executable(mako_calibrate mako_calibrate.cpp)
target_link_libraries(mako_calibrate PRIVATE mako_glm mako_dsv41 mako_text)
set_source_files_properties(mako_calibrate.cpp PROPERTIES
    COMPILE_OPTIONS "--embed-dir=${PROJECT_SOURCE_DIR}/data;-Wno-c23-extensions"
    OBJECT_DEPENDS "${PROJECT_SOURCE_DIR}/data/calibration.txt")

add_executable(logit_compare logit_compare.cpp)

if(MAKO_REF_SRC AND MAKO_REF_BUILD)
    add_subdirectory(ref_dump)
endif()

add_executable(mako_serve mako_serve.cpp)
target_link_libraries(mako_serve PRIVATE mako_server mako_dsv41 mako_app)

add_executable(prefill_sweep prefill_sweep.cpp)
target_link_libraries(prefill_sweep PRIVATE mako_glm mako_text mako_app)
target_compile_definitions(prefill_sweep PRIVATE MAKO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

add_executable(render_template render_template.cpp)
target_link_libraries(render_template PRIVATE mako_chat mako_gguf)

add_executable(engram_rows engram_rows.cpp)
target_link_libraries(engram_rows PRIVATE mako_dsv41)

add_executable(ds_run ds_run.cpp)
target_link_libraries(ds_run PRIVATE mako_dsv41 mako_gen mako_app)

add_executable(vision_run vision_run.cpp)
target_link_libraries(vision_run PRIVATE mako_vision mako_mtl)

add_executable(vision_chat vision_chat.cpp)
target_link_libraries(vision_chat PRIVATE mako_vision mako_glm mako_dsv41 mako_text mako_chat mako_gen)

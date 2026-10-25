add_executable(gemv_test gemv_test.cpp)
target_link_libraries(gemv_test PRIVATE mako_glm)

add_executable(tokenizer_test tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE mako_text)

add_executable(chat_test chat_test.cpp)
target_link_libraries(chat_test PRIVATE mako_chat mako_text)

add_executable(server_test server_test.cpp)
target_link_libraries(server_test PRIVATE mako_server glaze_headers)

add_executable(template_cases_test template_cases_test.cpp)
target_link_libraries(template_cases_test PRIVATE mako_chat mako_gguf)

add_executable(sampler_test sampler_test.cpp)
target_link_libraries(sampler_test PRIVATE mako_gen)

add_executable(gemm_test gemm_test.cpp)
target_link_libraries(gemm_test PRIVATE mako_mtl)

add_executable(regress_test regress_test.cpp)
target_link_libraries(regress_test PRIVATE mako_glm mako_text)
target_compile_definitions(regress_test PRIVATE MAKO_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data"
                                                MAKO_CALIBRATION="${CMAKE_SOURCE_DIR}/data/calibration.txt")

# ctest -L kernels | tokenizer | sampler | server | e2e; every test exits 77 (skipped) without MAKO_MODEL.
add_test(NAME tokenizer COMMAND tokenizer_test "${MAKO_MODEL}" ${CMAKE_CURRENT_SOURCE_DIR}/data/glm_tokenizer_cases.bin
                                ${CMAKE_CURRENT_SOURCE_DIR}/data/glm_tokenizer_expected.txt)
# DeepSeek-V4.1's tokenizer on the same cases (expected ids from the reference tokenizer).
add_test(NAME tokenizer_ds41 COMMAND tokenizer_test "${MAKO_DS41_MODEL}" ${CMAKE_CURRENT_SOURCE_DIR}/data/glm_tokenizer_cases.bin
                                     ${CMAKE_CURRENT_SOURCE_DIR}/data/ds41_tokenizer_expected.txt)
add_test(NAME template_ds41 COMMAND template_cases_test "${MAKO_DS41_MODEL}" ${CMAKE_CURRENT_SOURCE_DIR}/data/ds41_template)
add_test(NAME gemv COMMAND gemv_test "${MAKO_MODEL}")
add_test(NAME gemm COMMAND gemm_test "${MAKO_MODEL}")
foreach(c decode prefill parity spec session ablate long)
    add_test(NAME e2e_${c} COMMAND regress_test "${MAKO_MODEL}" ${c})
    set_tests_properties(e2e_${c} PROPERTIES LABELS e2e RESOURCE_LOCK gpu_model)
endforeach()
add_test(NAME sampler COMMAND sampler_test)
add_test(NAME server COMMAND server_test "${MAKO_MODEL}")
add_test(NAME chat COMMAND chat_test "${MAKO_MODEL}" ${CMAKE_CURRENT_SOURCE_DIR}/data/glm_chat_cases.bin)
set_tests_properties(tokenizer tokenizer_ds41 template_ds41 chat PROPERTIES LABELS tokenizer)
set_tests_properties(server PROPERTIES LABELS server)
set_tests_properties(sampler PROPERTIES LABELS sampler)
set_tests_properties(gemv gemm PROPERTIES LABELS kernels RESOURCE_LOCK gpu_model)
set_tests_properties(tokenizer tokenizer_ds41 template_ds41 chat server gemv gemm e2e_decode e2e_prefill e2e_parity e2e_spec e2e_session e2e_ablate e2e_long PROPERTIES SKIP_RETURN_CODE 77)

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

add_executable(dsv41_test dsv41_test.cpp)
target_link_libraries(dsv41_test PRIVATE mako_dsv41)

add_executable(engram_test engram_test.cpp)
target_link_libraries(engram_test PRIVATE mako_dsv41)

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
# DeepSeek-V4.1 engine checks (ablation sites and invariants; MAKO_DS41_ABLATION adds a real file's prefill check).
add_test(NAME dsv41 COMMAND dsv41_test "${MAKO_DS41_MODEL}" "${MAKO_DS41_ENGRAM}" ${MAKO_DS41_ABLATION})
set_tests_properties(dsv41 PROPERTIES LABELS e2e RESOURCE_LOCK gpu_model SKIP_RETURN_CODE 77)
# Engram hashing and row decoding (expected rows from the reference hasher, values decoded independently).
add_test(NAME engram_ds41 COMMAND engram_test "${MAKO_DS41_MODEL}" "${MAKO_DS41_ENGRAM}"
                                  ${CMAKE_CURRENT_SOURCE_DIR}/data/ds41_engram_expected.txt)
add_test(NAME gemv COMMAND gemv_test "${MAKO_MODEL}")
add_test(NAME gemm COMMAND gemm_test "${MAKO_MODEL}")
# DeepSeek-V4.1's expert formats (IQ3_S down projections, IQ2_S gate) against the GEMV.
add_test(NAME gemm_ds41 COMMAND gemm_test "${MAKO_DS41_MODEL}" blk.0.ffn_down_exps.weight blk.0.ffn_gate_exps.weight)
foreach(c decode prefill parity spec session ablate long)
    add_test(NAME e2e_${c} COMMAND regress_test "${MAKO_MODEL}" ${c})
    set_tests_properties(e2e_${c} PROPERTIES LABELS e2e RESOURCE_LOCK gpu_model)
endforeach()
add_test(NAME sampler COMMAND sampler_test)
add_test(NAME server COMMAND server_test "${MAKO_MODEL}")
add_test(NAME server_ds41 COMMAND server_test "${MAKO_DS41_MODEL}")
add_test(NAME chat COMMAND chat_test "${MAKO_MODEL}" ${CMAKE_CURRENT_SOURCE_DIR}/data/glm_chat_cases.bin)
set_tests_properties(tokenizer tokenizer_ds41 template_ds41 chat PROPERTIES LABELS tokenizer)
set_tests_properties(engram_ds41 PROPERTIES LABELS dsv41)
set_tests_properties(server server_ds41 PROPERTIES LABELS server)
set_tests_properties(sampler PROPERTIES LABELS sampler)
set_tests_properties(gemv gemm gemm_ds41 PROPERTIES LABELS kernels RESOURCE_LOCK gpu_model)
set_tests_properties(tokenizer tokenizer_ds41 template_ds41 engram_ds41 gemm_ds41 chat server server_ds41 gemv gemm e2e_decode e2e_prefill e2e_parity e2e_spec e2e_session e2e_ablate e2e_long PROPERTIES SKIP_RETURN_CODE 77)

add_executable(quant_test quant_test.cpp)
target_link_libraries(quant_test PRIVATE mako_mtl mako_gguf)
add_test(NAME quant COMMAND quant_test)

add_executable(presets_test presets_test.cpp)
target_link_libraries(presets_test PRIVATE mako_app)
add_test(NAME presets COMMAND presets_test ${CMAKE_CURRENT_BINARY_DIR})

add_executable(gemv_own_test gemv_own_test.cpp)
target_link_libraries(gemv_own_test PRIVATE mako_mtl)
add_test(NAME gemv_own COMMAND gemv_own_test)

# DeepSeek-V4.1 against the reference engine's logits (tests/data/ds41_*): short prose, a 24K-token prompt's tail, the
# miss penalty, dense Q6_K, and the drafter (MAKO_DS41_DSPARK).
add_executable(ds41_ref_test ds41_ref_test.cpp)
target_link_libraries(ds41_ref_test PRIVATE mako_dsv41 mako_gen)
target_compile_definitions(ds41_ref_test PRIVATE MAKO_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
foreach(m short long penalty q6k dspark)
    add_test(NAME ds41_${m} COMMAND ds41_ref_test "${MAKO_DS41_MODEL}" "${MAKO_DS41_ENGRAM}" ${m} "${MAKO_DS41_DSPARK}")
    set_tests_properties(ds41_${m} PROPERTIES LABELS e2e RESOURCE_LOCK gpu_model SKIP_RETURN_CODE 77)
endforeach()

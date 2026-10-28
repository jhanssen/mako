add_executable(gemv_test gemv_test.cpp)
target_link_libraries(gemv_test PRIVATE mako_glm)

add_executable(tokenizer_test tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE mako_text)

add_executable(chat_test chat_test.cpp)
target_link_libraries(chat_test PRIVATE mako_chat mako_text)

add_executable(server_test server_test.cpp)
target_link_libraries(server_test PRIVATE mako_server glaze_headers)

add_executable(server_http_test server_http_test.cpp)
target_link_libraries(server_http_test PRIVATE mako_server httplib_headers)

add_executable(template_cases_test template_cases_test.cpp)
target_link_libraries(template_cases_test PRIVATE mako_chat mako_gguf)

add_executable(dsv41_test dsv41_test.cpp)
target_link_libraries(dsv41_test PRIVATE mako_dsv41)

add_executable(engram_test engram_test.cpp)
target_link_libraries(engram_test PRIVATE mako_dsv41)

add_executable(template_test template_test.cpp)
target_link_libraries(template_test PRIVATE mako_chat)

add_executable(sampler_test sampler_test.cpp)
target_link_libraries(sampler_test PRIVATE mako_gen)

add_executable(loop_guard_test loop_guard_test.cpp)
target_link_libraries(loop_guard_test PRIVATE mako_gen)

add_executable(vision_prep_test vision_prep_test.cpp)
target_link_libraries(vision_prep_test PRIVATE mako_vision)

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
# The Engram tables: MAKO_DS41_ENGRAM, else the model's own shards.
if(MAKO_DS41_ENGRAM)
    set(DS41_ENGRAM "${MAKO_DS41_ENGRAM}")
else()
    set(DS41_ENGRAM "${MAKO_DS41_MODEL}")
endif()
# DeepSeek-V4.1 engine checks (ablation sites and invariants; MAKO_DS41_ABLATION adds a real file's prefill check).
add_test(NAME dsv41 COMMAND dsv41_test "${MAKO_DS41_MODEL}" "${DS41_ENGRAM}" ${MAKO_DS41_ABLATION})
set_tests_properties(dsv41 PROPERTIES LABELS e2e RESOURCE_LOCK gpu_model SKIP_RETURN_CODE 77)
# Engram hashing and row decoding (expected rows from the reference hasher, values decoded independently).
add_test(NAME engram_ds41 COMMAND engram_test "${MAKO_DS41_MODEL}" "${DS41_ENGRAM}"
                                  ${CMAKE_CURRENT_SOURCE_DIR}/data/ds41_engram_expected.txt)
add_test(NAME gemv COMMAND gemv_test "${MAKO_MODEL}")
add_test(NAME gemm COMMAND gemm_test "${MAKO_MODEL}")
add_test(NAME gemm_synthetic COMMAND gemm_test --synthetic)
set_tests_properties(gemm_synthetic PROPERTIES LABELS kernels RESOURCE_LOCK gpu_model)
# DeepSeek-V4.1's expert formats (IQ3_S down projections, IQ2_S gate) against the GEMV.
add_test(NAME gemm_ds41 COMMAND gemm_test "${MAKO_DS41_MODEL}" blk.0.ffn_down_exps.weight blk.0.ffn_gate_exps.weight)
foreach(c decode prefill parity spec session pool ablate long)
    add_test(NAME e2e_${c} COMMAND regress_test "${MAKO_MODEL}" ${c})
    set_tests_properties(e2e_${c} PROPERTIES LABELS e2e RESOURCE_LOCK gpu_model SKIP_RETURN_CODE 77)
endforeach()
add_test(NAME sampler COMMAND sampler_test)
add_test(NAME loop_guard COMMAND loop_guard_test)
add_test(NAME template COMMAND template_test)
add_test(NAME vision_prep COMMAND vision_prep_test ${CMAKE_CURRENT_SOURCE_DIR}/data/vision)
add_test(NAME server COMMAND server_test "${MAKO_MODEL}")
add_test(NAME server_ds41 COMMAND server_test "${MAKO_DS41_MODEL}")
add_test(NAME server_http COMMAND server_http_test "${MAKO_MODEL}")
add_test(NAME chat COMMAND chat_test "${MAKO_MODEL}" ${CMAKE_CURRENT_SOURCE_DIR}/data/glm_chat_cases.bin)
set_tests_properties(tokenizer tokenizer_ds41 template_ds41 chat PROPERTIES LABELS tokenizer)
set_tests_properties(engram_ds41 PROPERTIES LABELS dsv41)
set_tests_properties(server server_ds41 server_http PROPERTIES LABELS server)
# cpu: no GPU and no model (what CI runs on GitHub's macOS runners, which have no usable Metal device).
set_tests_properties(sampler PROPERTIES LABELS "sampler;cpu")
set_tests_properties(loop_guard PROPERTIES LABELS "sampler;cpu")
set_tests_properties(template PROPERTIES LABELS "tokenizer;cpu")
set_tests_properties(vision_prep PROPERTIES LABELS "vision;cpu")
set_tests_properties(gemv gemm gemm_ds41 PROPERTIES LABELS kernels RESOURCE_LOCK gpu_model)
set_tests_properties(tokenizer tokenizer_ds41 template_ds41 engram_ds41 gemm_ds41 chat server server_ds41 server_http gemv gemm PROPERTIES SKIP_RETURN_CODE 77)

add_executable(quant_test quant_test.cpp)
target_link_libraries(quant_test PRIVATE mako_mtl mako_gguf)
add_test(NAME quant COMMAND quant_test)
set_tests_properties(quant PROPERTIES LABELS kernels)

add_executable(presets_test presets_test.cpp)
target_link_libraries(presets_test PRIVATE mako_app)
add_test(NAME presets COMMAND presets_test ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(presets PROPERTIES LABELS cpu)

add_executable(cli_test cli_test.cpp)
target_link_libraries(cli_test PRIVATE mako_app)
add_test(NAME cli COMMAND cli_test ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(cli PROPERTIES LABELS cpu)

add_executable(engram_decode_test engram_decode_test.cpp)
target_link_libraries(engram_decode_test PRIVATE mako_mtl mako_gguf)
add_test(NAME engram_decode COMMAND engram_decode_test)
set_tests_properties(engram_decode PROPERTIES LABELS kernels)
add_executable(gemv_own_test gemv_own_test.cpp)
target_link_libraries(gemv_own_test PRIVATE mako_mtl)
add_test(NAME gemv_own COMMAND gemv_own_test)
add_executable(gemv_rows_test gemv_rows_test.cpp)
target_link_libraries(gemv_rows_test PRIVATE mako_mtl)
add_test(NAME gemv_rows COMMAND gemv_rows_test)
set_tests_properties(gemv_own gemv_rows PROPERTIES LABELS kernels)
add_executable(hc_test hc_test.cpp)
target_link_libraries(hc_test PRIVATE mako_mtl)
add_test(NAME hc COMMAND hc_test)
set_tests_properties(hc PROPERTIES LABELS kernels)

# DeepSeek-V4.1 against the reference engine's logits (tests/data/ds41_*): short prose, a 24K-token prompt's tail, the
# miss penalty, dense Q6_K, and the drafter (MAKO_DS41_DSPARK).
add_executable(ds41_ref_test ds41_ref_test.cpp)
target_link_libraries(ds41_ref_test PRIVATE mako_dsv41 mako_gen)
target_compile_definitions(ds41_ref_test PRIVATE MAKO_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
foreach(m short long penalty q6k dspark pool)
    add_test(NAME ds41_${m} COMMAND ds41_ref_test "${MAKO_DS41_MODEL}" "${DS41_ENGRAM}" ${m} "${MAKO_DS41_DSPARK}")
    set_tests_properties(ds41_${m} PROPERTIES LABELS e2e RESOURCE_LOCK gpu_model SKIP_RETURN_CODE 77)
endforeach()

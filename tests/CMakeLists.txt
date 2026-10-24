add_executable(gemv_test gemv_test.cpp)
target_link_libraries(gemv_test PRIVATE mako_glm)

add_executable(tokenizer_test tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE mako_text)

add_executable(gemm_test gemm_test.cpp)
target_link_libraries(gemm_test PRIVATE mako_mtl)

add_executable(regress_test regress_test.cpp)
target_link_libraries(regress_test PRIVATE mako_glm mako_text)
target_compile_definitions(regress_test PRIVATE MAKO_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data"
                                                MAKO_CALIBRATION="${CMAKE_SOURCE_DIR}/data/calibration.txt")

# ctest -L kernels | tokenizer | e2e; every test exits 77 (skipped) without MAKO_MODEL.
add_test(NAME tokenizer COMMAND tokenizer_test "${MAKO_MODEL}" ${CMAKE_CURRENT_SOURCE_DIR}/data/glm_tokenizer_cases.bin
                                ${CMAKE_CURRENT_SOURCE_DIR}/data/glm_tokenizer_expected.txt)
add_test(NAME gemv COMMAND gemv_test "${MAKO_MODEL}")
add_test(NAME gemm COMMAND gemm_test "${MAKO_MODEL}")
foreach(c decode prefill parity spec long)
    add_test(NAME e2e_${c} COMMAND regress_test "${MAKO_MODEL}" ${c})
    set_tests_properties(e2e_${c} PROPERTIES LABELS e2e RESOURCE_LOCK gpu_model)
endforeach()
set_tests_properties(tokenizer PROPERTIES LABELS tokenizer)
set_tests_properties(gemv gemm PROPERTIES LABELS kernels RESOURCE_LOCK gpu_model)
set_tests_properties(tokenizer gemv gemm e2e_decode e2e_prefill e2e_parity e2e_spec e2e_long PROPERTIES SKIP_RETURN_CODE 77)

add_executable(gemv_test gemv_test.cpp)
target_link_libraries(gemv_test PRIVATE mako_glm)

add_executable(tokenizer_test tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE mako_text)

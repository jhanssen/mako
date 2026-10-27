# Third-party Jinja engine (MIT) for the GGUF chat templates, with the JSON library it wraps. Its sources include a
# header for their assert macro only; a two-line stand-in is generated for it.
set(JINJA_DIR ${CMAKE_BINARY_DIR}/third_party/jinja) # from cmake/llama_cpp_sources.cmake
file(WRITE ${JINJA_DIR}/ggml.h "#pragma once\n#include <cstdlib>\n#define GGML_ASSERT(x) do { if (!(x)) std::abort(); } while (0)\n")

add_library(jinja_engine STATIC
    ${JINJA_DIR}/jinja/lexer.cpp ${JINJA_DIR}/jinja/parser.cpp ${JINJA_DIR}/jinja/runtime.cpp
    ${JINJA_DIR}/jinja/string.cpp ${JINJA_DIR}/jinja/value.cpp ${JINJA_DIR}/json.cpp ${JINJA_DIR}/unicode.cpp)
target_include_directories(jinja_engine SYSTEM PUBLIC ${JINJA_DIR})
target_compile_options(jinja_engine PRIVATE -w)

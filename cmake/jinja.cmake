# Third-party Jinja engine (MIT) for the GGUF chat templates. Its sources include a header for their assert macro only;
# a two-line stand-in is generated for it. The JSON value it reads its context from (json.h) is mako's own
# (src/chat/json), found ahead of the engine's directory.
set(JINJA_DIR ${CMAKE_BINARY_DIR}/third_party/jinja) # from cmake/llama_cpp_sources.cmake
file(WRITE ${JINJA_DIR}/ggml.h "#pragma once\n#include <cstdlib>\n#define GGML_ASSERT(x) do { if (!(x)) std::abort(); } while (0)\n")

add_library(jinja_engine STATIC
    ${JINJA_DIR}/jinja/lexer.cpp ${JINJA_DIR}/jinja/parser.cpp ${JINJA_DIR}/jinja/runtime.cpp
    ${JINJA_DIR}/jinja/string.cpp ${JINJA_DIR}/jinja/value.cpp ${JINJA_DIR}/unicode.cpp)
add_library(jinja_json STATIC ${CMAKE_SOURCE_DIR}/src/chat/json/json.cpp)
target_include_directories(jinja_json PUBLIC ${CMAKE_SOURCE_DIR}/src/chat/json)
target_link_libraries(jinja_engine PUBLIC jinja_json)
target_include_directories(jinja_engine SYSTEM PUBLIC ${JINJA_DIR})
target_compile_options(jinja_engine PRIVATE -w)

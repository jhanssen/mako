# Third-party Jinja engine (MIT) for the GGUF chat templates. Its sources include a header for their assert macro only;
# a two-line stand-in is generated for it. The JSON value it reads its context from (json.h) is mako's own
# (src/chat/json), found ahead of the engine's directory.
set(JINJA_DIR ${CMAKE_BINARY_DIR}/third_party/jinja) # fetched by mako-common (cmake/jinja_files.cmake)
file(WRITE ${JINJA_DIR}/ggml.h "#pragma once\n#include <cstdlib>\n#define GGML_ASSERT(x) do { if (!(x)) std::abort(); } while (0)\n")

# tojson's text counts as the context's (where control tokens are not recognized): templates render tool definitions
# and tool call arguments through it. Built from a patched copy; the original keeps its hash.
file(READ ${JINJA_DIR}/jinja/value.cpp _value_cpp)
set(_tojson_ret "    return mk_val<value_string>(json_str);\n}")
string(FIND "${_value_cpp}" "${_tojson_ret}" _at)
if(_at EQUAL -1)
    message(FATAL_ERROR "jinja value.cpp: tojson's return not found; update the patch in cmake/jinja.cmake")
endif()
string(REPLACE "${_tojson_ret}"
    "    auto out = mk_val<value_string>(json_str);\n    out->mark_input();\n    return out;\n}"
    _value_cpp "${_value_cpp}")
set(_value_patched ${JINJA_DIR}/jinja/value_input_json.cpp)
unset(_have)
if(EXISTS ${_value_patched})
    file(READ ${_value_patched} _have)
endif()
if(NOT _have STREQUAL _value_cpp)
    file(WRITE ${_value_patched} "${_value_cpp}")
endif()

add_library(jinja_engine STATIC
    ${JINJA_DIR}/jinja/lexer.cpp ${JINJA_DIR}/jinja/parser.cpp ${JINJA_DIR}/jinja/runtime.cpp
    ${JINJA_DIR}/jinja/string.cpp ${_value_patched} ${JINJA_DIR}/unicode.cpp)
add_library(jinja_json STATIC ${CMAKE_SOURCE_DIR}/src/chat/json/json.cpp)
target_include_directories(jinja_json PUBLIC ${CMAKE_SOURCE_DIR}/src/chat/json)
target_link_libraries(jinja_engine PUBLIC jinja_json)
target_include_directories(jinja_engine SYSTEM PUBLIC ${JINJA_DIR})
target_compile_options(jinja_engine PRIVATE -w)

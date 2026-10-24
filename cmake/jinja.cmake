# Third-party Jinja engine (MIT) for the GGUF chat templates, pinned by commit and hash, with the JSON library it
# wraps. Its sources include a header for their assert macro only; a two-line stand-in is generated for it.
set(JINJA_COMMIT d327bf1f0780f87b8c01b367b61853b0c291d7c4)
set(JINJA_DIR ${CMAKE_BINARY_DIR}/third_party/jinja)
set(_base https://raw.githubusercontent.com/ggml-org/llama.cpp/${JINJA_COMMIT})
set(_files
    "common/jinja/lexer.h|93aecd72cc952fab08f657aa2a70e9f58421672b19d3179f8bdf555e683a1cc7|jinja"
    "common/jinja/lexer.cpp|ca5674297a3788d6893cf2e83812bce7e32354800ab65e34ee78d334a28892c1|jinja"
    "common/jinja/parser.h|6225eb759da910f3dd7753aed71c6938131cfa04c609d744e6f1390396422316|jinja"
    "common/jinja/parser.cpp|5a1da297892cf62e9e412c954e0a9e74401d0df13c7acc113887b6d081e8a62d|jinja"
    "common/jinja/runtime.h|c816c0e75d9b7f9f8e11c3280285c81086563026b02ed74767adf4448fbfbbab|jinja"
    "common/jinja/runtime.cpp|d846f033f16185d24b733233cf61aca296e312f8a08a7765bd38b30e963b8524|jinja"
    "common/jinja/string.h|d32ad2a319398b8dbc1cc43b39dd10ab2eb0ce2eae9d0eaf3c2162777dd615dc|jinja"
    "common/jinja/string.cpp|6b1b13dea3f94c4d542c15250f793abd638839733fd96a9da6704bfdab662e86|jinja"
    "common/jinja/value.h|6fae65f5eb4b1c07bf2e4a900799982fd40bba8d9db8333aba564463fca553b5|jinja"
    "common/jinja/value.cpp|a3272e014ec80951b6b5603750078c399f935c0c75175e43178ae7fee749f40c|jinja"
    "common/jinja/utils.h|61ae8bc57a7a551bdfb294a23a0ef8b5ba8e3cf4d695389f0f0797e2d7689781|jinja"
    "common/json.h|a352f31307f1be2bdb34c027b876f42a1c5d1ea775cc8e4f11d703b16c40d22f|."
    "common/json.cpp|d8c556445cef8e7c93b0857f6ca7ae7ebd8295f169403ac66641b72f6b0598f2|."
    "common/unicode.h|219b9257ac567a3e6bcf0f552c9a9ddc01d46a15eb4d2829ae5e069e643f0ec5|."
    "common/unicode.cpp|191c00abab788e59e83e29fb877047fcbac1f4d6088c5e9962d38ac2e694686c|."
    "vendor/nlohmann/json.hpp|08f19bbcc66ff3082f5caf58867b1232987dba98d82e2b29d870722fce42776b|nlohmann"
    "LICENSE|94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d|.")
foreach(entry ${_files})
    string(REPLACE "|" ";" parts ${entry})
    list(GET parts 0 path)
    list(GET parts 1 hash)
    list(GET parts 2 sub)
    get_filename_component(name ${path} NAME)
    file(DOWNLOAD ${_base}/${path} ${JINJA_DIR}/${sub}/${name} EXPECTED_HASH SHA256=${hash})
endforeach()
file(WRITE ${JINJA_DIR}/ggml.h "#pragma once\n#include <cstdlib>\n#define GGML_ASSERT(x) do { if (!(x)) std::abort(); } while (0)\n")

add_library(jinja_engine STATIC
    ${JINJA_DIR}/jinja/lexer.cpp ${JINJA_DIR}/jinja/parser.cpp ${JINJA_DIR}/jinja/runtime.cpp
    ${JINJA_DIR}/jinja/string.cpp ${JINJA_DIR}/jinja/value.cpp ${JINJA_DIR}/json.cpp ${JINJA_DIR}/unicode.cpp)
target_include_directories(jinja_engine SYSTEM PUBLIC ${JINJA_DIR})
target_compile_options(jinja_engine PRIVATE -w)

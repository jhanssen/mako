# The Jinja engine's files from the llama.cpp release mako-common takes its GEMV kernels from (path | SHA-256 |
# destination under third_party/), fetched by mako-common's cmake/llama_cpp_sources.cmake.
set(MAKO_LLAMA_CPP_EXTRA_FILES
    "common/jinja/lexer.h|93aecd72cc952fab08f657aa2a70e9f58421672b19d3179f8bdf555e683a1cc7|jinja/jinja"
    "common/jinja/lexer.cpp|ca5674297a3788d6893cf2e83812bce7e32354800ab65e34ee78d334a28892c1|jinja/jinja"
    "common/jinja/parser.h|6225eb759da910f3dd7753aed71c6938131cfa04c609d744e6f1390396422316|jinja/jinja"
    "common/jinja/parser.cpp|5a1da297892cf62e9e412c954e0a9e74401d0df13c7acc113887b6d081e8a62d|jinja/jinja"
    "common/jinja/runtime.h|c816c0e75d9b7f9f8e11c3280285c81086563026b02ed74767adf4448fbfbbab|jinja/jinja"
    "common/jinja/runtime.cpp|d846f033f16185d24b733233cf61aca296e312f8a08a7765bd38b30e963b8524|jinja/jinja"
    "common/jinja/string.h|d32ad2a319398b8dbc1cc43b39dd10ab2eb0ce2eae9d0eaf3c2162777dd615dc|jinja/jinja"
    "common/jinja/string.cpp|6b1b13dea3f94c4d542c15250f793abd638839733fd96a9da6704bfdab662e86|jinja/jinja"
    "common/jinja/value.h|6fae65f5eb4b1c07bf2e4a900799982fd40bba8d9db8333aba564463fca553b5|jinja/jinja"
    "common/jinja/value.cpp|a3272e014ec80951b6b5603750078c399f935c0c75175e43178ae7fee749f40c|jinja/jinja"
    "common/jinja/utils.h|61ae8bc57a7a551bdfb294a23a0ef8b5ba8e3cf4d695389f0f0797e2d7689781|jinja/jinja"
    "common/unicode.h|219b9257ac567a3e6bcf0f552c9a9ddc01d46a15eb4d2829ae5e069e643f0ec5|jinja"
    "common/unicode.cpp|191c00abab788e59e83e29fb877047fcbac1f4d6088c5e9962d38ac2e694686c|jinja"
    "LICENSE|94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d|jinja")

# Writes ${OUT} with the source tree's commit; run at every build, rewrites only on change.
execute_process(COMMAND git rev-parse --short=12 HEAD WORKING_DIRECTORY ${SRC}
                OUTPUT_VARIABLE commit OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    set(commit "unknown")
else()
    execute_process(COMMAND git diff --quiet HEAD WORKING_DIRECTORY ${SRC} RESULT_VARIABLE dirty)
    if(NOT dirty EQUAL 0)
        string(APPEND commit "-dirty")
    endif()
endif()
file(CONFIGURE OUTPUT ${OUT} CONTENT "#pragma once\n#define MAKO_COMMIT \"${commit}\"\n")

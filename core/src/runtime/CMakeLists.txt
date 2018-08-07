set(runtime_inc
    ${CORE_INCLUDE_DIR}/hiveRuntime.h
    ${CORE_INCLUDE_DIR}/hiveServer.h
    ${CORE_INCLUDE_DIR}/hiveRemoteFunctions.h
    ${CORE_INCLUDE_DIR}/hiveEdtFunctions.h
    ${CORE_INCLUDE_DIR}/hiveDbFunctions.h
    ${CORE_INCLUDE_DIR}/hiveDbList.h
    ${CORE_INCLUDE_DIR}/hiveArrayDb.h
    ${CORE_INCLUDE_DIR}/hiveTerminationDetection.h
)

set(runtime_src
    hiveRuntime.c
    hiveServer.c
    hiveRemoteFunctions.c
    hiveEdtFunctions.c
    hiveDbFunctions.c
    hiveDbList.c
    hiveArrayDb.c
    hiveTerminationDetection.c
    shadAdapter.c
)

add_library(runtime OBJECT ${runtime_inc} ${runtime_src})
install(FILES ${runtime_inc} DESTINATION include)

set(runtime_inc
    ${CORE_INCLUDE_DIR}/artsRuntime.h
    ${CORE_INCLUDE_DIR}/artsServer.h
    ${CORE_INCLUDE_DIR}/artsRemoteFunctions.h
    ${CORE_INCLUDE_DIR}/artsEdtFunctions.h
    ${CORE_INCLUDE_DIR}/artsDbFunctions.h
    ${CORE_INCLUDE_DIR}/artsDbList.h
    ${CORE_INCLUDE_DIR}/artsArrayDb.h
    ${CORE_INCLUDE_DIR}/artsTerminationDetection.h
)

set(runtime_src
    artsRuntime.c
    artsServer.c
    artsRemoteFunctions.c
    artsEdtFunctions.c
    artsEventFunctions.c
    artsDbFunctions.c
    artsDbList.c
    artsArrayDb.c
    artsTerminationDetection.c
    shadAdapter.c
)

add_library(runtime OBJECT ${runtime_inc} ${runtime_src})
install(FILES ${runtime_inc} DESTINATION include)
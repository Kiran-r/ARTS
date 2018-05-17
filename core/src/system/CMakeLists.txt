set(system_inc
    ${CORE_INCLUDE_DIR}/hiveAbstractMachineModel.h
    ${CORE_INCLUDE_DIR}/hiveAtomics.h
    ${CORE_INCLUDE_DIR}/hiveMalloc.h
    ${CORE_INCLUDE_DIR}/hiveDebug.h
    ${CORE_INCLUDE_DIR}/hiveTimer.h
    ${CORE_INCLUDE_DIR}/hiveThreads.h
)

set(system_src
    hiveAbstractMachineModel.c
    hiveAtomics.c
    hiveMallocStandard.c
    hiveDebugFunctions.c
    hivePrintf.c
    hiveTimer.c
    hivePthreadLayer.c
)

add_library(system OBJECT ${system_src} )
if(HWLOC)
set(LIBS hive ${Hwloc_LIBRARIES})
target_link_libraries(system ${LIBS})
endif()
install(FILES ${system_inc} DESTINATION include)

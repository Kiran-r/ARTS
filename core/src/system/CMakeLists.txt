set(system_inc
    ${CORE_INCLUDE_DIR}/artsAbstractMachineModel.h
    ${CORE_INCLUDE_DIR}/artsAtomics.h
    ${CORE_INCLUDE_DIR}/artsDebug.h
    ${CORE_INCLUDE_DIR}/artsThreads.h
)

set(system_src
    artsAbstractMachineModel.c
    artsAtomics.c
    artsMallocStandard.c
    artsDebugFunctions.c
    artsPrintf.c
    artsTimer.c
    artsPthreadLayer.c
    hive_tMT.c
)

add_library(system OBJECT ${system_src} )
if(HWLOC)
set(LIBS arts ${Hwloc_LIBRARIES})
target_link_libraries(system ${LIBS})
endif()
install(FILES ${system_inc} DESTINATION include)

if(DEQUE MATCHES "chase_lev")
    set(common_lib_src ${common_lib_src} chaseLevDequeSimple.c)
elseif( DEQUE MATCHES "priority" )
    set(common_lib_src ${common_lib_src} chaseLevDequePriority.c)
else()
    message( FATAL_ERROR "Unsupported runtime deque chosen.")
endif()

set(common_inc
    ${CORE_INCLUDE_DIR}/artsDeque.h
    ${CORE_INCLUDE_DIR}/artsQueue.h
    ${CORE_INCLUDE_DIR}/artsHash.h
    ${CORE_INCLUDE_DIR}/artsArrayList.h
    ${CORE_INCLUDE_DIR}/artsTable.h
    ${CORE_INCLUDE_DIR}/artsLinkList.h
)

set(common_src
    ${common_lib_src}
    hashLock.c
    table.c
    linkList.c
    artsArrayList.c
    artsQueue.c
)

add_library(common OBJECT ${common_inc} ${common_src})
install(FILES ${common_inc} DESTINATION include)
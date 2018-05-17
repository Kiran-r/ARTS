if(DEQUE MATCHES "chase_lev")
    set(common_lib_src ${common_lib_src} chaseLevDequeSimple.c)
elseif( DEQUE MATCHES "priority" )
    set(common_lib_src ${common_lib_src} chaseLevDequePriority.c)
else()
    message( FATAL_ERROR "Unsupported runtime deque chosen.")
endif()

set(common_inc
    ${CORE_INCLUDE_DIR}/hiveDeque.h
    ${CORE_INCLUDE_DIR}/hiveQueue.h
    ${CORE_INCLUDE_DIR}/hiveHash.h
    ${CORE_INCLUDE_DIR}/hiveArrayList.h
    ${CORE_INCLUDE_DIR}/hiveTable.h
    ${CORE_INCLUDE_DIR}/hiveLinkList.h
)

set(common_src
    ${common_lib_src}
    hashLock.c
    table.c
    linkList.c
    hiveArrayList.c
    hiveQueue.c
)

add_library(common OBJECT ${common_inc} ${common_src})
install(FILES ${common_inc} DESTINATION include)
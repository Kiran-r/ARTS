set(gas_inc
    ${CORE_INCLUDE_DIR}/hiveGuid.h
    ${CORE_INCLUDE_DIR}/hiveOutOfOrder.h
    ${CORE_INCLUDE_DIR}/hiveOutOfOrderList.h
    ${CORE_INCLUDE_DIR}/hiveRouteTable.h
)

set(gas_src
    hiveGuid.c
    hiveOutOfOrder.c
    hiveOutOfOrderList.c
    hiveRouteTable.c
)

add_library(gas OBJECT ${gas_inc} ${gas_src} )
install(FILES ${gas_inc} DESTINATION include)
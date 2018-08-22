set(gas_inc
    ${CORE_INCLUDE_DIR}/artsGuid.h
    ${CORE_INCLUDE_DIR}/artsOutOfOrder.h
    ${CORE_INCLUDE_DIR}/artsOutOfOrderList.h
    ${CORE_INCLUDE_DIR}/artsRouteTable.h
)

set(gas_src
    artsGuid.c
    artsOutOfOrder.c
    artsOutOfOrderList.c
    artsRouteTable.c
)

add_library(gas OBJECT ${gas_inc} ${gas_src} )
install(FILES ${gas_inc} DESTINATION include)
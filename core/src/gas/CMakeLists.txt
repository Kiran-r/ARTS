set(gas_src
    hiveGuid.c
    hiveOutOfOrder.c
    hiveOutOfOrderList.c
    hiveRouteTable.c
)

add_library(gas OBJECT ${gas_src} )

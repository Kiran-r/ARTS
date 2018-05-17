set(network_inc
    ${CORE_INCLUDE_DIR}/hiveConnection.h
    ${CORE_INCLUDE_DIR}/hiveRemote.h
    ${CORE_INCLUDE_DIR}/hiveRemoteLauncher.h
    ${CORE_INCLUDE_DIR}/hiveRemoteProtocol.h
    ${CORE_INCLUDE_DIR}/hiveSocketServer.h
)

set(network_src
    hiveRemoteLauncherSSH.c
    hiveSocket.c
    hiveRemoteProtocol.c
    hiveSocketServer.c
)

add_library(network OBJECT ${network_inc} ${network_src})
install(FILES ${network_inc} DESTINATION include)

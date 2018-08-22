set(network_inc
    ${CORE_INCLUDE_DIR}/artsConnection.h
    ${CORE_INCLUDE_DIR}/artsRemote.h
    ${CORE_INCLUDE_DIR}/artsRemoteLauncher.h
    ${CORE_INCLUDE_DIR}/artsRemoteProtocol.h
    ${CORE_INCLUDE_DIR}/artsSocketServer.h
)

set(network_src
    artsRemoteLauncherSSH.c
    artsSocket.c
    artsRemoteProtocol.c
    artsSocketServer.c
)

add_library(network OBJECT ${network_inc} ${network_src})
install(FILES ${network_inc} DESTINATION include)

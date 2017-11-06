set(network_src
    hiveRemoteLauncherSSH.c
    hiveSocket.c
    hiveRemoteProtocol.c
    hiveSocketServer.c
)

add_library(network OBJECT ${network_src})


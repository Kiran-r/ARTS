set(runtime_src
    hiveRuntime.c
    hiveServer.c
    hiveRemoteFunctions.c
    hiveEdtFunctions.c
    hiveDbFunctions.c
    hiveDbList.c
)

add_library(runtime OBJECT ${runtime_src})

set(system_src
    hiveAbstractMachineModel.c
    hiveAtomics.c
    hiveMallocStandard.c
    hivePthreadLayer.c
    hiveDebugFunctions.c
    hivePrintf.c
    hiveTimer.c
)

add_library(system OBJECT ${system_src} )

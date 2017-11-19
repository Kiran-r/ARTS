set(system_src
    hiveAbstractMachineModel.c
    hiveAtomics.c
    hiveMallocStandard.c
    hiveDebugFunctions.c
    hivePrintf.c
    hiveTimer.c
    hivePthreadLayer.c
)

add_library(system OBJECT ${system_src} )

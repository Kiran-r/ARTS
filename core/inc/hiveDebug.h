#ifndef HIVEDEBUG_H
#define HIVEDEBUG_H
#ifdef __cplusplus
extern "C" {
#endif

void hiveDebugPrintStack();
void hiveDebugGenerateSegFault();
void hiveTurnOnCoreDumps();
char * getBackTrace(unsigned int skip);
#ifdef __cplusplus
}
#endif

#endif

#ifndef ARTSDEBUG_H
#define ARTSDEBUG_H
#ifdef __cplusplus
extern "C" {
#endif

void artsDebugPrintStack();
void artsDebugGenerateSegFault();
void artsTurnOnCoreDumps();
char * getBackTrace(unsigned int skip);
#ifdef __cplusplus
}
#endif

#endif

#ifndef HIVEDEBUG_H
#define HIVEDEBUG_H

void hiveDebugPrintStack();
void hiveDebugGenerateSegFault();
void hiveTurnOnCoreDumps();
char * getBackTrace(unsigned int skip);

#endif

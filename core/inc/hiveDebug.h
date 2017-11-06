#ifndef HIVEDEBUG_H
#define HIVEDEBUG_H

void hiveDebugPrintStack();
void hiveDebugGenerateSegFault();
char * getBackTrace(unsigned int skip);

#endif

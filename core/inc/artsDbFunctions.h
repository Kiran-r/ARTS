#ifndef ARTSDBFUNCTIONS_H
#define	ARTSDBFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"

void artsDbCreateInternal(artsGuid_t guid, void *addr, uint64_t size, uint64_t packetSize, artsType_t mode);    
void acquireDbs(struct artsEdt * edt);
void releaseDbs(unsigned int depc, artsEdtDep_t * depv);
bool artsAddDbDuplicate(struct artsDb * db, unsigned int rank, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void prepDbs(unsigned int depc, artsEdtDep_t * depv);
void internalPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epochGuid, unsigned int rank);

#ifdef __cplusplus
}
#endif
#endif	/* artsDBFUNCTIONS_H */


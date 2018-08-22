#ifndef ARTSDBFUNCTIONS_H
#define	ARTSDBFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
    
void artsDbCreateInternal(artsGuid_t guid, void *addr, u64 size, u64 packetSize, artsType_t mode);    
    
artsGuid_t artsDbCreate(void **addr, u64 size, artsType_t mode);
void * artsDbCreateWithGuid(artsGuid_t guid, u64 size);
void * artsDbCreateWithGuidAndData(artsGuid_t guid, void * data, u64 size);
artsGuid_t artsDbCreateRemote(unsigned int route, u64 size, artsType_t mode);

void * artsDbResize(artsGuid_t guid, unsigned int size, bool copy);
void artsDbMove(artsGuid_t dbGuid, unsigned int rank);

void artsDbDestroy(artsGuid_t guid);
void artsDbDestroySafe(artsGuid_t guid, bool remote);

void acquireDbs(struct artsEdt * edt);
void releaseDbs(unsigned int depc, artsEdtDep_t * depv);

bool artsAddDbDuplicate(struct artsDb * db, unsigned int rank, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void prepDbs(unsigned int depc, artsEdtDep_t * depv);

void internalPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epochGuid, unsigned int rank);
void artsPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void artsPutInDbAt(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);
void artsPutInDbEpoch(void * ptr, artsGuid_t epochGuid, artsGuid_t dbGuid, unsigned int offset, unsigned int size);

void artsGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void artsGetFromDbAt(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);

#ifdef __cplusplus
}
#endif
#endif	/* artsDBFUNCTIONS_H */


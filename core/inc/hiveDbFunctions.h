#ifndef HIVEDBFUNCTIONS_H
#define	HIVEDBFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
    
hiveGuid_t hiveDbCreate(void **addr, u64 size, hiveDbAccessMode_t mode);
void * hiveDbCreateWithGuid(hiveGuid_t guid, u64 size, hiveDbAccessMode_t mode);
void * hiveDbCreateWithGuidAndData(hiveGuid_t guid, void * data, u64 size, hiveDbAccessMode_t mode);
hiveGuid_t hiveDbCreateRemote(unsigned int route, u64 size, hiveDbAccessMode_t mode);

void * hiveDbResize(hiveGuid_t guid, unsigned int size, bool copy);
void hiveDbMove(hiveGuid_t dbGuid, unsigned int rank);

void hiveDbDestroy(hiveGuid_t guid);
void hiveDbDestroySafe(hiveGuid_t guid, bool remote);

void acquireDbs(struct hiveEdt * edt);
void releaseDbs(unsigned int depc, hiveEdtDep_t * depv);

bool hiveAddDbDuplicate(struct hiveDb * db, unsigned int rank, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode);
void prepDbs(unsigned int depc, hiveEdtDep_t * depv);

void internalPutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, hiveGuid_t epochGuid, unsigned int rank);
void hivePutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void hivePutInDbAt(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);
void hivePutInDbEpoch(void * ptr, hiveGuid_t epochGuid, hiveGuid_t dbGuid, unsigned int offset, unsigned int size);

void hiveGetFromDb(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void hiveGetFromDbAt(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);

#ifdef __cplusplus
}
#endif
#endif	/* hiveDBFUNCTIONS_H */


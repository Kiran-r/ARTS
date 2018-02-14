#ifndef HIVEDBFUNCTIONS_H
#define	HIVEDBFUNCTIONS_H

hiveGuid_t hiveDbCreateRemote(unsigned int route, u64 size, bool pin);
hiveGuid_t hiveDbCreate(void **addr, u64 size, bool pin);
void * hiveDbCreateWithGuid(hiveGuid_t guid, u64 size, bool pin);
void * hiveDbResize(hiveGuid_t guid, unsigned int size, bool copy);
void hiveDbDestroy(hiveGuid_t guid);
void hiveDbDestroySafe(hiveGuid_t guid, bool remote);
hiveGuid_t hiveDbAssignGuid(void * ptr);
hiveGuid_t hiveDbAssignGuidForRank(void * ptr, unsigned int rank);
hiveGuid_t hiveDbReassignGuid(hiveGuid_t guid);
void hiveDbCleanExt(hiveGuid_t guid, bool removeLocal);
void hiveDbCleanLocalOnlyExt(hiveGuid_t guid);
void acquireDbs(struct hiveEdt * edt);
void checkIfLocalDbIsOutOfDate(unsigned int slot, void ** ptr, hiveGuid_t guid, unsigned int mode, struct hiveEdt * edt);
void releaseDbs(unsigned int depc, hiveEdtDep_t * depv);
bool hiveAddDbDuplicate(struct hiveDb * db, unsigned int rank, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode);
void prepDbs(unsigned int depc, hiveEdtDep_t * depv);
void hiveGetFromDb(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void * hiveDbCreateWithGuidAndData(hiveGuid_t guid, void * data, u64 size, bool pin);
void hivePutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void hivePutInDbEpoch(void * ptr, hiveGuid_t epochGuid, hiveGuid_t dbGuid, unsigned int offset, unsigned int size);
void internalPutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, hiveGuid_t epoch);
#endif	/* hiveDBFUNCTIONS_H */


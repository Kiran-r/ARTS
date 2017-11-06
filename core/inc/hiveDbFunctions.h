#ifndef HIVEDBFUNCTIONS_H
#define	HIVEDBFUNCTIONS_H

hiveGuid_t hiveDbCreate(void **addr, u64 size);
void * hiveDbCreateWithGuid(hiveGuid_t guid, u64 size);
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

#endif	/* hiveDBFUNCTIONS_H */


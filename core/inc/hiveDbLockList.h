#ifndef HIVEDBLOCKLIST_H
#define HIVEDBLOCKLIST_H
#ifdef __cplusplus
extern "C" {
#endif
struct hiveDbLockList;
struct hiveDbLockList * hiveDbLockListGroupNew(unsigned int listSize);
void hiveDbLockListNew(struct hiveDbLockList *list);
struct hiveDbLockList * hiveDbLockListGet(struct hiveDbLockList *linkList, unsigned int position);
void hiveDbLockListDelete(void *linkList);
bool hiveDbLockListPush(struct hiveDbLockList *list, void *item, unsigned int rank, bool shared);
void ** hiveDbLockListPop( struct hiveDbLockList * list, unsigned int ** rankList, unsigned int * size );
void * hiveDbLockListPeek( struct hiveDbLockList * list );
void hiveDbLockListDeleteItem( void * item );
#ifdef __cplusplus
}
#endif

#endif

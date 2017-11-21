#ifndef HIVEROUTETABLE_H
#define HIVEROUTETABLE_H

#include "hive.h"
#include "hiveOutOfOrderList.h"

struct hiveRouteInvalidate
{
    int size;
    int used;
    struct hiveRouteInvalidate * next;
    unsigned int data[];
};

typedef enum
{
    noKey = 0,
    anyKey,
    deletedKey,   //deleted only
    allocatedKey, //reserved, available, or requested
    availableKey, //available only 
    requestedKey, //available but reserved (means so one else has the valid copy)
    reservedKey,  //reserved only      
} itemState;

struct hiveRouteItem;
struct hiveRouteTable;

struct hiveRouteTable *hiveRouteTableListNew(unsigned int listSize, unsigned int routeTableSize, unsigned int shift);
struct hiveRouteTable *hiveRouteTableListGetRouteTable(struct hiveRouteTable * routeTableList, unsigned int position);
void hiveRouteTableListDelete(struct hiveRouteTable *routeTableList);
//void hiveRouteTableNew(struct hiveRouteTable *routeTable, unsigned int size, unsigned int shift, unsigned int func);
void hiveRouteTableDelete(struct hiveRouteTable *routeTable);
void * hiveRouteTableAddItem(void* item, hiveGuid_t key, unsigned int route, bool used);
bool hiveRouteTableAddItemRace(void * item, hiveGuid_t key, unsigned int route, bool used);
bool hiveRouteTableAddItemAtomic(void * item, hiveGuid_t key, unsigned int route);
bool hiveRouteTableDeleteItem(hiveGuid_t key);
bool hiveRouteTableRemoveItem(hiveGuid_t key);
bool hiveRouteTableInvalidateItem(hiveGuid_t key);
void * hiveRouteTableLookupItem(hiveGuid_t key);
void * hiveRouteTableLookupInvalidItem(hiveGuid_t key);
int hiveRouteTableLookupRank(hiveGuid_t key);
bool hiveRouteTableUpdateItem(hiveGuid_t key, void * data, unsigned int rank, itemState state);
struct hiveDbFrontierIterator * hiveRouteTableGetRankDuplicates(hiveGuid_t key, unsigned int rank);
bool hiveRouteTableAddSent(hiveGuid_t key, void * edt, unsigned int slot, bool aggregate);
bool hiveRouteTableAddOO(hiveGuid_t key, void * data, unsigned int rank );
void hiveRouteTableFireOO(hiveGuid_t key, void (*callback)(void *, void*) );
void hiveRouteTableFireSent(hiveGuid_t key, void (*callback)(void *, void*) );
unsigned int hiveRouteTablePopEw(hiveGuid_t key );
bool hiveRouteTablePushEw(hiveGuid_t key, unsigned int rank );
void hiveRouteTableAddRankDuplicate(hiveGuid_t key, unsigned int rank);
void hiveRouteTableResetSent(hiveGuid_t key);
void hiveRouteTableResetOO(hiveGuid_t key);
bool hiveRouteTableClearItem(hiveGuid_t key);
void * hiveRouteTableCreateLocalEntry( struct hiveRouteTable * routeTable, void * item, unsigned int rank );
bool hiveRouteTableLockGuid(hiveGuid_t key);
bool hiveRouteTableLockGuidRace(hiveGuid_t key, unsigned int rank);
itemState hiveRouteTableLookupItemWithState(hiveGuid_t key, void *** data, itemState min, bool inc);
itemState getItemState(struct hiveRouteItem * item);
bool hiveRouteTableReturnDb(hiveGuid_t key, bool markToDelete);
void * hiveRouteTableLookupDb(hiveGuid_t key, int * rank);
int hiveRouteTableSetRank(hiveGuid_t key, int rank);
void ** hiveRouteTableGetOOList(hiveGuid_t key, struct hiveOutOfOrderList ** list);
void hiveRouteTableDecItem(hiveGuid_t key, void * data);
void ** hiveRouteTableReserve(hiveGuid_t key, bool * dec, itemState * state);

#endif

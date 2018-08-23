#ifndef ARTSROUTETABLE_H
#define ARTSROUTETABLE_H
#ifdef __cplusplus
extern "C" {
#endif

#include "artsOutOfOrderList.h"

struct artsRouteInvalidate
{
    int size;
    int used;
    struct artsRouteInvalidate * next;
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

struct artsRouteItem;
struct artsRouteTable;

struct artsRouteTable *artsRouteTableListNew(unsigned int listSize, unsigned int routeTableSize, unsigned int shift);
struct artsRouteTable *artsRouteTableListGetRouteTable(struct artsRouteTable * routeTableList, unsigned int position);
void artsRouteTableListDelete(struct artsRouteTable *routeTableList);
//void artsRouteTableNew(struct artsRouteTable *routeTable, unsigned int size, unsigned int shift, unsigned int func);
void artsRouteTableDelete(struct artsRouteTable *routeTable);
void * artsRouteTableAddItem(void* item, artsGuid_t key, unsigned int route, bool used);
bool artsRouteTableAddItemRace(void * item, artsGuid_t key, unsigned int route, bool used);
bool artsRouteTableAddItemAtomic(void * item, artsGuid_t key, unsigned int route);
bool artsRouteTableDeleteItem(artsGuid_t key);
bool artsRouteTableRemoveItem(artsGuid_t key);
bool artsRouteTableInvalidateItem(artsGuid_t key);
void * artsRouteTableLookupItem(artsGuid_t key);
void * artsRouteTableLookupInvalidItem(artsGuid_t key);
int artsRouteTableLookupRank(artsGuid_t key);
bool artsRouteTableUpdateItem(artsGuid_t key, void * data, unsigned int rank, itemState state);
struct artsDbFrontierIterator * artsRouteTableGetRankDuplicates(artsGuid_t key, unsigned int rank);
bool artsRouteTableAddSent(artsGuid_t key, void * edt, unsigned int slot, bool aggregate);
bool artsRouteTableAddOO(artsGuid_t key, void * data);
void artsRouteTableFireOO(artsGuid_t key, void (*callback)(void *, void*) );
void artsRouteTableFireSent(artsGuid_t key, void (*callback)(void *, void*) );
unsigned int artsRouteTablePopEw(artsGuid_t key );
bool artsRouteTablePushEw(artsGuid_t key, unsigned int rank );
void artsRouteTableAddRankDuplicate(artsGuid_t key, unsigned int rank);
void artsRouteTableResetSent(artsGuid_t key);
void artsRouteTableResetOO(artsGuid_t key);
bool artsRouteTableClearItem(artsGuid_t key);
void * artsRouteTableCreateLocalEntry( struct artsRouteTable * routeTable, void * item, unsigned int rank );
bool artsRouteTableLockGuid(artsGuid_t key);
bool artsRouteTableLockGuidRace(artsGuid_t key, unsigned int rank);
itemState artsRouteTableLookupItemWithState(artsGuid_t key, void *** data, itemState min, bool inc);
itemState getItemState(struct artsRouteItem * item);
bool artsRouteTableReturnDb(artsGuid_t key, bool markToDelete);
void * artsRouteTableLookupDb(artsGuid_t key, int * rank);
int artsRouteTableSetRank(artsGuid_t key, int rank);
void ** artsRouteTableGetOOList(artsGuid_t key, struct artsOutOfOrderList ** list);
void artsRouteTableDecItem(artsGuid_t key, void * data);
void ** artsRouteTableReserve(artsGuid_t key, bool * dec, itemState * state);
#ifdef __cplusplus
}
#endif

#endif

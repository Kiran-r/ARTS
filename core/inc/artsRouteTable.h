/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,* 
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
#ifndef ARTSROUTETABLE_H
#define ARTSROUTETABLE_H
#ifdef __cplusplus
extern "C" {
#endif

#include "artsOutOfOrderList.h"

//These are for the lock for each item in the RT
#define reservedItem  0x8000000000000000
#define availableItem 0x4000000000000000
#define deleteItem    0x2000000000000000
#define statusMask    (reservedItem | availableItem | deleteItem)

#define maxItem       0x1FFFFFFFFFFFFFFF
#define countMask     ~(reservedItem | availableItem | deleteItem)
#define checkMaxItem(x) (((x & countMask) + 1) < maxItem)
#define getCount(x)   (x & countMask)

#define isDel(x)       ( x & deleteItem )
#define isRes(x)       ( (x & reservedItem ) && !(x & availableItem) && !(x & deleteItem) )
#define isAvail(x)     ( (x & availableItem) && !(x & reservedItem ) && !(x & deleteItem) )
#define isReq(x)  ( (x & reservedItem ) &&  (x & availableItem) && !(x & deleteItem) )

#define shouldDelete(x) (isDel(x) && !getCount(x))

#define collisionResolves 8

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

struct artsRouteItem
{
    artsGuid_t key;
    void * data;
    volatile uint64_t lock;
    unsigned int rank;
    struct artsOutOfOrderList ooList;
} __attribute__ ((aligned));

typedef void (*setRouteItem_t) (struct artsRouteItem * item, void * data);
typedef void (*freeRouteItem_t) (struct artsRouteItem * item);

//Add padding around locks...
struct artsRouteTable
{
    struct artsRouteItem * data;
    unsigned int size;
    unsigned int shift;
    struct artsRouteTable * next;
    volatile unsigned readerLock;
    volatile unsigned writerLock;
    setRouteItem_t setFunc;
    freeRouteItem_t freeFunc;
} __attribute__ ((aligned));

typedef struct {
    uint64_t index;
    struct artsRouteTable * table;
} artsRouteTableIterator;

bool decItem(struct artsRouteTable * routeTable, struct artsRouteItem * item);

struct artsRouteTable * artsNewRouteTable(unsigned int routeTableSize, unsigned int shift);

void * artsRouteTableAddItem(void* item, artsGuid_t key, unsigned int route, bool used);
bool internalRouteTableAddItemRace(struct artsRouteTable * routeTable, void * item, artsGuid_t key, unsigned int rank, bool usedRes, bool usedAvail);
bool artsRouteTableAddItemRace(void * item, artsGuid_t key, unsigned int route, bool used);

void * artsRouteTableLookupItem(artsGuid_t key);
int artsRouteTableLookupRank(artsGuid_t key);
bool artsRouteTableRemoveItem(artsGuid_t key);
bool artsRouteTableInvalidateItem(artsGuid_t key);

struct artsRouteItem * artsRouteTableSearchForKey(struct artsRouteTable *routeTable, artsGuid_t key, itemState state);
bool artsRouteTableUpdateItem(artsGuid_t key, void * data, unsigned int rank, itemState state);
struct artsDbFrontierIterator * artsRouteTableGetRankDuplicates(artsGuid_t key, unsigned int rank);
bool artsRouteTableAddSent(artsGuid_t key, void * edt, unsigned int slot, bool aggregate);
void artsRouteTableAddRankDuplicate(artsGuid_t key, unsigned int rank);

itemState artsRouteTableLookupItemWithState(artsGuid_t key, void *** data, itemState min, bool inc);
itemState getItemState(struct artsRouteItem * item);

int artsRouteTableSetRank(artsGuid_t key, int rank);

void ** artsRouteTableReserve(artsGuid_t key, bool * dec, itemState * state);

void artsRouteTableDecItem(artsGuid_t key, void * data);
struct artsRouteItem * getItemFromData(artsGuid_t key, void * data);

void * internalRouteTableLookupDb(struct artsRouteTable * routeTable, artsGuid_t key, int * rank);
void * artsRouteTableLookupDb(artsGuid_t key, int * rank);
bool internalRouteTableReturnDb(struct artsRouteTable * routeTable, artsGuid_t key, bool markToDelete, bool doDelete);
bool artsRouteTableReturnDb(artsGuid_t key, bool markToDelete);

bool artsRouteTableAddOO(artsGuid_t key, void * data, bool inc);
void artsRouteTableFireOO(artsGuid_t key, void (*callback)(void *, void*));
void artsRouteTableResetOO(artsGuid_t key);
void ** artsRouteTableGetOOList(artsGuid_t key, struct artsOutOfOrderList ** list);

artsRouteTableIterator * artsNewRouteTableIterator(struct artsRouteTable * table);
void artsResetRouteTableIterator(artsRouteTableIterator * iter, struct artsRouteTable * table);
struct artsRouteItem * artsRouteTableIterate(artsRouteTableIterator * iter);
void artsPrintItem(struct artsRouteItem * item);

uint64_t artsCleanUpRouteTable(struct artsRouteTable * routeTable, uint64_t sizeToClean, bool cleanZeros);

#ifdef __cplusplus
}
#endif

#endif

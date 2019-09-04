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

#include "artsRouteTable.h"
#include "artsAtomics.h"
#include "artsOutOfOrder.h"
#include "artsGuid.h"
#include "artsGlobals.h"
#include "artsDbList.h"
#include "artsDebug.h"
#include "artsCounter.h"
#ifdef USE_GPU
#include "artsGpuStream.h"
#endif

#define DPRINTF
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

#define collisionResolves 8
#define initInvalidateSize 128
#define guidLockSize 1024
volatile unsigned int guidLock[guidLockSize] = {0};


void freeItem(struct artsRouteItem * item, unsigned int gpu)
{
#ifdef USE_GPU
    if (gpu)
        artsGpuFree(item->data, gpu-1);
    else
#endif
        artsFree(item->data);
    artsOutOfOrderListDelete(&item->ooList);
    item->data = NULL;
    item->key = 0;
    item->lock = 0;
}

bool markReserve(struct artsRouteItem * item, bool markUse)
{
    if(markUse)
    {
        uint64_t mask = reservedItem + 1;
        return !artsAtomicCswapU64(&item->lock, 0, mask);
    }
    else
        return !artsAtomicFetchOrU64(&item->lock, reservedItem);
}

bool markRequested(struct artsRouteItem * item)
{
    uint64_t local, temp;
    while(1)
    {
        local = item->lock;
        if((local & reservedItem) || (local & deleteItem))
            return false;
        else
        {
            temp = local | reservedItem;
            if(local == artsAtomicCswapU64(&item->lock, local, temp))
                    return true;
        }
    }
}

bool markWrite(struct artsRouteItem * item)
{
    uint64_t local, temp;
    while(1)
    {
        local = item->lock;
        if(local & reservedItem)
        {
            temp = (local & ~reservedItem) | availableItem;
            if(local == artsAtomicCswapU64(&item->lock, local, temp))
                return true;
        }
        else
            return false;
    }
}

bool markDelete(struct artsRouteItem * item)
{
    uint64_t res = artsAtomicFetchOrU64(&item->lock, deleteItem);
    return (res & deleteItem) != 0;
}

inline void printState(struct artsRouteItem * item)
{
    if(item)
    {
        uint64_t local = item->lock;
        if(isReq(local))
            PRINTF("%lu: reserved-available %p\n", item->key, local);
        else if(isRes(local))
            PRINTF("%lu: reserved\n", item->key, local);
        else if(isAvail(local))
            PRINTF("%lu: available %p\n", item->key, local);
        else if(isDel(local))
            PRINTF("%lu: deleted %p\n", item->key, local);
    }
    else
        PRINTF("NULL ITEM\n");
}

//11000 & 11100 = 11000, 10000 & 11100 = 10000, 11100 & 11100 = 11000
bool checkItemState(struct artsRouteItem * item, itemState state)
{
    if(item)
    {
        uint64_t local = item->lock;
        switch(state)
        {
            case reservedKey:
                return isRes(local);

            case requestedKey:
                return isReq(local);

            case availableKey:
                return isAvail(local);

            case allocatedKey:
                return isRes(local) || isAvail(local) || isReq(local);

            case deletedKey:
                return isDel(local);

            case anyKey:
                return local!=0;

            default:
                return false;
        }
    }
    return false;
}

inline bool checkMinItemState(struct artsRouteItem * item, itemState state)
{
    if(item)
    {
        uint64_t local = item->lock;
        itemState actualState = noKey;

        if(isDel(local))
            actualState = deletedKey;

        else if(isRes(local))
            actualState = reservedKey;

        else if(isReq(local))
            actualState = requestedKey;

        else if(isAvail(local))
            actualState = availableKey;

        return (actualState && actualState>=state);
    }
    return false;
}

itemState getItemState(struct artsRouteItem * item)
{
    if(item)
    {
        uint64_t local = item->lock;

        if(isRes(local))
            return reservedKey;
        if(isAvail(local))
            return availableKey;

        if(isReq(local))
            return requestedKey;

        if(isDel(local))
            return deletedKey;
    }
    return noKey;
}

bool incItem(struct artsRouteItem * item, unsigned int count)
{
    while(1)
    {
        uint64_t local = item->lock;
        if(!(local & deleteItem) && checkMaxItem(local))
        {
            if(local == artsAtomicCswapU64(&item->lock, local, local + count))
            {
                return true;
            }
        }
        else
            break;
    }
    return false;
}

bool decItem(struct artsRouteItem * item, unsigned int gpu)
{
   uint64_t local = item->lock;
   if(getCount(local) == 0)
       artsDebugGenerateSegFault();
    local = artsAtomicSubU64(&item->lock, 1);
    if(shouldDelete(local))
    {
        freeItem(item, gpu);
        return true;
    }
    return false;
}

void readerTableLock(struct artsRouteTable *  table)
{
    while(1)
    {
        while(table->writerLock);
        artsAtomicFetchAdd(&table->readerLock, 1U);
        if(table->writerLock==0)
            break;
        artsAtomicSub(&table->readerLock, 1U);
    }
}

void readerTableUnlock(struct artsRouteTable *  table)
{
    artsAtomicSub(&table->readerLock, 1U);
}

inline void writerTableLock(struct artsRouteTable *  table)
{
    while(artsAtomicCswap(&table->writerLock, 0U, 1U) == 0U);
    while(table->readerLock);
    return;
}

bool writerTryTableLock(struct artsRouteTable *  table)
{
    if(artsAtomicCswap(&table->writerLock, 0U, 1U) == 0U)
    {
        while(table->readerLock);
        return true;
    }
    return false;
}

void writeTableUnlock(struct artsRouteTable *  table)
{
    artsAtomicSwap(&table->writerLock, 0U);
}

uint64_t urand64()
{
    uint64_t hi = lrand48();
    uint64_t md = lrand48();
    uint64_t lo = lrand48();
    uint64_t res = (hi << 42) + (md << 21) + lo;
    return res;
}

#define hash64(x, y)       ( (uint64_t)(x) * y )

static inline uint64_t getRouteTableKey(uint64_t x, unsigned int shift)
{
    uint64_t hash = 14695981039346656037U;
    switch (shift)
    {
        /*case 5:
            hash *= 31;
        case 6:
            hash *= 61;
        case 7:
            hash *= 127;
        case 8:
            hash *= 251;
        case 9:
            hash *= 509;*/
        case 10:
            hash *= 1021;
        case 11:
            hash *= 2039;
        case 12:
            hash *= 4093;
        case 13:
            hash *= 8191;
        case 14:
            hash *= 16381;
        case 15:
            hash *= 32749;
        case 16:
            hash *= 65521;
        case 17:
            hash *= 131071;
        case 18:
            hash *= 262139;
        case 19:
            hash *= 524287;
        case 20:
            hash *= 1048573;
        case 21:
            hash *= 2097143;
        case 22:
            hash *= 4194301;
        case 31:
            hash *= 2147483647;
        case 32:
            hash *= 4294967291;
    }

    return (hash64(x, hash) >> (64-shift))*collisionResolves;
}
extern uint64_t numTables;
extern uint64_t maxGuid;
extern uint64_t keysPerThread;
extern uint64_t minGlobalGuidThread;
extern uint64_t maxGlobalGuidThread;

static inline struct artsRouteTable * artsGetRouteTable(artsGuid_t guid)
{
    artsGuid raw = (artsGuid) guid;
    uint64_t key = raw.fields.key;
    if(keysPerThread)
    {
        uint64_t globalThread = (key / keysPerThread);
        if(minGlobalGuidThread <= globalThread && globalThread < maxGlobalGuidThread)
            return artsNodeInfo.routeTable[globalThread - minGlobalGuidThread];
    }
    return artsNodeInfo.remoteRouteTable;
}

void artsRouteTableNew(struct artsRouteTable * routeTable, unsigned int size, unsigned int shift)
{
    routeTable->data =
        (struct artsRouteItem *) artsCalloc(collisionResolves*size * sizeof (struct artsRouteItem));
    routeTable->size = size;
    routeTable->shift = shift;
}

struct artsRouteTable * artsRouteTableListNew(unsigned int listSize, unsigned int routeTableSize, unsigned int shift)
{
    struct artsRouteTable *routeTableList = (struct artsRouteTable *) artsCalloc(sizeof(struct artsRouteTable) * listSize);
    for (int i = 0; i < listSize; i++)
        artsRouteTableNew(routeTableList + i, routeTableSize, shift);
    return routeTableList;
}

struct artsRouteItem * artsRouteTableSearchForKey(struct artsRouteTable *routeTable, artsGuid_t key, itemState state)
{
    struct artsRouteTable * current = routeTable;
    struct artsRouteTable * next;
    uint64_t keyVal;
    while(current)
    {
        keyVal =  getRouteTableKey((uint64_t)key, current->shift);
        for(int i=0; i<collisionResolves; i++ )
        {
            if(checkItemState(&current->data[keyVal], state))
            {
                if(current->data[keyVal].key == key)
                {
                    return &current->data[keyVal];
                }
            }
            keyVal++;
        }
        readerTableLock(current);
        next = current->next;
        readerTableUnlock(current);
        current = next;
    }
    return NULL;
}

struct artsRouteItem * artsGpuRouteTableSearchForKey(artsGuid_t key, int gpuId)
{
    return artsRouteTableSearchForKey(artsNodeInfo.gpuRouteTable[gpuId], key, availableKey);
}

struct artsRouteItem * artsRouteTableSearchForEmpty(struct artsRouteTable * routeTable, artsGuid_t key, bool markUsed)
{
    struct artsRouteTable * current = routeTable;
    struct artsRouteTable * next;
    uint64_t keyVal;
    while(current != NULL)
    {
        keyVal = getRouteTableKey((uint64_t) key, current->shift);
        for(int i=0; i<collisionResolves; i++)
        {
            if(!current->data[keyVal].lock)
            {
                if(markReserve(&current->data[keyVal], markUsed))
                {
                    current->data[keyVal].key = key;
                    DPRINTF("%lu %p %lu\n", key, current, keyVal);
                    return &current->data[keyVal];
                }
            }
            keyVal++;
        }

        readerTableLock(current);
        next = current->next;
        readerTableUnlock(current);

        if(!next)
        {
            if(writerTryTableLock(current))
            {
                DPRINTF("LS Resize %d %d %p %p %d %ld\n", keyVal, 2*current->size, current, routeTable);
                next = artsCalloc(sizeof(struct artsRouteTable));
                artsRouteTableNew((struct artsRouteTable *)next, 2*current->size, current->shift+1);
                current->next = next;
                writeTableUnlock(current);
            }
            else
            {
                readerTableLock(current);
                next = current->next;
                readerTableUnlock(current);
            }

        }
        current = next;
    }
    PRINTF("Route table search in impossible state: producing a segfault now %p ...", routeTable);
    artsDebugGenerateSegFault();
    return NULL;
}

void * internalRouteTableAddItem(struct artsRouteTable * routeTable, void* item, artsGuid_t key, unsigned int rank, bool used)
{
    struct artsRouteItem * location = artsRouteTableSearchForEmpty(routeTable, key, used);

    location->data = item;
    location->rank = rank;
    markWrite(location);
    return location;
}

void * artsRouteTableAddItem(void* item, artsGuid_t key, unsigned int rank, bool used)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    return internalRouteTableAddItem(routeTable, item, key, rank, used);
}

bool artsRouteTableRemoveItem(artsGuid_t key)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * item = artsRouteTableSearchForKey(routeTable, key, availableKey);
    if(item)
    {
        if(markDelete(item))
        {
            freeItem(item, 0);
        }
    }
    return 0;
}

//This locks the guid so it is useful when multiple people have the guid ahead of time
//The guid doesn't need to be locked if no one knows about it
bool internalRouteTableAddItemRace(struct artsRouteTable * routeTable, void * item, artsGuid_t key, unsigned int rank, bool usedRes, bool usedAvail)
{
    unsigned int pos = (unsigned int)(((uint64_t)key) % (uint64_t)guidLockSize);

    bool ret = false;
    struct artsRouteItem * found = NULL;
    while(!found)
    {
        if(guidLock[pos] == 0)
        {
            if(!artsAtomicCswap(&guidLock[pos], 0U, 1U))
            {
                found = artsRouteTableSearchForKey(routeTable, key, allocatedKey);
                if(found)
                {
                    if(checkItemState(found, reservedKey))
                    {
                        found->data = item;
                        found->rank = rank;
                        markWrite(found);
                        if(usedRes)
                            incItem(found, 1);
                        ret = true;
                    }
                    else if(usedAvail && checkItemState(found, availableKey))
                        incItem(found, 1);
                }
                else
                {
                    found = internalRouteTableAddItem(routeTable, item, key, rank, usedRes);
                    ret = true;
                }
                guidLock[pos] = 0U;
            }
        }
        else
            found = artsRouteTableSearchForKey(routeTable, key, availableKey);
    }
//    PRINTF("found: %lu %p\n", key, found);
    return ret;
}

bool artsRouteTableAddItemRace(void * item, artsGuid_t key, unsigned int rank, bool used)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    return internalRouteTableAddItemRace(routeTable, item, key, rank, used, false);
}

bool artsGpuRouteTableAddItemRace(void * item, artsGuid_t key, unsigned int rank, unsigned int gpuId)
{
    struct artsRouteTable * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    //For the gpus it should always mark as used
    return internalRouteTableAddItemRace(routeTable, item, key, rank, true, true);
}

//This is used for the send aggregation
bool artsRouteTableReserveItemRace(artsGuid_t key, struct artsRouteItem ** item, bool used)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    unsigned int pos = (unsigned int)(((uint64_t)key) % (uint64_t)guidLockSize);
    bool ret = false;
    *item = NULL;
    while(!(*item))
    {
        if(guidLock[pos] == 0)
        {
            if(!artsAtomicCswap(&guidLock[pos], 0U, 1U))
            {
                *item = artsRouteTableSearchForKey(routeTable, key, allocatedKey);
                if(!(*item))
                {
                    *item = artsRouteTableSearchForEmpty(routeTable, key, used);
                    ret = true;
                    DPRINTF("RES: %lu %p\n", key, routeTable);
                }
                guidLock[pos] = 0U;
            }
        }
        else
        {
            *item = artsRouteTableSearchForKey(routeTable, key, allocatedKey);
        }
    }
//    printState(artsRouteTableSearchForKey(routeTable, key, anyKey));
    return ret;
}

//This does the send aggregation
bool artsRouteTableAddSent(artsGuid_t key, void * edt, unsigned int slot, bool aggregate)
{
    struct artsRouteItem * item = NULL;
    bool sendReq;
    //I shouldn't be able to get to here if the db hasn't already been created
    //and I am the owner node thus item can't be null... or so it should be
    if(artsGuidGetRank(key) == artsGlobalRankId)
    {
        struct artsRouteTable * routeTable = artsGetRouteTable(key);
        item = artsRouteTableSearchForKey(routeTable, key, allocatedKey);
        sendReq = markRequested(item);
    }
    else
    {
        sendReq = artsRouteTableReserveItemRace(key, &item, true);
        if(!sendReq && !incItem(item, 1))
            PRINTF("Item marked for deletion before it has arrived %u...", sendReq);
    }
    artsOutOfOrderHandleDbRequestWithOOList(&item->ooList, &item->data, edt, slot);
    return sendReq || !aggregate;
}

void * artsRouteTableLookupItem(artsGuid_t key)
{
    void * ret = NULL;
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * location = artsRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
        ret = location->data;
    return ret;
}

itemState artsRouteTableLookupItemWithState(artsGuid_t key, void *** data, itemState min, bool inc)
{
    void * ret = NULL;
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * location = artsRouteTableSearchForKey(routeTable, key, min);
    if(location)
    {
        if(inc)
        {
            if(!incItem(location, 1))
            {
                *data = NULL;
                return noKey;
            }
        }
        *data = &location->data;
        return getItemState(location);
    }
    return noKey;
}

uint64_t artsLookupGpuDb(artsGuid_t key)
{
    uint64_t ret = 0;
    for (int i=0; i<artsNodeInfo.gpu; ++i)
    {
        struct artsRouteTable * gpuRouteTable = artsNodeInfo.gpuRouteTable[i];
        struct artsRouteItem * location = artsRouteTableSearchForKey(gpuRouteTable, key, availableKey);
        if(location)
            ret |= 1<<i;
    }
    return ret;
}

void * internalRouteTableLookupDb(struct artsRouteTable * routeTable, artsGuid_t key, int * rank)
{
    *rank = -1;
    void * ret = NULL;
    struct artsRouteItem * location = artsRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
    {
        *rank = location->rank;
        if(incItem(location, 1))
            ret = location->data;
    }
    return ret;
}

void * artsRouteTableLookupDb(artsGuid_t key, int * rank)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    return internalRouteTableLookupDb(routeTable, key, rank);
}

void * artsGpuRouteTableLookupDb(artsGuid_t key, int * rank, int gpuId)
{
    struct artsRouteTable * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    return internalRouteTableLookupDb(routeTable, key, rank);
}

bool internalRouteTableReturnDb(struct artsRouteTable * routeTable, artsGuid_t key, bool markToDelete, bool doDelete, unsigned int gpuId)
{
    struct artsRouteItem * location = artsRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
    {
        //Only mark it for deletion if it is the last one
        //Why make it unusable to other if there is still other
        //tasks that may benifit
        if(markToDelete && (getCount(location->lock) == 1))
        {
            //This should work if there is only one outstanding left... me.  The decItem needs to sub 1 to delete
            uint64_t compVal = availableItem + 1;
            uint64_t newVal = (availableItem | deleteItem) + 1;
            uint64_t oldVal = artsAtomicCswapU64(&location->lock, compVal, newVal);
            //Successfully marked to delete, but we don't want to delete it now
            if(!doDelete && (compVal == oldVal))
                return false;
        }
        return decItem(location, gpuId);
    }
    return false;
}

bool artsRouteTableReturnDb(artsGuid_t key, bool markToDelete)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    bool isRemote = artsGuidGetRank(key) != artsGlobalRankId;
    return internalRouteTableReturnDb(routeTable, key, isRemote, isRemote, 0);
}

bool artsGpuRouteTableReturnDb(artsGuid_t key, bool markToDelete, unsigned int gpuId)
{
    struct artsRouteTable * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    return internalRouteTableReturnDb(routeTable, key, true, false, gpuId+1);
}

int artsRouteTableLookupRank(artsGuid_t key)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * location = artsRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
        return location->rank;
    return -1;
}

int artsRouteTableSetRank(artsGuid_t key, int rank)
{
    int ret = -1;
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * location = artsRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
    {
        ret = location->rank;
        location->rank = rank;
    }
    return ret;
}

void artsRouteTableFireOO(artsGuid_t key, void (*callback)(void *, void*))
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * item = artsRouteTableSearchForKey(routeTable, key, availableKey);
    if(item != NULL)
        artsOutOfOrderListFireCallback(&item->ooList, item->data, callback);
}

bool artsRouteTableAddOO(artsGuid_t key, void * data, bool inc)
{
    struct artsRouteItem * item = NULL;
    if(artsRouteTableReserveItemRace(key, &item, true) || checkItemState(item, reservedKey))
    {
        if(inc)
            incItem(item, 1);
        bool res = artsOutOfOrderListAddItem( &item->ooList, data );
        return res;
    }
    return false;
}

void artsRouteTableResetOO(artsGuid_t key)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * item = artsRouteTableSearchForKey(routeTable, key, anyKey);
    artsOutOfOrderListReset(&item->ooList);
}

void ** artsRouteTableGetOOList(artsGuid_t key, struct artsOutOfOrderList ** list)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * item = artsRouteTableSearchForKey(routeTable, key, availableKey);
    if(item != NULL)
    {
        *list = &item->ooList;
        return &item->data;
    }
}

//This is just a wrapper for outside consumption...
void ** artsRouteTableReserve(artsGuid_t key, bool * dec, itemState *state)
{
    bool res;
    *dec = false;
    struct artsRouteItem * item = NULL;
    while(1)
    {
        res = artsRouteTableReserveItemRace(key, &item, true);
        if(!res)
        {
            //Check to make sure we can use it
            if(incItem(item, 1))
            {
                *dec = true;
                break;
            }
            //If we were not keep trying...
        }
        else //we were successful in reserving
            break;
    }
    if(item)
        *state = getItemState(item);
    return &item->data;
}

struct artsRouteItem * getItemFromData(artsGuid_t key, void * data)
{
    if(data)
    {
        struct artsRouteItem * item = (struct artsRouteItem*)((char*) data - sizeof(artsGuid_t));
        if(key == item->key)
            return item;
    }
    return NULL;
}

void artsRouteTableDecItem(artsGuid_t key, void * data)
{
    if(data)
    {
        decItem(getItemFromData(key, data), 0);
    }
}

artsRouteTableIterator * artsNewRouteTableIterator(struct artsRouteTable * table)
{
    artsRouteTableIterator * ret = (artsRouteTableIterator *) artsCalloc(sizeof(artsRouteTableIterator));
    ret->table = table;
    return ret;
}

void artsResetRouteTableIterator(artsRouteTableIterator * iter, struct artsRouteTable * table)
{
    iter->table = table;
    iter->index = 0;
}

struct artsRouteItem * artsRouteTableIterate(artsRouteTableIterator * iter)
{
    struct artsRouteTable * current = iter->table;
    struct artsRouteTable * next;
    while(current != NULL)
    {
        for(uint64_t i=iter->index; i<current->size*collisionResolves; i++)
        {
            DPRINTF("i: %lu ", i);
            // artsPrintItem(&current->data[i]);
            if(current->data[i].lock)
            {
                iter->index = i+1;
                iter->table = current;
                return &current->data[i];
            }
        }
        iter->index = 0;
        readerTableLock(current);
        next = current->next;
        readerTableUnlock(current);
        current = next;
    }
    return NULL;
}

void artsPrintItem(struct artsRouteItem * item)
{
    if(item)
    {
        uint64_t local = item->lock;
        PRINTF("GUID: %lu DATA: %p RANK: %u LOCK: %p COUNT: %lu Res: %u Req: %u Avail: %u Del: %u\n", 
            item->key, item->data, item->rank, local, getCount(local),
            isRes(local)!=0, isReq(local)!=0, isAvail(local)!=0, isDel(local)!=0);
    }
}

/*This takes three parameters to regulate what is deleted.  This will only clean up DBs!
1.  sizeToClean - this is the desired space to clean up.  The gc will continue untill it
    it reaches this size or it has made a full pass across the RT.  Passing -1 will make the gc
    clean up the entire RT.
2.  cleanZeros - this flag indicates if we should delete data that is not being used by anyone.
    Will delete up to sizeToClean.
3.  gpuId - the id of which GPU this RT belongs.  This is the contiguous id [0 - numGpus-1].
    Pass -1 for a host RT.
Returns the size of the memory freed!
*/ 
uint64_t internalCleanUpRouteTable(struct artsRouteTable * routeTable, uint64_t sizeToClean, bool cleanZeros, int gpuId)
{
    uint64_t freedSize = 0;
    artsRouteTableIterator iter;
    artsResetRouteTableIterator(&iter, routeTable);

    struct artsRouteItem * item = artsRouteTableIterate(&iter);
    while(item && freedSize < sizeToClean)
    {
        artsPrintItem(item);
        artsType_t type = artsGuidGetType(item->key);
        //These are DB types
        if(type > ARTS_BUFFER && type < ARTS_LAST_TYPE)
        {
            struct artsDb * db = item->data;
            uint64_t dbSize = db->header.size;
            
            if(isDel(item->lock))
            {
                if(decItem(item, gpuId+1))
                    freedSize+=dbSize;
            }
            else if(cleanZeros && !getCount(item->lock))
            {
                artsAtomicCswapU64(&item->lock, availableItem, 1+(availableItem | deleteItem));
                if(decItem(item, gpuId+1))
                    freedSize+=dbSize;
            }
        }
        item = artsRouteTableIterate(&iter);
    }
    return freedSize;
}

//See internalCleanUpRouteTable!
uint64_t artsCleanUpGpuRouteTable(unsigned int sizeToClean, bool cleanZeros, unsigned int gpuId)
{
    struct artsRouteTable * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    return internalCleanUpRouteTable(routeTable, sizeToClean, cleanZeros, gpuId);
}

//To cleanup --------------------------------------------------------------------------->

bool artsRouteTableUpdateItem(artsGuid_t key, void * data, unsigned int rank, itemState state)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    bool ret = false;
    struct artsRouteItem * found = NULL;
    while(!found)
    {
        found = artsRouteTableSearchForKey(routeTable, key, state);
        if(found)
        {
            found->data = data;
            found->rank = rank;
            markWrite(found);
            ret = true;
        }
    }
    return ret;
}

bool artsRouteTableInvalidateItem(artsGuid_t key)
{
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * location = artsRouteTableSearchForKey(routeTable, key, allocatedKey);
    if(location)
    {
        markDelete(location);
        if(shouldDelete(location->lock))
        {
            freeItem(location, 0);
            return true;
        }
        DPRINTF("Marked %lu as invalid %lu\n", key, location->lock);
    }
    return false;
}

void artsRouteTableAddRankDuplicate(artsGuid_t key, unsigned int rank)
{

}

struct artsDbFrontierIterator * artsRouteTableGetRankDuplicates(artsGuid_t key, unsigned int rank)
{
    struct artsDbFrontierIterator * iter = NULL;
    struct artsRouteTable * routeTable = artsGetRouteTable(key);
    struct artsRouteItem * location = artsRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
    {
        if(rank != -1)
        {
            //Blocks until the OO is done firing
            artsOutOfOrderListReset(&location->ooList);
            location->rank = rank;
        }
        struct artsDb * db = location->data;
        iter = artsCloseFrontier(db->dbList);
    }
    return iter;
}

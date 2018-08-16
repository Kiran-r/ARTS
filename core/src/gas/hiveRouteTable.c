#include "hive.h"
#include "hiveMalloc.h"
#include "hiveAtomics.h"
#include "hiveOutOfOrderList.h"
#include "hiveOutOfOrder.h"
#include "hiveRouteTable.h"
#include "hiveCounter.h"
#include "hiveGuid.h"
#include "hiveGlobals.h"
#include "hiveDbList.h"
#include "hiveDebug.h"

#define DPRINTF
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

#define collisionResolves 8
#define initInvalidateSize 128
#define guidLockSize 1024
volatile unsigned int guidLock[guidLockSize] = {0};

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

struct hiveRouteItem
{
    hiveGuid_t key;
    void * data;
    volatile u64 lock;
    unsigned int rank;
    struct hiveOutOfOrderList ooList;
} __attribute__ ((aligned));

//Add padding around locks...
struct hiveRouteTable
{
    struct hiveRouteItem * data;
    unsigned int size;
    unsigned int currentSize;
    unsigned int shift;
    u64 func;
    struct hiveRouteTable * next;
    volatile unsigned readerLock;
    volatile unsigned writerLock;
} __attribute__ ((aligned));

void freeItem(struct hiveRouteItem * item)
{
    hiveFree(item->data);
    hiveOutOfOrderListDelete(&item->ooList);
    item->data = NULL;
    item->key = 0;
    item->lock = 0;
}

bool markReserve(struct hiveRouteItem * item, bool markUse)
{
    if(markUse)
    {
        u64 mask = reservedItem + 1;
        return !hiveAtomicCswapU64(&item->lock, 0, mask);
    }
    else
        return !hiveAtomicFetchOrU64(&item->lock, reservedItem);
}

bool markRequested(struct hiveRouteItem * item)
{
    u64 local, temp;
    while(1)
    {
        local = item->lock;
        if((local & reservedItem) || (local & deleteItem))
            return false;
        else
        {
            temp = local | reservedItem;
            if(local == hiveAtomicCswapU64(&item->lock, local, temp))
                    return true;
        }
    }
}

bool markWrite(struct hiveRouteItem * item)
{
    u64 local, temp;
    while(1)
    {
        local = item->lock;
        if(local & reservedItem)
        {
            temp = (local & ~reservedItem) | availableItem;
            if(local == hiveAtomicCswapU64(&item->lock, local, temp))
                return true;
        }
        else
            return false;
    }
}

bool markDelete(struct hiveRouteItem * item)
{
    u64 res = hiveAtomicFetchOrU64(&item->lock, deleteItem);
    return (res & deleteItem) != 0;
}

inline void printState(struct hiveRouteItem * item)
{
    if(item)
    {
        u64 local = item->lock;
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
bool checkItemState(struct hiveRouteItem * item, itemState state)
{
    if(item)
    {
        u64 local = item->lock;
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

inline bool checkMinItemState(struct hiveRouteItem * item, itemState state)
{
    if(item)
    {
        u64 local = item->lock;
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

itemState getItemState(struct hiveRouteItem * item)
{
    if(item)
    {
        u64 local = item->lock;

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

bool incItem(struct hiveRouteItem * item)
{
    while(1)
    {
        u64 local = item->lock;
        if(!(local & deleteItem) && checkMaxItem(local))
        {
            if(local == hiveAtomicCswapU64(&item->lock, local, local + 1))
            {
                return true;
            }
        }
        else
            break;
    }
    return false;
}

bool decItem(struct hiveRouteItem * item)
{
    u64 local = hiveAtomicSubU64(&item->lock, 1);
    if(shouldDelete(local))
    {
        freeItem(item);
        return true;
    }
    return false;
}

void readerTableLock(struct hiveRouteTable *  table)
{
    while(1)
    {
        while(table->writerLock);
        hiveAtomicFetchAdd(&table->readerLock, 1U);
        if(table->writerLock==0)
            break;
        hiveAtomicSub(&table->readerLock, 1U);
    }
}

void readerTableUnlock(struct hiveRouteTable *  table)
{
    hiveAtomicSub(&table->readerLock, 1U);
}

inline void writerTableLock(struct hiveRouteTable *  table)
{
    while(hiveAtomicCswap(&table->writerLock, 0U, 1U) == 0U);
    while(table->readerLock);
    return;
}

bool writerTryTableLock(struct hiveRouteTable *  table)
{
    if(hiveAtomicCswap(&table->writerLock, 0U, 1U) == 0U)
    {
        while(table->readerLock);
        return true;
    }
    return false;
}

void writeTableUnlock(struct hiveRouteTable *  table)
{
    hiveAtomicSwap(&table->writerLock, 0U);
}

uint64_t urand64()
{
    uint64_t hi = lrand48();
    uint64_t md = lrand48();
    uint64_t lo = lrand48();
    uint64_t res = (hi << 42) + (md << 21) + lo;
    return res;
}

#define hash64(x, y)       ( (u64)(x) * y )

static inline u64 getRouteTableKey(u64 x, unsigned int shift, u64 func)
{
    u64 hash = func;
    hash = 14695981039346656037U;
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

static inline struct hiveRouteTable * hiveGetRouteTable(hiveGuid_t guid)
{
    hiveGuid bytes = (hiveGuid)guid;
    if( bytes.fields.local && hiveNodeInfo.routeTable[bytes.fields.thread])
        return  hiveNodeInfo.routeTable[bytes.fields.thread];
    return hiveNodeInfo.remoteRouteTable;
}

void hiveRouteTableNew(struct hiveRouteTable * routeTable, unsigned int size, unsigned int shift, unsigned int func, bool overide)
{
    routeTable->data =
        (struct hiveRouteItem *) hiveCalloc(collisionResolves*size * sizeof (struct hiveRouteItem));
    routeTable->size = size;
    routeTable->shift = shift;
    routeTable->func = func;
}

struct hiveRouteTable * hiveRouteTableListNew(unsigned int listSize, unsigned int routeTableSize, unsigned int shift)
{
    struct hiveRouteTable *routeTableList = (struct hiveRouteTable *) hiveCalloc(sizeof(struct hiveRouteTable) * listSize);
    for (int i = 0; i < listSize; i++)
        hiveRouteTableNew(routeTableList + i, routeTableSize, shift, 0, true);
    return routeTableList;
}

struct hiveRouteItem * hiveRouteTableSearchForKey(struct hiveRouteTable *routeTable, hiveGuid_t key, itemState state)
{
    struct hiveRouteTable * current = routeTable;
    struct hiveRouteTable * next;
    u64 keyVal;
    while(current)
    {
        keyVal =  getRouteTableKey((u64)key, current->shift, current->func);
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

struct hiveRouteItem * hiveRouteTableSearchForEmpty(struct hiveRouteTable * routeTable, hiveGuid_t key, bool markUsed)
{
    struct hiveRouteTable * current = routeTable;
    struct hiveRouteTable * next;
    u64 keyVal;
    while(current != NULL)
    {
        keyVal = getRouteTableKey((u64) key, current->shift, current->func);
        for(int i=0; i<collisionResolves; i++)
        {
            if(!current->data[keyVal].lock)
            {
                if(markReserve(&current->data[keyVal], markUsed))
                {
                    current->data[keyVal].key = key;
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
                DPRINTF("LS Resize %d %d %p %p %d %ld\n", keyVal, 2*current->size, current, routeTable, current->currentSize, current->func);
                next = hiveCalloc(sizeof(struct hiveRouteTable));
                hiveRouteTableNew((struct hiveRouteTable *)next, 2*current->size, current->shift+1, urand64(), false);
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
    hiveDebugGenerateSegFault();
    return NULL;
}

void * hiveRouteTableAddItem(void* item, hiveGuid_t key, unsigned int rank, bool used)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForEmpty(routeTable, key, used);

    location->data = item;
    location->rank = rank;
    markWrite(location);
    return location;
}

bool hiveRouteTableRemoveItem(hiveGuid_t key)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * item = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(item)
    {
        if(markDelete(item))
        {
            freeItem(item);
        }
    }
    return 0;
}

//This locks the guid so it is useful when multiple people have the guid ahead of time
//The guid doesn't need to be locked if no one knows about it
bool hiveRouteTableAddItemRace(void * item, hiveGuid_t key, unsigned int rank, bool used)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    unsigned int pos = (unsigned int)(((u64)key) % (u64)guidLockSize);

    bool ret = false;
    struct hiveRouteItem * found = NULL;
    while(!found)
    {
        if(guidLock[pos] == 0)
        {
            if(!hiveAtomicCswap(&guidLock[pos], 0U, 1U))
            {
                found = hiveRouteTableSearchForKey(routeTable, key, allocatedKey);
                if(found)
                {
                    if(checkItemState(found, reservedKey))
                    {
                        found->data = item;
                        found->rank = rank;
                        markWrite(found);
                        if(used)
                            incItem(found);
                        ret = true;
                    }
                }
                else
                {
                    found = hiveRouteTableAddItem(item, key, rank, used);
                    ret = true;
                }
                guidLock[pos] = 0U;
            }
        }
        else
            found = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    }
//    PRINTF("found: %lu %p\n", key, found);
    return ret;
}

//This is used for the send aggregation
bool hiveRouteTableReserveItemRace(hiveGuid_t key, struct hiveRouteItem ** item, bool used)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    unsigned int pos = (unsigned int)(((u64)key) % (u64)guidLockSize);
    bool ret = false;
    *item = NULL;
    while(!(*item))
    {
        if(guidLock[pos] == 0)
        {
            if(!hiveAtomicCswap(&guidLock[pos], 0U, 1U))
            {
                *item = hiveRouteTableSearchForKey(routeTable, key, allocatedKey);
                if(!(*item))
                {
                    *item = hiveRouteTableSearchForEmpty(routeTable, key, used);
                    ret = true;
                    DPRINTF("RES: %lu %p\n", key, routeTable);
                }
                guidLock[pos] = 0U;
            }
        }
        else
        {
            *item = hiveRouteTableSearchForKey(routeTable, key, allocatedKey);
        }
    }
//    printState(hiveRouteTableSearchForKey(routeTable, key, anyKey));
    return ret;
}

//This does the send aggregation
bool hiveRouteTableAddSent(hiveGuid_t key, void * edt, unsigned int slot, bool aggregate)
{
    struct hiveRouteItem * item = NULL;
    bool sendReq;
    //I shouldn't be able to get to here if the db hasn't already been created
    //and I am the owner node thus item can't be null... or so it should be
    if(hiveGuidGetRank(key) == hiveGlobalRankId)
    {
        struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
        item = hiveRouteTableSearchForKey(routeTable, key, allocatedKey);
        sendReq = markRequested(item);
    }
    else
    {
        sendReq = hiveRouteTableReserveItemRace(key, &item, true);
        if(!sendReq && !incItem(item))
            PRINTF("Item marked for deletion before it has arrived %u...", sendReq);
    }
    hiveOutOfOrderHandleDbRequestWithOOList(&item->ooList, &item->data, edt, slot);
    return sendReq || !aggregate;
}

void * hiveRouteTableLookupItem(hiveGuid_t key)
{
    void * ret = NULL;
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
        ret = location->data;
    return ret;
}

itemState hiveRouteTableLookupItemWithState(hiveGuid_t key, void *** data, itemState min, bool inc)
{
    void * ret = NULL;
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, min);
    if(location)
    {
        if(inc)
        {
            if(!incItem(location))
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

void * hiveRouteTableLookupDb(hiveGuid_t key, int * rank)
{
    *rank = -1;
    void * ret = NULL;
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
    {
        *rank = location->rank;
        if(incItem(location))
            ret = location->data;
    }
    return ret;
}

bool hiveRouteTableReturnDb(hiveGuid_t key, bool markToDelete)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
    {
        if(markToDelete && hiveGuidGetRank(key) != hiveGlobalRankId)
        {
            //Only mark it for deletion if it is the last one
            //Why make it unusable to other if there is still other
            //tasks that may benifit
            if(!getCount(location->lock))
            {
                hiveAtomicCswapU64(&location->lock, availableItem, (availableItem | deleteItem));
            }
        }
        return decItem(location);
    }
    return false;
}

int hiveRouteTableLookupRank(hiveGuid_t key)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
        return location->rank;
    return -1;
}

int hiveRouteTableSetRank(hiveGuid_t key, int rank)
{
    int ret = -1;
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
    {
        ret = location->rank;
        location->rank = rank;
    }
    return ret;
}

void hiveRouteTableFireOO(hiveGuid_t key, void (*callback)(void *, void*))
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * item = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(item != NULL)
        hiveOutOfOrderListFireCallback(&item->ooList, item->data, callback);
}

bool hiveRouteTableAddOO(hiveGuid_t key, void * data)
{
    struct hiveRouteItem * item = NULL;
    if(hiveRouteTableReserveItemRace(key, &item, true) || checkItemState(item, reservedKey))
    {
        bool res = hiveOutOfOrderListAddItem( &item->ooList, data );
        return res;
    }
    return false;
}

void hiveRouteTableResetOO(hiveGuid_t key)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * item = hiveRouteTableSearchForKey(routeTable, key, anyKey);
    hiveOutOfOrderListReset(&item->ooList);
}

void ** hiveRouteTableGetOOList(hiveGuid_t key, struct hiveOutOfOrderList ** list)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * item = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(item != NULL)
    {
        *list = &item->ooList;
        return &item->data;
    }
}

//This is just a wrapper for outside consumption...
void ** hiveRouteTableReserve(hiveGuid_t key, bool * dec, itemState *state)
{
    bool res;
    *dec = false;
    struct hiveRouteItem * item = NULL;
    while(1)
    {
        res = hiveRouteTableReserveItemRace(key, &item, true);
        if(!res)
        {
            //Check to make sure we can use it
            if(incItem(item))
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

struct hiveRouteItem * getItemFromData(hiveGuid_t key, void * data)
{
    struct hiveRouteItem * item = (struct hiveRouteItem*)((char*) data - sizeof(hiveGuid_t));
    if(key == item->key)
        return item;
    return NULL;
}

void hiveRouteTableDecItem(hiveGuid_t key, void * data)
{
    if(data)
    {
        decItem(getItemFromData(key, data));
    }
}

//To cleanup --------------------------------------------------------------------------->

bool hiveRouteTableUpdateItem(hiveGuid_t key, void * data, unsigned int rank, itemState state)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    bool ret = false;
    struct hiveRouteItem * found = NULL;
    while(!found)
    {
        found = hiveRouteTableSearchForKey(routeTable, key, state);
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

bool hiveRouteTableInvalidateItem(hiveGuid_t key)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, allocatedKey);
    if(location)
    {
        markDelete(location);
        if(shouldDelete(location->lock))
        {
            freeItem(location);
            return true;
        }
        DPRINTF("Marked %lu as invalid %lu\n", key, location->lock);
    }
    return false;
}

void hiveRouteTableAddRankDuplicate(hiveGuid_t key, unsigned int rank)
{

}

struct hiveDbFrontierIterator * hiveRouteTableGetRankDuplicates(hiveGuid_t key, unsigned int rank)
{
    struct hiveDbFrontierIterator * iter = NULL;
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
    {
        if(rank != -1)
        {
            //Blocks until the OO is done firing
            hiveOutOfOrderListReset(&location->ooList);
            location->rank = rank;
        }
        struct hiveDb * db = location->data;
        iter = hiveCloseFrontier(db->dbList);
    }
    return iter;
}

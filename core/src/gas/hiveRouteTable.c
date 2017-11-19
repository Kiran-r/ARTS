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
#define maxItem       0xDFFFFFFFFFFFFFFF
#define statusMask    (reservedItem | availableItem | deleteItem)

struct hiveRouteItem
{
    hiveGuid_t key;
    void * data;
    volatile u64 lock;
    unsigned int rank;
    unsigned int requested;
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

bool markWrite(struct hiveRouteItem * item)
{
    u64 res = hiveAtomicFetchOrU64(&item->lock, availableItem);
    u64 mask = statusMask;
    u64 validMask = reservedItem;
    return (res & mask) != validMask;
}

bool markDelete(struct hiveRouteItem * item)
{
    u64 res = hiveAtomicFetchOrU64(&item->lock, deleteItem);
    u64 mask = statusMask;
    u64 validMask = reservedItem | availableItem;
    return (res & mask) != validMask;
}

inline void printState(struct hiveRouteItem * item)
{
    if(item)
    {
        u64 mask = statusMask;
        u64 res = item->lock & mask;
        if(res == reservedItem)
            PRINTF("%lu: reserved\n", item->key);
        else if(res == (reservedItem | availableItem))
            PRINTF("%lu: available\n", item->key);
        else if((res == (reservedItem | availableItem | deleteItem)))
            PRINTF("%lu: deleted\n", item->key);
    }
    PRINTF("NULL ITEM\n");
}

//11000 & 11100 = 11000, 10000 & 11100 = 10000, 11100 & 11100 = 11000
bool checkItemState(struct hiveRouteItem * item, itemState state)
{
    if(item)
    {
        u64 mask = statusMask;
        u64 res = item->lock & mask;
        switch(state)
        {
            case reservedKey:
                return res == reservedItem;

            case availableKey:
                return res == (reservedItem | availableItem);

            case allocatedKey:
                return (res == reservedItem) || (res == (reservedItem | availableItem));

            case deletedKey:
                return (res == (reservedItem | availableItem | deleteItem));

            case anyKey:
                return res != 0;

            default:
                return false;
        }
    }
    return false;
}

itemState getItemState(struct hiveRouteItem * item)
{
    if(item)
    {
        u64 mask = statusMask;
        u64 res = item->lock & mask;

        if(res == reservedItem)
            return reservedKey;

        else if(res == (reservedItem | availableItem))
            return availableKey;

        else if((res == (reservedItem | availableItem | deleteItem)))
            return deletedKey;
    }
    return noKey;
}

bool incItem(struct hiveRouteItem * item)
{
    u64 mask = deleteItem;
    while(1)
    {
        u64 local = item->lock;
        if(!(local & mask) && (local + 1 < maxItem))
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
    u64 mask = statusMask;
    if(mask == hiveAtomicSubU64(&item->lock, 1))
    {
        hiveOutOfOrderListDelete(&item->ooList);
        hiveFree(item->data);
        item->data = NULL;
        item->lock = 0;
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
    if( bytes.fields.rank == hiveGlobalRankId && bytes.fields.key >= hiveGuidMin && bytes.fields.key <= hiveGuidMax && hiveNodeInfo.routeTable[bytes.fields.thread])
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
            hiveOutOfOrderListDelete(&item->ooList);
            item->data = NULL;
            item->lock = 0;
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
                    PRINTF("RES: %lu %p\n", key, routeTable);
                }
                guidLock[pos] = 0U;
            }
        }
        else
        {
            *item = hiveRouteTableSearchForKey(routeTable, key, allocatedKey);
        }
    }
//    printState(hiveRouteTableSearchForKey(routeTable, key, reservedKey));
    return ret;
}

//This does the send aggregation
bool hiveRouteTableAddSent(hiveGuid_t key, void * edt, unsigned int slot, bool aggregate, bool update)
{
    struct hiveRouteItem * item = NULL;
    bool sendReq;
    if(update)
    {
        struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
        item = hiveRouteTableSearchForKey(routeTable, key, allocatedKey);
        sendReq = (hiveAtomicCswap(&item->requested, 0, item->rank+1) == 0);
    }
    else
    {
        sendReq = hiveRouteTableReserveItemRace(key, &item, true);

        if(!sendReq && !incItem(item))
            PRINTF("Item marked for deletion before it has arrived...");
    }
    hiveOutOfOrderHandleRemoteDbRequest(&item->ooList, &item->data, edt, slot);
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

itemState hiveRouteTableLookupItemWithState(hiveGuid_t key, void ** data)
{
    void * ret = NULL;
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, anyKey);
    if(location)
    {
        *data = location->data;
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

bool hiveRouteTableReturnDb(hiveGuid_t key)
{
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, availableKey);
    if(location)
    {
        if(hiveGuidGetRank(key) != hiveGlobalRankId)
        {
            u64 validMask = (reservedItem | availableItem) + 1;
            u64 local = location->lock;
            if(local == validMask)
            {
                hiveAtomicCswapU64(&location->lock, local, (local | deleteItem));
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

bool hiveRouteTableAddOO(hiveGuid_t key, void * data, unsigned int rank )
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
    if(item == NULL)
        PRINTF("ER: %ld\n", key);
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

//To cleanup --------------------------------------------------------------------------->

bool hiveRouteTableUpdateItem(hiveGuid_t key, void * data, unsigned int rank)
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
                found = hiveRouteTableSearchForKey(routeTable, key, reservedKey);
                if(found)
                {
                    found->data = data;
                    found->rank = rank;
                    markWrite(found);
                    ret = true;
                }
                guidLock[pos] = 0U;
            }
        }
    }
    return ret;
}


void * hiveRouteTableInvalidateItem(hiveGuid_t key)
{
    void * ret = NULL;
    struct hiveRouteTable * routeTable = hiveGetRouteTable(key);
    struct hiveRouteItem * location = hiveRouteTableSearchForKey(routeTable, key, allocatedKey);

    if(location)
    {
        if(markDelete(location))
        {
            hiveOutOfOrderListDelete(&location->ooList);
            hiveFree(location->data);
            location->data = NULL;
            location->lock = 0;
        }
    }
    return ret;
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
            PRINTF("Setting %lu %u -> %u\n", key, location->rank, rank);
            //Blocks until the OO is done firing
            hiveOutOfOrderListReset(&location->ooList);
            location->rank = rank;
        }
        else
        {
            PRINTF("DID not update rank\n");
        }
        struct hiveDb * db = location->data;
        iter = hiveCloseFrontier(db->dbList);
    }
    return iter;
}

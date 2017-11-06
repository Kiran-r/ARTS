#include "hive.h"
#include "hiveMalloc.h"
#include "hiveAtomics.h"
#include "hiveHash.h"

struct hiveHashItem
{
    void *data;
    hiveGuid_t key;
    volatile unsigned int lock;
} __attribute__ ((aligned));

struct hiveHash
{
    struct hiveHashItem * data;
    unsigned int size;
    //unsigned int use;
    //unsigned int col;
    unsigned int shift;
    struct hiveHash * next;
    unsigned int lock;
} __attribute__ ((aligned));

struct hiveHash *
hiveHashListNew(unsigned int listSize, unsigned int hashSize, unsigned int shift)
{
    int i;
    struct hiveHash *hashList =
        (struct hiveHash *) hiveCalloc(sizeof (struct hiveHash) * listSize);

    for (i = 0; i < listSize; i++)
        hiveHashNew(hashList + i, hashSize, shift);

    return hashList;
}

struct hiveHash *
hiveHashListGetHash(struct hiveHash *hashList, unsigned int position)
{
    return hashList + position;
}

void
hiveHashNew(struct hiveHash *hash, unsigned int size, unsigned int shift)
{
    hash->data =
        (struct hiveHashItem *) hiveCalloc(size *
                                         sizeof (struct hiveHashItem));
    hash->size = size;
    hash->shift = shift;
    //hash->use =0;
    //hash->col =0;
    hash->next = NULL;
    hash->lock=0U;
}

void
hiveHashDelete(struct hiveHash *hash)
{
    struct hiveHash * next;
    struct hiveHash * last;

    next = hash->next;

    while( next != NULL )
    {
        last = next;
        next = next->next;
        hiveFree( last->data );
        hiveFree( last );
    }

    hiveFree(hash->data);
}

void
hiveHashListDelete(struct hiveHash *hash)
{
    /*struct hiveHash * next;
    struct hiveHash * last;

    next = hash->next;

    while( next != NULL )
    {
        last = next;
        next = next->next;
        hiveFree( last->data );
        hiveFree( last );
    }*/

    hiveFree(hash->data);
}

#define hash64(x)       ( (u64)(x) * 14695981039346656037U )
//#define H_BITS          16   // Hashtable size = 2 ^ 4 = 16
#define H_SHIFT_64      ( 64 - H_BITS )

static inline u64 getHashKey( u64 x, unsigned int shift )
{
    //return hash64(x) >> H_SHIFT_64;
    return hash64(x) >> (64-shift);
}

static inline void hiveHashKeySet(struct hiveHash *current, hiveGuid_t key, u64 keyVal)
{
    current->data[ keyVal ].key = key;
}
static inline void hiveHashLock(struct hiveHash *current, hiveGuid_t keyVal)
{
    current->data[ keyVal ].lock = 1U;
}

void *
hiveHashAddItem(struct hiveHash *hash, void *item, hiveGuid_t key)
{
    volatile struct hiveHash * volatile current = hash;
    volatile struct hiveHash * volatile next;
    u64 keyVal;
    while( 1 )
    {
        keyVal =  getHashKey( (u64) key, current->shift);
        //PRINTF("Hash %d/%d %d %ld\n", current->use, current->size, current->col, keyVal);
        if( current->data[ keyVal ].lock == 0U  )
        {
            unsigned int old = hiveAtomicSwap( &current->data[keyVal].lock, 2U  );
            if(old == 0U)
            {
                current->data[ keyVal ].data = item;
                current->data[ keyVal ].key = key;
                current->data[ keyVal ].lock = 1U;
                break;
            }
            else
            {
                while(current->data[ keyVal ].lock != 1U){}
                if( current->data[ keyVal ].key == key  )
                    break;

                if( current->next == NULL )
                {
                    unsigned int old = hiveAtomicSwap( &current->lock, 1U  );
                    if( old == 0U)
                    {
                        //PRINTF("Resize %d %d\n", keyVal, 2*current->size);
                        next = hiveMalloc( sizeof( struct hiveHash  ) );

                        hiveHashNew( (struct hiveHash *)next, 2*current->size, current->shift+1 );

                        current->next = (struct hiveHash *)next;
                    }
                    else
                        while( current->next==NULL  ) {}
                }
                current = current->next;

            }
            //current->use++;
            //PRINTF("Added %ld\n", key);
            //hiveHashKeySet(current, key, keyVal);
            // hiveHashLock(current, keyVal);

            //detectError(hash);

        }
        else if( current->data[ keyVal ].key == key  )
        {
            //PRINTF("Already in table %ld\n", key);
            //return false;
            break;
        }
        else
        {
            while(current->data[ keyVal ].lock != 1U){}
            if( current->data[ keyVal ].key == key  )
                break;
            if( current->next == NULL )
            {
                unsigned int old = hiveAtomicSwap( &current->lock, 1U  );
                if( old == 0U)
                {
                    //break;
                    //PRINTF("Resize %d %d %p %p\n", keyVal, 2*current->size, current, hash);
                    next = hiveMalloc( sizeof( struct hiveHash  ) );

                    hiveHashNew( (struct hiveHash *)next, 2*current->size, current->shift+1 );

                    current->next = (struct hiveHash *)next;
                }
                else
                    while( current->next==NULL  ) {}
            }

            //current->col++;
            current = current->next;
        }
    }
    return current->data[ keyVal ].data;
}

bool
hiveHashDeleteItem(struct hiveHash *hash, hiveGuid_t key)
{
    volatile struct hiveHash * volatile current = hash;
    u64 keyVal;
    while( current != NULL  )
    {
        keyVal =  getHashKey( (u64) key, current->shift);
        if( current->data[ keyVal ].key == key  )
        {
            current->data[ keyVal ].lock = 0U;
            return true;
        }
        else
        {
            current = current->next;
        }
    }

    return false;
}

void*
hiveHashLookupItem(struct hiveHash *hash, hiveGuid_t key)
{
    volatile struct hiveHash * volatile current = hash;
    u64 keyVal;
    while( current != NULL  )
    {
        keyVal =  getHashKey( (u64) key, current->shift);
        if( current->data[keyVal].lock == 1U )
        {
            if( current->data[ keyVal ].key == key  )
            {
                //hiveDebugGenerateSegFault();
                //PRINTF("Found %ld\n", key);
                return (void *)current->data[ keyVal ].data;
            }
            else
            {
                current = current->next;
            }
        }
        else
        {
            current = current->next;
        }
    }

    //PRINTF("Did not find %ld\n", key);
    return NULL;
}

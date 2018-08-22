#include "arts.h"
#include "artsMalloc.h"
#include "artsAtomics.h"
#include "artsHash.h"

struct artsHashItem
{
    void *data;
    artsGuid_t key;
    volatile unsigned int lock;
} __attribute__ ((aligned));

struct artsHash
{
    struct artsHashItem * data;
    unsigned int size;
    //unsigned int use;
    //unsigned int col;
    unsigned int shift;
    struct artsHash * next;
    unsigned int lock;
} __attribute__ ((aligned));

struct artsHash *
artsHashListNew(unsigned int listSize, unsigned int hashSize, unsigned int shift)
{
    int i;
    struct artsHash *hashList =
        (struct artsHash *) artsCalloc(sizeof (struct artsHash) * listSize);

    for (i = 0; i < listSize; i++)
        artsHashNew(hashList + i, hashSize, shift);

    return hashList;
}

struct artsHash *
artsHashListGetHash(struct artsHash *hashList, unsigned int position)
{
    return hashList + position;
}

void
artsHashNew(struct artsHash *hash, unsigned int size, unsigned int shift)
{
    hash->data =
        (struct artsHashItem *) artsCalloc(size *
                                         sizeof (struct artsHashItem));
    hash->size = size;
    hash->shift = shift;
    //hash->use =0;
    //hash->col =0;
    hash->next = NULL;
    hash->lock=0U;
}

void
artsHashDelete(struct artsHash *hash)
{
    struct artsHash * next;
    struct artsHash * last;

    next = hash->next;

    while( next != NULL )
    {
        last = next;
        next = next->next;
        artsFree( last->data );
        artsFree( last );
    }

    artsFree(hash->data);
}

void
artsHashListDelete(struct artsHash *hash)
{
    /*struct artsHash * next;
    struct artsHash * last;

    next = hash->next;

    while( next != NULL )
    {
        last = next;
        next = next->next;
        artsFree( last->data );
        artsFree( last );
    }*/

    artsFree(hash->data);
}

#define hash64(x)       ( (u64)(x) * 14695981039346656037U )
//#define H_BITS          16   // Hashtable size = 2 ^ 4 = 16
#define H_SHIFT_64      ( 64 - H_BITS )

static inline u64 getHashKey( u64 x, unsigned int shift )
{
    //return hash64(x) >> H_SHIFT_64;
    return hash64(x) >> (64-shift);
}

static inline void artsHashKeySet(struct artsHash *current, artsGuid_t key, u64 keyVal)
{
    current->data[ keyVal ].key = key;
}
static inline void artsHashLock(struct artsHash *current, artsGuid_t keyVal)
{
    current->data[ keyVal ].lock = 1U;
}

void *
artsHashAddItem(struct artsHash *hash, void *item, artsGuid_t key)
{
    volatile struct artsHash * volatile current = hash;
    volatile struct artsHash * volatile next;
    u64 keyVal;
    while( 1 )
    {
        keyVal =  getHashKey( (u64) key, current->shift);
        //PRINTF("Hash %d/%d %d %ld\n", current->use, current->size, current->col, keyVal);
        if( current->data[ keyVal ].lock == 0U  )
        {
            unsigned int old = artsAtomicSwap( &current->data[keyVal].lock, 2U  );
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
                    unsigned int old = artsAtomicSwap( &current->lock, 1U  );
                    if( old == 0U)
                    {
                        //PRINTF("Resize %d %d\n", keyVal, 2*current->size);
                        next = artsMalloc( sizeof( struct artsHash  ) );

                        artsHashNew( (struct artsHash *)next, 2*current->size, current->shift+1 );

                        current->next = (struct artsHash *)next;
                    }
                    else
                        while( current->next==NULL  ) {}
                }
                current = current->next;

            }
            //current->use++;
            //PRINTF("Added %ld\n", key);
            //artsHashKeySet(current, key, keyVal);
            // artsHashLock(current, keyVal);

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
                unsigned int old = artsAtomicSwap( &current->lock, 1U  );
                if( old == 0U)
                {
                    //break;
                    //PRINTF("Resize %d %d %p %p\n", keyVal, 2*current->size, current, hash);
                    next = artsMalloc( sizeof( struct artsHash  ) );

                    artsHashNew( (struct artsHash *)next, 2*current->size, current->shift+1 );

                    current->next = (struct artsHash *)next;
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
artsHashDeleteItem(struct artsHash *hash, artsGuid_t key)
{
    volatile struct artsHash * volatile current = hash;
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
artsHashLookupItem(struct artsHash *hash, artsGuid_t key)
{
    volatile struct artsHash * volatile current = hash;
    u64 keyVal;
    while( current != NULL  )
    {
        keyVal =  getHashKey( (u64) key, current->shift);
        if( current->data[keyVal].lock == 1U )
        {
            if( current->data[ keyVal ].key == key  )
            {
                //artsDebugGenerateSegFault();
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

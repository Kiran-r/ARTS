#include "hive.h"
#include "hiveMalloc.h"
#include "hiveAtomics.h"
#include "hiveTable.h"
#include <string.h>

struct hiveTable
{
    void ** data;
    unsigned int size;
    //unsigned int elementSize;
    unsigned int lock;
    //unsigned int use;
    //unsigned int col;
    //unsigned int shift;
    struct hiveTable * next;
} __attribute__ ((aligned));

struct hiveTable *
hiveTableListNew(unsigned int listSize, unsigned int tableLength)
{
    int i;
    struct hiveTable *tableList =
        (struct hiveTable *) hiveCalloc(sizeof (struct hiveTable) * listSize);

    for (i = 0; i < listSize; i++)
        hiveTableNew(tableList + i, tableLength);

    return tableList;
}

extern inline struct hiveTable *
hiveTableListGetTable(struct hiveTable *tableList, unsigned int position)
{
    return tableList + position;
}

void
hiveTableNew(struct hiveTable *table, unsigned int tableSize)
{
    table->data =  hiveCalloc( tableSize * sizeof(void *) );
    table->size = tableSize;
    //table->elementSize = elementSize;
    //hash->shift = shift;
    //hash->use =0;
    //hash->col =0;
    table->next = NULL;
}

void
hiveTableDelete(struct hiveTable *table)
{
    struct hiveTable * next;
    struct hiveTable * last;

    next = table->next;

    while( next != NULL )
    {
        last = next;
        next = next->next;
        hiveFree( last->data );
        hiveFree( last );
    }

    hiveFree(table->data);
}

void
hiveTableListDelete(struct hiveTable *table)
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
    //FIXME: Free Full List
    hiveFree(table->data);
}


void
hiveTableAddItem(struct hiveTable *table, void *item, unsigned int pos, unsigned int size)
{
    volatile struct hiveTable * volatile current = table;
    volatile struct hiveTable * volatile next;

    while( 1 )
    {
        if( pos < current->size  )
        {
            //PRINTF("current add %p %p %d\n", current, item, pos);
            void * ptr = hiveMalloc( size );
            memcpy(ptr, item, size);
            current->data[pos] = ptr;
            //memcpy( current->data+pos*current->elementSize, item, current->elementSize );
            //current->data[ keyVal ].data = item;
            //current->data[ keyVal ].key = key;
            //current->data[ keyVal ].lock = 1U;
            //current->use++;
            //PRINTF("Added %ld\n", key);
            //hiveHashKeySet(current, key, keyVal);
            // hiveHashLock(current, keyVal);

            //detectError(hash);

            break;
        }
        else
        {
            if( current->next == NULL )
            {
                unsigned int old = hiveAtomicSwap( &current->lock, 1U  );
                if( old == 0U)
                {
                    //break;
                    //PRINTF("Resize %d %d\n", keyVal, 2*current->size);
                    next = hiveMalloc( sizeof( struct hiveTable  ) );

                    hiveTableNew( (struct hiveTable *)next, 2*current->size );

                    current->next = (struct hiveTable *)next;
                }
                else
                    while( current->next==NULL  ) {}
            }

            //current->col++;
            pos-= current->size;
            current = current->next;
        }
    }
}

void*
hiveTableLookupItem(struct hiveTable *table, unsigned int pos)
{
    volatile struct hiveTable * volatile current = table;

    while( current != NULL  )
    {
        if( pos < current->size  )
        {
            //PRINTF("current look %pi %p\n", current, current->data[pos]);
            //PRINTF("Found %ld\n", key);
            return (void *)current->data[ pos ];
        }
        else
        {
            //PRINTF("current look fail %p\n", current);
            pos-= current->size;
            current = current->next;
        }
    }

    //PRINTF("Did not find %ld\n", key);
    return NULL;
}

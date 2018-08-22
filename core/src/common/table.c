#include "arts.h"
#include "artsMalloc.h"
#include "artsAtomics.h"
#include "artsTable.h"
#include <string.h>

struct artsTable
{
    void ** data;
    unsigned int size;
    //unsigned int elementSize;
    unsigned int lock;
    //unsigned int use;
    //unsigned int col;
    //unsigned int shift;
    struct artsTable * next;
} __attribute__ ((aligned));

struct artsTable *
artsTableListNew(unsigned int listSize, unsigned int tableLength)
{
    int i;
    struct artsTable *tableList =
        (struct artsTable *) artsCalloc(sizeof (struct artsTable) * listSize);

    for (i = 0; i < listSize; i++)
        artsTableNew(tableList + i, tableLength);

    return tableList;
}

extern inline struct artsTable *
artsTableListGetTable(struct artsTable *tableList, unsigned int position)
{
    return tableList + position;
}

void
artsTableNew(struct artsTable *table, unsigned int tableSize)
{
    table->data =  artsCalloc( tableSize * sizeof(void *) );
    table->size = tableSize;
    //table->elementSize = elementSize;
    //hash->shift = shift;
    //hash->use =0;
    //hash->col =0;
    table->next = NULL;
}

void
artsTableDelete(struct artsTable *table)
{
    struct artsTable * next;
    struct artsTable * last;

    next = table->next;

    while( next != NULL )
    {
        last = next;
        next = next->next;
        artsFree( last->data );
        artsFree( last );
    }

    artsFree(table->data);
}

void
artsTableListDelete(struct artsTable *table)
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
    //FIXME: Free Full List
    artsFree(table->data);
}


void
artsTableAddItem(struct artsTable *table, void *item, unsigned int pos, unsigned int size)
{
    volatile struct artsTable * volatile current = table;
    volatile struct artsTable * volatile next;

    while( 1 )
    {
        if( pos < current->size  )
        {
            //PRINTF("current add %p %p %d\n", current, item, pos);
            void * ptr = artsMalloc( size );
            memcpy(ptr, item, size);
            current->data[pos] = ptr;
            //memcpy( current->data+pos*current->elementSize, item, current->elementSize );
            //current->data[ keyVal ].data = item;
            //current->data[ keyVal ].key = key;
            //current->data[ keyVal ].lock = 1U;
            //current->use++;
            //PRINTF("Added %ld\n", key);
            //artsHashKeySet(current, key, keyVal);
            // artsHashLock(current, keyVal);

            //detectError(hash);

            break;
        }
        else
        {
            if( current->next == NULL )
            {
                unsigned int old = artsAtomicSwap( &current->lock, 1U  );
                if( old == 0U)
                {
                    //break;
                    //PRINTF("Resize %d %d\n", keyVal, 2*current->size);
                    next = artsMalloc( sizeof( struct artsTable  ) );

                    artsTableNew( (struct artsTable *)next, 2*current->size );

                    current->next = (struct artsTable *)next;
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
artsTableLookupItem(struct artsTable *table, unsigned int pos)
{
    volatile struct artsTable * volatile current = table;

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

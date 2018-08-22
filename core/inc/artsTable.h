#ifndef ARTSTABLE_H
#define ARTSTABLE_H

#include "arts.h"

struct artsTable;

struct artsTable *artsTableListNew(unsigned int listSize, unsigned int tableSize);
struct artsTable *artsTableListGetTable(struct artsTable * tableList, unsigned int position);
void artsTableListDelete(struct artsTable *tableList);
void artsTableNew(struct artsTable *table, unsigned int size);
void artsTableAddItem(struct artsTable *table, void *item, unsigned int pos, unsigned int size);
void* artsTableLookupItem(struct artsTable *table, unsigned int pos);

#endif

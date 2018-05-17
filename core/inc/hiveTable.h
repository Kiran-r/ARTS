#ifndef HIVETABLE_H
#define HIVETABLE_H

#include "hive.h"

struct hiveTable;

struct hiveTable *hiveTableListNew(unsigned int listSize, unsigned int tableSize);
struct hiveTable *hiveTableListGetTable(struct hiveTable * tableList, unsigned int position);
void hiveTableListDelete(struct hiveTable *tableList);
void hiveTableNew(struct hiveTable *table, unsigned int size);
void hiveTableAddItem(struct hiveTable *table, void *item, unsigned int pos, unsigned int size);
void* hiveTableLookupItem(struct hiveTable *table, unsigned int pos);

#endif

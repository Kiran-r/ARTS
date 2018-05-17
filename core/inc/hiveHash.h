#ifndef HIVEHASH_H
#define HIVEHASH_H

#include "hive.h"

struct hiveHash;

struct hiveHash *hiveHashListNew(unsigned int listSize, unsigned int hashSize, unsigned int shift);
struct hiveHash *hiveHashListGetHash(struct hiveHash * hashList, unsigned int position);
void hiveHashListDelete(struct hiveHash *hashList);
void hiveHashNew(struct hiveHash *hash, unsigned int size, unsigned int shift);
void hiveHashDelete(struct hiveHash *hash);
void * hiveHashAddItem(struct hiveHash *hash, void *item, hiveGuid_t key);
bool hiveHashDeleteItem(struct hiveHash *hash, hiveGuid_t key);
void* hiveHashLookupItem(struct hiveHash *hash, hiveGuid_t key);

#endif

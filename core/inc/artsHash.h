#ifndef ARTSHASH_H
#define ARTSHASH_H

#include "arts.h"

struct artsHash;

struct artsHash *artsHashListNew(unsigned int listSize, unsigned int hashSize, unsigned int shift);
struct artsHash *artsHashListGetHash(struct artsHash * hashList, unsigned int position);
void artsHashListDelete(struct artsHash *hashList);
void artsHashNew(struct artsHash *hash, unsigned int size, unsigned int shift);
void artsHashDelete(struct artsHash *hash);
void * artsHashAddItem(struct artsHash *hash, void *item, artsGuid_t key);
bool artsHashDeleteItem(struct artsHash *hash, artsGuid_t key);
void* artsHashLookupItem(struct artsHash *hash, artsGuid_t key);

#endif

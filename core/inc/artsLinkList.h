#ifndef ARTSLINKLIST_H
#define ARTSLINKLIST_H

struct artsLinkList;

struct artsLinkList * artsLinkListGroupNew(unsigned int listSize);
struct artsLinkList * artsLinkListGet(struct artsLinkList *linkList, unsigned int position);
void artsLinkListDelete(void *linkList);
void artsLinkListPushBack(struct artsLinkList *list, void *item, unsigned int size);
void * artsLinkListPopFront( struct artsLinkList * list, void ** freePos );
void artsLinkListDeleteItem( void * item );
void * artsLinkListNewItem(unsigned int size);

#endif

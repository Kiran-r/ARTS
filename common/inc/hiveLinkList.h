#ifndef HIVELINKLIST_H
#define HIVELINKLIST_H


struct hiveLinkList;

struct hiveLinkList *
hiveLinkListGroupNew(unsigned int listSize);

struct hiveLinkList * hiveLinkListGet(struct hiveLinkList *linkList, unsigned int position);

void hiveLinkListDelete(void *linkList);

void hiveLinkListPushBack(struct hiveLinkList *list, void *item, unsigned int size);

void * hiveLinkListPopFront( struct hiveLinkList * list, void ** freePos );

void hiveLinkListDeleteItem( void * item );

void * hiveLinkListNewItem(unsigned int size);

#endif

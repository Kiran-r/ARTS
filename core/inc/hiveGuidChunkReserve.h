#ifndef HIVEGUIDCHUNKRESERVE_H
#define HIVEGUIDCHUNKRESERVE_H

#define NODE_CHUNK_LIMIT 1048576
#define THREAD_CHUNK_LIMIT 256
#define KEY_CHUNK_LIMIT 32768
#define ROUTE_LIMIT 1048576 
#define ROUTE_CHUNK_SIZE 512 


struct hiveGuidReserveNodeTable
{
    unsigned int size; 
    struct hiveGuidReserveTable * next;
    char elements[];
};

struct hiveGuidReserveThreadTable
{
    unsigned int size; 
    struct hiveGuidReserveTable * next;
    char destinations[];
};

struct hiveGuidReserveThreadTable
{
    unsigned int size;
    unsigned int key[];
    //----------------------------
    unsigned int nodeBits[];
    unsigned int threadBits[];

    struct hiveGuidReserveThreadTable * next;
}


#endif

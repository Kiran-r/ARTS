#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
//
//hiveGuid_t sink(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
//{
//    unsigned int sum = 0;
//    for(unsigned int i=0; i<depc; i++)
//    {
//        unsigned int * ptr = depv[i].ptr;
//        for(unsigned int j = 0; j<ptr[0]; j++)
//        {
//            sum+=ptr[1+j];
//        }
//    }
//    unsigned int check = 0;
//    for(unsigned int i=0; i<paramv[0]; i++)
//        check+=i;
//    PRINTF("NUM: %u BLOCKS: %u SUM: %u CHECK: %u\n", paramv[0], depc, sum, check);
//    hiveShutdown();
//}
//
//hiveGuid_t source(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
//{
//    hivePercolateEdt(sink, 1, 1, paramv, paramc-1, (hiveGuid_t *)&paramv[1]);
//}
//
//void initPerNode(unsigned int nodeId, int argc, char** argv)
//{
//    
//}
//
//void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
//{   
//    if(!nodeId && !workerId)
//    {
//        unsigned int num = atoi(argv[1]);
//        hiveGuid_t * guids = hiveMalloc(sizeof(hiveGuid_t)*num);
//        guids[0] = num;
//        
//        unsigned int i = 0;
//        unsigned int j = 2;
//        unsigned int k = 1;
//        while(i<num)
//        {
//            unsigned int * ptr = NULL;
//            guids[k++] = hiveDbCreate((void**)&ptr, sizeof(unsigned int)*j, false);
//            ptr[0] = j-1;
//            for(unsigned int l=1; l<j; l++)
//            {
//                if(i<num)
//                    ptr[l] = i++;
//                else
//                    ptr[l] = 0;
//            }
//            j++;
//        }
//        hiveEdtCreate(source, 0, k, (u64*)guids, 0, NULL);
//    }
//}
//
int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}

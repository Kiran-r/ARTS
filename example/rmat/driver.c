/* 
 * File:   main.c
 * Author: suet688
 *
 * Created on June 18, 2015, 1:59 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include "compat.h"
#include "user_settings.h"
#include "graph_generator.h"
#include "rmat.h"
#include "prng.h"
#include "xalloc.h"
#include "seq-csr.h"

static packed_edge * restrict IJ;
static int64_t nvtx_scale;
static int64_t nedge;
int64_t edgefactor;
int64_t SCALE;

int main(int argc, char** argv) 
{
    if(argc!=8)
    {
        printf("<scale> <edgefactor> <A> <B> <C> <filename> <partitions>\n");
        return 0;
    }
    init_random();
    
    SCALE = (int64_t) atoi(argv[1]);
    edgefactor = (int64_t) atoi(argv[2]);
    double A = strtod(argv[3], NULL);
    double B = strtod(argv[4], NULL);
    double C = strtod(argv[5], NULL);
    int64_t partitions = (int64_t) atoi(argv[7]);
    nvtx_scale = ((int64_t)1)<<SCALE;
    nedge = nvtx_scale * edgefactor;
    printf("Creating graph Size: %d Edges: %d A: %lf B: %lf C: %lf D: %lf File: %s\n",
            nvtx_scale, nedge, A, B, C, (1-A-B-C), argv[6]);
    IJ = xmalloc_large_ext (nedge * sizeof (*IJ));
    rmat_edgelist(IJ, nedge, SCALE, A, B, C);
    create_graph_from_edgelist (IJ, nedge);
//    print_graph();
    write_partitioned_graph_to_file(argv[6], partitions);
//    write_graph_to_file(argv[6]);
    destroy_graph ();
    xfree_large (IJ);
    return 0;
}


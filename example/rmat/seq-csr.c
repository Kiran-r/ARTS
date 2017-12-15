/* -*- mode: C; mode: folding; fill-column: 70; -*- */
/* Copyright 2010,  Georgia Institute of Technology, USA. */
/* See COPYING for license. */
#define _FILE_OFFSET_BITS 64
#define _THREAD_SAFE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <assert.h>

#include "seq-csr.h"
#include "xalloc.h"
#define MAXFILENAMESIZE 128
#define MINVECT_SIZE 2

static int64_t maxvtx, nv, sz;
static int64_t * restrict xoff; /* Length 2*nv+2 */
static int64_t * restrict xadjstore; /* Length MINVECT_SIZE + (xoff[nv] == nedge) */
static int64_t * restrict xadj;

static void
find_nv (const struct packed_edge * restrict IJ, const int64_t nedge)
{
  int64_t k;

  maxvtx = -1;
  for (k = 0; k < nedge; ++k) {
    if (get_v0_from_edge(&IJ[k]) > maxvtx)
      maxvtx = get_v0_from_edge(&IJ[k]);
    if (get_v1_from_edge(&IJ[k]) > maxvtx)
      maxvtx = get_v1_from_edge(&IJ[k]);
  }
  nv = 1+maxvtx;
}

static int
alloc_graph (int64_t nedge)
{
  sz = (2*nv+2) * sizeof (*xoff);
  xoff = xmalloc_large_ext (sz);
  if (!xoff) return -1;
  return 0;
}

static void
free_graph (void)
{
  xfree_large (xadjstore);
  xfree_large (xoff);
}

#define XOFF(k) (xoff[2*(k)])
#define XENDOFF(k) (xoff[1+2*(k)])

static int
setup_deg_off (const struct packed_edge * restrict IJ, int64_t nedge)
{
  int64_t k, accum;
  for (k = 0; k < 2*nv+2; ++k)
    xoff[k] = 0;
  for (k = 0; k < nedge; ++k) {
    int64_t i = get_v0_from_edge(&IJ[k]);
    int64_t j = get_v1_from_edge(&IJ[k]);
    if (i != j) { /* Skip self-edges. */
      if (i >= 0) ++XOFF(i);
      if (j >= 0) ++XOFF(j);
    }
  }
  accum = 0;
  for (k = 0; k < nv; ++k) {
    int64_t tmp = XOFF(k);
    if (tmp < MINVECT_SIZE) tmp = MINVECT_SIZE;
    XOFF(k) = accum;
    accum += tmp;
  }
  XOFF(nv) = accum;
  for (k = 0; k < nv; ++k)
    XENDOFF(k) = XOFF(k);
  if (!(xadjstore = xmalloc_large_ext ((accum + MINVECT_SIZE) * sizeof (*xadjstore))))
    return -1;
  xadj = &xadjstore[MINVECT_SIZE]; /* Cheat and permit xadj[-1] to work. */
  for (k = 0; k < accum + MINVECT_SIZE; ++k)
    xadjstore[k] = -1;
  return 0;
}

static void
scatter_edge (const int64_t i, const int64_t j)
{
  int64_t where;
  where = XENDOFF(i)++;
  xadj[where] = j;
}

static int
i64cmp (const void *a, const void *b)
{
  const int64_t ia = *(const int64_t*)a;
  const int64_t ib = *(const int64_t*)b;
  if (ia < ib) return -1;
  if (ia > ib) return 1;
  return 0;
}

static void
pack_vtx_edges (const int64_t i)
{
  int64_t kcur, k;
  if (XOFF(i)+1 >= XENDOFF(i)) return;
  qsort (&xadj[XOFF(i)], XENDOFF(i)-XOFF(i), sizeof(*xadj), i64cmp);
  kcur = XOFF(i);
  for (k = XOFF(i)+1; k < XENDOFF(i); ++k)
    if (xadj[k] != xadj[kcur])
      xadj[++kcur] = xadj[k];
  ++kcur;
  for (k = kcur; k < XENDOFF(i); ++k)
    xadj[k] = -1;
  XENDOFF(i) = kcur;
}

static void
pack_edges (void)
{
  int64_t v;

  for (v = 0; v < nv; ++v)
    pack_vtx_edges (v);
}

static void
gather_edges (const struct packed_edge * restrict IJ, int64_t nedge)
{
  int64_t k;

  for (k = 0; k < nedge; ++k) {
    int64_t i = get_v0_from_edge(&IJ[k]);
    int64_t j = get_v1_from_edge(&IJ[k]);
    if (i >= 0 && j >= 0 && i != j) {
      scatter_edge (i, j);
      scatter_edge (j, i);
    }
  }

  pack_edges ();
}

int 
create_graph_from_edgelist (struct packed_edge *IJ, int64_t nedge)
{
  find_nv (IJ, nedge);
  if (alloc_graph (nedge)) return -1;
  if (setup_deg_off (IJ, nedge)) {
    xfree_large (xoff);
    return -1;
  }
  gather_edges (IJ, nedge);
  return 0;
}

int
make_bfs_tree (int64_t *bfs_tree_out, int64_t *max_vtx_out,
	       int64_t srcvtx)
{
  int64_t * restrict bfs_tree = bfs_tree_out;
  int err = 0;

  int64_t * restrict vlist = NULL;
  int64_t k1, k2;

  *max_vtx_out = maxvtx;

  vlist = xmalloc_large (nv * sizeof (*vlist));
  if (!vlist) return -1;

  for (k1 = 0; k1 < nv; ++k1)
    bfs_tree[k1] = -1;

  vlist[0] = srcvtx;
  bfs_tree[srcvtx] = srcvtx;
  k1 = 0; k2 = 1;
  while (k1 != k2) {
    const int64_t oldk2 = k2;
    int64_t k;
    for (k = k1; k < oldk2; ++k) {
      const int64_t v = vlist[k];
      const int64_t veo = XENDOFF(v);
      int64_t vo;
      for (vo = XOFF(v); vo < veo; ++vo) {
	const int64_t j = xadj[vo];
	if (bfs_tree[j] == -1) {
	  bfs_tree[j] = v;
	  vlist[k2++] = j;
	}
      }
    }
    k1 = oldk2;
  }

  xfree_large (vlist);

  return err;
}

void
destroy_graph (void)
{
  free_graph ();
}

void
print_graph (void)
{
    int64_t i, j;
    for(i=0; i<nv; i++)
    {
        printf("%d {", i);
        int64_t start = XOFF(i);
        int64_t end = XENDOFF(i);
        for(j=start; j<end; j++)
        {
            printf(" %d ", xadj[j]);
        }
        printf("}\n");
    }
}

void
stat_graph (int64_t * rowSize, int64_t * columnSize)
{
    int64_t localRow = 0;
    int64_t localColumn = 0;
    
    int64_t i, j;
    for(i=0; i<nv; i++)
    {
        int64_t temp = XENDOFF(i) - XOFF(i);
        if(temp)
        {
            localRow++;
            localColumn+=temp;
        }
    }
    *rowSize=localRow;
    *columnSize=localColumn;
}

void
stat_block (int64_t * rowSize, int64_t * columnSize, int64_t start, int64_t end)
{
    int64_t localRow = 0;
    int64_t localColumn = 0;
    
    int64_t i, j;
    for(i=start; i<end; i++)
    {
        int64_t temp = XENDOFF(i) - XOFF(i);
        if(temp)
        {
            localRow++;
            localColumn+=temp;
        }
    }
    *rowSize=localRow;
    *columnSize=localColumn;
}

void
write_graph_to_file(char * filename)
{
    FILE * fp = fopen(filename,"w");
    if(fp)
    {
        int64_t row, column;
        stat_graph(&row, &column);
        fprintf(fp, "%" PRId64 " %" PRId64 " %" PRId64 "\n", nv, row, column);
        int64_t i, j;
        for(i=0; i<nv; i++)
        {
            int64_t start = XOFF(i);
            int64_t end = XENDOFF(i);
            fprintf(fp, "%" PRId64 "", end-start);
            for(j=start; j<end; j++)
            {
                fprintf(fp, " %" PRId64 "", xadj[j]);
            }
            fprintf(fp, "\n");
        }
    }
    else
        printf("Failed to open %s\n", filename);
    fclose(fp);
}

void
write_partitioned_graph_to_file(char * prefix, int64_t partitions)
{
    int64_t blockSize = nv/partitions;
    char filename[MAXFILENAMESIZE];
    int64_t k;
    for(k=0; k<partitions; k++)
    {
        sprintf(filename,"%s.%d", prefix, k);
        FILE * fp = fopen(filename,"w");
        if(fp)
        {
            int64_t row, column;
            int64_t start = k*blockSize;
            int64_t end = (k==partitions-1) ? nv : (k+1)*blockSize;
            stat_block(&row, &column, start, end);
            fprintf(fp, "%" PRId64 " %" PRId64 " %" PRId64 "\n", end-start, row, column);
            int64_t i, j;
            for(i=start; i<end; i++)
            {
                int64_t start = XOFF(i);
                int64_t end = XENDOFF(i);
                fprintf(fp, "%" PRId64 "", end-start);
                for(j=start; j<end; j++)
                {
                    fprintf(fp, " %" PRId64 "", xadj[j]);
                }
                fprintf(fp, "\n");
            }
        }
        else
            printf("Failed to open %s\n", filename);
        fclose(fp);
    }
}
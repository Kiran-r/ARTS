/* 
 * File:   seq-csr.h
 * Author: suet688
 *
 * Created on June 18, 2015, 2:27 PM
 */

#ifndef SEQ_CSR_H
#define	SEQ_CSR_H

#ifdef	__cplusplus
extern "C" {
#endif

#include "compat.h"
#include "graph_generator.h"

int create_graph_from_edgelist (struct packed_edge *IJ, int64_t nedge);
void print_graph (void);
void destroy_graph (void);
void write_graph_to_file(char * filename);
void write_partitioned_graph_to_file(char * prefix, int64_t partitions);

#ifdef	__cplusplus
}
#endif

#endif	/* SEQ_CSR_H */


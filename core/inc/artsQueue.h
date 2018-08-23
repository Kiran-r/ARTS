// Copyright (c) 2013, Adam Morrison and Yehuda Afek.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in
//    the documentation and/or other materials provided with the
//    distribution.
//  * Neither the name of the Tel Aviv University nor the names of the
//    author of this software may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ARTSQUEUE_H
#define	ARTSQUEUE_H

#include "arts.h"

#define Object          uint64_t
#define RING_POW        (9)
#define RING_SIZE       (1ull << RING_POW)
#define ALIGNMENT       8

typedef struct RingNode
{
    volatile uint64_t val;
    volatile uint64_t idx;
    uint64_t pad[14];
} RingNode __attribute__ ((aligned (128)));

typedef struct RingQueue
{
    volatile int64_t head __attribute__ ((aligned (128)));
    volatile int64_t tail __attribute__ ((aligned (128)));
    struct RingQueue *next __attribute__ ((aligned (128)));
    RingNode array[RING_SIZE];
} RingQueue __attribute__ ((aligned (128)));

typedef struct artsQueue
{
    RingQueue * head;
    RingQueue * tail;
} artsQueue __attribute__ ((aligned (128)));

artsQueue * artsNewQueue();
void enqueue(Object arg, artsQueue * queue);
Object dequeue(artsQueue * queue);

int close_crq(RingQueue *rq, const uint64_t t, const int tries);
uint64_t node_index(uint64_t i) __attribute__ ((pure));
void fixState(RingQueue *rq);
uint64_t set_unsafe(uint64_t i) __attribute__ ((pure));
int is_empty(uint64_t v) __attribute__ ((pure));
uint64_t node_unsafe(uint64_t i) __attribute__ ((pure));
int crq_is_closed(uint64_t t) __attribute__ ((pure));
void init_ring(RingQueue *r);
uint64_t tail_index(uint64_t t) __attribute__ ((pure));

#endif	/* artsQUEUE_H */

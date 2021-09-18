#pragma once

#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <vector>
#include <stdint.h>

using vid_t = uint32_t;
using eid_t = uint32_t;

/**
 * The Graph class stores various information about the graph, using the Compressed Sparse
 * Row (CSR) format.
 */
class Graph {
public:
    uint32_t n;         //number of vertices
    uint32_t m;         //number of edges

    vid_t *adj;         //adjacency array
    eid_t *num_edges;   //starting postion of the vertice N in the adjacency array and edgeId array
    eid_t *edge_id;     //edgeId array
    int32_t *degree;    //degree of vertices

    void free_graph() {
        if (this->adj != nullptr)
            free(this->adj);

        if (this->num_edges != nullptr)
            free(this->num_edges);

        if (this->edge_id != nullptr)
            free(this->edge_id);

        /*if (this->degree != nullptr)
            free(this->degree);
        */
    }
};

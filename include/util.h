#pragma once
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <omp.h>
#include "log.h"
#include "timer.h"

#define max(a,b) ((a)>(b)?(a):(b))

void parseArg(int argc, char *argv[]) {

    int res = getopt(argc, argv, "f:");
    if (res == -1 || res != 'f') {
        printf("Please enter the parameters correctly\n");
        exit(-1);
    }
}

void read_data_from_file(char *filename, vid_t *&u, vid_t *&v, int *&degree, int &edge_num, int &max_vertex) {
    int file_fd = open(filename, O_RDONLY);
    if (file_fd < 0) {
        printf("Could not open input file-{%s}", filename);
        exit(-1);
    }

    struct stat statbuf;
    stat(filename, &statbuf);
    size_t file_size = statbuf.st_size;
    int num_edges_estimate = file_size / 8;

    char *pmem_base = (char *)mmap(nullptr, file_size, PROT_READ, MAP_SHARED, file_fd, 0);
    int num_thread = omp_get_max_threads();
    uint64_t avg = file_size / num_thread;
    uint64_t beg_offset[num_thread], end_offset[num_thread];
    for (auto i = 0; i < num_thread; i++){
        beg_offset[i] = (i == 0) ? 0 : end_offset[i - 1] + 1;

        uint64_t es_end_offset = (i + 1) * avg;
        while (*(pmem_base + es_end_offset) != '\n' && *(pmem_base + es_end_offset)) {
            es_end_offset++;
        }
        end_offset[i] = es_end_offset;
    }
    uint64_t num_edges_thread[num_thread];
    uint64_t off_write_thread[num_thread];
#pragma omp parallel
    {
        auto tid = omp_get_thread_num();
        int local_num_edges_es = num_edges_estimate / num_thread;
        vid_t *local_u = (vid_t *)malloc(local_num_edges_es * sizeof(vid_t));
        vid_t *local_v = (vid_t *)malloc(local_num_edges_es * sizeof(vid_t));
        int read_line = 0;
        for (auto off = beg_offset[tid]; off <= end_offset[tid] && *(pmem_base + off); off++) {

            while ((*(pmem_base + off) < '0' || *(pmem_base + off) > '9') && *(pmem_base + off)) {
                off++;
            }

            if (!*(pmem_base + off)) {
                break;
            }

            int u = 0;
            while (*(pmem_base + off) >= '0' && *(pmem_base + off) <= '9') {
                u = u * 10 + (*(pmem_base + off) - '0');
                off++;
            }

            while (*(pmem_base + off) < '0' || *(pmem_base + off) > '9') {
                off++;
            }

            int v = 0;
            while (*(pmem_base + off) >= '0' && *(pmem_base + off) <= '9') {
                v = v * 10 + (*(pmem_base + off) - '0');
                off++;
            }

            while (*(pmem_base + off) != '\n' && *(pmem_base + off)) {
                off++;
            }
            local_u[read_line] = u;
            local_v[read_line] = v;
            read_line++;
        }
        num_edges_thread[tid] = read_line;
        if (tid == num_thread - 1) {
            max_vertex = local_v[read_line - 1];
        }
#pragma omp barrier

#pragma omp single
        {
            edge_num = num_edges_thread[0];
            off_write_thread[0] = 0;
            for (auto i = 1; i < num_thread; i++) {
                off_write_thread[i] = off_write_thread[i - 1] + num_edges_thread[i - 1];
                edge_num += num_edges_thread[i];
            }
            u = (vid_t *)malloc(edge_num * sizeof(vid_t));
            v = (vid_t *)malloc(edge_num * sizeof(vid_t));
            degree = (int *)malloc((max_vertex + 1) * sizeof(int));
            memset(degree, 0, (max_vertex + 1) * sizeof(int));
        }
#pragma omp barrier
        Timer timer;
        memcpy(u + off_write_thread[tid], local_u, num_edges_thread[tid] * sizeof(vid_t));
        memcpy(v + off_write_thread[tid], local_v, num_edges_thread[tid] * sizeof(vid_t));
#pragma omp barrier
        if (tid ==0) log_debug("memcpy time:%.5lf sec", timer.timeAndReset());
    }

    Timer timer;
#pragma omp parallel for
    for (int i = 0; i < edge_num; i++) {
        __sync_fetch_and_add(&degree[v[i]], 1);
    }
    log_debug("calc degree time:%.5lf sec", timer.timeAndReset());

}

int get_upper_k(int *degree, vid_t *u, vid_t *v, int n, int m) {

    int *degree_num = (int *)malloc(n * sizeof(int));
    memset(degree_num, 0, n * sizeof(int));

#pragma omp parallel for
    for (int i = 0; i <= n; i++) {
        __sync_fetch_and_add(&degree_num[degree[i]], 1);
    }

    int upper_k = 0;
    int total_num = 0;
    for (int i = n - 1; i >= 0; i--) {
        int cur = i + 1;
        total_num += degree_num[i];
        if (total_num >= cur) {
            upper_k = cur;
            break;
        }
    }
    return upper_k;
}


void init_graph(Graph *g, vid_t *u, vid_t *v, int *degree, int edge_num, int max_vertex, int lower_k) {
    Timer timer;
    int *tmp_degree = (int *)malloc((max_vertex + 1) * sizeof(int));
    memcpy(tmp_degree, degree, (max_vertex + 1) * sizeof(int));
    //calc k core
#pragma omp parallel for
        for (auto i = 0; i < edge_num; i++) {
            if (tmp_degree[u[i]] <= lower_k || tmp_degree[v[i]] <= lower_k) {
                tmp_degree[u[i]]--;
            }
        }

#pragma omp parallel for
        for (auto i = 0; i < edge_num; i++) {
            if (tmp_degree[u[i]] > lower_k && tmp_degree[v[i]] > lower_k) {
                //__sync_fetch_and_add(&g->num_edges[v[i] + 1], 1);
                g->adj[i] = u[i];
            }
        }
    log_debug("calc k core. time:.%.5lf sec", timer.timeAndReset());

    int m = 0;
    for (int i = 0; i < edge_num; i++) {
        if (g->adj[i] != 0) {
            g->adj[m++] = g->adj[i];
            g->num_edges[v[i] + 1]++;
        }
    }
    log_debug("init adj time:%.5lf sec", timer.timeAndReset());

    for (int i = 1; i <= max_vertex + 1; i++) {
        g->num_edges[i] += g->num_edges[i - 1];
    }
    log_debug("init num_edges time:%.5lf sec", timer.timeAndReset());

    g->n = max_vertex + 1;
    g->m = m;
}

void output_result(int *edge_support, long num_edges) {
    int max_support = 0;
    int num_edges_with_max_support = 0;
    for (long i = 0; i < num_edges; i++) {
        if (max_support < edge_support[i]) {
            max_support = edge_support[i];
            num_edges_with_max_support = 1;
        } else if (max_support == edge_support[i]) {
            num_edges_with_max_support++;
        }
    }
    printf("kmax = %d, Edges in kmax-truss = %d\n", max_support + 2, num_edges_with_max_support);
}


int get_max_support(int *edge_support, long num_edges, int &max_edges_num) {
    int max_support = 0;
    for (long i = 0; i < num_edges; i++) {
        if (max_support < edge_support[i]) {
            max_support = edge_support[i];
            max_edges_num = 1;
        } else if (max_support == edge_support[i]) {
            max_edges_num++;
        }
    }
    return max_support;
}

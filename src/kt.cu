#include "kt.h"
#include "log.h"
#include "timer.h"
#include "edge.h"
#include <bitset>

#define min(a, b) ((a)<(b)?(a):(b))

__inline__ __device__ void swap(eid_t &t1, eid_t &t2) {
    eid_t tmp = t1;
    t1 = t2;
    t2 = tmp;
}

__inline__ __device__ int linear_search(vid_t *arr, int beg, int end, int val) {
    for (auto i = beg; i < end; i++) {
        if (arr[i] >= val) {
            return i;
        }
    }
    return end;
}

__inline__ __device__ int binary_search(vid_t *arr, int beg, int end, int val) {
    while (beg < end) {
        int mid = ((uint64_t)beg + end) / 2;
        if (*(arr + mid) >= val) {
            end = mid;
        }
        else {
            beg = mid + 1;
        }
    }
    return beg;
}

__inline__ __device__ int binary_search_v2(vid_t *arr, int beg, int end, int val) {
    int l = beg, r = end - 1;
    while (l <= r) {
        int mid = ((uint64_t)l + r) / 2;
        if (*(arr + mid) == val) {
            return mid;
        }
        else if (*(arr + mid) > val) {
            r = mid - 1;
        }
        else {
            l = mid + 1;
        }
    }
    return -1;
}

__global__ void start_edge_kernel(eid_t *dev_num_edges, vid_t *dev_adj, eid_t *dev_start_edge, vid_t n) {

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (vid_t i = tid; i <= n; i += stride) {
        eid_t j = dev_num_edges[i];
        eid_t endIndex = dev_num_edges[i + 1];

        while (j < endIndex) {
            if (dev_adj[j] > i)
                break;
            j++;
        }
        dev_start_edge[i] = j;
    }
}

void init_start_edge(eid_t *dev_num_edges, vid_t *dev_adj, eid_t *dev_start_edge, vid_t n) {
    dim3 blockSize(64);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    start_edge_kernel<<<gridSize, blockSize>>>(dev_num_edges, dev_adj, dev_start_edge, n);
    cudaDeviceSynchronize();
}

__global__ void tc_kernel(eid_t *dev_num_edges, vid_t *dev_adj, eid_t *dev_edge_id, eid_t *dev_start_edge, int *dev_edge_support,
                          bool *dev_processed, vid_t n, int mul) {

    uint32_t b_id = blockIdx.x / mul;
    uint32_t b_stride = gridDim.x / mul;
    uint32_t t_id = threadIdx.x + (blockIdx.x % mul) * blockDim.x;
    uint32_t t_stride = blockDim.x * mul;
    for (int u = b_id; u <= n; u += b_stride) {

        for (eid_t j = dev_num_edges[u] + t_id; j < dev_start_edge[u]; j += t_stride) {
            vid_t v = dev_adj[j];

            for (eid_t k = dev_num_edges[v + 1] - 1; k >= dev_start_edge[v]; k--) {
                vid_t w = dev_adj[k];
                if (w <= u) break;
                int inx = binary_search_v2(dev_adj, dev_start_edge[u], dev_num_edges[u + 1], w);
                if (inx != -1) {
                    eid_t e1 = dev_edge_id[inx], e2 = dev_edge_id[j], e3 = dev_edge_id[k];
                    atomicAdd(&dev_edge_support[e1], 1);
                    atomicAdd(&dev_edge_support[e2], 1);
                    atomicAdd(&dev_edge_support[e3], 1);
                }
            }
        }
    }
}

void triangleCounting(eid_t *dev_num_edges, vid_t *dev_adj, eid_t *dev_edge_id, eid_t *dev_start_edge, int *dev_edge_support,
                      bool *dev_processed, vid_t n) {

    dim3 blockSize(1024);
    int mul = 2;
    dim3 gridSize(n * mul);
    tc_kernel<<<gridSize, blockSize>>>(dev_num_edges, dev_adj, dev_edge_id, dev_start_edge, dev_edge_support,dev_processed, n, mul);
    cudaDeviceSynchronize();
}

__global__ void count_zero_tri_kernel(int *dev_edge_support, bool *dev_processed, eid_t n, uint32_t *count) {

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;

    uint32_t local_count = 0;
    for (eid_t i = tid; i < n; i += stride) {
        if (dev_edge_support[i] == 0 && !dev_processed[i]) {
            local_count++;
            dev_processed[i] = true;
        }
    }
    atomicAdd(count, local_count);
}

int count_num_edges_with_zero_tri(int *dev_edge_support, bool *dev_processed, eid_t n) {
    dim3 blockSize(64);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    uint32_t count;
    uint32_t *dev_count;
    cudaMalloc((void **)&dev_count, sizeof(uint32_t));
    count_zero_tri_kernel<<<gridSize, blockSize>>>(dev_edge_support, dev_processed, n, dev_count);
    cudaDeviceSynchronize();
    cudaMemcpy((void *)&count, dev_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(dev_count);
    return count;
}

__global__ void scan_edges_kernel(eid_t num_edges, int *dev_edge_support, int level, eid_t *dev_curr, unsigned int *dev_curr_tail,
                                  bool *dev_in_curr, bool *dev_processed) {
    // Size of cache line
    const long BUFFER_SIZE_BYTES = 2048;
    const long BUFFER_SIZE = BUFFER_SIZE_BYTES / sizeof(vid_t);

    vid_t buff[BUFFER_SIZE];
    long index = 0;

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (long i = tid; i < num_edges; i += stride) {
        if (dev_edge_support[i] <= level && !dev_processed[i]) {
            buff[index] = i;
            dev_in_curr[i] = true;
            index++;

            if (index >= BUFFER_SIZE) {
                long tempIdx = atomicAdd(dev_curr_tail, BUFFER_SIZE);

                for (long j = 0; j < BUFFER_SIZE; j++) {
                    dev_curr[tempIdx + j] = buff[j];
                }
                index = 0;
            }
        }
    }

    if (index > 0) {
        long tempIdx = atomicAdd(dev_curr_tail, index);

        for (long j = 0; j < index; j++) {
            dev_curr[tempIdx + j] = buff[j];
        }
    }
}

void scan_sub_level(eid_t num_edges, int *dev_edge_support, int level, eid_t *dev_curr, unsigned int *dev_curr_tail,
                        bool *dev_in_curr, bool *dev_processed) {
    dim3 blockSize(64);
    dim3 gridSize((num_edges + blockSize.x - 1) / blockSize.x);
    scan_edges_kernel<<<gridSize, blockSize>>>(num_edges, dev_edge_support, level, dev_curr, dev_curr_tail, dev_in_curr, dev_processed);
    cudaDeviceSynchronize();
}

__inline__ __device__ void update_support(eid_t edge_inx, int level, int *dev_edge_support, eid_t *dev_next,
                                          bool *dev_in_next, unsigned int *dev_next_tail) {
    int cur = atomicSub(&dev_edge_support[edge_inx], 1);
    //当e的支持度为level+1时，由于减1，所以当前e这条边的支持度为level，加入next队列等待下一轮处理
    if (cur == (level + 1)) {
        auto insert_inx = atomicAdd(dev_next_tail, 1);
        dev_next[insert_inx] = edge_inx;
        dev_in_next[edge_inx] = true;
    }

    if (cur <= level) {
        atomicAdd(&dev_edge_support[edge_inx], 1);
    }
}

__global__ void process_edges_kernel(eid_t *dev_num_edges, vid_t *dev_adj, eid_t *dev_edge_id, eid_t *dev_curr, bool *dev_in_curr,
                                     unsigned int *dev_curr_tail, int *dev_edge_support, int level, eid_t *dev_next, bool *dev_in_next,
                                     unsigned int *dev_next_tail, bool *dev_processed, Edge *dev_id_to_edge, eid_t *dev_start_edge) {

    uint32_t b_id = blockIdx.x;
    uint32_t b_stride = gridDim.x;
    uint32_t t_id = threadIdx.x;
    uint32_t t_stride = blockDim.x;

    for (auto i = b_id; i < *dev_curr_tail; i += b_stride) {
        //process edge <u,v>
        eid_t e1 = dev_curr[i];

        Edge edge = dev_id_to_edge[e1];

        vid_t u = edge.u;
        vid_t v = edge.v;

        eid_t u_start = dev_num_edges[u], u_end = dev_num_edges[u + 1];
        eid_t v_start = dev_num_edges[v], v_end = dev_num_edges[v + 1];
        if ((u_end - u_start) > (v_end - v_start)) {
            swap(u_start, v_start);
            swap(u_end, v_end);
        }

        for (int j = u_start + t_id; j < u_end; j += t_stride) {
            int w = dev_adj[j];
            int inx = binary_search_v2(dev_adj, v_start, v_end, w);
            if (inx == -1)
                continue;

            eid_t e2 = dev_edge_id[j];  //<v,w>
            eid_t e3 = dev_edge_id[inx];//<u,w>

            bool is_peel_e2 = !dev_in_curr[e2];
            bool is_peel_e3 = !dev_in_curr[e3];

            if (is_peel_e2 || is_peel_e3) {
                if ((!dev_processed[e2]) && (!dev_processed[e3])) {
                    if (is_peel_e2 && is_peel_e3) {
                        update_support(e2, level, dev_edge_support, dev_next, dev_in_next, dev_next_tail);
                        update_support(e3, level, dev_edge_support, dev_next, dev_in_next, dev_next_tail);
                    } else if (is_peel_e2) {
                        if (e1 < e3) {
                            update_support(e2, level, dev_edge_support, dev_next, dev_in_next, dev_next_tail);
                        }
                    } else {
                        if (e1 < e2) {
                            update_support(e3, level, dev_edge_support, dev_next, dev_in_next, dev_next_tail);
                        }
                    }
                }
            }
        }
    }
}

__global__ void update_processed_kernel(bool *dev_in_curr, unsigned int *dev_curr_tail, eid_t *dev_curr, bool *dev_processed) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (auto i = tid; i < *dev_curr_tail; i += stride) {
        eid_t e = dev_curr[i];

        dev_processed[e] = true;
        dev_in_curr[e] = false;
    }
}

void sub_level_process(eid_t *dev_num_edges, vid_t *dev_adj, eid_t *dev_edge_id, eid_t *dev_curr, bool *dev_in_curr,
                           unsigned int *dev_curr_tail, int *dev_edge_support, int level, eid_t *dev_next, bool *dev_in_next,
                           unsigned int *dev_next_tail, bool *dev_processed, Edge *dev_id_to_edge, eid_t *dev_start_edge, unsigned int cur_tail) {

    dim3 blockSize(1024);
    dim3 gridSize((cur_tail + blockSize.x - 1) / blockSize.x);

    process_edges_kernel<<<cur_tail, blockSize>>>(dev_num_edges, dev_adj, dev_edge_id, dev_curr, dev_in_curr, dev_curr_tail, dev_edge_support, level,
                         dev_next, dev_in_next, dev_next_tail, dev_processed, dev_id_to_edge, dev_start_edge);
    cudaDeviceSynchronize();
    update_processed_kernel<<<gridSize, blockSize>>>(dev_in_curr, dev_curr_tail, dev_curr, dev_processed);
    cudaDeviceSynchronize();
}

__global__ void get_upper_kernel(eid_t *dev_num_edges, vid_t *dev_adj, eid_t *dev_upper_start, eid_t *sum_upper, int n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (vid_t u = tid; u <= n; u += stride) {
        dev_upper_start[u] = (dev_num_edges[u + 1] - dev_num_edges[u] < 64)
                ? linear_search(dev_adj, dev_num_edges[u], dev_num_edges[u + 1], u)
                : binary_search(dev_adj, dev_num_edges[u], dev_num_edges[u + 1], u);

        sum_upper[u] = dev_num_edges[u + 1] - dev_upper_start[u];
    }
}

__global__ void get_eid_ptr_kernel(eid_t *dev_eid_ptr, eid_t *dev_sum_upper, int n) {
    dev_eid_ptr[0] = 0;
    for (vid_t u = 0; u <= n; u++) {
        dev_eid_ptr[u + 1] = dev_eid_ptr[u] + dev_sum_upper[u];
    }
}

__global__ void get_edge_kernel(eid_t *dev_num_edges, vid_t *dev_adj, eid_t *dev_upper_start, eid_t *dev_edge_id, Edge *dev_id_to_edge,
                    eid_t *dev_eid_ptr, int n) {

    uint32_t b_id = blockIdx.x;
    uint32_t b_stride = gridDim.x;
    uint32_t t_id = threadIdx.x;
    uint32_t t_stride = blockDim.x;

    for (vid_t u = b_id; u <= n; u += b_stride) {

        for (eid_t j = dev_num_edges[u] + t_id; j < dev_num_edges[u + 1]; j += t_stride) {
            vid_t v = dev_adj[j];
            if (u < v) {
                Edge e = Edge(u, v);

                eid_t edge_id = dev_eid_ptr[u] + (j - dev_upper_start[u]);
                dev_edge_id[j] = edge_id;
                dev_id_to_edge[edge_id] = e;
            }
            else {
                int off = binary_search(dev_adj, dev_upper_start[v], dev_num_edges[v + 1], u);
                eid_t edge_id = dev_eid_ptr[v] + (off - dev_upper_start[v]);
                dev_edge_id[j] = edge_id;
            }
        }
    }
}

void init_eid_and_get_edge(eid_t *dev_num_edges, vid_t *dev_adj, eid_t *edge_id, Edge *id_to_edge, int n) {

    eid_t *upper_start, *sum_upper, *eid_ptr;
    cudaMalloc((void **)&upper_start, (n + 1) * sizeof(eid_t));
    cudaMalloc((void **)&sum_upper, (n + 1) * sizeof(eid_t));
    cudaMalloc((void **)&eid_ptr, (n + 2) * sizeof(eid_t));
    dim3 blockSize(64);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    get_upper_kernel<<<gridSize, blockSize>>>(dev_num_edges, dev_adj, upper_start, sum_upper, n);
    cudaDeviceSynchronize();

    get_eid_ptr_kernel<<<1, 1>>>(eid_ptr, sum_upper, n);
    cudaDeviceSynchronize();
    blockSize = 1024;
    get_edge_kernel<<<n, blockSize>>>(dev_num_edges, dev_adj, upper_start, edge_id, id_to_edge, eid_ptr, n);
    cudaDeviceSynchronize();

    cudaFree(upper_start);
    cudaFree(sum_upper);
    cudaFree(eid_ptr);
}

void init_device_variable(eid_t numEdges, eid_t m, vid_t n, Graph *g, int *&edge_support, eid_t *&dev_num_edges,
                          eid_t *&dev_edge_id, vid_t *&dev_adj, int *&dev_edge_support, Edge *&dev_id_to_edge, bool *&dev_processed,
                          eid_t *&dev_start_edge, eid_t *&dev_curr, bool *&dev_in_curr, eid_t *&dev_next, bool *&dev_in_next,
                          unsigned int *&dev_curr_tail, unsigned int *&dev_next_tail) {
    cudaMalloc((void **)&dev_num_edges, (n + 1) * sizeof(eid_t));
    cudaMalloc((void **)&dev_edge_id, m * sizeof(eid_t));
    cudaMalloc((void **)&dev_adj, m * sizeof(vid_t));
    cudaMalloc((void **)&dev_edge_support, numEdges * sizeof(int));
    cudaMalloc((void **)&dev_id_to_edge, numEdges * sizeof(Edge));

    cudaMalloc((void **)&dev_processed, numEdges * sizeof(bool));
    cudaMalloc((void **)&dev_start_edge, (n + 1) * sizeof(eid_t));

    cudaMalloc((void **)&dev_curr, numEdges * sizeof(eid_t));
    cudaMalloc((void **)&dev_in_curr, numEdges * sizeof(bool));
    cudaMalloc((void **)&dev_next, numEdges * sizeof(eid_t));
    cudaMalloc((void **)&dev_in_next, numEdges * sizeof(bool));

    cudaMalloc((void **)&dev_curr_tail, sizeof(unsigned int));
    cudaMalloc((void **)&dev_next_tail, sizeof(unsigned int));

    cudaMemcpy((void *)dev_num_edges, (void *)g->num_edges, (n + 1) * sizeof(eid_t), cudaMemcpyHostToDevice);
    //cudaMemcpy((void *)dev_edge_id, (void *)g->edge_id, m * sizeof(eid_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dev_adj, (void *)g->adj, m * sizeof(vid_t), cudaMemcpyHostToDevice);
    //cudaMemcpy((void *)dev_id_to_edge, (void *)id_to_edge, numEdges * sizeof(Edge), cudaMemcpyHostToDevice);

    cudaMemset((void *)dev_processed, 0, numEdges * sizeof(bool));
    cudaMemset((void *)dev_edge_support, 0, numEdges * sizeof(int));

    cudaMemset((void *)dev_in_curr, 0, sizeof(unsigned int));
    cudaMemset((void *)dev_in_next, 0, sizeof(unsigned int));
    cudaMemset((void *)dev_curr_tail, 0, sizeof(unsigned int));
    cudaMemset((void *)dev_next_tail, 0, sizeof(unsigned int));
}

void calc_kmax_truss(Graph *g, int *edge_support, int beg_level, bool &is_peel_much) {
    Timer timer;
    eid_t numEdges = g->m / 2;
    eid_t m = g->m;
    vid_t n = g->n;

    int level = 0;
    eid_t remain = numEdges;

    unsigned int curr_tail = 0;

    log_debug("begin init device variable");
    eid_t *dev_num_edges = nullptr, *dev_edge_id = nullptr, *dev_start_edge = nullptr, *dev_curr = nullptr, *dev_next = nullptr;
    vid_t *dev_adj = nullptr;
    int *dev_edge_support = nullptr;
    Edge *dev_id_to_edge = nullptr;
    bool *dev_processed = nullptr, *dev_in_curr = nullptr, *dev_in_next = nullptr;
    unsigned int *dev_curr_tail = nullptr, *dev_next_tail = nullptr;

    init_device_variable(numEdges, m, n, g, edge_support, dev_num_edges, dev_edge_id, dev_adj, dev_edge_support,
                         dev_id_to_edge, dev_processed, dev_start_edge, dev_curr, dev_in_curr, dev_next, dev_in_next,
                         dev_curr_tail, dev_next_tail);
    log_debug("end of init device variable . time:%.5lf sec", timer.timeAndReset());

    log_debug("begin init eid and get edge.");
    init_eid_and_get_edge(dev_num_edges, dev_adj, dev_edge_id, dev_id_to_edge, n);
    log_debug("init eid and get edge. time:%.5lf sec", timer.timeAndReset());

    log_debug("begin init start_edge");
    init_start_edge(dev_num_edges, dev_adj, dev_start_edge, n);
    log_debug("end of init start_edge. time:%.5lf sec", timer.timeAndReset());

    log_debug("begin triangleCounting.");
    triangleCounting(dev_num_edges, dev_adj, dev_edge_id, dev_start_edge, dev_edge_support, dev_processed, n);
    log_debug("triangleCounting time:%.5lf sec", timer.timeAndReset());

    log_debug("begin get zero_tri num.");
    int num_edges_zero_tri = count_num_edges_with_zero_tri(dev_edge_support, dev_processed, numEdges);
    log_debug("end of get zero_tri num. num:%d, time:%.5lf sec", num_edges_zero_tri, timer.timeAndReset());
    remain -= num_edges_zero_tri;

    level = beg_level;
    log_debug("begin calc k-max truss.");
    while (remain > 0) {
        scan_sub_level(numEdges, dev_edge_support, level, dev_curr, dev_curr_tail, dev_in_curr, dev_processed);
        cudaMemcpy((void *)&curr_tail, (void *)dev_curr_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        while (curr_tail > 0) {
            if (remain == curr_tail) {
                remain = 0;
                break;
            }
            if (level <= 1) {
                is_peel_much = false;
            }
            remain -= curr_tail;
            log_debug("level-%d curr_tail:%d remain:%d", level, curr_tail, remain);
            sub_level_process(dev_num_edges, dev_adj, dev_edge_id, dev_curr, dev_in_curr, dev_curr_tail, dev_edge_support,
                                  level, dev_next, dev_in_next, dev_next_tail, dev_processed, dev_id_to_edge, dev_start_edge, curr_tail);

            eid_t *temp_curr = dev_curr;
            dev_curr = dev_next;
            dev_next = temp_curr;

            bool *temp_in_curr = dev_in_curr;
            dev_in_curr = dev_in_next;
            dev_in_next = temp_in_curr;

            cudaMemcpy((void *)dev_curr_tail, (void *)dev_next_tail, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            cudaMemset((void *)dev_next_tail, 0, sizeof(unsigned int));
            cudaMemcpy((void *)&curr_tail, (void *)dev_curr_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        }
        level = level + 1;
    }
    log_debug("calc k-max truss time:%.5lf sec", timer.timeAndReset());

    cudaMemcpy((void *)edge_support, (void *)dev_edge_support, numEdges * sizeof(int), cudaMemcpyDeviceToHost);

    log_debug("begin free variable");
    cudaFree(dev_num_edges);
    cudaFree(dev_edge_id);
    cudaFree(dev_adj);
    cudaFree(dev_edge_support);
    cudaFree(dev_id_to_edge);
    cudaFree(dev_processed);
    cudaFree(dev_start_edge);
    cudaFree(dev_curr);
    cudaFree(dev_next);
    cudaFree(dev_in_curr);
    cudaFree(dev_in_next);
    cudaFree(dev_curr_tail);
    cudaFree(dev_next_tail);
    log_debug("end of free variable. time:%.5lf sec", timer.timeAndReset());
}

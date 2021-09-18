#include "kt.h"
#include "util.h"
using namespace std;

int main(int argc, char *argv[]) {
    if (string(argv[argc - 1]) != "debug") {
        log_set_quiet(true);
    }

    Timer timer;
    log_debug("max thread:%d", omp_get_max_threads());

    parseArg(argc, argv);
    char *file_path = optarg;

    Graph g;
    int edge_num, max_vertex;
    vid_t *u = nullptr, *v = nullptr;
    int *degree = nullptr;
    log_debug("begin read data file.");
    read_data_from_file(file_path, u, v, degree, edge_num, max_vertex);
    log_debug("edge_num:%d, max_vertex:%d", edge_num, max_vertex);
    log_debug("read data time:%.5lf sec", timer.timeAndReset());

    g.num_edges = (eid_t *)malloc((max_vertex + 3) * sizeof(eid_t));
    g.adj = (vid_t *)malloc(edge_num * sizeof(vid_t));
    int *edge_support = (int *) malloc(edge_num / 2 * sizeof(int));

    int upper_k = get_upper_k(degree, u, v, max_vertex, edge_num);
    log_debug("upper_k:%d. time:%.5lf sec", upper_k, timer.timeAndReset());
    int lower_k = upper_k / 3;
    log_debug("lower_k:%d", lower_k);
    int beg_level = 1;

    init_graph(&g, u, v, degree, edge_num, max_vertex, lower_k);
    log_debug("init graph, n:%d, m:%d. time:%.5lf sec", g.n, g.m, timer.timeAndReset());

    log_debug("begin compute kmax truss. beg_level:%d lower_k:%d", beg_level, lower_k);
    bool is_peel_much = true;
    calc_kmax_truss(&g, edge_support, beg_level, is_peel_much);
    log_debug("calc kmax truss end. is_peel_much:%d, time:%.5lf sec", is_peel_much, timer.timeAndReset());

    int max_edges_num = 0;
    int max_sup = get_max_support(edge_support, g.m / 2, max_edges_num);
    log_debug("max_sup:%d. time:%.5lf sec", max_sup, timer.timeAndReset());

    beg_level = max_sup <= 1 ? 1 : (max_sup - 1);
    lower_k = beg_level * (is_peel_much ? 3 : 1);
    while (1) {
        memset(g.num_edges, 0, (max_vertex + 3) * sizeof(eid_t));
        memset(g.adj, 0, edge_num * sizeof(vid_t));
        init_graph(&g, u, v, degree, edge_num, max_vertex, lower_k);
        log_debug("init graph, n:%d, m:%d. time:%.5lf sec", g.n, g.m, timer.timeAndReset());

        log_debug("begin compute kmax truss. beg_level:%d, lower_k:%d", beg_level, lower_k);
        calc_kmax_truss(&g, edge_support, beg_level, is_peel_much);
        log_debug("calc kmax truss end. time:%.5lf sec", timer.timeAndReset());
        int tmp_max_edges_num = 0;
        int tmp_max_sup = get_max_support(edge_support, g.m / 2, tmp_max_edges_num);
        if (tmp_max_sup <= beg_level || ((tmp_max_sup == max_sup) && (tmp_max_edges_num < max_edges_num))) {
            lower_k /= 2;
            log_debug("peel too many edges, recalculate. lower_k:%d-------------------", lower_k);
        }
        else {
            max_sup = tmp_max_sup;
            max_edges_num = tmp_max_edges_num;
            break;
        }
    }

    printf("kmax = %d, Edges in kmax-truss = %d\n", max_sup + 2, max_edges_num);
    free(u);
    free(v);
    free(degree);
    g.free_graph();
    log_debug("total time:%.5lf sec", timer.t_time());
}

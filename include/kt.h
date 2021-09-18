#pragma once

#include <cstring>
#include <string>
#include "graph.h"
#include "timer.h"

void calc_kmax_truss(Graph *g, int *edge_support, int beg_level, bool &is_peel_much);


#pragma once

struct Edge {
    vid_t u;
    vid_t v;

    __host__ __device__
    Edge() {
        this->u = 0;
        this->v = 0;
    }

    __host__ __device__
    Edge(vid_t u, vid_t v) {
        this->u = u;
        this->v = v;
    }
};
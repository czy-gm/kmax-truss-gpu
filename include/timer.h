#pragma once
#include <sys/time.h>

class Timer {
public:
    struct timeval f_tp;
    struct timeval tp;

    Timer() {
        gettimeofday(&tp, nullptr);
        f_tp = tp;
    }

    double time() {
        struct timeval cur;
        gettimeofday(&cur, nullptr);
        return ((double)(cur.tv_sec) + cur.tv_usec * 1e-6) - ((double)(tp.tv_sec) + tp.tv_usec * 1e-6);
    }

    double timeAndReset() {
        struct timeval cur;
        gettimeofday(&cur, nullptr);
        auto tmp = tp;
        tp = cur;
        return ((double)(cur.tv_sec) + cur.tv_usec * 1e-6) - ((double)(tmp.tv_sec) + tmp.tv_usec * 1e-6);
    }

    double t_time() {
        struct timeval cur;
        gettimeofday(&cur, nullptr);
        return ((double)(cur.tv_sec) + cur.tv_usec * 1e-6) - ((double)(f_tp.tv_sec) + f_tp.tv_usec * 1e-6);
    }

    void reset() {
        gettimeofday(&tp, nullptr);
    }

};
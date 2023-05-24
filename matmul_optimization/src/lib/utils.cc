#include <sys/time.h>

#include "matmul.h"

namespace matmul {

float MatmulOperator::interval_to_us(struct timeval *start, struct timeval *end) {
    float us_seconds = (end->tv_sec - start->tv_sec) * 1000000 + (end->tv_usec - start->tv_usec);
    return us_seconds;
}

}  // namespace matmul

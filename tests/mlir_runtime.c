#include <string.h>
#include <stdlib.h>

void memrefCopy(long size, void* src, void* dst) {
    memcpy(dst, src, size);
}

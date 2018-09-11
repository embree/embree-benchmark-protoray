/*
 * Copyright (C) 2010-2018 Attila T. Afra <attila.afra@gmail.com>
 */

#include <immintrin.h>
#include "tasking.h"

// This improves TBB performance a bit (avoids calling into the kernel)
#if !defined(_WIN32)
int sched_yield()
{
    _mm_pause();
    return 0;
}
#endif

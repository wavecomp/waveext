/*
 * Copyright (c) 2010-2018 Wave Computing, Inc. and its applicable licensors.   
 * All rights reserved; provided, that any files identified as open source shall
 * be governed by the specific open source license(s) applicable to such files. 
 *
 * For any files associated with distributions under the Apache 2.0 license, 
 * full attribution to The Apache Software Foundation is given via the license 
 * below.
 */
/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include <stdint.h>
#include <stdlib.h>

#define USE_VECTOR_INSTRUCTIONS

#ifdef USE_VECTOR_INSTRUCTIONS
#include <emmintrin.h>
#include <smmintrin.h>
#endif

void mm_nt(int32_t *a_m, int a_x, int a_y, int32_t *b_m, int b_x, int b_y, int32_t *z_m, int shift, bool stochastic)
{
    int i, j, k;

    if (stochastic) {  // stochastic rounding
#if defined(USE_VECTOR_INSTRUCTIONS)
        __m128i x0, x1, xr;
        uint32_t rnd[4];
        rnd[0] = rand();  // random seed
        rnd[1] = rand();
        rnd[2] = rand();
        rnd[3] = rand();
        __m128i xrnd = _mm_load_si128((__m128i*)rnd);
        __m128i xc1 = _mm_set1_epi32(1664525);
        __m128i xc2 = _mm_set1_epi32(1013904223);
        __m128i *a, *b;
    
        for (i = 0; i < a_x; i++) {
            b = (__m128i *)b_m;
            for (j = 0; j < b_x; j++) {
                __m128i xv = _mm_setzero_si128();
                a = (__m128i *)(a_m + i * a_y);
                for (k = 0; k < a_y>>2; k++) {
                    x0 = _mm_load_si128(a++);
                    x1 = _mm_load_si128(b++);
                    xr = _mm_srli_epi32(xrnd, 32 - shift);
                    x0 = _mm_mullo_epi32(x0, x1);
                    xrnd = _mm_mullo_epi32(xrnd, xc1);
                    x0 = _mm_add_epi32(x0, xr);
                    xrnd = _mm_add_epi32(xrnd, xc2);
                    x0 = _mm_srai_epi32(x0, shift);
                    xv = _mm_add_epi32(xv, x0);
                }
                int32_t t[4];
                xv = _mm_hadd_epi32(xv, xv);
                xv = _mm_hadd_epi32(xv, xv);
                _mm_storeu_si128((__m128i*)t, xv);
                *z_m++ = t[0];
            }
        }
#else
        int32_t round, tmp;
        uint32_t rnd = rand();

        for (i = 0; i < a_x; i++) {
            for (j = 0; j < b_x; j++) {
                int32_t v = 0;
                for (k = 0; k < a_y; k++) {
                    tmp = a_m[i*a_y+k] * b_m[j*b_y+k];
                    round = rnd >> (32 - shift);
                    tmp = (tmp + round) >> shift;
                    rnd = rnd * 1664525 + 1013904223;
                    v += tmp;
                }
                *z_m++ = v;
            }
        }
#endif
    } else {  // convergent rounding
        int32_t round = 1 << (shift - 1);
        int32_t mask = (round << 1) - 1;

#if defined(USE_VECTOR_INSTRUCTIONS)
        __m128i x0, x1;
        __m128i xr = _mm_set1_epi32(round);
        __m128i xm = _mm_set1_epi32(mask);
        __m128i xo = _mm_set1_epi32(1);
        __m128i *a, *b;
    
        for (i = 0; i < a_x; i++) {
            b = (__m128i *)b_m;
            for (j = 0; j < b_x; j++) {
                __m128i xv = _mm_setzero_si128();
                a = (__m128i *)(a_m + i * a_y);
                for (k = 0; k < a_y>>2; k++) {
                    x0 = _mm_load_si128(a++);
                    x1 = _mm_load_si128(b++);
                    x0 = _mm_mullo_epi32(x0, x1);
                    x1 = _mm_and_si128(x0, xm);
                    x0 = _mm_add_epi32(x0, xr);
                    x1 = _mm_cmpeq_epi32(x1, xr);
                    x0 = _mm_srai_epi32(x0, shift);
                    x1 = _mm_sub_epi32(x1, xo);
                    x0 = _mm_and_si128(x0, x1);
                    xv = _mm_add_epi32(xv, x0);
                }
                int32_t t[4];
                xv = _mm_hadd_epi32(xv, xv);
                xv = _mm_hadd_epi32(xv, xv);
                _mm_storeu_si128((__m128i*)t, xv);
                *z_m++ = t[0];
            }
        }
#else
        int32_t r, tmp;

        for (i = 0; i < a_x; i++) {
            for (j = 0; j < b_x; j++) {
                int32_t v = 0;
                for (k = 0; k < a_y; k++) {
                    tmp = a_m[i*a_y+k] * b_m[j*b_y+k];
                    r = tmp & mask;
                    tmp = (tmp + round) >> shift;
                    if (r == round)
                        tmp &= ~1;
                    v += tmp;
                }
                *z_m++ = v;
            }
        }
#endif
    }
}

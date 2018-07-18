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

#pragma once

#include "Eigen/Core"


template<typename T>
void fp_clear(T& m_out, int x, int y)
{
    for (int j = 0; j < x; j++) {
        for (int k = 0; k < y; k++) {
            m_out(j, k) = 0.f;
        }
    }
}

template<typename Tin, typename Tout> static inline
void mm_nn(bool zero, Tin& a_m, int a_x, int a_y, Tin& b_m, int b_x, int b_y, Tout& z_m)
{
    // M[i] : A.dim[0]
    // N[j] : B.dim[1]
    // K[k] : A.dim[1] == B.dim[0]
    // output: (A.dim[0], B.dim[1])
    for (int i = 0; i < a_x; i++) {
        for (int j = 0; j < b_y; j++) {
            // printf("generating out(%i, %i)\n", i, j);
            float val = 0.f;
            for (int k = 0; k < a_y; k++) {
                val += a_m(i, k) * b_m(k, j);
            }
            z_m(i, j) = zero ? val : z_m(i, j) + val;
        }
    }
}

template<typename Tin, typename Tout> static inline
void mm_tn(bool zero, Tin& a_m, int a_x, int a_y, Tin& b_m, int b_x, int b_y, Tout& z_m)
{
    // A (grad) is transposed       (T,N)
    // M[i] : A.dim[1]
    // N[j] : B.dim[1]
    // K[k] : A.dim[0] == B.dim[0]
    // (k, i); (k, j)
    // output: (A.dim[1], B.dim[1])
    // KS: there's probably an easier way to clear this tensor
    if (zero) {
        fp_clear(z_m, a_y, b_y);
    }
    for (int i = 0; i < a_y; i++) {
        for (int k = 0; k < a_x; k++) {
            for (int j = 0; j < b_y; j++) {
                z_m(i, j) += a_m(k, i) * b_m(k, j);
            }
        }
    }        
}

template<typename Tin, typename Tout> static inline
void mm_nt(bool zero, Tin& a_m, int a_x, int a_y, Tin& b_m, int b_x, int b_y, Tout& z_m)
{
    // B (weights) is transposed    (N,T)
    // M[i] : A.dim[0]
    // N[j] : B.dim[0]
    // K[k] : A.dim[1] == B.dim[1]
    // (i, k); (j, k)
    // output: (A.dim[0], B.dim[0])
    for (int i = 0; i < a_x; i++) {
        for (int j = 0; j < b_x; j++) {            
            float val = 0.f;
            for (int k = 0; k < a_y; k++) {
                val += a_m(i, k) * b_m(j, k);
            }
            z_m(i, j) = zero ? val : z_m(i, j) + val;
        }
    }
}

template<typename Tin, typename Tout> static inline
void mm_tt(bool zero, Tin& a_m, int a_x, int a_y, Tin& b_m, int b_x, int b_y, Tout& z_m)
{
    // A, B is transposed       (T,T)
    // M[i] : A.dim[1]
    // N[j] : B.dim[0]
    // K[k] : A.dim[0] == B.dim[1]
    // (k, i); (j, k)
    // output: (A.dim[1], B.dim[0])
    // KS: there's probably an easier way to clear this tensor
    if (zero) {
        fp_clear(z_m, a_y, b_x);
    }    
    for (int i = 0; i < a_y; i++) {
        for (int k = 0; k < a_x; k++) {
            for (int j = 0; j < b_x; j++) {
                z_m(i, j) += a_m(k, i) * b_m(j, k);
            }
        }
    }   
}

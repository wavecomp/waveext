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

#include "dyn_fx_pt.h"

template<typename T>
void dfx_clear(const fxbp& bp_set, T& m_out, int rows, int cols)
{
    for (int j = 0; j < rows; j++) {
        for (int k = 0; k < cols; k++) {
            m_out(j, k).set_fxbp(bp_set);
            m_out(j, k) = 0;
        }
    }
}

template<typename Tin, typename Tout> static inline
void mm_nn(bool zero, const fxbp& bp, Tin& a_m, int a_x, int a_y, Tin& b_m, int b_x, int b_y, Tout& z_m)
{
    // M[i] : A.dim[0]
    // N[j] : B.dim[1]
    // K[k] : A.dim[1] == B.dim[0]
    // output: (A.dim[0], B.dim[1])
    for (int i = 0; i < a_x; i++) {
        for (int j = 0; j < b_y; j++) {
            DynFxPoint v(bp);
            for (int k = 0; k < a_y; k++) {
                v += a_m(i, k) * b_m(k, j);
            }
            z_m(i, j) = zero ? v : z_m(i, j) + v;
        }
    }
}

template<typename Tin, typename Tout> static inline
void mm_tn(bool zero, const fxbp& bp, Tin& a_m, int a_x, int a_y, Tin& b_m, int b_x, int b_y, Tout& z_m)
{
    // A (grad) is transposed       (T,N)
    // M[i] : A.dim[1]
    // N[j] : B.dim[1]
    // K[k] : A.dim[0] == B.dim[0]
    // (k, i); (k, j)
    // output: (A.dim[1], B.dim[1])
    // KS: there's probably an easier way to clear this tensor
    if (zero) {
        dfx_clear(bp, z_m, a_y, b_y);
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
void mm_nt(bool zero, const fxbp& bp, Tin& a_m, int a_x, int a_y, Tin& b_m, int b_x, int b_y, Tout& z_m)
{
    // B (weights) is transposed    (N,T)
    // M[i] : A.dim[0]
    // N[j] : B.dim[0]
    // K[k] : A.dim[1] == B.dim[1]
    // (i, k); (j, k)
    // output: (A.dim[0], B.dim[0])
    for (int i = 0; i < a_x; i++) {
        for (int j = 0; j < b_x; j++) {            
            DynFxPoint v(bp);
            for (int k = 0; k < a_y; k++) {
                v += a_m(i, k) * b_m(j, k);
            }
            z_m(i, j) = zero ? v : z_m(i, j) + v;
        }
    }
}

template<typename Tin, typename Tout> static inline
void mm_tt(bool zero, const fxbp& bp, Tin& a_m, int a_x, int a_y, Tin& b_m, int b_x, int b_y, Tout& z_m)
{
    // A, B is transposed       (T,T)
    // M[i] : A.dim[1]
    // N[j] : B.dim[0]
    // K[k] : A.dim[0] == B.dim[1]
    // (k, i); (j, k)
    // output: (A.dim[1], B.dim[0])
    // KS: there's probably an easier way to clear this tensor
    if (zero) {
        dfx_clear(bp, z_m, a_y, b_x);
    }    
    for (int i = 0; i < a_y; i++) {
        for (int k = 0; k < a_x; k++) {
            for (int j = 0; j < b_x; j++) {
                z_m(i, j) += a_m(k, i) * b_m(j, k);
            }
        }
    }   
}


// Helper functions
// These functions are commonly used to copy in FP arrays and prepare
// BP matrixes for compute.
template<typename Tout>
void fp2dfx(fxbp& dest_bp, Tout& m_out, const float* flat_arr)
{
    // Destination binary point will be reset to whatever best represents
    // the array to copy.
    if (dest_bp.m_bp == -1 || !dest_bp.m_initialized) {
        dest_bp.set_range_fp(flat_arr, flat_arr + m_out.size());
    }

    for (int i = 0; i < m_out.rows(); i++) {
        for (int j = 0; j < m_out.cols(); j++) {
            m_out(i, j).set_fxbp(dest_bp);
            m_out(i, j) = flat_arr[i*m_out.cols() + j];
        }
    }
}

template<typename Tout>
void fp2dfx(int dest_wl, Tout& m_out, const float* flat_arr)
{
    fxbp dest_bp(-1, dest_wl);
    fp2dfx(dest_bp, m_out, flat_arr);
}


template<typename Tin>
void dfx2fp(fxbp& out_bp, float* conv_out, const Tin& m_out)
{
    // Destination binary point will be reset to whatever best represents
    // the array to copy.
    if (out_bp.m_bp == -1 || !out_bp.m_initialized) {
        out_bp.set_range_dfx(m_out.data(), m_out.data()+m_out.size());
    }
    
    for (int i = 0; i < m_out.rows(); i++) {
        for (int j = 0; j < m_out.cols(); j++) {
            // Individual matrix values are quantized as they are converted
            // out to FP.
            DynFxPoint v(out_bp);
            v = m_out(i, j);
            conv_out[i*m_out.cols() + j] = v.to_fp();
        }
    }
}

template<typename Tin>
void dfx2fp(int out_wl, float* conv_out, const Tin& m_out)
{
    fxbp out_bp(-1, out_wl);
    dfx2fp(out_bp, conv_out, m_out);
}

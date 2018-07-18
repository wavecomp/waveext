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

#include <assert.h>
#include <math.h>
#include <memory>
#include <iostream>
#include <x86intrin.h>

// Forward declarations
class fxbp;
class DynFxPoint;

/* Subclass to represent a shared binary point.
 */
class fxbp
{
public:
    fxbp()                      {m_bp = 0; m_wl = 16; m_initialized = false;}
    fxbp(int32_t b)             {initialize(b, 16);}
    fxbp(int32_t b, int32_t w)  {assert(w <= 32); initialize(b, w);}

    fxbp(float f) {
        int32_t w = 16;
        initialize(w - get_int_bits(f) - 2, w);
    }

    fxbp(float f, int32_t w){
        assert(w <= 32);
        initialize(w - get_int_bits(f) - 2, w);
    }

    void initialize(int32_t b, int32_t w) { m_bp = b; m_wl = w; m_initialized = true; }
    
    template <typename Ti> 
        void set_range_fp(Ti start, Ti end);

    template <typename Ti> 
        void set_range_dfx(Ti start, Ti end);

    void set_range_int32(const int32_t* start, const int32_t* end, int32_t src_bp);

    float               epsilon() const {return (float)((uint32_t)1 << m_bp);}
    static int32_t      get_int_bits(float v);
    
    int32_t m_bp;
    int32_t m_wl;
    bool m_initialized;
};

/* Dynamic Fixed Point type, multi-precision.
 */
class DynFxPoint
{
public:
    DynFxPoint()        {m_v = 0;}
    DynFxPoint(fxbp bp) {m_fxbp = bp; m_v = 0;}

    // Direct initialization ctors
    DynFxPoint(const int& i) { operator=(i); }
    DynFxPoint(const float& f) { operator=(f); }

    // explicit cast operators
    explicit operator float() const { return to_fp(); }
    explicit operator int() const { return get_value() >> m_fxbp.m_bp; }

    void        set_fxbp(fxbp);
    void        set_fxbp(uint32_t b, uint32_t w);
    fxbp        get_fxbp() const {return m_fxbp;}
    
    DynFxPoint& operator=(const float &f);
    DynFxPoint& operator=(const DynFxPoint &v);
    DynFxPoint& operator=(const int &v);
    int32_t&    operator()() {return m_v;}

    DynFxPoint  operator+(const DynFxPoint &y) const;
    DynFxPoint  operator-(const DynFxPoint &y) const;
    DynFxPoint  operator-() const;
    DynFxPoint  operator*(const DynFxPoint &y) const;
    DynFxPoint  operator/(const DynFxPoint &y) const;
    //DynFxPoint  operator%(const DynFxPoint &y) const;

    bool        operator>(const DynFxPoint &y) const;
    bool        operator>=(const DynFxPoint &y) const;
    bool        operator<(const DynFxPoint &y) const;
    bool        operator<=(const DynFxPoint &y) const;
    bool        operator==(const DynFxPoint &y) const;
    bool        operator!=(const DynFxPoint &y) const;

    DynFxPoint& operator+=(const DynFxPoint &y);
    DynFxPoint& operator-=(const DynFxPoint &y);
    DynFxPoint& operator*=(const DynFxPoint &y);
    DynFxPoint& operator/=(const DynFxPoint &y);

    DynFxPoint& operator++();
    DynFxPoint  operator++(int unused);
    DynFxPoint& operator--();
    DynFxPoint  operator--(int unused);
    
    int32_t     get_value() const;
    float       to_fp() const;
    void        set_bits(const int32_t& v) {m_v = v;}
    void        adjust_bp(int bp);

    // public globals
    static void quantize(DynFxPoint* v);

    friend std::ostream &operator<<(std::ostream& strm, const DynFxPoint& obj) {
        strm << obj.to_fp();
        return strm;
    }
    
private:
    int32_t     with_params(unsigned p, unsigned bp, bool round=true) const;
    int64_t     with_params(int64_t v, unsigned p, unsigned bp, bool round=true) const;
    int32_t     round_adjustment(int32_t v, unsigned r) const;
    void        adjust_precision(unsigned p);    
    int32_t     sat_eval(int64_t v) const;
    int32_t     sat_eval(int p, int64_t v) const;
    
    int32_t     m_v;
    fxbp        m_fxbp;
};

template <typename Ti> 
void fxbp::set_range_fp(Ti start, Ti end)
{
    m_initialized = true;
    // Reset the BP based on the range of all tensor values. The goal is to set 
    // the BP to allow the highest/lowest valued elements to be below/above the
    // saturation threshold of the integer field.
    float vmax = 1e-10;
    for (auto it = start; it != end; ++it) {
        float v = fabs(*it);
        vmax = fmax(vmax, v);
    }
    int32_t int_bits = get_int_bits(vmax);
    // Int can be no larger than WL - 1. If so, we will saturate some values in
    // the tensor in order to constrain the precision.
    int32_t fx_int_max = m_wl - 1;
    // Float32 base type can only represent 23 bits of fraction no matter what.
    fx_int_max = fx_int_max > 23 ? 23 : fx_int_max;
    // An int field of -16 yields a BP of 31, which is the max value before
    // some functions begin to return invalid results (max word size == 32).
    const int32_t fx_int_min = -16;
    int_bits = int_bits > fx_int_max ? fx_int_max : int_bits;
    int_bits = int_bits < fx_int_min ? fx_int_min : int_bits;
    
    // FX word format: |signed integer | fraction |
    // BP is the fraction bits 
    m_bp = m_wl - 1 - int_bits;
    assert(m_bp < 32);
    assert(m_bp >= 0);    
}

template <typename Ti> 
void fxbp::set_range_dfx(Ti start, Ti end)
{
    m_initialized = true;
    // We need to set based on the min BP (max number of int bits).
    int32_t fx_int_max = 0;
    int32_t fx_lzc = 1000;
    
    for (auto it = start; it != end; ++it) {
        const fxbp& src_bp = it->get_fxbp();
        int32_t int_bits = 32 - src_bp.m_bp - 1;
        fx_int_max = int_bits > fx_int_max ? int_bits : fx_int_max;
        int32_t v = it->get_value();
        v = v >= 0 ? v : -v;
        int32_t int_lz = __builtin_clz(v) - 1;
        fx_lzc = int_lz < fx_lzc ? int_lz : fx_lzc;
    }
    m_bp = m_wl - (fx_int_max - fx_lzc) - 1;
}

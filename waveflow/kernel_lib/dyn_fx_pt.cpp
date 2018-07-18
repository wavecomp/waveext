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

#include <assert.h>
#include <math.h>
#include <memory>
#include <iostream>

#include "dyn_fx_pt.h"

int32_t fxbp::get_int_bits(float v)
{
    union {
        int32_t i;
        float f;
    } u;
    u.f = fabs(v);
    return (u.i >> 23) - 126;
}

void fxbp::set_range_int32(const int32_t* start, const int32_t* end, int32_t src_bp)
{
    m_initialized = true;
    // We need to set based on the min BP (max number of int bits).
    int32_t fx_int_max = 0;
    int32_t fx_lzc = 1000;
    int32_t maxv = 0x80000000;

    for (auto it = start; it != end; ++it) {
        int32_t v = *it;
        v = v >= 0 ? v : -v;
        maxv = v > maxv ? v : maxv;
    }

    int32_t int_bits = 31 - src_bp;
    fx_int_max = int_bits > fx_int_max ? int_bits : fx_int_max;

    int32_t int_lz = __builtin_clz(maxv) - 1;
    fx_lzc = int_lz < fx_lzc ? int_lz : fx_lzc;

    m_bp = m_wl - (fx_int_max - fx_lzc) - 1;
}

// Note: WON'T modify value. The value must be processed by the caller, if 
// necessary.
void DynFxPoint::set_fxbp(fxbp bp)
{
    m_fxbp = bp;
}

void DynFxPoint::set_fxbp(uint32_t b, uint32_t w)
{
    m_fxbp.m_wl = w;
    m_fxbp.m_bp = b;
    m_fxbp.m_initialized = true;
}

// Convert this BP number to an FP number.
int32_t DynFxPoint::get_value() const
{
    assert(m_fxbp.m_initialized);    
    return m_v;
}

// Convert this BP number to an FP number.
float DynFxPoint::to_fp() const
{
    assert(m_fxbp.m_initialized);
    return (float)m_v / m_fxbp.epsilon();
}

// Version which sets based on a float value.
DynFxPoint& DynFxPoint::operator=(const float &v)
{
    if(!m_fxbp.m_initialized)
    {
        m_fxbp = v;
    }
    m_v  = roundf(v * m_fxbp.epsilon());
    m_v = sat_eval(m_v);
    return *this;
}

// Note that this assignment may not be bit equivalent; it will always use the
// current BP value, even if different from the RHS.
DynFxPoint& DynFxPoint::operator=(const DynFxPoint &v)
{
    // If the current object has no assigned BP data, we will copy the BP
    // settings from the RHS.
    if (!m_fxbp.m_initialized) {
        m_fxbp = v.m_fxbp;
    }
    if (m_fxbp.m_bp == v.m_fxbp.m_bp) {
        if (m_fxbp.m_wl >= v.m_fxbp.m_wl)
            m_v = v.m_v;
        else
            m_v = sat_eval(v.m_v);
    } else
        m_v = v.with_params(m_fxbp.m_wl, m_fxbp.m_bp);
    return *this;
}

// Version which sets based on a float value.
DynFxPoint& DynFxPoint::operator=(const int &v)
{
    if(!m_fxbp.m_initialized)
    {
        m_fxbp.initialize(16, 32);
    }
    int va = abs(v);
    m_v = (va << m_fxbp.m_bp);
    // If v is negative, negate the entire value
    if(v < 0)
        std::cout << " value is negative: " << v << std::endl;
    m_v = (v < 0) ? -m_v : m_v;
    m_v = sat_eval(m_v);
    return *this;
}

DynFxPoint DynFxPoint::operator+(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    // Copy operand A of RHS. This also inherits the BP.
    DynFxPoint dfx(*this);
    // Call the increment operator.
    dfx += y;
    return dfx;
}

DynFxPoint& DynFxPoint::operator+=(const DynFxPoint &y)
{
    assert(m_fxbp.m_initialized);
    // Addition always promotes to 32 bits. Adjust the BP will be based on the 
    // current object. 
    // *** Note that adding 2 BP numbers with different BP values 
    // is discouraged, and usually represents a logical bug. ***
    m_fxbp.m_wl = 32;
    int32_t y_v;
    if (y.m_fxbp.m_bp == m_fxbp.m_bp)
        y_v = y.m_v;
    else
        y_v = y.with_params(m_fxbp.m_wl, m_fxbp.m_bp);
    int64_t ovf_check = (int64_t)m_v + y_v;
    // Detect overflow
    m_v = sat_eval(ovf_check);
    return *this;
}

DynFxPoint& DynFxPoint::operator++()
{
    assert(m_fxbp.m_initialized);
    DynFxPoint y;
    y.set_fxbp(0, 16);
    y.m_v = 1;
    (*this) += y;
    return *this;
}

DynFxPoint DynFxPoint::operator++(int unused)
{
    assert(m_fxbp.m_initialized);
    DynFxPoint temp = *this;
    ++(*this);
    return temp;
}

DynFxPoint DynFxPoint::operator-(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    // Copy operand A of RHS. This also inherits the BP.
    DynFxPoint dfx(*this);
    // Call the decrement operator.
    dfx -= y;
    return dfx;
}

DynFxPoint DynFxPoint::operator-() const
{
    assert(m_fxbp.m_initialized);
    // Copy operand A of RHS. This also inherits the BP.
    DynFxPoint dfx(*this);
    dfx.m_fxbp = m_fxbp;
    dfx.m_v = -dfx.m_v;
    return dfx;
}

DynFxPoint& DynFxPoint::operator-=(const DynFxPoint &y)
{
    assert(m_fxbp.m_initialized);
    // Addition always promotes to 32 bits. Adjust the BP will be based on the 
    // current object. 
    // *** Note that adding 2 BP numbers with different BP values 
    // is discouraged, and usually represents a logical bug. ***
    m_fxbp.m_wl = 32;
    int32_t y_v;
    if (y.m_fxbp.m_bp == m_fxbp.m_bp)
        y_v = y.m_v;
    else
        y_v = y.with_params(m_fxbp.m_wl, m_fxbp.m_bp);
    // Detect overflow
    int64_t ovf_check = (int64_t)m_v - y_v;
    m_v = sat_eval(ovf_check);
    return *this;
}

DynFxPoint& DynFxPoint::operator--()
{
    assert(m_fxbp.m_initialized);
    DynFxPoint y;
    y.set_fxbp(0, 16);
    y.m_v = 1;
    (*this) -= y;
    return *this;
}

DynFxPoint DynFxPoint::operator--(int unused)
{
    assert(m_fxbp.m_initialized);
    DynFxPoint temp = *this;
    --(*this);
    return temp;
}

DynFxPoint DynFxPoint::operator*(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    // Copy operand A of RHS.
    DynFxPoint dfx(*this);
    dfx.m_fxbp.m_bp += y.m_fxbp.m_bp;
    if (m_fxbp.m_wl + y.m_fxbp.m_wl <= 32) {
        dfx.m_v *= y.m_v;
        dfx.m_fxbp.m_wl = 32;
    } else {
        int32_t desired_bp;
        int64_t ovf_check = (int64_t)m_v * (int64_t)y.m_v;
        if (ovf_check < 0) {
            desired_bp = __builtin_clzll(-ovf_check) + dfx.m_fxbp.m_bp - 33;
        } else {
            desired_bp = __builtin_clzll(ovf_check) + dfx.m_fxbp.m_bp - 33;
        }
        dfx.m_v = (int32_t)dfx.with_params(ovf_check, 32, desired_bp);
        dfx.set_fxbp(desired_bp, 32);
    }

    return dfx;
}

DynFxPoint& DynFxPoint::operator*=(const DynFxPoint &y)
{
    assert(m_fxbp.m_initialized);
    // Multiplication always promotes to 32 bits.
    // Binary point is always the sum of the two operand integer widths. 
    // However, we need to re-adjust the product back to the user's specified
    // BP after multiplying. Do this before saturation.
    int desired_bp = m_fxbp.m_bp;
    m_fxbp.m_bp += y.m_fxbp.m_bp;
    int64_t ovf_check = (int64_t)m_v * (int64_t)y.m_v;
    m_v = with_params(ovf_check, m_fxbp.m_wl, desired_bp);
    m_fxbp.m_bp = desired_bp;
    return *this;
}

DynFxPoint DynFxPoint::operator/(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    // Copy operand A of RHS.
    DynFxPoint dfx(*this);
    // Generate a private FX object; will be replaced by the caller in nearly
    // all cases.
    float fp = to_fp();
    int req_int_bits = fxbp::get_int_bits(fp / y.to_fp());
    int int_bits = fxbp::get_int_bits(fp);
    req_int_bits = int_bits > req_int_bits ? int_bits : req_int_bits;
    int bp = 30 - req_int_bits;
    dfx.m_v = dfx.with_params(32, bp);
    dfx.set_fxbp(bp, 32);
    dfx /= y;
    return dfx;
}

DynFxPoint& DynFxPoint::operator/=(const DynFxPoint &y)
{
    assert(m_fxbp.m_initialized);
    // Division always promotes to 32 bits.
    int32_t new_wl = 32;
    // Increase the BP by whatever margin a new 32-bit word length can give us.
    // This helps a lot to get best precision out of the divide.
    int desired_wl = m_fxbp.m_wl;
    int desired_bp = m_fxbp.m_bp;
    const int a_xtra_frac = new_wl - m_fxbp.m_wl;
    int new_bp = m_fxbp.m_bp - y.m_fxbp.m_bp + a_xtra_frac;
    // Division may underflow if the divisor BP is large compared to the 
    // dividend.
    new_bp = new_bp < 0 ? 0 : new_bp;
    int32_t a = with_params(new_wl, m_fxbp.m_bp + a_xtra_frac);
    int64_t ovf_check = (int64_t)a / (int64_t)y.m_v;
    set_fxbp(new_bp, new_wl);
    m_v = with_params(ovf_check, desired_wl, desired_bp);
    set_fxbp(desired_bp, desired_wl);
    return *this;
}

// NOTE: The tensorflow mathematical definition of modulo is unknown. It does
// *not* match the definition given in the API documentation. Until the 
// definition is clear, this function can't be implemented properly.
/*
DynFxPoint DynFxPoint::operator%(const Dynamic Fixed Point &y) const
{
    assert(false);
    assert(m_fxbp);
    // DEBUG
    float a_fp = to_fp();
    float b_fp = y.to_fp();
    // Copy operand A of RHS.
    DynFxPoint dfx(*this);
    // Generate a private FX object; will be replaced by the caller in nearly
    // all cases.
    fxbp_ptr bp = std::make_shared<fxbp>();
    // Modulo always promotes to 32 bits. 
    bp->m_wl = 32;
    dfx.set_fxbp(bp);
    // Tensorflow definition: the remainder of division. 
    // z = x - floor( x / y ) * y
    const int a_xtra_frac = bp->m_wl - m_fxbp->m_wl;
    bp->m_bp = m_fxbp->m_bp - y.m_fxbp->m_bp + a_xtra_frac;
    // Division may underflow if the divisor BP is large compared to the 
    // dividend.
    bp->m_bp = bp->m_bp < 0 ? 0 : bp->m_bp;
    int32_t a = with_params(bp->m_wl, m_fxbp->m_bp + a_xtra_frac, false);
    dfx.m_v = a / y.m_v;
    float wtf = dfx.to_fp();
    // abs() and truncate
    bool neg = dfx.m_v < 0;
    dfx.m_v *= neg ? -1 : 1;
    dfx.m_v = dfx.with_params(bp->m_wl, 0, false);
    bp->m_bp = 0;
    wtf = dfx.to_fp();
    dfx *= y;
    wtf = dfx.to_fp();
#if 0
    dfx.m_v *= (dfx.m_v < 0) ? -1 : 1;
    wtf = dfx.to_fp();
    // Next we truncate the integer field and leave only the fraction.
    // int new_bp = bp->m_wl - 2;
    // dfx.m_v = dfx.with_params(bp->m_wl, new_bp, false);
    // bp->m_bp = new_bp;
    dfx.m_v &= (1 << bp->m_bp) - 1;
    // restore sign
    dfx.m_v *= (neg && dfx.m_v > 0) ? -1 : 1;
#else
    // Negate original sign.
    // dfx.m_v *= (!neg && dfx.m_v > 0) ? 1 : -1;
    wtf = dfx.to_fp();
    // Add x
    dfx += y;
#endif
    wtf = dfx.to_fp();
    float wtf2 = dfx.to_fp();
    return dfx;
}
*/


// Transforms the value to the equivalent value with the given word length and
// binary point parameters.
int64_t DynFxPoint::with_params(int64_t v, unsigned p, unsigned bp, bool round) const
{
    assert(m_fxbp.m_initialized);
    int64_t nv = v;
    // Compute the shift effect from the BP. 
    int32_t wl_diff = bp - m_fxbp.m_bp;
    // Positive diffs shift left.
    if (wl_diff >= 0) {
        nv = nv << wl_diff;
    } else {
        // This will be compiled as an arithmetic shift (due to sign).
        nv = nv >> (-wl_diff);        
        nv += round ? round_adjustment(v, -wl_diff) : 0;
    }
    // Saturation check
    return sat_eval(p, nv);
}

// Transforms the value to the equivalent value with the given word length and
// binary point parameters.
int32_t DynFxPoint::with_params(unsigned p, unsigned bp, bool round) const
{
    assert(m_fxbp.m_initialized);
    return with_params(m_v, p, bp, round);
}

// Computes the assigned rounding for value v with rounded bits r.
int32_t DynFxPoint::round_adjustment(int32_t v, unsigned r) const
{
    assert(r > 0);
    int32_t round_mask = (1 << r) - 1;
    int32_t rv = v & round_mask;
    int32_t mid_point = 1 << (r-1);
    int32_t rc = 0;
    // string rounding_mode = WaveConfigOp::m_rounding_mode;

    // if(rounding_mode.compare("convergent") == 0) {
        // If we are not a tie, round to nearest.
        if (rv != mid_point) {
            rc = rv > mid_point ? 1 : 0;
        } else {
            // The next highest bit beyond the round field determines even/odd
            int odd = (v >> r) & 1;
            // odd: round up; even: round down
            rc = odd ? 1 : 0;
        }
    // } else if(rounding_mode.compare("stochastic") == 0) {
    //     std::mt19937 &mt = RandomGenerator::Instance().get();
    //     std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    //     uint32_t rand_num = dist(mt) & round_mask;
    //     rc = rand_num < rv ? 1 : 0;
    // }
    // else {
    //     printf("rounding mode: unknown !!!\n");
    //     assert(0);
    // }
    return rc;
}

// Evaluates an over-sized value to see if it will saturate on the upper or 
// lower bound given the current FX settings.
int32_t DynFxPoint::sat_eval(int p, int64_t v) const
{
    assert(p > 0);
    int64_t sat_max = (1 << (p-1)) - 1;
    int64_t sat_min = -1 ^ sat_max;
    int64_t x = v > sat_max ? sat_max : v;
    x = v < sat_min ? sat_min : x;
    return (int32_t)x;
}

int32_t DynFxPoint::sat_eval(int64_t v) const
{
    assert(m_fxbp.m_initialized);
    return sat_eval(m_fxbp.m_wl, v);
}

// Sets a new binary point, and adjusts the data value appropriately.
void DynFxPoint::adjust_bp(int bp)
{
    m_v = with_params(m_fxbp.m_wl, bp);
    m_fxbp.m_bp = bp;
}

bool DynFxPoint::operator==(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    // Note: this operator will not compare with an epsilon! There are more than
    // one encoding which would compare bit-equal, which we account for here.
    // However, two numbers may be very close (with an epsilon), and those cases
    // will not compare equal. Fixed point arithmetic does have a precise 
    // notion of accuracy, and in that domain epsilon is not recommended.
    
    // Find min and max BP
    int max_bp = m_fxbp.m_bp > y.m_fxbp.m_bp ? m_fxbp.m_bp : y.m_fxbp.m_bp;
    int min_bp = m_fxbp.m_bp < y.m_fxbp.m_bp ? m_fxbp.m_bp : y.m_fxbp.m_bp;
    
    // First compare by min BP (largest int).
    int32_t a_v = with_params(m_fxbp.m_wl, min_bp, false);
    int32_t b_v = y.with_params(y.m_fxbp.m_wl, min_bp, false);

    // If there's a tie, we may need to compare on max BP
    if (a_v == b_v) {
        a_v = with_params(m_fxbp.m_wl, max_bp, false);
        b_v = y.with_params(y.m_fxbp.m_wl, max_bp, false);
    }
    return a_v == b_v;
}

bool DynFxPoint::operator!=(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    return !(*this == y);
}

bool DynFxPoint::operator>(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    // Find min and max BP
    int max_bp = m_fxbp.m_bp > y.m_fxbp.m_bp ? m_fxbp.m_bp : y.m_fxbp.m_bp;
    int min_bp = m_fxbp.m_bp < y.m_fxbp.m_bp ? m_fxbp.m_bp : y.m_fxbp.m_bp;
    
    // First compare by min BP (largest int).
    int32_t a_v = with_params(m_fxbp.m_wl, min_bp);
    int32_t b_v = y.with_params(y.m_fxbp.m_wl, min_bp);
    
    // If there's a tie, we may need to compare on max BP
    if (a_v == b_v) {
        a_v = with_params(m_fxbp.m_wl, max_bp);
        b_v = y.with_params(y.m_fxbp.m_wl, max_bp);
    }
    return a_v > b_v;
}

bool DynFxPoint::operator>=(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    if (*this == y) {
        return true;
    }
    return *this > y;
}

bool DynFxPoint::operator<(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    // Find min and max BP
    int max_bp = m_fxbp.m_bp > y.m_fxbp.m_bp ? m_fxbp.m_bp : y.m_fxbp.m_bp;
    int min_bp = m_fxbp.m_bp < y.m_fxbp.m_bp ? m_fxbp.m_bp : y.m_fxbp.m_bp;
    
    // First compare by min BP (largest int).
    int32_t a_v = with_params(m_fxbp.m_wl, min_bp);
    int32_t b_v = y.with_params(y.m_fxbp.m_wl, min_bp);
    
    // If there's a tie, we may need to compare on max BP
    if (a_v == b_v) {
        a_v = with_params(m_fxbp.m_wl, max_bp);
        b_v = y.with_params(y.m_fxbp.m_wl, max_bp);
    }
    return a_v < b_v;
}

bool DynFxPoint::operator<=(const DynFxPoint &y) const
{
    assert(m_fxbp.m_initialized);
    if (*this == y) {
        return true;
    }
    return *this < y;
}

#if 1
// Static function to quantize a BP tensor. This version will quantize to
// 16-bits. This could be enhanced to quantize to a parameterized precision.
void DynFxPoint::quantize(DynFxPoint* v)
{
    assert(v);
}
#endif

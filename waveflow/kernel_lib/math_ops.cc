// $HEADER$
// $LICENSE$

#include <ieee754.h>
#include <stdio.h>

#include <cstdint>

#include "math_ops.h"

namespace WaveCompMathOps {
  
void print_double(double d_) {
  ieee754_double d  = {d_};
  printf("%01x %d %08x %08x\n", d.ieee.negative, d.ieee.exponent,
	 d.ieee.mantissa0, d.ieee.mantissa1);
}

void print_single(float f_) {
  ieee754_float f  = {f_};
  printf("%01x %d %08x\n", f.ieee.negative, f.ieee.exponent,
	f.ieee.mantissa);
}

int32_t scale_32_t(int32_t x, int amt);

int16_t to_int16_t(float f_, int point) {
  ieee754_float f  = {f_};
  uint32_t m = uint32_t(0x800000) + f.ieee.mantissa;
  uint32_t adj = 23 + 127 - f.ieee.exponent  - point;
  //printf("adj %d %08x \n", adj, m >> adj);
  //printf("%01x %d %08x\n", f.ieee.negative, f.ieee.exponent,
  //	 m);
  //uint32_t m_ = m >> adj;
  uint32_t m_ = scale_32_t(m, -adj);
  return f.ieee.negative? -1 * m_: m_;
}


float to_float(int16_t i, int point) {
  float f = float(i);
  for (int i=1; i<=point; i++)
    f = f / 2.0;
  return f;
}

int16_t scale_16_t(int16_t x, int amt) {
  if (x == 0) return 0;
  else if (amt == 0) return x;
  else if (amt > 0) return x << amt;
  else {
    amt = -amt;
    int sign = (x > 0)? 1 : -1;
    if (x == 0x8000) {
      x = x / 2;
      amt = amt - 1;
    }
    if (sign == -1) x = -x;
    int16_t t1 = x >> amt;
    uint16_t t2 = (uint16_t) x << (16 - amt);
    if (t2 == 0x8000) {
      if (t1 & 1)
	return sign * (t1 + 1);
      else
	 return sign * t1;
    }
    else if (t2 > 0x8000) return  sign * t1 + sign;
    return sign * t1;
  }
  
}

int32_t scale_32_t(int32_t x, int amt) {
  if (x == 0) return 0;
  else if (amt == 0) return x;
  else if (amt > 0) return x << amt;
  else {
    amt = -amt;
    int sign = (x > 0)? 1 : -1;
    if (x == 0x80000000) {
      x = x / 2;
      amt = amt - 1;
    }
    if (sign == -1) x = -x;
    int32_t t1 = x >> amt;
    uint32_t t2 = (uint32_t) x << (16 - amt);
    if (t2 == 0x80000000) {
      if (t1 & 1)
	return sign * (t1 + 1);
      else
	 return sign * t1;
    }
    else if (t2 > 0x80000000) return  sign * t1 + sign;
    return sign * t1;
  }
  
}

}


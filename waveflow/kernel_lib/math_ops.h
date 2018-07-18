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
#ifndef WAVECOMP_MATH_OPS_H_
#define WAVECOMP_MATH_OPS_H_

#include <cstdint>

namespace WaveCompMathOps {

  void print_double(double d_);

  void print_single(float f_);

  int16_t to_int16_t(float f_, int point);

  float to_float(int16_t i, int point);

  int16_t scale_16_t(int16_t x, int amt);

  int32_t scale_32_t(int32_t x, int amt);


}



#endif // WAVECOMP_MATH_OPS_H_


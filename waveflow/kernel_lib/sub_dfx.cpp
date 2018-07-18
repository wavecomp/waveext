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
#include "binop_dfx.h"

using namespace tensorflow;

REGISTER_OP("WaveSubDfx")
    .Input("a: float")
    .Input("b: float")
    .Output("z: float")
    .Attr("data_format: string = 'NHWC'")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);


class WaveSubDfxOp : public WaveBinDfxOp {
  public:

  explicit WaveSubDfxOp(OpKernelConstruction* ctx) : WaveBinDfxOp(ctx) {
      // KS: for right now, we will handle only NHWC. TF seems only to have
      // this implemented internally, so it should be the only thing we see.
      string data_format_str;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
      OP_REQUIRES(ctx, (data_format_str == "NHWC"),
          errors::InvalidArgument(
          "wave_sub requires the data_format attribute to be NHWC"));
      if (m_show_banner) {
          printf("WaveSubDfxOp() init\n");
          m_show_banner = false;
      }
  }

  void binop(DynFxPoint& c, const DynFxPoint& a, const DynFxPoint& b) override
  {
      c = a - b;
  }

  private:
  static bool m_show_banner;
};

bool WaveSubDfxOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveSubDfx").Device(DEVICE_CPU), WaveSubDfxOp);
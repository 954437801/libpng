/* filter_write_lsx_intrinsics.c - LSX optimized WRITE-side filter functions
 *
 * 与 loongarch/filter_lsx_intrinsics.c（上游，只覆盖 READ 侧 7 个 unfilter）互补：
 * 这里补的是编码侧 png_write_find_filter() 的热点。
 *
 * 实测依据（box188，4K PNG15，perf -F 999，5 轮）：
 *   png_write_find_filter 占整次编码 46.62%，其中大部分是 5 种滤波的
 *   "应用 + sum" 逐字节循环。
 *
 * 语义等价性（关键）：
 *   C 版 png_setup_up_row() 内是
 *       v = (png_byte)(((int)*rp - (int)*pp) & 0xff);
 *       sum += (v < 128) ? v : 256 - v;      // = |int8(v)|
 *       if (sum > lmins) break;              // 早停
 *   向量版不做早停，但返回值语义保持："真实 sum <= lmins 时精确相等，
 *   否则返回一个 > lmins 的值"。调用方只判断 sum < mins（此时 lmins == mins），
 *   因此早停与否不影响滤波选择结果，只影响多算的字节数。
 *
 * 本文件仅做 UP 与 NONE 两种（无像素依赖 / 纯 abs 累加）。
 * SUB/AVG/PAETH 有跨像素依赖，暂不在此文件内。
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

#include "../pngpriv.h"

#ifdef PNG_WRITE_SUPPORTED

#if PNG_LOONGARCH_LSX_IMPLEMENTATION == 1

#include <lsxintrin.h>
#include <sys/auxv.h>

#define LA_HWCAP_LSX (1 << 4)

/* 运行时 LSX 检测（缓存）；无 LSX 的机器走函数内的标量回退 */
static int
png_write_lsx_enabled(void)
{
   static int cached = -1;

   if (cached < 0)
      cached = (getauxval(AT_HWCAP) & LA_HWCAP_LSX) != 0;

   return cached;
}

/* acc += 16 个 u8 的 |int8| 之和（两级加宽到 4×u32，单 lane 不会溢出：
 * 4K 行 15360 字节 / 4 lane * 128 ≈ 491k < 2^32） */
static __m128i
lsx_acc_abs_sum(__m128i acc, __m128i v)
{
   __m128i a = __lsx_vabsd_b(v, __lsx_vldi(0));
   __m128i s16 = __lsx_vhaddw_hu_bu(a, a);
   __m128i s32 = __lsx_vhaddw_wu_hu(s16, s16);

   return __lsx_vadd_w(acc, s32);
}

static size_t
lsx_reduce_u32x4(__m128i acc)
{
   uint32_t lane[4];

   __lsx_vst(acc, lane, 0);

   return (size_t)lane[0] + lane[1] + lane[2] + lane[3];
}

/* ==== BEGIN fn=png_write_none_sum_lsx ==== */
/* png_write_find_filter() 里 NONE 滤波的 sum；C 版无早停，这里严格等价。 */
size_t
png_write_none_sum_lsx(png_const_bytep rp, size_t row_bytes)
{
   size_t i = 0, sum = 0;
   __m128i acc;

   if (!png_write_lsx_enabled())
   {
      for (; i < row_bytes; i++)
      {
         png_byte v = rp[i];

         sum += (v < 128) ? v : 256 - v;
      }
      return sum;
   }

   acc = __lsx_vldi(0);
   for (; i + 16 <= row_bytes; i += 16)
      acc = lsx_acc_abs_sum(acc, __lsx_vld(rp + i, 0));
   sum = lsx_reduce_u32x4(acc);

   for (; i < row_bytes; i++)
   {
      png_byte v = rp[i];

      sum += (v < 128) ? v : 256 - v;
   }

   return sum;
}
/* ==== END fn=png_write_none_sum_lsx ==== */

/* ==== BEGIN fn=png_setup_up_row_lsx ==== */
/* 写 UP 滤波（dp = try_row + 1）并返回 sum。不早停，见文件头语义说明。 */
size_t
png_setup_up_row_lsx(png_bytep dp, png_const_bytep rp, png_const_bytep pp,
    size_t row_bytes)
{
   size_t i = 0, sum = 0;
   __m128i acc;

   if (!png_write_lsx_enabled())
   {
      for (; i < row_bytes; i++)
      {
         png_byte v = (png_byte)(((int)rp[i] - (int)pp[i]) & 0xff);

         dp[i] = v;
         sum += (v < 128) ? v : 256 - v;
      }
      return sum;
   }

   acc = __lsx_vldi(0);
   for (; i + 16 <= row_bytes; i += 16)
   {
      __m128i d = __lsx_vsub_b(__lsx_vld(rp + i, 0), __lsx_vld(pp + i, 0));

      __lsx_vst(d, dp + i, 0);
      acc = lsx_acc_abs_sum(acc, d);
   }
   sum = lsx_reduce_u32x4(acc);

   for (; i < row_bytes; i++)
   {
      png_byte v = (png_byte)(((int)rp[i] - (int)pp[i]) & 0xff);

      dp[i] = v;
      sum += (v < 128) ? v : 256 - v;
   }

   return sum;
}
/* ==== END fn=png_setup_up_row_lsx ==== */

/* ==== BEGIN fn=png_setup_sub_row_lsx ==== */
/* SUB 滤波：dp[i] = rp[i] - rp[i-bpp]，是**逐字节**运算、与像素边界无关，
 * 所以 bpp=3/4/6/8 共用同一段 16 字节对齐减法（只要 i >= bpp）。
 * 前 bpp 字节按 C 版原样拷贝（对应 C 的第一个循环）。 */
size_t
png_setup_sub_row_lsx(png_bytep dp, png_const_bytep rp, png_uint_32 bpp,
    size_t row_bytes)
{
   size_t i = 0, sum = 0;
   __m128i acc;

   for (; i < bpp && i < row_bytes; i++)
   {
      png_byte v = rp[i];

      dp[i] = v;
      sum += (v < 128) ? v : 256 - v;
   }

   if (!png_write_lsx_enabled())
   {
      for (; i < row_bytes; i++)
      {
         png_byte v = (png_byte)(((int)rp[i] - (int)rp[i - bpp]) & 0xff);

         dp[i] = v;
         sum += (v < 128) ? v : 256 - v;
      }
      return sum;
   }

   acc = __lsx_vldi(0);
   for (; i + 16 <= row_bytes; i += 16)
   {
      __m128i d = __lsx_vsub_b(__lsx_vld(rp + i, 0),
          __lsx_vld(rp + i - bpp, 0));

      __lsx_vst(d, dp + i, 0);
      acc = lsx_acc_abs_sum(acc, d);
   }
   sum += lsx_reduce_u32x4(acc);

   for (; i < row_bytes; i++)
   {
      png_byte v = (png_byte)(((int)rp[i] - (int)rp[i - bpp]) & 0xff);

      dp[i] = v;
      sum += (v < 128) ? v : 256 - v;
   }

   return sum;
}
/* ==== END fn=png_setup_sub_row_lsx ==== */

/* ==== BEGIN fn=png_setup_avg_row_lsx ==== */
/* AVG 滤波：
 *   前 bpp 字节  dp[i] = rp[i] - (pp[i] / 2)
 *   其余       dp[i] = rp[i] - ((rp[i-bpp] + pp[i]) >> 1)   （floor）
 * 向量实现用 vavgr_bu（ceil 平均）再减 ``(a^b)&1`` 修正为 floor，
 * 已通过 avgprobe 抽样穷举验证（bad=0）。
 */
size_t
png_setup_avg_row_lsx(png_bytep dp, png_const_bytep rp, png_const_bytep pp,
    png_uint_32 bpp, size_t row_bytes)
{
   size_t i = 0, sum = 0;
   __m128i acc, one;

   for (; i < bpp && i < row_bytes; i++)
   {
      png_byte v = (png_byte)(((int)rp[i] - ((int)pp[i] / 2)) & 0xff);

      dp[i] = v;
      sum += (v < 128) ? v : 256 - v;
   }

   if (!png_write_lsx_enabled())
   {
      for (; i < row_bytes; i++)
      {
         png_byte v = (png_byte)(((int)rp[i] -
             (((int)rp[i - bpp] + (int)pp[i]) >> 1)) & 0xff);

         dp[i] = v;
         sum += (v < 128) ? v : 256 - v;
      }
      return sum;
   }

   acc = __lsx_vldi(0);
   one = __lsx_vldi(1);
   for (; i + 16 <= row_bytes; i += 16)
   {
      __m128i a = __lsx_vld(rp + i - bpp, 0);
      __m128i b = __lsx_vld(pp + i, 0);
      __m128i avg = __lsx_vsub_b(__lsx_vavgr_bu(a, b),
          __lsx_vand_v(__lsx_vxor_v(a, b), one));
      __m128i d = __lsx_vsub_b(__lsx_vld(rp + i, 0), avg);

      __lsx_vst(d, dp + i, 0);
      acc = lsx_acc_abs_sum(acc, d);
   }
   sum += lsx_reduce_u32x4(acc);

   for (; i < row_bytes; i++)
   {
      png_byte v = (png_byte)(((int)rp[i] -
          (((int)rp[i - bpp] + (int)pp[i]) >> 1)) & 0xff);

      dp[i] = v;
      sum += (v < 128) ? v : 256 - v;
   }

   return sum;
}
/* ==== END fn=png_setup_avg_row_lsx ==== */

#endif /* PNG_LOONGARCH_LSX_IMPLEMENTATION == 1 */
#endif /* PNG_WRITE_SUPPORTED */

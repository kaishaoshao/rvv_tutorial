
#include "Simd_Common.h"

#include <assert.h>

#include <iostream>
#include <cmath>
#include <time.h>
#include <iomanip> // for std::setprecision

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace wx::Simd
{
#ifdef SIMD_RVV_ENABLE    
    namespace Rvv
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t)*(5 * width + A));
                    in0 = (uint16_t*)_p;
                    in1 = in0 + width;
                    out0 = in1 + width;
                    out1 = out0 + width;
                    dst = out1 + width + HA;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * in0;
                uint16_t * in1;
                uint16_t * out0;
                uint16_t * out1;
                uint16_t * dst;
            private:
                void *_p;
            };

            // struct RvvMask
            // {
            //     RvvMask()
            //     {
            //         size_t vlmax = __riscv_vsetvlmax_e8m1();  // 获取最大向量长度
            //         uint8_t flag[vlmax];
            //         for(size_t i = 0; i < vlmax; i += 2)
            //         {
            //             flag[i] = 0x01;
            //             flag[i + 1] = 0x00;
            //         }

            //         vec_mask = __riscv_vlm_v_b8(flag, vlmax);
            //     }

            //     vbool8_t vec_mask;
            // };

            // RvvMask my_mask;

            size_t vlmax = __riscv_vsetvlmax_e8m1();  // 获取最大向量长度
            int8_t mask_flag[] = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, \
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
            // vbool8_t vec_mask = __riscv_vmseq_vx_i8m1_b8(__riscv_vle8_v_i8m1(mask_flag, vlmax), 1, vlmax);
        }

        // @brief 加载vl个数据到向量寄存器，并由uint8_t扩展为uint16_t.
        // @param vl 单次可以处理的数据个数
        SIMD_INLINE vuint16m2_t LoadUnpacked(const void * src, size_t vl)
        {
            // vuint16m2_t __riscv_vzext_vf2_u16m2(vuint8m1_t vs2, size_t vl);
            return __riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1((const uint8_t *)src, vl), vl);

            // TODO: 将8位整数向量 vint8m1_t vs2 转换为16位整数向量 vint16m2_t 
            //??? __riscv_vwcvt_x_x_v_i16m2跟__riscv_vzext_vf2_u16m2比起来怎么样???
            // vint16m2_t __riscv_vwcvt_x_x_v_i16m2(vint8m1_t vs2, size_t vl);
        }

        template<bool align> SIMD_INLINE void FirstRow5x5(vuint16m2_t src, Buffer & buffer, size_t offset, size_t vl)
        {
            __riscv_vse16_v_u16m2(buffer.in0 + offset, src, vl);
            __riscv_vse16_v_u16m2(buffer.in1 + offset, __riscv_vmul_vx_u16m2(src, 5, vl), vl);
        }

        template<bool align> SIMD_INLINE void FirstRow5x5(const uint8_t * src, Buffer & buffer, size_t offset, size_t vl)
        {
            FirstRow5x5<align>(LoadUnpacked(src + offset, vl), buffer, offset, vl);
        }

        template<bool align> SIMD_INLINE void MainRowY5x5(vuint16m2_t odd, vuint16m2_t even, Buffer & buffer, size_t offset, size_t vl)
        {
            vuint16m2_t cp = __riscv_vmul_vx_u16m2(odd, 4, vl);
            vuint16m2_t c0 = __riscv_vle16_v_u16m2(buffer.in0 + offset, vl);
            vuint16m2_t c1 = __riscv_vle16_v_u16m2(buffer.in1 + offset, vl);

            // Store<align>((__m128i*)(buffer.dst + offset), _mm_add_epi16(even, _mm_add_epi16(c1, _mm_add_epi16(cp, _mm_mullo_epi16(c0, K16_0006)))));

            // vuint16m2_t result = __riscv_vmul_vx_u16m2(c0, 6, vl);
            vuint16m2_t result = __riscv_vmadd_vx_u16m2(c0, 6, cp, vl);
            // vuint16m2_t __riscv_vadd_vv_u16m2(vuint16m2_t vs2, vuint16m2_t vs1, size_t vl);
            result = __riscv_vadd_vv_u16m2(result, c1, vl);
            result = __riscv_vadd_vv_u16m2(result, even, vl);
            __riscv_vse16_v_u16m2(buffer.dst + offset, result, vl);


            // Store<align>((__m128i*)(buffer.out1 + offset), _mm_add_epi16(c0, cp));

            __riscv_vse16_v_u16m2(buffer.out1 + offset, __riscv_vadd_vv_u16m2(c0, cp, vl), vl);


            // Store<align>((__m128i*)(buffer.out0 + offset), even);

            __riscv_vse16_v_u16m2(buffer.out0 + offset, even, vl);
        }

        template<bool align> SIMD_INLINE void MainRowY5x5(const uint8_t *odd, const uint8_t *even, Buffer & buffer, size_t offset, size_t vl)
        {
            MainRowY5x5<align>(LoadUnpacked(odd + offset, vl), LoadUnpacked(even + offset, vl), buffer, offset, vl);
            // offset += HA;
            // MainRowY5x5<align>(LoadUnpacked(odd + offset, vl), LoadUnpacked(even + offset, vl), buffer, offset, vl);
        }

        template <bool align, bool compensation> SIMD_INLINE vuint16m2_t MainRowX5x5(uint16_t * dst, size_t vl)
        {
            // __m128i t0 = _mm_loadu_si128((__m128i*)(dst - 2));
            // __m128i t1 = _mm_loadu_si128((__m128i*)(dst - 1));
            // __m128i t2 = Load<align>((__m128i*)dst);
            // __m128i t3 = _mm_loadu_si128((__m128i*)(dst + 1));
            // __m128i t4 = _mm_loadu_si128((__m128i*)(dst + 2));


            // vuint16m2_t __riscv_vle16_v_u16m2(const uint16_t *rs1, size_t vl);
            vuint16m2_t t0 = __riscv_vle16_v_u16m2(dst - 2, vl);
            vuint16m2_t t1 = __riscv_vle16_v_u16m2(dst - 1, vl);
            vuint16m2_t t2 = __riscv_vle16_v_u16m2(dst, vl);
            vuint16m2_t t3 = __riscv_vle16_v_u16m2(dst + 1, vl);
            vuint16m2_t t4 = __riscv_vle16_v_u16m2(dst + 2, vl);


            // t2 = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(t2, K16_0006), _mm_mullo_epi16(_mm_add_epi16(t1, t3), K16_0004)), _mm_add_epi16(t0, t4));

            t2 = __riscv_vadd_vv_u16m2(__riscv_vmadd_vx_u16m2(__riscv_vadd_vv_u16m2(t1, t3, vl), \
                4, __riscv_vmul_vx_u16m2(t2, 6, vl), vl), __riscv_vadd_vv_u16m2(t0, t4, vl), vl);


            // return DivideBy256<compensation>(t2);

            // 将加法常量添加到向量
            // vuint16m2_t __riscv_vadd_vx_u16m2(vuint16m2_t vs2, uint16_t rs1, size_t vl);
            t2 = __riscv_vadd_vx_u16m2(t2, 128, vl);

            // 执行带符号右移操作 (val_int_plus_const + 128) >> 8
            // vint16m2_t __riscv_vsra_vx_i16m2(vint16m2_t vs2, size_t rs1, size_t vl);
            // vuint16m2_t __riscv_vsrl_vx_u16m2(vuint16m2_t vs2, size_t rs1, size_t vl);
            // vuint16m2_t __riscv_vssrl_vx_u16m2(vuint16m2_t vs2, size_t rs1, unsigned int vxrm, size_t vl);
            // return __riscv_vsra_vx_u16m2(t2, 8, vl);
            return __riscv_vsrl_vx_u16m2(t2, 8, vl);
        }

        template <bool align, bool compensation> SIMD_INLINE void MainRowX5x5(Buffer & buffer, size_t offset, uint8_t *dst, size_t vl)
        {
            vuint16m2_t t0 = MainRowX5x5<align, compensation>(buffer.dst + offset, vl);
            // __m128i t1 = MainRowX5x5<align, compensation>(buffer.dst + offset + HA);

            // t0 = _mm_packus_epi16(_mm_and_si128(_mm_packus_epi16(t0, t1), K16_00FF), K_ZERO);

            // TODO: t0 = _mm_packus_epi16(_mm_and_si128(_mm_packus_epi16(t0, t1), K16_00FF), K_ZERO); //???
        
            // vuint8m1_t __riscv_vnclipu_wx_u8m1(vuint16m2_t vs2, size_t rs1, unsigned int vxrm, size_t vl);
            vuint8m1_t vec_u8_result = __riscv_vnclipu_wx_u8m1(t0, 0, 0, vl);
        #if 0   // method 1 
            //vfgt
            // vuint16m2_t = __riscv_vzext_vf2_u16m2(vec_u8_result, vl);
            // vint8m1_t __riscv_vand_vv_i8m1(vint8m1_t vs2, vint8m1_t vs1, size_t vl);
            // vuint8m1_t __riscv_vand_vv_u8m1(vuint8m1_t vs2, vuint8m1_t vs1, size_t vl);

            // 如何将vuint8m1_t里面的每相邻两个元素uint8_t合并为一个uint16_t, 最终得到一个vl/2长度的vuint16_t向量?
            // 然后将vl/2长度的vuint16_t向量裁剪或者右移8位narrow为 vl/2 的vuint8m1_t向量。
            // 关于0x00FF 转换成向量可以提前弄好在vlmax大小的向量，然后这里只应用vand即可
            uint8_t flag[vl];
            /*for(int i = 0; i < vl; i+= 2)
            {
                flag[i] = 0xff;
                // if((i+1) < vl)
                flag[i + 1] = 0x00;
            }
            vuint8m1_t vec_flag = __riscv_vle8_v_u8m1(flag, vl);
            vec_u8_result = __riscv_vand_vv_u8m1(vec_u8_result, vec_flag, vl);
            // the end.
            */
            __riscv_vse8_v_u8m1(flag, vec_u8_result, vl);
            for(int j = 0, i = 0; i < vl; i+= 2)
            {
                dst[j++] = flag[i];
                
            }
        #else // method 2: more better.
            // vuint16m2_t __riscv_vand_vx_u16m2(vuint16m2_t vs2, uint16_t rs1, size_t vl);
            // vuint16m2_t vec_u16_result = __riscv_vand_vx_u16m2(t0, 0x00FF, vl);
            // vuint8m1_t __riscv_vncvt_x_x_w_u8m1(vuint16m2_t vs2, size_t vl);

            // vbool8_t __riscv_vlm_v_b8(const uint8_t *rs1, size_t vl);
            // vbool16_t __riscv_vlm_v_b16(const uint8_t *rs1, size_t vl);

            // vuint8m1_t __riscv_vcompress_vm_u8m1(vuint8m1_t vs2, vbool8_t vs1, size_t vl);
            // PS: [TODO] vec_mask可以在最外层调用的地方加载，提供给所有的图片使用，效率更高
            vbool8_t vec_mask = __riscv_vmseq_vx_i8m1_b8(__riscv_vle8_v_i8m1(mask_flag, vlmax), 1, vlmax);
            vec_u8_result = __riscv_vcompress_vm_u8m1(vec_u8_result, vec_mask, vl);
            __riscv_vse8_v_u8m1(dst, vec_u8_result, vl / 2);

        #endif



            // _mm_storel_epi64 函数的功能是将128位SSE寄存器中的低64位数据存储到内存中的指定位置
            // void _mm_storel_epi64 (__m128i* mem_addr, __m128i a)
            // Description
            // Store 64-bit integer from the first element of a into memory.
            // Operation
            // MEM[mem_addr+63:mem_addr] := a[63:0]

            // _mm_storel_epi64((__m128i*)dst, t0);

            // __riscv_vse8_v_u8m1(dst, vec_u8_result, vl);
        }

        template <bool compensation> void ReduceGray5x5(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= A);

            size_t alignedWidth = Simd::AlignLo(srcWidth, A);
            size_t bufferDstTail = Simd::AlignHi(srcWidth - A, 2);

            Buffer buffer(Simd::AlignHi(srcWidth, A));

            for (size_t vl, col = 0; col < alignedWidth; col += vl)
            {
                vl = __riscv_vsetvl_e8m1(alignedWidth - col);
                FirstRow5x5<true>(src, buffer, col, vl);
            }
            // if (alignedWidth != srcWidth) // 针对宽度未对齐的情况，把最后A个元素再按照非对齐方式存储计算一次
            //     FirstRow5x5<false>(src, buffer, srcWidth - A);
            src += srcStride;

            for (size_t row = 1; row <= srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t *odd = src - (row < srcHeight ? 0 : srcStride);
                const uint8_t *even = odd + (row < srcHeight - 1 ? srcStride : 0);

                for (size_t vl, col = 0; col < alignedWidth; col += vl)
                {
                    vl = __riscv_vsetvl_e8m1(alignedWidth - col);
                    MainRowY5x5<true>(odd, even, buffer, col, vl);
                }
                // if (alignedWidth != srcWidth)
                    // MainRowY5x5<false>(odd, even, buffer, srcWidth - A);

                Swap(buffer.in0, buffer.out0);
                Swap(buffer.in1, buffer.out1);

                buffer.dst[-2] = buffer.dst[0];
                buffer.dst[-1] = buffer.dst[0];
                buffer.dst[srcWidth] = buffer.dst[srcWidth - 1];
                buffer.dst[srcWidth + 1] = buffer.dst[srcWidth - 1];

                for (size_t vl, srcCol = 0, dstCol = 0; srcCol < alignedWidth; srcCol += vl, dstCol += (vl / 2))
                {
                    vl = __riscv_vsetvl_e8m1(alignedWidth - srcCol);
                    MainRowX5x5<true, compensation>(buffer, srcCol, dst + dstCol, vl);
                }
                // if (alignedWidth != srcWidth)
                    // MainRowX5x5<false, compensation>(buffer, bufferDstTail, dst + dstWidth - HA);
            }
        }

        void ReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray5x5<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray5x5<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif

}


int main(int argc, char *argv[])
{
#ifdef SIMD_SSE41_ENABLE     
    std::cout << "sizeof(__m128)=" << sizeof(__m128)  << " sizeof(__m128i)=" << sizeof(__m128i)
        << " sizeof(float)=" << sizeof(float) << " sizeof(__m128) / sizeof(float)=" << sizeof(__m128) / sizeof(float) 
        << " sizeof(long long)=" << sizeof(long long) << " sizeof(long)=" << sizeof(long) << std::endl;
#endif

    // test 
    // wx::Simd::Rvv::RvvMask mask1;
#if 0    
    size_t vlmax = __riscv_vsetvlmax_e8m1();  // 获取最大向量长度
    int8_t flag[vlmax];
    for(size_t i = 0; i < vlmax; i += 2)
    {
        flag[i] = 0x01;
        flag[i + 1] = 0x00;
    }

    // vbool8_t vec_mask = __riscv_vlm_v_b8(flag, vlmax);
    // vbool8_t __riscv_vmseq_vv_i8m1_b8(vint8m1_t vs2, vint8m1_t vs1, size_t vl);
    // vbool8_t __riscv_vmseq_vx_i8m1_b8(vint8m1_t vs2, int8_t rs1, size_t vl);
    vbool8_t vec_mask = __riscv_vmseq_vx_i8m1_b8(__riscv_vle8_v_i8m1(flag, vlmax), 1, vlmax);
#else
    vbool8_t vec_mask = __riscv_vmseq_vx_i8m1_b8(__riscv_vle8_v_i8m1(wx::Simd::Rvv::mask_flag, wx::Simd::Rvv::vlmax), 1, wx::Simd::Rvv::vlmax);    
#endif

    uint8_t aaa[32] = { 0x01, 0x02, 0x03, 0x04, 0x01, 0x02, 0x03, 0x04, 0x01, 0x02, 0x03, 0x04, 0x01, 0x02, 0x03, 0x04, 0x01, 0x02, 0x03, 0x04, 0x01, 0x02, 0x03, 0x04, 0x01, 0x02, 0x03, 0x04, 0x01, 0x02, 0x03, 0x04 };
    size_t vl = __riscv_vsetvl_e8m1(32); //4;
    vuint8m1_t vec_aaa = __riscv_vle8_v_u8m1(aaa, vl);
    vec_aaa = __riscv_vcompress_vm_u8m1(vec_aaa, vec_mask, vl);
    uint8_t bbb[32];
    __riscv_vse8_v_u8m1(bbb, vec_aaa, vl);
    std::cout << "vl=" << vl << " bbb[0]=" << bbb[0] << std::endl;
    std::cout << "test vcompress: bbb[]=";
    for(size_t i = 0; i < vl; i++) std::cout << (int)bbb[i] << "  "; std::cout << std::endl;
    // return 0;
    // the end.

    size_t srcWidth = 640;
    size_t srcHeight = 480;

    cv::Mat image_gray = cv::imread("../assets/1726299898318057216.png", cv::IMREAD_GRAYSCALE);

    if (image_gray.empty() ) {
		std::cout << "can't read image!!" << std::endl;
		return -1;
	}

    clock_t clock_start = clock();
    cv::Mat mipmap(cv::Size(srcWidth + srcWidth / 2, srcHeight), CV_8UC1);
    image_gray.copyTo(mipmap(cv::Rect(0, 0, srcWidth, srcHeight)));

    int stride = mipmap.step[0]; // 输出图像第0行的总字节数。本例即为960.
    // std::cout << "stride=" << stride << std::endl;
    int srcW = srcWidth;
    int srcH = srcHeight;
    int dstW = srcW / 2;
    int dstH = srcH / 2;
    uint8_t* src_begin = mipmap.data;
    uint8_t* dst_begin = src_begin + srcW;
    wx::Simd::Rvv::ReduceGray5x5(src_begin, srcW, srcH, stride, dst_begin, dstW, dstH, stride, 1);
    for(int Level=1; Level<4; Level++)
    {
        srcW = dstW;
        srcH = dstH;
        dstW /= 2;
        dstH /= 2;
        src_begin = dst_begin;
        dst_begin += srcH*stride;
        wx::Simd::Rvv::ReduceGray5x5(src_begin, srcW, srcH, stride, dst_begin, dstW, dstH, stride, 1);
    }

    clock_t clock_end = clock();
    std::cout << std::fixed << std::setprecision(3) << "risc-v vector image pyramid:" << " clock=" << (clock_end - clock_start) << " cycles\n";

    cv::imshow("mipmap", mipmap);
    cv::waitKey(0);
    
    return 0;
}
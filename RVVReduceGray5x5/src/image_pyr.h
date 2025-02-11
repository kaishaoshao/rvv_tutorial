/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Image pyramid implementation stored as mipmap
*/

#pragma once

// #include <basalt/image/image.h>
#include "image.h"

#ifdef _USE_RVV_INTRINSIC_
#include <time.h>
#endif

#define _RVV_PYRAMID_

namespace basalt {

/// @brief Image pyramid that stores levels as mipmap
///
/// \image html mipmap.jpeg
/// Computes image pyramid (see \ref subsample) and stores it as a mipmap
/// (https://en.wikipedia.org/wiki/Mipmap).
template <typename T, class Allocator = DefaultImageAllocator<T>>
class ManagedImagePyr {
 public:
  using PixelType = T;
  using Ptr = std::shared_ptr<ManagedImagePyr<T, Allocator>>;

  /// @brief Default constructor.
  inline ManagedImagePyr() {}

  /// @brief Construct image pyramid from other image.
  ///
  /// @param other image to use for the pyramid level 0
  /// @param num_level number of levels for the pyramid
  inline ManagedImagePyr(const ManagedImage<T>& other, size_t num_levels) {
    setFromImage(other, num_levels);
  }

  inline ManagedImagePyr(const cv::Mat& other, size_t num_levels) {
    std::cout << "TODO: crate image pyramid" << std::endl;
    int srcWidth = other.cols;
    int srcHeight = other.rows;
    orig_w = srcWidth;
    // 初始化mipmap图像大小
    image.Reinitialise(srcWidth + srcWidth / 2, srcHeight);

    // set level 0
    // image_gray.copyTo(mipmap(cv::Rect(0, 0, srcWidth, srcHeight)));
    Image<uint8_t> img = lvl_internal(0);
    const uint8_t* data_in = other.ptr();
    // uint8_t* data_out = img.ptr;
    // std::cout << "img.h=" << img.h << ", img.w=" << img.w << std::endl;

    size_t i = 0;
    for (size_t r = 0; r < img.h; ++r) {
      uint8_t* row = img.RowPtr(r);

      for(size_t c = 0; c < img.w; ++c) {
        // int val = row[c];
        // val = val >> 8;
        // data_out[i++] = val;

        row[c] = data_in[i++];
      }
    }

  #if 0
    {
      // test
      const uint8_t* data_in = nullptr;
      uint8_t* data_out = nullptr;

      const Image<const uint8_t> img = mipmap();
      cv::Mat pyramid_image(img.h, img.w, CV_8UC1);
        data_in = img.ptr;
        data_out = pyramid_image.ptr();

        size_t full_size = img.size();  // forw_img.cols * forw_img.rows;
        for (size_t i = 0; i < full_size; i++) {
          int val = data_in[i];
          // val = val >> 8;
          data_out[i] = val;
        }

        cv::imshow("1 pyramid image", pyramid_image);
        // cv::imshow("other", other);
        cv::waitKey(0);
    }

    {
      // test 2
      const Image<const uint8_t> img = lvl(0);
      // const Image<const uint16_t> img = pyr.lvl(0);

      const uint8_t* data_in = nullptr;
      uint8_t* data_out = nullptr;

      // cv::Mat cv_image;
      // cv_image = cv::Mat::zeros(img.h, img.w, CV_8UC1);  // CV_8UC3
      cv::Mat cv_image(img.h, img.w, CV_8UC1);
      data_out = cv_image.ptr();

      size_t i = 0;
      for (size_t r = 0; r < img.h; ++r) {
        const uint8_t* row = img.RowPtr(r);

        for(size_t c = 0; c < img.w; ++c) {
          int val = row[c];
          // val = val >> 8;
          data_out[i++] = val;
        }
      }


      cv::imshow("2 pyramid image", cv_image);
      cv::waitKey(0);
    }

    // return ;
  #endif  

    // set other levels
    // int stride = mipmap.step[0]; // 输出图像第0行的总字节数。本例即为960.
    int stride = mipmap().pitch;
    // std::cout << "stride=" << stride << std::endl;
    int srcW = srcWidth;
    int srcH = srcHeight;
    int dstW = srcW / 2;
    int dstH = srcH / 2;
    uint8_t* src_begin = img.ptr;
    uint8_t* dst_begin = src_begin + srcW;
    wx::Simd::Rvv::ReduceGray5x5(src_begin, srcW, srcH, stride, dst_begin, dstW, dstH, stride, 1);
    for(int Level=1; Level<num_levels; Level++)
    {
        srcW = dstW;
        srcH = dstH;
        dstW /= 2;
        dstH /= 2;
        src_begin = dst_begin;
        dst_begin += srcH*stride;
        wx::Simd::Rvv::ReduceGray5x5(src_begin, srcW, srcH, stride, dst_begin, dstW, dstH, stride, 1);
    }
  #if 1
    {
      // test 3
      const uint8_t* data_in = nullptr;
      uint8_t* data_out = nullptr;

      const Image<const uint8_t> img = mipmap();
      cv::Mat pyramid_image(img.h, img.w, CV_8UC1);
        data_in = img.ptr;
        data_out = pyramid_image.ptr();

        size_t full_size = img.size();  // forw_img.cols * forw_img.rows;
        for (size_t i = 0; i < full_size; i++) {
          int val = data_in[i];
          // val = val >> 8;
          data_out[i] = val;
        }

        cv::imshow("3 pyramid image", pyramid_image);
        cv::waitKey(0);
    }
  #endif  
  }

  /// @brief Set image pyramid from other image.
  ///
  /// @param other image to use for the pyramid level 0
  /// @param num_level number of levels for the pyramid
  inline void setFromImage(const ManagedImage<T>& other, size_t num_levels) {
    orig_w = other.w;
    // 初始化mipmap图像大小
    image.Reinitialise(other.w + other.w / 2, other.h);
    image.Fill(0); // 图像的每一个位置赋值0
    lvl_internal(0).CopyFrom(other); // 给第0层赋值原始图像数据

    #if 0
    clock_t clock_start = clock();
    #endif
    for (size_t i = 0; i < num_levels; i++) {
      const Image<const T> l = lvl(i); // 返回mipmap图像中第i层金字塔图像对象
      // lvl_internal功能跟lvl一样，都是返回mipmap中指定层图像对象，区别在于后者是const类型的。
      Image<T> lp1 = lvl_internal(i + 1);
      subsample(l, lp1); // 下采样
    }

    #if 0
    clock_t clock_end = clock();
#if !defined(_RVV_PYRAMID_)
    std::cout << std::fixed << std::setprecision(3) << "general image pyramid:" << " clock=" << (clock_end - clock_start) << " cycles\n";
#else    
    std::cout << std::fixed << std::setprecision(3) << "risc-v vector image pyramid:" << " clock=" << (clock_end - clock_start) << " cycles\n";
#endif    
    #endif
  }

  /// @brief Extrapolate image after border with reflection.
  static inline int border101(int x, int h) { // 用反射来推断边界后的图像。
    return h - 1 - std::abs(h - 1 - x);
  }
#if !defined(_USE_RISCV_V)
  /// @brief Subsample the image twice in each direction.
  ///
  /// Subsampling is done by convolution with Gaussian kernel
  /// \f[
  /// \frac{1}{256}
  /// \begin{bmatrix}
  ///   1 & 4 & 6 & 4 & 1
  /// \\4 & 16 & 24 & 16 & 4
  /// \\6 & 24 & 36 & 24 & 6
  /// \\4 & 16 & 24 & 16 & 4
  /// \\1 & 4 & 6 & 4 & 1
  /// \\ \end{bmatrix}
  /// \f]
  /// and removing every even-numbered row and column.
  static void subsample(const Image<const T>& img, Image<T>& img_sub) {
    static_assert(std::is_same<T, uint16_t>::value ||
                  std::is_same<T, uint8_t>::value);

    constexpr int kernel[5] = {1, 4, 6, 4, 1}; // 5的卷积核

    // accumulator
    ManagedImage<int> tmp(img_sub.h, img.w); // 中间变量 高度缩小一半  宽度先不变

    // Vertical convolution
    {
      for (int r = 0; r < int(img_sub.h); r++) { // 遍历高度
        const T* row_m2 = img.RowPtr(std::abs(2 * r - 2));
        const T* row_m1 = img.RowPtr(std::abs(2 * r - 1));
        const T* row = img.RowPtr(2 * r);
        const T* row_p1 = img.RowPtr(border101(2 * r + 1, img.h));
        const T* row_p2 = img.RowPtr(border101(2 * r + 2, img.h));

        for (int c = 0; c < int(img.w); c++) { // 遍历宽度
          tmp(r, c) = kernel[0] * int(row_m2[c]) + kernel[1] * int(row_m1[c]) +
                      kernel[2] * int(row[c]) + kernel[3] * int(row_p1[c]) +
                      kernel[4] * int(row_p2[c]);
        }
      }
    }

    // Horizontal convolution
    {
      for (int c = 0; c < int(img_sub.w); c++) {
        const int* row_m2 = tmp.RowPtr(std::abs(2 * c - 2));
        const int* row_m1 = tmp.RowPtr(std::abs(2 * c - 1));
        const int* row = tmp.RowPtr(2 * c);
        const int* row_p1 = tmp.RowPtr(border101(2 * c + 1, tmp.h));
        const int* row_p2 = tmp.RowPtr(border101(2 * c + 2, tmp.h));

        for (int r = 0; r < int(tmp.w); r++) {
          int val_int = kernel[0] * row_m2[r] + kernel[1] * row_m1[r] +
                        kernel[2] * row[r] + kernel[3] * row_p1[r] +
                        kernel[4] * row_p2[r];
          // 1 << 7 表示128，256的一半。相当于(int)(fVal + 0.5)四舍五入向上取整。              
          T val = ((val_int + (1 << 7)) >> 8); // 这个位置是除以256   这里采用四舍五入的思想了
          img_sub(c, r) = val; // 最后再赋值一下
        }
      }
    }
  }

#elif defined(_U16M1_)
  // use rvv 
  static void subsample(const Image<const T>& img, Image<T>& img_sub) {
    static_assert(std::is_same<T, uint16_t>::value ||
                  std::is_same<T, uint8_t>::value);

    constexpr int kernel[5] = {1, 4, 6, 4, 1};

    // accumulator
    ManagedImage<int> tmp(img_sub.h, img.w);

    // Vertical convolution
    {
      for (int r = 0; r < int(img_sub.h); r++) {
        const T* row_m2 = img.RowPtr(std::abs(2 * r - 2));
        const T* row_m1 = img.RowPtr(std::abs(2 * r - 1));
        const T* row = img.RowPtr(2 * r);
        const T* row_p1 = img.RowPtr(border101(2 * r + 1, img.h));
        const T* row_p2 = img.RowPtr(border101(2 * r + 2, img.h));
#if 0
        for (int c = 0; c < int(img.w); c++) {
          tmp(r, c) = kernel[0] * int(row_m2[c]) + kernel[1] * int(row_m1[c]) +
                      kernel[2] * int(row[c]) + kernel[3] * int(row_p1[c]) +
                      kernel[4] * int(row_p2[c]);
        }
#else
        size_t n = img.w;
        #if 0
        for (size_t vl; n > 0; n -= vl) // , a += vl, b += vl, c += vl
        {
          vl = __riscv_vsetvl_e16m2(n);

          // vuint16m2_t __riscv_vle16_v_u16m2(const uint16_t *rs1, size_t vl);
          vuint16m2_t vec_row_m2 = __riscv_vle16_v_u16m2(row_m2, vl);
          vuint16m2_t vec_row_m1 = __riscv_vle16_v_u16m2(row_m1, vl);
          vuint16m2_t vec_row = __riscv_vle16_v_u16m2(row, vl);
          vuint16m2_t vec_row_p1 = __riscv_vle16_v_u16m2(row_p1, vl);
          vuint16m2_t vec_row_p2 = __riscv_vle16_v_u16m2(row_p2, vl);

          // vuint16m2_t __riscv_vmul_vx_u16m2(vuint16m2_t vs2, uint16_t rs1, size_t vl);
          vuint16m2_t result = __riscv_vmul_vx_u16m2(vec_row_m2, kernel[0], vl);
          // vuint16m2_t __riscv_vmadd_vx_u16m2(vuint16m2_t vd, uint16_t rs1,vuint16m2_t vs2, size_t vl);
          result = __riscv_vmadd_vx_u16m2(result, kernel[1], vec_row_m1, vl);
          result = __riscv_vmadd_vx_u16m2(result, kernel[2], vec_row, vl);
          result = __riscv_vmadd_vx_u16m2(result, kernel[3], vec_row_p1, vl);
          result = __riscv_vmadd_vx_u16m2(result, kernel[4], vec_row_p2, vl);

          // void __riscv_vse16_v_u16m2(uint16_t *rs1, vuint16m2_t vs3, size_t vl);
          __riscv_vse16_v_u16m2(tmp.RowPtr(r), result, vl);

          // vint32m2_t __riscv_vwcvt_x_x_v_i32m2(vint16m1_t vs2, size_t vl); ???
          // vint32m2_t __riscv_vsext_vf2_i32m2(vint16m1_t vs2, size_t vl);
          
        }
        #endif

        // int * p_tmp_r = tmp.RowPtr(r)
        int tmp_array[32];
        //
        for (size_t vl, i = 0; i < n; i += vl) // , a += vl, b += vl, c += vl
        {
          vl = __riscv_vsetvl_e16m1(n - i);

          vint32m2_t vec_row_m2 = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf2_u32m2(__riscv_vle16_v_u16m1(row_m2 + i, vl), vl));
          vint32m2_t vec_row_m1 = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf2_u32m2(__riscv_vle16_v_u16m1(row_m1 + i, vl), vl));
          vint32m2_t vec_row = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf2_u32m2(__riscv_vle16_v_u16m1(row + i, vl), vl));
          vint32m2_t vec_row_p1 = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf2_u32m2(__riscv_vle16_v_u16m1(row_p1 + i, vl), vl));
          vint32m2_t vec_row_p2 = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf2_u32m2(__riscv_vle16_v_u16m1(row_p2 + i, vl), vl));

          /*
          std::cout << "uint16_t row_m2: ";
          for(size_t j = 0; j < vl; j++)
          std::cout << row_m2[i+j] << "  "; std::cout << std::endl;

          int int_row_m2[vl];
          std::cout << "int row_m2:      ";
          __riscv_vse32_v_i32m2(int_row_m2, vec_row_m2, vl);
          for(size_t j = 0; j < vl; j++)
          std::cout << int_row_m2[j] << "  "; std::cout << std::endl;
          */

          // vint32m2_t __riscv_vmul_vx_i32m2(vint32m2_t vs2, int32_t rs1, size_t vl);
          vint32m2_t result = __riscv_vmul_vx_i32m2(vec_row_m2, kernel[0], vl);
          // vint32m2_t __riscv_vmadd_vx_i32m2(vint32m2_t vd, int32_t rs1, vint32m2_t vs2, size_t vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_m1, kernel[1], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row, kernel[2], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_p1, kernel[3], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_p2, kernel[4], result, vl);

          /*if(r == 0 && i == 0)
          {
            int int_result[vl];
            std::cout << "int_result:      ";
            __riscv_vse32_v_i32m2(int_result, result, vl);
            for(size_t j = 0; j < vl; j++)
            std::cout << int_result[j] << "  "; std::cout << std::endl;
          }*/

          // void __riscv_vse32_v_i32m2(int32_t *rs1, vint32m2_t vs3, size_t vl);
          // __riscv_vse32_v_i32m2(tmp.RowPtr(r) + i, result, vl);
          // __riscv_vse32_v_i32m2(p_tmp_r + i, result, vl);

          __riscv_vse32_v_i32m2(tmp_array, result, vl);
          for(size_t j = 0; j < vl; j++)
            tmp(r, i + j) = tmp_array[j];

          /*if(r == 0 && i == 0)
          {
            std::cout << "tmp.RowPtr:      ";
            for(size_t j = 0; j < vl; j++)
            std::cout << tmp.RowPtr(r)[j] << "  "; std::cout << std::endl;

            std::cout << "a ce 666:         ";
            for(size_t j = 0; j < vl; j++)
            std::cout << tmp(r, j) << "  "; std::cout << std::endl;
          }*/

          // vint32m2_t __riscv_vwcvt_x_x_v_i32m2(vint16m1_t vs2, size_t vl); ???
          // vint32m2_t __riscv_vsext_vf2_i32m2(vint16m1_t vs2, size_t vl);
          
        }
#endif
        /*if(r == 0)
        {
            for(int i = 0; i < img.w; i++)
            {
              if(i % 16 == 0) std::cout << std::endl;
              std::cout << tmp(r, i) << " ";
            }

            std::cout << std::endl;
        }*/

      } // for (int r = 0 ...)
    }

    // Horizontal convolution
    {
      // 定义加法常量
      const int shift_const = (1 << 7);  // 即 128

      for (int c = 0; c < int(img_sub.w); c++) {
        const int* row_m2 = tmp.RowPtr(std::abs(2 * c - 2));
        const int* row_m1 = tmp.RowPtr(std::abs(2 * c - 1));
        const int* row = tmp.RowPtr(2 * c);
        const int* row_p1 = tmp.RowPtr(border101(2 * c + 1, tmp.h));
        const int* row_p2 = tmp.RowPtr(border101(2 * c + 2, tmp.h));
#if 0
        for (int r = 0; r < int(tmp.w); r++) {
          int val_int = kernel[0] * row_m2[r] + kernel[1] * row_m1[r] +
                        kernel[2] * row[r] + kernel[3] * row_p1[r] +
                        kernel[4] * row_p2[r];
          T val = ((val_int + (1 << 7)) >> 8);
          img_sub(c, r) = val;
        }
#else
        
        uint16_t tmp_array[32];
        size_t n = tmp.w;
        // for (size_t vl; n > 0; n -= vl) // , a += vl, b += vl, c += vl
        for (size_t vl, i = 0; i < n; i += vl)
        {
          vl = __riscv_vsetvl_e32m2(n - i);

          // vint32m2_t __riscv_vle32_v_i32m2(const int32_t *rs1, size_t vl);
          vint32m2_t vec_row_m2 = __riscv_vle32_v_i32m2(row_m2 + i, vl);
          vint32m2_t vec_row_m1 = __riscv_vle32_v_i32m2(row_m1 + i, vl);
          vint32m2_t vec_row = __riscv_vle32_v_i32m2(row + i, vl);
          vint32m2_t vec_row_p1 = __riscv_vle32_v_i32m2(row_p1 + i, vl);
          vint32m2_t vec_row_p2 = __riscv_vle32_v_i32m2(row_p2 + i, vl);

          // vint32m2_t __riscv_vmul_vx_i32m2(vint32m2_t vs2, int32_t rs1, size_t vl);
          vint32m2_t result = __riscv_vmul_vx_i32m2(vec_row_m2, kernel[0], vl);
          // vint32m2_t __riscv_vmadd_vx_i32m2(vint32m2_t vd, int32_t rs1, vint32m2_t vs2, size_t vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_m1, kernel[1], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row, kernel[2], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_p1, kernel[3], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_p2, kernel[4], result, vl);

          
          // vint32m2_t __riscv_vadd_vx_i32m2(vint32m2_t vs2, int32_t rs1, size_t vl);
          // vint32m2_t __riscv_vssra_vx_i32m2(vint32m2_t vs2, size_t rs1, unsigned int vxrm,size_t vl);
          // vint32m2_t __riscv_vsra_vx_i32m2(vint32m2_t vs2, size_t rs1, size_t vl);

          // 将加法常量添加到向量
          result = __riscv_vadd_vx_i32m2(result, shift_const, vl);

          // 执行带符号右移操作 (val_int_plus_const + 128) >> 8
          result = __riscv_vsra_vx_i32m2(result, 8, vl);


          // void __riscv_vse32_v_i32m2(int32_t *rs1, vint32m2_t vs3, size_t vl);
          // __riscv_vse32_v_i32m2(img_sub.RowPtr(r) + i, result, vl); // Store to output image

          // vuint16m1_t __riscv_vnclipu_wx_u16m1(vuint32m2_t vs2, size_t rs1, unsigned int vxrm, size_t vl);

          vuint16m1_t vec_u16_result = __riscv_vnclipu_wx_u16m1(__riscv_vreinterpret_v_i32m2_u32m2(result), 0, 0, vl); // ok.
          // __riscv_vse16_v_u16m1(img_sub.RowPtr(c) + i, vec_u16_result, vl);

          __riscv_vse16_v_u16m1(tmp_array, vec_u16_result, vl);
          for(size_t j = 0; j < vl; j++)
            img_sub(c, i + j) = tmp_array[j];

        }
#endif
      } // for(int c = 0; ...)
    }
  }
  // the end.
#else //defined(_U16M2_)
  static void subsample(const Image<const T>& img, Image<T>& img_sub) {
    static_assert(std::is_same<T, uint16_t>::value ||
                  std::is_same<T, uint8_t>::value);

    constexpr int kernel[5] = {1, 4, 6, 4, 1};

    // accumulator
    ManagedImage<int> tmp(img_sub.h, img.w);

    // Vertical convolution
    {
      for (int r = 0; r < int(img_sub.h); r++) {
        const T* row_m2 = img.RowPtr(std::abs(2 * r - 2));
        const T* row_m1 = img.RowPtr(std::abs(2 * r - 1));
        const T* row = img.RowPtr(2 * r);
        const T* row_p1 = img.RowPtr(border101(2 * r + 1, img.h));
        const T* row_p2 = img.RowPtr(border101(2 * r + 2, img.h));
#if 0
        for (int c = 0; c < int(img.w); c++) {
          tmp(r, c) = kernel[0] * int(row_m2[c]) + kernel[1] * int(row_m1[c]) +
                      kernel[2] * int(row[c]) + kernel[3] * int(row_p1[c]) +
                      kernel[4] * int(row_p2[c]);
        }
#else
        size_t n = img.w;

        // int * p_tmp_r = tmp.RowPtr(r)
        int tmp_array[32];
        //
        for (size_t vl, i = 0; i < n; i += vl) // , a += vl, b += vl, c += vl
        {
          vl = __riscv_vsetvl_e16m2(n - i);

          vint32m4_t vec_row_m2 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2_u32m4(__riscv_vle16_v_u16m2(row_m2 + i, vl), vl));
          vint32m4_t vec_row_m1 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2_u32m4(__riscv_vle16_v_u16m2(row_m1 + i, vl), vl));
          vint32m4_t vec_row = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2_u32m4(__riscv_vle16_v_u16m2(row + i, vl), vl));
          vint32m4_t vec_row_p1 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2_u32m4(__riscv_vle16_v_u16m2(row_p1 + i, vl), vl));
          vint32m4_t vec_row_p2 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2_u32m4(__riscv_vle16_v_u16m2(row_p2 + i, vl), vl));

          vint32m4_t result = __riscv_vmul_vx_i32m4(vec_row_m2, kernel[0], vl);
          result = __riscv_vmadd_vx_i32m4(vec_row_m1, kernel[1], result, vl);
          result = __riscv_vmadd_vx_i32m4(vec_row, kernel[2], result, vl);
          result = __riscv_vmadd_vx_i32m4(vec_row_p1, kernel[3], result, vl);
          result = __riscv_vmadd_vx_i32m4(vec_row_p2, kernel[4], result, vl);

          // void __riscv_vse32_v_i32m2(int32_t *rs1, vint32m2_t vs3, size_t vl);
          // __riscv_vse32_v_i32m2(tmp.RowPtr(r) + i, result, vl);
          // __riscv_vse32_v_i32m2(p_tmp_r + i, result, vl);

          __riscv_vse32_v_i32m4(tmp_array, result, vl);
          for(size_t j = 0; j < vl; j++)
            tmp(r, i + j) = tmp_array[j];


          // vint32m2_t __riscv_vwcvt_x_x_v_i32m2(vint16m1_t vs2, size_t vl); ???
          // vint32m2_t __riscv_vsext_vf2_i32m2(vint16m1_t vs2, size_t vl);
          
        }
#endif
        /*if(r == 0)
        {
            for(int i = 0; i < img.w; i++)
            {
              if(i % 16 == 0) std::cout << std::endl;
              std::cout << tmp(r, i) << " ";
            }

            std::cout << std::endl;
        }*/

      } // for (int r = 0 ...)
    }

    // Horizontal convolution
    {
      // 定义加法常量
      const int shift_const = (1 << 7);  // 即 128

      for (int c = 0; c < int(img_sub.w); c++) {
        const int* row_m2 = tmp.RowPtr(std::abs(2 * c - 2));
        const int* row_m1 = tmp.RowPtr(std::abs(2 * c - 1));
        const int* row = tmp.RowPtr(2 * c);
        const int* row_p1 = tmp.RowPtr(border101(2 * c + 1, tmp.h));
        const int* row_p2 = tmp.RowPtr(border101(2 * c + 2, tmp.h));
#if 0
        for (int r = 0; r < int(tmp.w); r++) {
          int val_int = kernel[0] * row_m2[r] + kernel[1] * row_m1[r] +
                        kernel[2] * row[r] + kernel[3] * row_p1[r] +
                        kernel[4] * row_p2[r];
          T val = ((val_int + (1 << 7)) >> 8);
          img_sub(c, r) = val;
        }
#else
        
        uint16_t tmp_array[32];
        size_t n = tmp.w;
        // for (size_t vl; n > 0; n -= vl) // , a += vl, b += vl, c += vl
        for (size_t vl, i = 0; i < n; i += vl)
        {
          vl = __riscv_vsetvl_e32m2(n - i);

          // vint32m2_t __riscv_vle32_v_i32m2(const int32_t *rs1, size_t vl);
          vint32m2_t vec_row_m2 = __riscv_vle32_v_i32m2(row_m2 + i, vl);
          vint32m2_t vec_row_m1 = __riscv_vle32_v_i32m2(row_m1 + i, vl);
          vint32m2_t vec_row = __riscv_vle32_v_i32m2(row + i, vl);
          vint32m2_t vec_row_p1 = __riscv_vle32_v_i32m2(row_p1 + i, vl);
          vint32m2_t vec_row_p2 = __riscv_vle32_v_i32m2(row_p2 + i, vl);

          // vint32m2_t __riscv_vmul_vx_i32m2(vint32m2_t vs2, int32_t rs1, size_t vl);
          vint32m2_t result = __riscv_vmul_vx_i32m2(vec_row_m2, kernel[0], vl);
          // vint32m2_t __riscv_vmadd_vx_i32m2(vint32m2_t vd, int32_t rs1, vint32m2_t vs2, size_t vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_m1, kernel[1], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row, kernel[2], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_p1, kernel[3], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_p2, kernel[4], result, vl);

          
          // vint32m2_t __riscv_vadd_vx_i32m2(vint32m2_t vs2, int32_t rs1, size_t vl);
          // vint32m2_t __riscv_vssra_vx_i32m2(vint32m2_t vs2, size_t rs1, unsigned int vxrm,size_t vl);
          // vint32m2_t __riscv_vsra_vx_i32m2(vint32m2_t vs2, size_t rs1, size_t vl);

          // 将加法常量添加到向量
          result = __riscv_vadd_vx_i32m2(result, shift_const, vl);

          // 执行带符号右移操作 (val_int_plus_const + 128) >> 8
          result = __riscv_vsra_vx_i32m2(result, 8, vl);


          // void __riscv_vse32_v_i32m2(int32_t *rs1, vint32m2_t vs3, size_t vl);
          // __riscv_vse32_v_i32m2(img_sub.RowPtr(r) + i, result, vl); // Store to output image

          // vuint16m1_t __riscv_vnclipu_wx_u16m1(vuint32m2_t vs2, size_t rs1, unsigned int vxrm, size_t vl);

          vuint16m1_t vec_u16_result = __riscv_vnclipu_wx_u16m1(__riscv_vreinterpret_v_i32m2_u32m2(result), 0, 0, vl); // ok.
          // __riscv_vse16_v_u16m1(img_sub.RowPtr(c) + i, vec_u16_result, vl);

          __riscv_vse16_v_u16m1(tmp_array, vec_u16_result, vl);
          for(size_t j = 0; j < vl; j++)
            img_sub(c, i + j) = tmp_array[j];

        }
#endif
      } // for(int c = 0; ...)
    }
  }
#endif

  /// @brief Return const image of the certain level
  ///
  /// @param lvl level to return
  /// @return const image of with the pyramid level
  inline const Image<const T> lvl(size_t lvl) const {
    size_t x = (lvl == 0) ? 0 : orig_w;
    size_t y = (lvl <= 1) ? 0 : (image.h - (image.h >> (lvl - 1)));
    size_t width = (orig_w >> lvl);
    size_t height = (image.h >> lvl);

    return image.SubImage(x, y, width, height);
  }

  /// @brief Return const image of underlying mipmap
  ///
  /// @return const image of of the underlying mipmap representation which can
  /// be for example used for visualization
  inline const Image<const T> mipmap() const {
    return image.SubImage(0, 0, image.w, image.h);
  }

  /// @brief Return coordinate offset of the image in the mipmap image.
  ///
  /// @param lvl level to return
  /// @return offset coordinates (2x1 vector)
  template <typename S>
  inline Eigen::Matrix<S, 2, 1> lvl_offset(size_t lvl) {
    size_t x = (lvl == 0) ? 0 : orig_w;
    size_t y = (lvl <= 1) ? 0 : (image.h - (image.h >> (lvl - 1)));

    return Eigen::Matrix<S, 2, 1>(x, y);
  }

 protected:
  /// @brief Return image of the certain level
  ///
  /// @param lvl level to return
  /// @return image of with the pyramid level
  inline Image<T> lvl_internal(size_t lvl) {
    // 这里的x和y是金字塔中的每层图像在一整幅图像中的位置，即表示缩放后的图像块所在的位置；
    // 换句话说是将金字塔所有层的图像拼接在同一个图像里面。
    size_t x = (lvl == 0) ? 0 : orig_w; // orig_w = 第0层图像的宽度
    size_t y = (lvl <= 1) ? 0 : (image.h - (image.h >> (lvl - 1)));
    size_t width = (orig_w >> lvl); // 缩放
    size_t height = (image.h >> lvl); // 缩放

    // basalt的构建图像金字塔是自己实现的，其形成的图像金字塔图像数据是保存在同一张图里的
    //（在默认的配置参数中，basalt默认图像金字塔构建三层-事实上他构建多了就容易出问题，会超出图像的索引，所以这个参数不能随便调整）。
    // 关于索引：
    // x = 0, orig_w， orig_w
    // y = 0， 0， h/2
    // 因此这里可以看到索引是与图像进行对应上的

    return image.SubImage(x, y, width, height);
  }

  size_t orig_w;          ///< Width of the original image (level 0)
  ManagedImage<T> image;  ///< Pyramid image stored as a mipmap
};

}  // namespace basalt

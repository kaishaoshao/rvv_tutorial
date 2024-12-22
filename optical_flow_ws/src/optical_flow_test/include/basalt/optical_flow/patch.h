/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

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
*/
#pragma once

#include <Eigen/Dense>
#include <sophus/se2.hpp>

#include <basalt/image/image.h>
#include <basalt/optical_flow/patterns.h>

// calc effiency: e32m2 > e32m8 > no_rvv > e32m4

// #define _USE_RISCV_V
#define _CALC_TIME_
#include <time.h>

#ifdef _USE_RISCV_V
#include <riscv_vector.h>

#define _E32M2_
// #define _E32M4_
// #define _E32M8_

// #define _REDUCTION_METHOD2_
// #define _REDUCTION_METHOD3_
// #define _REDUCTION_METHOD4_ // the best one but not need now. 2024-12-11.

// void *memcpy_vec(void *restrict destination, const void *restrict source, size_t n) {
void *memcpy_vec(void *destination, const void *source, size_t n) {
  unsigned char *dst = static_cast<unsigned char *>(destination);
  const unsigned char *src = static_cast<const unsigned char *>(source);
  // copy data byte by byte
  for (size_t vl; n > 0; n -= vl, src += vl, dst += vl) {
    vl = __riscv_vsetvl_e8m8(n);
    vuint8m8_t vec_src = __riscv_vle8_v_u8m8(src, vl);
    __riscv_vse8_v_u8m8(dst, vec_src, vl);
  }
  return destination;
}

#endif

namespace basalt {

template <typename Scalar, typename Pattern>
struct OpticalFlowPatch {
  static constexpr int PATTERN_SIZE = Pattern::PATTERN_SIZE; // 对于Pattern51来说，PATTERN_SIZE应该是52

  typedef Eigen::Matrix<int, 2, 1> Vector2i;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 1, 2> Vector2T;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 1> VectorP;

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 2> MatrixP2;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 3> MatrixP3;
  typedef Eigen::Matrix<Scalar, 3, PATTERN_SIZE> Matrix3P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 4> MatrixP4;
  typedef Eigen::Matrix<int, 2, PATTERN_SIZE> Matrix2Pi;

  static const Matrix2P pattern2;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OpticalFlowPatch() = default;

  // @img: 金字塔指定层的图像
  // @pos: 对应层的坐标
  OpticalFlowPatch(const Image<const uint16_t> &img, const Vector2 &pos) {
    setFromImage(img, pos);
  }

  template <typename ImgT>
  static void setData(const ImgT &img, const Vector2 &pos, Scalar &mean,
                      VectorP &data, const Sophus::SE2<Scalar> *se2 = nullptr) {
    int num_valid_points = 0;
    Scalar sum = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      Vector2 p;
      if (se2) {
        p = pos + (*se2) * pattern2.col(i);
      } else {
        p = pos + pattern2.col(i);
      };

      if (img.InBounds(p, 2)) {
        Scalar val = img.template interp<Scalar>(p);
        data[i] = val;
        sum += val;
        num_valid_points++;
      } else {
        data[i] = -1;
      }
    }

    mean = sum / num_valid_points;
    data /= mean;
  }

  // 雅可比计算：(在对应pattern对应原图位置中的雅克比)
  // 这部分在slam14光流小节中也实现了，只不过slam14是算一下正向的再算一下反向的光流，而basalt只用反向光流。
  // 目的：减小计算量反向光流是在上一帧上计算，因此只需要计算一遍
  // 理论部分: 对应单个像素的光流 ∂r/∂se2 = ∂I/∂pix * ∂pix/∂se2
  // ∂I/∂pix 表示图像梯度，因为图像是个离散的表达，因此这部分其实是采用f'(x) = f(x+Δx)−f(x) / Δx   f'(x) = \frac{f(x+\Delta x) - f(x)}{\Delta x}
  // 进行计算的，简单说，就是相邻像素差就是图像梯度了，但是为了保证精度，basalt做了线性插值。

#if !defined(_USE_RISCV_V)
  template <typename ImgT>
  static void setDataJacSe2(const ImgT &img, const Vector2 &pos, Scalar &mean,
                            VectorP &data, MatrixP3 &J_se2) {
    //- 雅可比是残差对状态量的偏导，这里的残差构建和几何雅可比，似乎都用到了扰动模型
    //- 正向光流法，求雅可比的时候，用的是第二个图像I2处的梯度
    // 本算法用的是反向光流法：即用第一个图像I1的梯度来代替
    // r = I2 - I1
    // J = ∂r/∂se2 = - ∂I/∂xi * ∂xi/∂se2

    int num_valid_points = 0;
    Scalar sum = 0;
    Vector3 grad_sum_se2(0, 0, 0);

    Eigen::Matrix<Scalar, 2, 3> Jw_se2; // 2 * 3的矩阵, 这个属于几何雅可比
    Jw_se2.template topLeftCorner<2, 2>().setIdentity(); // 左上角2 * 2设为单位阵，即前面两列由单位阵占据

    #if 0
    auto start = std::chrono::steady_clock::now();
    #endif

    #if 1
    clock_t clock_start = clock();
    #endif

    // 对于每个pattern内部的点进行计算
    for (int i = 0; i < PATTERN_SIZE; i++) { // PATTERN_SIZE=52的时候，表示patch里面有52个点，pattern2里面是坐标的偏移量
      Vector2 p = pos + pattern2.col(i); // 位于图像的位置，点的位置加上pattern里面的偏移量，得到在patch里面的新的位姿

      // Fill jacobians with respect to SE2 warp 对Jw_se2的第2列（列下标从0开始的,也即最后一列）赋值 //- 下面两行完全是为了构建几何雅可比。
      Jw_se2(0, 2) = -pattern2(1, i); // 取pattern2的第1行，第i列。 对于Pattern51来说，pattern2表示的是2*52的矩阵
      Jw_se2(1, 2) = pattern2(0, i); // 取pattern2的第0行，第i列

      if (img.InBounds(p, 2)) { // 判断加了偏移量的点p是否在图像内，border=2
        // valGrad[0]表示图像强度，valGrad[1]表示x方向梯度，valGrad[0]表示y方向梯度
        Vector3 valGrad = img.template interpGrad<Scalar>(p); // interp是interpolation的缩写，表示利用双线性插值计算图像灰度和图像梯度 ( x方向梯度, y方向梯度 )
        data[i] = valGrad[0]; // 赋值图像灰度值
        sum += valGrad[0]; // 统计总图像强度
        // J_se2在Pattern51的情况下是52*3，每一行是1*3. //?具体含义有待补充：其实这一部分是，梯度*几何雅可比
        J_se2.row(i) = valGrad.template tail<2>().transpose() * Jw_se2; // 链式法则: 取valGrad的后2位元素，即图像梯度，列向量转置后，变成1*2，再乘以2*3 
        grad_sum_se2 += J_se2.row(i); // 所有行的梯度相加
        num_valid_points++;
      } else {
        data[i] = -1;
      }
    }

    // std::cout << std::setprecision(3) << std::fixed << "1 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "2 sum=" << sum << "  grad_sum_se2=" << grad_sum_se2.transpose() << std::endl;

    mean = sum / num_valid_points; // 总灰度除以有效点数，得到平均亮度值，可以消除曝光时间引起的图像光度尺度变化，但无法消除光度的偏移变化

    const Scalar mean_inv = num_valid_points / sum; // 平均亮度的逆

    // std::cout << std::setprecision(7) << std::fixed << "22 mean_inv=" << mean_inv << "  num_valid_points=" << num_valid_points << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "23 data=" << data.transpose() << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "24 J_se2.row(0)=" << J_se2.row(0) << std::endl;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (data[i] >= 0) { // 如果patch里面序号i对应的点的图像强度大于等于0，
        J_se2.row(i) -= grad_sum_se2.transpose() * data[i] / sum; //? //TODO: -= 和 /  谁的优先级更高
        data[i] *= mean_inv;
      } else { // 否则无效的图像强度，该行直接置为0
        J_se2.row(i).setZero();
      }
    }
    J_se2 *= mean_inv; // 至此，完成了梯度雅可比和几何雅可比的链式法则求偏导的整个过程。

    #if 0
    auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << std::fixed << "setDataJacSe2: " << diff.count() << " s\n";
    // std::cout << std::fixed << "setDataJacSe2: " << std::chrono::duration_cast<std::chrono::milliseconds>((end - start).count()) << " ms\n";
    auto diff = end - start;
    std::cout << std::setprecision(6) << std::fixed << "setDataJacSe2:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    #endif

    #if 1
    clock_t clock_end = clock();
    auto clock_diff = clock_end - clock_start;
    clock_sum += clock_diff;
    #endif

    #if 0
    std::cout << std::setprecision(3) << std::fixed << "3 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    #endif

  }

#elif defined(_E32M8_) //&& !defined(CONDITION_2)
  template <typename ImgT>
  static void setDataJacSe2(const ImgT &img, const Vector2 &pos, Scalar &mean,
                            VectorP &data, MatrixP3 &J_se2) {
    //- 雅可比是残差对状态量的偏导，这里的残差构建和几何雅可比，似乎都用到了扰动模型
    //- 正向光流法，求雅可比的时候，用的是第二个图像I2处的梯度
    // 本算法用的是反向光流法：即用第一个图像I1的梯度来代替
    // r = I2 - I1
    // J = ∂r/∂se2 = - ∂I/∂xi * ∂xi/∂se2

    int num_valid_points = 0;
    Scalar sum = 0;
    Vector3 grad_sum_se2(0, 0, 0);

    Eigen::Matrix<Scalar, 2, 3> Jw_se2; // 2 * 3的矩阵, 这个属于几何雅可比
    Jw_se2.template topLeftCorner<2, 2>().setIdentity(); // 左上角2 * 2设为单位阵，即前面两列由单位阵占据

    #if 0
    auto start = std::chrono::steady_clock::now();
    #endif

    #if 1
    clock_t clock_start = clock();
    #endif

    J_se2.setZero();
    // uint8_t valid_index[PATTERN_SIZE] = { 0 };
    Scalar coefficient[PATTERN_SIZE] = { 0 };

    Scalar val[PATTERN_SIZE] = { 0 };

    Scalar grad_x[PATTERN_SIZE] = { 0 };
    Scalar grad_y[PATTERN_SIZE] = { 0 };

    Scalar Jw_se2_11[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_21[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_12[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_22[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_13[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_23[PATTERN_SIZE] = { 0 };
    
    Scalar *p_grad_x = grad_x;
    Scalar *p_grad_y = grad_y;

    Scalar*p_Jw_se2_11 = Jw_se2_11;
    Scalar*p_Jw_se2_21 = Jw_se2_21;
    Scalar*p_Jw_se2_12 = Jw_se2_12;
    Scalar*p_Jw_se2_22 = Jw_se2_22;
    Scalar*p_Jw_se2_13 = Jw_se2_13;
    Scalar*p_Jw_se2_23 = Jw_se2_23;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      //
      Vector2 p = pos + pattern2.col(i);
      Jw_se2(0, 2) = -pattern2(1, i);
      Jw_se2(1, 2) = pattern2(0, i);
      if (img.InBounds(p, 2)) {
        Vector3 valGrad = img.template interpGrad<Scalar>(p);
        data[i] = valGrad[0];
        // sum += valGrad[0];
        // J_se2.row(i) = valGrad.template tail<2>().transpose() * Jw_se2;
        // grad_sum_se2 += J_se2.row(i);

        // valid_index[num_valid_points] = i;

        val[i] = valGrad[0];

        grad_x[i] = valGrad[1];
        grad_y[i] = valGrad[2];

        Jw_se2_11[i] = Jw_se2(0, 0);
        Jw_se2_21[i] = Jw_se2(1, 0);
        Jw_se2_12[i] = Jw_se2(0, 1);
        Jw_se2_22[i] = Jw_se2(1, 1);
        Jw_se2_13[i] = Jw_se2(0, 2);
        Jw_se2_23[i] = Jw_se2(1, 2);

        num_valid_points++;

        coefficient[i] = 1;
      } else {
        data[i] = -1;

        val[i] = 0;

        grad_x[i] = 0;
        grad_y[i] = 0;

        Jw_se2_11[i] = 0;
        Jw_se2_21[i] = 0;
        Jw_se2_12[i] = 0;
        Jw_se2_22[i] = 0;
        Jw_se2_13[i] = 0;
        Jw_se2_23[i] = 0;

        coefficient[i] = 0;
      }
    }

    // fetch 3 columns
    Scalar *p_J_se2_11 = J_se2.data();
    Scalar *p_J_se2_12 = J_se2.data() + 52;
    Scalar *p_J_se2_13 = J_se2.data() + 104;

    int n = PATTERN_SIZE;
    for (size_t vl; n > 0; n -= vl) // , a += vl, b += vl, c += vl
    {
      vl = __riscv_vsetvl_e32m8(n);
      vfloat32m8_t vec_grad_x = __riscv_vle32_v_f32m8(p_grad_x, vl);
      vfloat32m8_t vec_grad_y = __riscv_vle32_v_f32m8(p_grad_y, vl);

      /*
       * vd * vs1 + vs2
       vfloat32m8_t __riscv_vfmadd_vv_f32m8(vfloat32m8_t vd, vfloat32m8_t vs1,
                                     vfloat32m8_t vs2, size_t vl);
       */
      vfloat32m8_t vec_result_a11 = __riscv_vfmadd_vv_f32m8(vec_grad_x, __riscv_vle32_v_f32m8(p_Jw_se2_11, vl), __riscv_vfmul_vv_f32m8(vec_grad_y, __riscv_vle32_v_f32m8(p_Jw_se2_21, vl), vl), vl);
      vfloat32m8_t vec_result_a12 = __riscv_vfmadd_vv_f32m8(vec_grad_x, __riscv_vle32_v_f32m8(p_Jw_se2_12, vl), __riscv_vfmul_vv_f32m8(vec_grad_y, __riscv_vle32_v_f32m8(p_Jw_se2_22, vl), vl), vl);
      vfloat32m8_t vec_result_a13 = __riscv_vfmadd_vv_f32m8(vec_grad_x, __riscv_vle32_v_f32m8(p_Jw_se2_13, vl), __riscv_vfmul_vv_f32m8(vec_grad_y, __riscv_vle32_v_f32m8(p_Jw_se2_23, vl), vl), vl);

      __riscv_vse32_v_f32m8(p_J_se2_11, vec_result_a11, vl);
      __riscv_vse32_v_f32m8(p_J_se2_12, vec_result_a12, vl);
      __riscv_vse32_v_f32m8(p_J_se2_13, vec_result_a13, vl);

      p_grad_x += vl;
      p_grad_y += vl;

      p_Jw_se2_11 += vl;
      p_Jw_se2_21 += vl;

      p_Jw_se2_12 += vl;
      p_Jw_se2_22 += vl;

      p_Jw_se2_13 += vl;
      p_Jw_se2_23 += vl;

      p_J_se2_11 += vl;
      p_J_se2_12 += vl;
      p_J_se2_13 += vl;

    }

    // std::cout << "sum=" << sum << "  J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "1 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;

    // TODO: grad_sum_se2 += J_se2.row(i);  sum += valGrad[0];

    // another method is reduction
    // 初始化一些变量
    size_t vlmax = __riscv_vsetvlmax_e32m8();  // 获取最大向量长度
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);  // 初始化零向量
    vfloat32m8_t vec_one = __riscv_vfmv_v_f_f32m8(1, vlmax);  // 初始化一向量
    vfloat32m8_t vec_s = __riscv_vfmv_v_f_f32m8(0, vlmax);  // 初始化累加向量

    vfloat32m8_t vec_s1 = __riscv_vfmv_v_f_f32m8(0, vlmax);
    vfloat32m8_t vec_s2 = __riscv_vfmv_v_f_f32m8(0, vlmax);
    vfloat32m8_t vec_s3 = __riscv_vfmv_v_f_f32m8(0, vlmax);

    Scalar *p_val = val;
    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;
    // 处理整个向量数据
    n = PATTERN_SIZE;
    for (size_t vl; n > 0; n -= vl) {
        vl = __riscv_vsetvl_e32m8(n);  // 根据剩余元素数设置向量长度
        // std::cout << "vl=" << vl << std::endl;

        // 从数据中加载向量
        vfloat32m8_t vec_val = __riscv_vle32_v_f32m8(p_val, vl);

        vfloat32m8_t vec_J_se2_11 = __riscv_vle32_v_f32m8(p_J_se2_11, vl);
        vfloat32m8_t vec_J_se2_12 = __riscv_vle32_v_f32m8(p_J_se2_12, vl);
        vfloat32m8_t vec_J_se2_13 = __riscv_vle32_v_f32m8(p_J_se2_13, vl);

        vbool4_t mask = __riscv_vmfne_vf_f32m8_b4(vec_val, 0, vl);

        vbool4_t mask_J_se2_11 = __riscv_vmfne_vf_f32m8_b4(vec_J_se2_11, 0, vl);
        vbool4_t mask_J_se2_12 = __riscv_vmfne_vf_f32m8_b4(vec_J_se2_12, 0, vl);
        vbool4_t mask_J_se2_13 = __riscv_vmfne_vf_f32m8_b4(vec_J_se2_13, 0, vl);

        // 执行归约求和
        vec_s = __riscv_vfmacc_vv_f32m8_tumu(mask, vec_s, vec_val, vec_one, vl);

        vec_s1 = __riscv_vfmacc_vv_f32m8_tumu(mask_J_se2_11, vec_s1, vec_J_se2_11, vec_one, vl);
        vec_s2 = __riscv_vfmacc_vv_f32m8_tumu(mask_J_se2_12, vec_s2, vec_J_se2_12, vec_one, vl);
        vec_s3 = __riscv_vfmacc_vv_f32m8_tumu(mask_J_se2_13, vec_s3, vec_J_se2_13, vec_one, vl);

        p_val += vl;

        p_J_se2_11 += vl;
        p_J_se2_12 += vl;
        p_J_se2_13 += vl;
    }

    // 最终归约求和
    vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m8_f32m1(vec_s, vec_zero, vlmax);

    vfloat32m1_t vec_sum1 = __riscv_vfredusum_vs_f32m8_f32m1(vec_s1, vec_zero, vlmax);
    vfloat32m1_t vec_sum2 = __riscv_vfredusum_vs_f32m8_f32m1(vec_s2, vec_zero, vlmax);
    vfloat32m1_t vec_sum3 = __riscv_vfredusum_vs_f32m8_f32m1(vec_s3, vec_zero, vlmax);
    
    // 提取最终的结果
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    
    grad_sum_se2[0] = __riscv_vfmv_f_s_f32m1_f32(vec_sum1);
    grad_sum_se2[1] = __riscv_vfmv_f_s_f32m1_f32(vec_sum2);
    grad_sum_se2(2) = __riscv_vfmv_f_s_f32m1_f32(vec_sum3);

    // std::cout << std::setprecision(3) << std::fixed << "2 sum=" << sum << "  grad_sum_se2=" << grad_sum_se2.transpose() << std::endl;

    mean = sum / num_valid_points; // 总灰度除以有效点数，得到平均亮度值，可以消除曝光时间引起的图像光度尺度变化，但无法消除光度的偏移变化

    const Scalar mean_inv = num_valid_points / sum; // 平均亮度的逆

    const Scalar constant1 = grad_sum_se2(0) / sum;
    const Scalar constant2 = grad_sum_se2(1) / sum;
    const Scalar constant3 = grad_sum_se2(2) / sum;

    float* pval = data.data();

    Scalar *p_coefficient = coefficient;

    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;

    n = PATTERN_SIZE;
    
    // std::cout << std::setprecision(7) << std::fixed << "22 mean_inv=" << mean_inv << "  num_valid_points=" << num_valid_points << "  vlmax=" << vlmax << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "23 data=" << data.transpose() << std::endl;

    for (size_t vl; n > 0; n -= vl)
    {
      vl = __riscv_vsetvl_e32m8(n);
      
      vfloat32m8_t vec_val = __riscv_vle32_v_f32m8(pval, vl);
      vfloat32m8_t vec_coefficient = __riscv_vle32_v_f32m8(p_coefficient, vl);
      vec_val = __riscv_vfmul_vv_f32m8(vec_val, vec_coefficient, vl);

      vfloat32m8_t vec_result_a11 = __riscv_vfmul_vf_f32m8(vec_val, constant1, vl);
      vfloat32m8_t vec_result_a12 = __riscv_vfmul_vf_f32m8(vec_val, constant2, vl);
      vfloat32m8_t vec_result_a13 = __riscv_vfmul_vf_f32m8(vec_val, constant3, vl);

      vfloat32m8_t vec_J_se2_11 = __riscv_vle32_v_f32m8(p_J_se2_11, vl);
      vfloat32m8_t vec_J_se2_12 = __riscv_vle32_v_f32m8(p_J_se2_12, vl);
      vfloat32m8_t vec_J_se2_13 = __riscv_vle32_v_f32m8(p_J_se2_13, vl);

      vec_result_a11 = __riscv_vfsub_vv_f32m8(vec_J_se2_11, vec_result_a11, vl);
      vec_result_a12 = __riscv_vfsub_vv_f32m8(vec_J_se2_12, vec_result_a12, vl);
      vec_result_a13 = __riscv_vfsub_vv_f32m8(vec_J_se2_13, vec_result_a13, vl);

      vec_result_a11 = __riscv_vfmul_vf_f32m8(vec_result_a11, mean_inv, vl);
      vec_result_a12 = __riscv_vfmul_vf_f32m8(vec_result_a12, mean_inv, vl);
      vec_result_a13 = __riscv_vfmul_vf_f32m8(vec_result_a13, mean_inv, vl);   

      __riscv_vse32_v_f32m8(p_J_se2_11, vec_result_a11, vl);
      __riscv_vse32_v_f32m8(p_J_se2_12, vec_result_a12, vl);
      __riscv_vse32_v_f32m8(p_J_se2_13, vec_result_a13, vl);
     
      vfloat32m8_t vec_result = __riscv_vfmul_vf_f32m8(vec_val, mean_inv, vl);

      __riscv_vse32_v_f32m8(pval, vec_result, vl);

      pval += vl;
      
      p_coefficient += vl;

      p_J_se2_11 += vl;
      p_J_se2_12 += vl;
      p_J_se2_13 += vl;
    }

    #if 0
    auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << std::fixed << "setDataJacSe2: " << diff.count() << " s\n";
    // std::cout << std::fixed << "setDataJacSe2: " << std::chrono::duration_cast<std::chrono::milliseconds>((end - start).count()) << " ms\n";
    auto diff = end - start;
    std::cout << std::setprecision(6) << std::fixed << "RVV setDataJacSe2:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    #endif

    #if 1
    clock_t clock_end = clock();
    auto clock_diff = clock_end - clock_start;
    clock_sum += clock_diff;
    #endif


    #if 0
    std::cout << std::setprecision(3) << std::fixed << "3 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    #endif

  }

#elif defined(_E32M4_)
template <typename ImgT>
  static void setDataJacSe2(const ImgT &img, const Vector2 &pos, Scalar &mean,
                            VectorP &data, MatrixP3 &J_se2) {
    //- 雅可比是残差对状态量的偏导，这里的残差构建和几何雅可比，似乎都用到了扰动模型
    //- 正向光流法，求雅可比的时候，用的是第二个图像I2处的梯度
    // 本算法用的是反向光流法：即用第一个图像I1的梯度来代替
    // r = I2 - I1
    // J = ∂r/∂se2 = - ∂I/∂xi * ∂xi/∂se2

    int num_valid_points = 0;
    Scalar sum = 0;
    Vector3 grad_sum_se2(0, 0, 0);

    Eigen::Matrix<Scalar, 2, 3> Jw_se2; // 2 * 3的矩阵, 这个属于几何雅可比
    Jw_se2.template topLeftCorner<2, 2>().setIdentity(); // 左上角2 * 2设为单位阵，即前面两列由单位阵占据

    #if 0
    auto start = std::chrono::steady_clock::now();
    #endif

    #if 1
    clock_t clock_start = clock();
    #endif

    J_se2.setZero();
    // uint8_t valid_index[PATTERN_SIZE] = { 0 };
    Scalar coefficient[PATTERN_SIZE] = { 0 };

    Scalar val[PATTERN_SIZE] = { 0 };

    Scalar grad_x[PATTERN_SIZE] = { 0 };
    Scalar grad_y[PATTERN_SIZE] = { 0 };

    Scalar Jw_se2_11[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_21[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_12[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_22[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_13[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_23[PATTERN_SIZE] = { 0 };
    
    Scalar *p_grad_x = grad_x;
    Scalar *p_grad_y = grad_y;

    Scalar*p_Jw_se2_11 = Jw_se2_11;
    Scalar*p_Jw_se2_21 = Jw_se2_21;
    Scalar*p_Jw_se2_12 = Jw_se2_12;
    Scalar*p_Jw_se2_22 = Jw_se2_22;
    Scalar*p_Jw_se2_13 = Jw_se2_13;
    Scalar*p_Jw_se2_23 = Jw_se2_23;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      //
      Vector2 p = pos + pattern2.col(i);
      Jw_se2(0, 2) = -pattern2(1, i);
      Jw_se2(1, 2) = pattern2(0, i);
      if (img.InBounds(p, 2)) {
        Vector3 valGrad = img.template interpGrad<Scalar>(p);
        data[i] = valGrad[0];
        // sum += valGrad[0];
        // J_se2.row(i) = valGrad.template tail<2>().transpose() * Jw_se2;
        // grad_sum_se2 += J_se2.row(i);

        // valid_index[num_valid_points] = i;

        val[i] = valGrad[0];

        grad_x[i] = valGrad[1];
        grad_y[i] = valGrad[2];

        Jw_se2_11[i] = Jw_se2(0, 0);
        Jw_se2_21[i] = Jw_se2(1, 0);
        Jw_se2_12[i] = Jw_se2(0, 1);
        Jw_se2_22[i] = Jw_se2(1, 1);
        Jw_se2_13[i] = Jw_se2(0, 2);
        Jw_se2_23[i] = Jw_se2(1, 2);

        num_valid_points++;

        coefficient[i] = 1;
      } else {
        data[i] = -1;

        val[i] = 0;

        grad_x[i] = 0;
        grad_y[i] = 0;

        Jw_se2_11[i] = 0;
        Jw_se2_21[i] = 0;
        Jw_se2_12[i] = 0;
        Jw_se2_22[i] = 0;
        Jw_se2_13[i] = 0;
        Jw_se2_23[i] = 0;

        coefficient[i] = 0;
      }
    }

    // fetch 3 columns
    Scalar *p_J_se2_11 = J_se2.data();
    Scalar *p_J_se2_12 = J_se2.data() + 52;
    Scalar *p_J_se2_13 = J_se2.data() + 104;

    int n = PATTERN_SIZE;
    for (size_t vl; n > 0; n -= vl) // , a += vl, b += vl, c += vl
    {
      vl = __riscv_vsetvl_e32m4(n);
      vfloat32m4_t vec_grad_x = __riscv_vle32_v_f32m4(p_grad_x, vl);
      vfloat32m4_t vec_grad_y = __riscv_vle32_v_f32m4(p_grad_y, vl);

      /*
       * vd * vs1 + vs2
       vfloat32m8_t __riscv_vfmadd_vv_f32m8(vfloat32m8_t vd, vfloat32m8_t vs1,
                                     vfloat32m8_t vs2, size_t vl);
       */
      vfloat32m4_t vec_result_a11 = __riscv_vfmadd_vv_f32m4(vec_grad_x, __riscv_vle32_v_f32m4(p_Jw_se2_11, vl), __riscv_vfmul_vv_f32m4(vec_grad_y, __riscv_vle32_v_f32m4(p_Jw_se2_21, vl), vl), vl);
      vfloat32m4_t vec_result_a12 = __riscv_vfmadd_vv_f32m4(vec_grad_x, __riscv_vle32_v_f32m4(p_Jw_se2_12, vl), __riscv_vfmul_vv_f32m4(vec_grad_y, __riscv_vle32_v_f32m4(p_Jw_se2_22, vl), vl), vl);
      vfloat32m4_t vec_result_a13 = __riscv_vfmadd_vv_f32m4(vec_grad_x, __riscv_vle32_v_f32m4(p_Jw_se2_13, vl), __riscv_vfmul_vv_f32m4(vec_grad_y, __riscv_vle32_v_f32m4(p_Jw_se2_23, vl), vl), vl);

      __riscv_vse32_v_f32m4(p_J_se2_11, vec_result_a11, vl);
      __riscv_vse32_v_f32m4(p_J_se2_12, vec_result_a12, vl);
      __riscv_vse32_v_f32m4(p_J_se2_13, vec_result_a13, vl);

      p_grad_x += vl;
      p_grad_y += vl;

      p_Jw_se2_11 += vl;
      p_Jw_se2_21 += vl;

      p_Jw_se2_12 += vl;
      p_Jw_se2_22 += vl;

      p_Jw_se2_13 += vl;
      p_Jw_se2_23 += vl;

      p_J_se2_11 += vl;
      p_J_se2_12 += vl;
      p_J_se2_13 += vl;

    }

    // std::cout << "sum=" << sum << "  J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "1 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;

    // TODO: grad_sum_se2 += J_se2.row(i);  sum += valGrad[0];

    // another method is reduction
    // 初始化一些变量
    size_t vlmax = __riscv_vsetvlmax_e32m4();  // 获取最大向量长度
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);  // 初始化零向量
    vfloat32m4_t vec_one = __riscv_vfmv_v_f_f32m4(1, vlmax);  // 初始化一向量
    vfloat32m4_t vec_s = __riscv_vfmv_v_f_f32m4(0, vlmax);  // 初始化累加向量

    vfloat32m4_t vec_s1 = __riscv_vfmv_v_f_f32m4(0, vlmax);
    vfloat32m4_t vec_s2 = __riscv_vfmv_v_f_f32m4(0, vlmax);
    vfloat32m4_t vec_s3 = __riscv_vfmv_v_f_f32m4(0, vlmax);

    Scalar *p_val = val;
    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;
    // 处理整个向量数据
    n = PATTERN_SIZE;
    for (size_t vl; n > 0; n -= vl) {
        vl = __riscv_vsetvl_e32m4(n);  // 根据剩余元素数设置向量长度
        // std::cout << "vl=" << vl << std::endl;

        // 从数据中加载向量
        vfloat32m4_t vec_val = __riscv_vle32_v_f32m4(p_val, vl);

        vfloat32m4_t vec_J_se2_11 = __riscv_vle32_v_f32m4(p_J_se2_11, vl);
        vfloat32m4_t vec_J_se2_12 = __riscv_vle32_v_f32m4(p_J_se2_12, vl);
        vfloat32m4_t vec_J_se2_13 = __riscv_vle32_v_f32m4(p_J_se2_13, vl);

        vbool8_t mask = __riscv_vmfne_vf_f32m4_b8(vec_val, 0, vl);

        vbool8_t mask_J_se2_11 = __riscv_vmfne_vf_f32m4_b8(vec_J_se2_11, 0, vl);
        vbool8_t mask_J_se2_12 = __riscv_vmfne_vf_f32m4_b8(vec_J_se2_12, 0, vl);
        vbool8_t mask_J_se2_13 = __riscv_vmfne_vf_f32m4_b8(vec_J_se2_13, 0, vl);

        // 执行归约求和
        vec_s = __riscv_vfmacc_vv_f32m4_tumu(mask, vec_s, vec_val, vec_one, vl);

        vec_s1 = __riscv_vfmacc_vv_f32m4_tumu(mask_J_se2_11, vec_s1, vec_J_se2_11, vec_one, vl);
        vec_s2 = __riscv_vfmacc_vv_f32m4_tumu(mask_J_se2_12, vec_s2, vec_J_se2_12, vec_one, vl);
        vec_s3 = __riscv_vfmacc_vv_f32m4_tumu(mask_J_se2_13, vec_s3, vec_J_se2_13, vec_one, vl);

        p_val += vl;

        p_J_se2_11 += vl;
        p_J_se2_12 += vl;
        p_J_se2_13 += vl;
    }

    // 最终归约求和
    vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m4_f32m1(vec_s, vec_zero, vlmax);

    vfloat32m1_t vec_sum1 = __riscv_vfredusum_vs_f32m4_f32m1(vec_s1, vec_zero, vlmax);
    vfloat32m1_t vec_sum2 = __riscv_vfredusum_vs_f32m4_f32m1(vec_s2, vec_zero, vlmax);
    vfloat32m1_t vec_sum3 = __riscv_vfredusum_vs_f32m4_f32m1(vec_s3, vec_zero, vlmax);
    
    // 提取最终的结果
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    
    grad_sum_se2[0] = __riscv_vfmv_f_s_f32m1_f32(vec_sum1);
    grad_sum_se2[1] = __riscv_vfmv_f_s_f32m1_f32(vec_sum2);
    grad_sum_se2(2) = __riscv_vfmv_f_s_f32m1_f32(vec_sum3);

    // std::cout << std::setprecision(3) << std::fixed << "2 sum=" << sum << "  grad_sum_se2=" << grad_sum_se2.transpose() << std::endl;

    mean = sum / num_valid_points; // 总灰度除以有效点数，得到平均亮度值，可以消除曝光时间引起的图像光度尺度变化，但无法消除光度的偏移变化

    const Scalar mean_inv = num_valid_points / sum; // 平均亮度的逆

    const Scalar constant1 = grad_sum_se2(0) / sum;
    const Scalar constant2 = grad_sum_se2(1) / sum;
    const Scalar constant3 = grad_sum_se2(2) / sum;

    float* pval = data.data();

    Scalar *p_coefficient = coefficient;

    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;

    n = PATTERN_SIZE;
    
    // std::cout << std::setprecision(7) << std::fixed << "22 mean_inv=" << mean_inv << "  num_valid_points=" << num_valid_points << "  vlmax=" << vlmax << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "23 data=" << data.transpose() << std::endl;

    for (size_t vl; n > 0; n -= vl)
    {
      vl = __riscv_vsetvl_e32m4(n);
      
      vfloat32m4_t vec_val = __riscv_vle32_v_f32m4(pval, vl);
      vfloat32m4_t vec_coefficient = __riscv_vle32_v_f32m4(p_coefficient, vl);
      vec_val = __riscv_vfmul_vv_f32m4(vec_val, vec_coefficient, vl);

      vfloat32m4_t vec_result_a11 = __riscv_vfmul_vf_f32m4(vec_val, constant1, vl);
      vfloat32m4_t vec_result_a12 = __riscv_vfmul_vf_f32m4(vec_val, constant2, vl);
      vfloat32m4_t vec_result_a13 = __riscv_vfmul_vf_f32m4(vec_val, constant3, vl);

      vfloat32m4_t vec_J_se2_11 = __riscv_vle32_v_f32m4(p_J_se2_11, vl);
      vfloat32m4_t vec_J_se2_12 = __riscv_vle32_v_f32m4(p_J_se2_12, vl);
      vfloat32m4_t vec_J_se2_13 = __riscv_vle32_v_f32m4(p_J_se2_13, vl);

      vec_result_a11 = __riscv_vfsub_vv_f32m4(vec_J_se2_11, vec_result_a11, vl);
      vec_result_a12 = __riscv_vfsub_vv_f32m4(vec_J_se2_12, vec_result_a12, vl);
      vec_result_a13 = __riscv_vfsub_vv_f32m4(vec_J_se2_13, vec_result_a13, vl);

      vec_result_a11 = __riscv_vfmul_vf_f32m4(vec_result_a11, mean_inv, vl);
      vec_result_a12 = __riscv_vfmul_vf_f32m4(vec_result_a12, mean_inv, vl);
      vec_result_a13 = __riscv_vfmul_vf_f32m4(vec_result_a13, mean_inv, vl);   

      __riscv_vse32_v_f32m4(p_J_se2_11, vec_result_a11, vl);
      __riscv_vse32_v_f32m4(p_J_se2_12, vec_result_a12, vl);
      __riscv_vse32_v_f32m4(p_J_se2_13, vec_result_a13, vl);
     
      vfloat32m4_t vec_result = __riscv_vfmul_vf_f32m4(vec_val, mean_inv, vl);

      __riscv_vse32_v_f32m4(pval, vec_result, vl);

      pval += vl;
      
      p_coefficient += vl;

      p_J_se2_11 += vl;
      p_J_se2_12 += vl;
      p_J_se2_13 += vl;
    }

    #if 0
    auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << std::fixed << "setDataJacSe2: " << diff.count() << " s\n";
    // std::cout << std::fixed << "setDataJacSe2: " << std::chrono::duration_cast<std::chrono::milliseconds>((end - start).count()) << " ms\n";
    auto diff = end - start;
    std::cout << std::setprecision(6) << std::fixed << "RVV setDataJacSe2:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    #endif

    #if 1
    clock_t clock_end = clock();
    auto clock_diff = clock_end - clock_start;
    clock_sum += clock_diff;
    #endif


    #if 0
    std::cout << std::setprecision(3) << std::fixed << "3 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    #endif

  }

#elif defined(_E32M2_)
template <typename ImgT>
  static void setDataJacSe2(const ImgT &img, const Vector2 &pos, Scalar &mean,
                            VectorP &data, MatrixP3 &J_se2) {
    //- 雅可比是残差对状态量的偏导，这里的残差构建和几何雅可比，似乎都用到了扰动模型
    //- 正向光流法，求雅可比的时候，用的是第二个图像I2处的梯度
    // 本算法用的是反向光流法：即用第一个图像I1的梯度来代替
    // r = I2 - I1
    // J = ∂r/∂se2 = - ∂I/∂xi * ∂xi/∂se2

    int num_valid_points = 0;
    Scalar sum = 0;
    Vector3 grad_sum_se2(0, 0, 0);

    Eigen::Matrix<Scalar, 2, 3> Jw_se2; // 2 * 3的矩阵, 这个属于几何雅可比
    Jw_se2.template topLeftCorner<2, 2>().setIdentity(); // 左上角2 * 2设为单位阵，即前面两列由单位阵占据

    #if 0
    auto start = std::chrono::steady_clock::now();
    #endif

    #if 1
    clock_t clock_start = clock();
    #endif

    J_se2.setZero();
    // uint8_t valid_index[PATTERN_SIZE] = { 0 };
/*
 * comment on 2024-12-11    
    Scalar coefficient[PATTERN_SIZE] = { 0 };

    Scalar val[PATTERN_SIZE] = { 0 };
*/
    alignas(32) Scalar grad_x[PATTERN_SIZE] = { 0 };
    alignas(32) Scalar grad_y[PATTERN_SIZE] = { 0 };

    alignas(32) Scalar Jw_se2_11[PATTERN_SIZE] = { 0 };
    alignas(32) Scalar Jw_se2_21[PATTERN_SIZE] = { 0 };
    alignas(32) Scalar Jw_se2_12[PATTERN_SIZE] = { 0 };
    alignas(32) Scalar Jw_se2_22[PATTERN_SIZE] = { 0 };
    alignas(32) Scalar Jw_se2_13[PATTERN_SIZE] = { 0 };
    alignas(32) Scalar Jw_se2_23[PATTERN_SIZE] = { 0 };

    Scalar *p_grad_x = grad_x;
    Scalar *p_grad_y = grad_y;

    Scalar*p_Jw_se2_11 = Jw_se2_11;
    Scalar*p_Jw_se2_21 = Jw_se2_21;
    Scalar*p_Jw_se2_12 = Jw_se2_12;
    Scalar*p_Jw_se2_22 = Jw_se2_22;
    Scalar*p_Jw_se2_13 = Jw_se2_13;
    Scalar*p_Jw_se2_23 = Jw_se2_23;
/*
 * method 1:
    for (int i = 0; i < PATTERN_SIZE; i++) {
      //
      Vector2 p = pos + pattern2.col(i);
      Jw_se2(0, 2) = -pattern2(1, i);
      Jw_se2(1, 2) = pattern2(0, i);
      if (img.InBounds(p, 2)) {
        Vector3 valGrad = img.template interpGrad<Scalar>(p);
        data[i] = valGrad[0];
        // sum += valGrad[0];
        // J_se2.row(i) = valGrad.template tail<2>().transpose() * Jw_se2;
        // grad_sum_se2 += J_se2.row(i);

        // valid_index[num_valid_points] = i;

        val[i] = valGrad[0];

        grad_x[i] = valGrad[1];
        grad_y[i] = valGrad[2];

        Jw_se2_11[i] = Jw_se2(0, 0);
        Jw_se2_21[i] = Jw_se2(1, 0);
        Jw_se2_12[i] = Jw_se2(0, 1);
        Jw_se2_22[i] = Jw_se2(1, 1);
        Jw_se2_13[i] = Jw_se2(0, 2);
        Jw_se2_23[i] = Jw_se2(1, 2);

        num_valid_points++;

        coefficient[i] = 1;
      } else {
        data[i] = -1;

        val[i] = 0;

        grad_x[i] = 0;
        grad_y[i] = 0;

        Jw_se2_11[i] = 0;
        Jw_se2_21[i] = 0;
        Jw_se2_12[i] = 0;
        Jw_se2_22[i] = 0;
        Jw_se2_13[i] = 0;
        Jw_se2_23[i] = 0;

        coefficient[i] = 0;
      }
    }
*/
    // TODO: method 2
    alignas(32) Scalar u[PATTERN_SIZE];
    alignas(32) Scalar v[PATTERN_SIZE];
    // alignas(32) Scalar result[PATTERN_SIZE];
    Scalar *result = data.data();
    const Scalar *p_point2d = pattern2.data();

    for (int i = 0, j = 0; i < PATTERN_SIZE; i++) {
      j = 2 * i;
      u[i] = p_point2d[j] + pos[0];
      v[i] = p_point2d[j + 1] + pos(1);
      // std::cout << "1 u[" << i << "]=" << u[i] << " v[" << i << "]=" << v[i] << std::endl;

      Jw_se2_23[i] = p_point2d[j];
      Jw_se2_13[i] = -p_point2d[j + 1];
      Jw_se2_11[i] = 1;
      Jw_se2_22[i] = 1;
      //
    }

    // method 1: take all valid points into array alone. 
    // and only apply these good points with bilinear interpolation of rvv.

    size_t n = PATTERN_SIZE;
    size_t w = img.w;
    size_t h = img.h;
    Scalar border = 2.0;
    Scalar offset(1);

    size_t vlmax = __riscv_vsetvlmax_e32m2(); // return 16
    vfloat32m2_t vec_zero = __riscv_vfmv_v_f_f32m2(0, vlmax);
    vfloat32m2_t vec_minus_one = __riscv_vfmv_v_f_f32m2(-1, vlmax);
    
    // mehtod 2: use mask & merge
    for(size_t vl, i = 0; i < n; i += vl)
    {
        // 根据剩余元素数设置向量长度
        vl = __riscv_vsetvl_e32m2(n - i);

        // 从数据中加载向量
        vfloat32m2_t vec_point2d_x = __riscv_vle32_v_f32m2(u + i, vl);
        vfloat32m2_t vec_point2d_y = __riscv_vle32_v_f32m2(v + i, vl);

        // >= border
        vbool16_t mask = __riscv_vmfge_vf_f32m2_b16(vec_point2d_x, border, vl);

        // < (w - border - offset)
        mask = __riscv_vmflt_vf_f32m2_b16_m(mask, vec_point2d_x, (w - border - offset), vl);

        mask = __riscv_vmfge_vf_f32m2_b16_m(mask, vec_point2d_y, border, vl);
        mask = __riscv_vmflt_vf_f32m2_b16_m(mask, vec_point2d_y, (h - border - offset), vl);

        vec_point2d_x= __riscv_vmerge_vvm_f32m2(vec_zero, vec_point2d_x, mask, vl);
        vec_point2d_y= __riscv_vmerge_vvm_f32m2(vec_zero, vec_point2d_y, mask, vl);
        /// storage
        __riscv_vse32_v_f32m2(u + i, vec_point2d_x, vl);
        __riscv_vse32_v_f32m2(v + i, vec_point2d_y, vl);

        num_valid_points += __riscv_vcpop_m_b16(mask, vl);
        // std::cout << "vl=" << vl << " num_valid_points=" << num_valid_points << std::endl;
    }

    sum = img.computePatchInterpGrad(u, v, result, grad_x, grad_y, PATTERN_SIZE);
    /*
     * comment this section. put some codes back on 2024-12-11.
    // set data[i] = -1 when point is outlier.
    for(size_t vl, i = 0; i < n; i += vl)
    {
      // 根据剩余元素数设置向量长度
      vl = __riscv_vsetvl_e32m2(n - i);
      vfloat32m2_t vec_point2d_x = __riscv_vle32_v_f32m2(u + i, vl);
      vfloat32m2_t vec_data = __riscv_vle32_v_f32m2(result + i, vl);

      // > 0
      vbool16_t mask = __riscv_vmfgt_vf_f32m2_b16(vec_point2d_x, 0, vl); // 2d点有效时，data[i] > 0;
      vec_data= __riscv_vmerge_vvm_f32m2(vec_minus_one, vec_data, mask, vl); // 否则data[i] = -1.

      __riscv_vse32_v_f32m2(&result[i], vec_data, vl);
    }*/
    // TODO THE END.

    vfloat32m1_t vec_s1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // 初始化累加向量
    vfloat32m1_t vec_s2 = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // 初始化累加向量
    vfloat32m1_t vec_s3 = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // 初始化累加向量

    // fetch 3 columns
    Scalar *p_J_se2_11 = J_se2.data();
    Scalar *p_J_se2_12 = J_se2.data() + 52;
    Scalar *p_J_se2_13 = J_se2.data() + 104;

    // int n = PATTERN_SIZE;
    for (size_t vl; n > 0; n -= vl) // , a += vl, b += vl, c += vl
    {
      vl = __riscv_vsetvl_e32m2(n);
      vfloat32m2_t vec_grad_x = __riscv_vle32_v_f32m2(p_grad_x, vl);
      vfloat32m2_t vec_grad_y = __riscv_vle32_v_f32m2(p_grad_y, vl);

      /*
       * vd * vs1 + vs2
       vfloat32m2_t __riscv_vfmadd_vv_f32m2(vfloat32m2_t vd, vfloat32m2_t vs1,
                                     vfloat32m2_t vs2, size_t vl);
       */
      vfloat32m2_t vec_result_a11 = __riscv_vfmadd_vv_f32m2(vec_grad_x, __riscv_vle32_v_f32m2(p_Jw_se2_11, vl), __riscv_vfmul_vv_f32m2(vec_grad_y, __riscv_vle32_v_f32m2(p_Jw_se2_21, vl), vl), vl);
      vfloat32m2_t vec_result_a12 = __riscv_vfmadd_vv_f32m2(vec_grad_x, __riscv_vle32_v_f32m2(p_Jw_se2_12, vl), __riscv_vfmul_vv_f32m2(vec_grad_y, __riscv_vle32_v_f32m2(p_Jw_se2_22, vl), vl), vl);
      vfloat32m2_t vec_result_a13 = __riscv_vfmadd_vv_f32m2(vec_grad_x, __riscv_vle32_v_f32m2(p_Jw_se2_13, vl), __riscv_vfmul_vv_f32m2(vec_grad_y, __riscv_vle32_v_f32m2(p_Jw_se2_23, vl), vl), vl);

      __riscv_vse32_v_f32m2(p_J_se2_11, vec_result_a11, vl);
      __riscv_vse32_v_f32m2(p_J_se2_12, vec_result_a12, vl);
      __riscv_vse32_v_f32m2(p_J_se2_13, vec_result_a13, vl);

      vec_s1 = __riscv_vfredusum_vs_f32m2_f32m1(vec_result_a11, vec_s1, vl);
      vec_s2 = __riscv_vfredusum_vs_f32m2_f32m1(vec_result_a12, vec_s2, vl);
      vec_s3 = __riscv_vfredusum_vs_f32m2_f32m1(vec_result_a13, vec_s3, vl);

      p_grad_x += vl;
      p_grad_y += vl;

      p_Jw_se2_11 += vl;
      p_Jw_se2_21 += vl;

      p_Jw_se2_12 += vl;
      p_Jw_se2_22 += vl;

      p_Jw_se2_13 += vl;
      p_Jw_se2_23 += vl;

      p_J_se2_11 += vl;
      p_J_se2_12 += vl;
      p_J_se2_13 += vl;

    }

    // 提前到这里来计算，因此后面的归约操作可以全部去掉。
    grad_sum_se2[0] = __riscv_vfmv_f_s_f32m1_f32(vec_s1);
    grad_sum_se2[1] = __riscv_vfmv_f_s_f32m1_f32(vec_s2);
    grad_sum_se2(2) = __riscv_vfmv_f_s_f32m1_f32(vec_s3);

    // std::cout << "sum=" << sum << "  J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "1 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;

    // TODO: grad_sum_se2 += J_se2.row(i);  sum += valGrad[0];

    // another method is reduction
    #if defined(_REDUCTION_METHOD2_)
    // method 2: use mask
    // 初始化一些变量
    size_t vlmax = __riscv_vsetvlmax_e32m2();  // 获取最大向量长度
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);  // 初始化零向量
    vfloat32m2_t vec_one = __riscv_vfmv_v_f_f32m2(1, vlmax);  // 初始化一向量
    vfloat32m2_t vec_s = __riscv_vfmv_v_f_f32m2(0, vlmax);  // 初始化累加向量

    vfloat32m2_t vec_s1 = __riscv_vfmv_v_f_f32m2(0, vlmax);
    vfloat32m2_t vec_s2 = __riscv_vfmv_v_f_f32m2(0, vlmax);
    vfloat32m2_t vec_s3 = __riscv_vfmv_v_f_f32m2(0, vlmax);

    Scalar *p_val = val;
    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;
    // 处理整个向量数据
    n = PATTERN_SIZE;
    for (size_t vl; n > 0; n -= vl) {
        vl = __riscv_vsetvl_e32m2(n);  // 根据剩余元素数设置向量长度
        // std::cout << "vl=" << vl << std::endl;

        // 从数据中加载向量
        vfloat32m2_t vec_val = __riscv_vle32_v_f32m2(p_val, vl);

        vfloat32m2_t vec_J_se2_11 = __riscv_vle32_v_f32m2(p_J_se2_11, vl);
        vfloat32m2_t vec_J_se2_12 = __riscv_vle32_v_f32m2(p_J_se2_12, vl);
        vfloat32m2_t vec_J_se2_13 = __riscv_vle32_v_f32m2(p_J_se2_13, vl);

        vbool16_t mask = __riscv_vmfne_vf_f32m2_b16(vec_val, 0, vl);

        vbool16_t mask_J_se2_11 = __riscv_vmfne_vf_f32m2_b16(vec_J_se2_11, 0, vl);
        vbool16_t mask_J_se2_12 = __riscv_vmfne_vf_f32m2_b16(vec_J_se2_12, 0, vl);
        vbool16_t mask_J_se2_13 = __riscv_vmfne_vf_f32m2_b16(vec_J_se2_13, 0, vl);

        // 执行归约求和
        vec_s = __riscv_vfmacc_vv_f32m2_tumu(mask, vec_s, vec_val, vec_one, vl);

        vec_s1 = __riscv_vfmacc_vv_f32m2_tumu(mask_J_se2_11, vec_s1, vec_J_se2_11, vec_one, vl);
        vec_s2 = __riscv_vfmacc_vv_f32m2_tumu(mask_J_se2_12, vec_s2, vec_J_se2_12, vec_one, vl);
        vec_s3 = __riscv_vfmacc_vv_f32m2_tumu(mask_J_se2_13, vec_s3, vec_J_se2_13, vec_one, vl);

        p_val += vl;

        p_J_se2_11 += vl;
        p_J_se2_12 += vl;
        p_J_se2_13 += vl;
    }

    // 最终归约求和
    vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m2_f32m1(vec_s, vec_zero, vlmax);

    vfloat32m1_t vec_sum1 = __riscv_vfredusum_vs_f32m2_f32m1(vec_s1, vec_zero, vlmax);
    vfloat32m1_t vec_sum2 = __riscv_vfredusum_vs_f32m2_f32m1(vec_s2, vec_zero, vlmax);
    vfloat32m1_t vec_sum3 = __riscv_vfredusum_vs_f32m2_f32m1(vec_s3, vec_zero, vlmax);
    
    // 提取最终的结果
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    
    grad_sum_se2[0] = __riscv_vfmv_f_s_f32m1_f32(vec_sum1);
    grad_sum_se2[1] = __riscv_vfmv_f_s_f32m1_f32(vec_sum2);
    grad_sum_se2(2) = __riscv_vfmv_f_s_f32m1_f32(vec_sum3);


    #elif defined(_REDUCTION_METHOD3_)
    // method 3: vfloat32m2_t __riscv_vfmacc_vf_f32m2(vfloat32m2_t vd, float rs1, vfloat32m2_t vs2, size_t vl);
    size_t vlmax = __riscv_vsetvlmax_e32m2();  // 获取最大向量长度
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);  // 初始化零向量
    // vfloat32m2_t vec_one = __riscv_vfmv_v_f_f32m2(1, vlmax);  // 初始化一向量
    vfloat32m2_t vec_s = __riscv_vfmv_v_f_f32m2(0, vlmax);  // 初始化累加向量

    vfloat32m2_t vec_s1 = __riscv_vfmv_v_f_f32m2(0, vlmax);
    vfloat32m2_t vec_s2 = __riscv_vfmv_v_f_f32m2(0, vlmax);
    vfloat32m2_t vec_s3 = __riscv_vfmv_v_f_f32m2(0, vlmax);

    Scalar *p_val = val;
    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;
    // 处理整个向量数据
    n = PATTERN_SIZE;
    for (size_t vl; n > 0; n -= vl) {
        vl = __riscv_vsetvl_e32m2(n);  // 根据剩余元素数设置向量长度
        // std::cout << "vl=" << vl << std::endl;

        // 从数据中加载向量
        vfloat32m2_t vec_val = __riscv_vle32_v_f32m2(p_val, vl);

        vfloat32m2_t vec_J_se2_11 = __riscv_vle32_v_f32m2(p_J_se2_11, vl);
        vfloat32m2_t vec_J_se2_12 = __riscv_vle32_v_f32m2(p_J_se2_12, vl);
        vfloat32m2_t vec_J_se2_13 = __riscv_vle32_v_f32m2(p_J_se2_13, vl);

        // vbool16_t mask = __riscv_vmfne_vf_f32m2_b16(vec_val, 0, vl);
        // vbool16_t mask_J_se2_11 = __riscv_vmfne_vf_f32m2_b16(vec_J_se2_11, 0, vl);
        // vbool16_t mask_J_se2_12 = __riscv_vmfne_vf_f32m2_b16(vec_J_se2_12, 0, vl);
        // vbool16_t mask_J_se2_13 = __riscv_vmfne_vf_f32m2_b16(vec_J_se2_13, 0, vl);

        // 执行归约求和
        // vec_s = __riscv_vfmacc_vv_f32m2_tumu(mask, vec_s, vec_val, vec_one, vl);

        // vec_s1 = __riscv_vfmacc_vv_f32m2_tumu(mask_J_se2_11, vec_s1, vec_J_se2_11, vec_one, vl);
        // vec_s2 = __riscv_vfmacc_vv_f32m2_tumu(mask_J_se2_12, vec_s2, vec_J_se2_12, vec_one, vl);
        // vec_s3 = __riscv_vfmacc_vv_f32m2_tumu(mask_J_se2_13, vec_s3, vec_J_se2_13, vec_one, vl);

        vec_s = __riscv_vfmacc_vf_f32m2(vec_s, 1.0, vec_val, vl);
        vec_s1 = __riscv_vfmacc_vf_f32m2(vec_s1, 1.0, vec_J_se2_11, vl);
        vec_s2 = __riscv_vfmacc_vf_f32m2(vec_s2, 1.0, vec_J_se2_12, vl);
        vec_s3 = __riscv_vfmacc_vf_f32m2(vec_s3, 1.0, vec_J_se2_13, vl);

        p_val += vl;

        p_J_se2_11 += vl;
        p_J_se2_12 += vl;
        p_J_se2_13 += vl;
    }

    // 最终归约求和
    vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m2_f32m1(vec_s, vec_zero, vlmax);

    vfloat32m1_t vec_sum1 = __riscv_vfredusum_vs_f32m2_f32m1(vec_s1, vec_zero, vlmax);
    vfloat32m1_t vec_sum2 = __riscv_vfredusum_vs_f32m2_f32m1(vec_s2, vec_zero, vlmax);
    vfloat32m1_t vec_sum3 = __riscv_vfredusum_vs_f32m2_f32m1(vec_s3, vec_zero, vlmax);
    
    // 提取最终的结果
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    
    grad_sum_se2[0] = __riscv_vfmv_f_s_f32m1_f32(vec_sum1);
    grad_sum_se2[1] = __riscv_vfmv_f_s_f32m1_f32(vec_sum2);
    grad_sum_se2(2) = __riscv_vfmv_f_s_f32m1_f32(vec_sum3);

    #elif defined(_REDUCTION_METHOD4_)
    // method 4: vfloat32m1_t __riscv_vfredusum_vs_f32m2_f32m1(vfloat32m2_t vs2, vfloat32m1_t vs1, size_t vl);
    vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // 初始化累加向量

    vfloat32m1_t vec_s1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // 初始化累加向量
    vfloat32m1_t vec_s2 = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // 初始化累加向量
    vfloat32m1_t vec_s3 = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // 初始化累加向量

    Scalar *p_val = val;
    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;
    // 处理整个向量数据
    n = PATTERN_SIZE;

    for (size_t vl; n > 0; n -= vl)
    {
        // 根据剩余元素数设置向量长度
        vl = __riscv_vsetvl_e32m2(n);
        // 从数据中加载向量
        vfloat32m2_t vec_val = __riscv_vle32_v_f32m2(p_val, vl);

        vfloat32m2_t vec_J_se2_11 = __riscv_vle32_v_f32m2(p_J_se2_11, vl);
        vfloat32m2_t vec_J_se2_12 = __riscv_vle32_v_f32m2(p_J_se2_12, vl);
        vfloat32m2_t vec_J_se2_13 = __riscv_vle32_v_f32m2(p_J_se2_13, vl);

        vec_s = __riscv_vfredusum_vs_f32m2_f32m1(vec_val, vec_s, vl);

        vec_s1 = __riscv_vfredusum_vs_f32m2_f32m1(vec_J_se2_11, vec_s1, vl);
        vec_s2 = __riscv_vfredusum_vs_f32m2_f32m1(vec_J_se2_12, vec_s2, vl);
        vec_s3 = __riscv_vfredusum_vs_f32m2_f32m1(vec_J_se2_13, vec_s3, vl);

        p_val += vl;

        p_J_se2_11 += vl;
        p_J_se2_12 += vl;
        p_J_se2_13 += vl;

        // float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        // std::cout << "sum=" << sum << std::endl;
    }

    sum = __riscv_vfmv_f_s_f32m1_f32(vec_s);
    grad_sum_se2[0] = __riscv_vfmv_f_s_f32m1_f32(vec_s1);
    grad_sum_se2[1] = __riscv_vfmv_f_s_f32m1_f32(vec_s2);
    grad_sum_se2(2) = __riscv_vfmv_f_s_f32m1_f32(vec_s3);

    #endif

    // std::cout << std::setprecision(3) << std::fixed << "2 sum=" << sum << "  grad_sum_se2=" << grad_sum_se2.transpose() << std::endl;

    mean = sum / num_valid_points; // 总灰度除以有效点数，得到平均亮度值，可以消除曝光时间引起的图像光度尺度变化，但无法消除光度的偏移变化

    const Scalar mean_inv = num_valid_points / sum; // 平均亮度的逆

    const Scalar constant1 = grad_sum_se2(0) / sum;
    const Scalar constant2 = grad_sum_se2(1) / sum;
    const Scalar constant3 = grad_sum_se2(2) / sum;

    float* pval = data.data();

    // Scalar *p_coefficient = coefficient; // comment on 2024-12-11

    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;

    n = PATTERN_SIZE;
    
    // std::cout << std::setprecision(7) << std::fixed << "22 mean_inv=" << mean_inv << "  num_valid_points=" << num_valid_points << "  vlmax=" << vlmax << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "23 data=" << data.transpose() << std::endl;

    Scalar *p_u = u;

    for (size_t vl; n > 0; n -= vl)
    {
      vl = __riscv_vsetvl_e32m2(n);
      
      vfloat32m2_t vec_val = __riscv_vle32_v_f32m2(pval, vl);
      /*
      // comment on 2024-12-11
      vfloat32m2_t vec_coefficient = __riscv_vle32_v_f32m2(p_coefficient, vl);
      vec_val = __riscv_vfmul_vv_f32m2(vec_val, vec_coefficient, vl);
      */

      vfloat32m2_t vec_result_a11 = __riscv_vfmul_vf_f32m2(vec_val, constant1, vl);
      vfloat32m2_t vec_result_a12 = __riscv_vfmul_vf_f32m2(vec_val, constant2, vl);
      vfloat32m2_t vec_result_a13 = __riscv_vfmul_vf_f32m2(vec_val, constant3, vl);

      vfloat32m2_t vec_J_se2_11 = __riscv_vle32_v_f32m2(p_J_se2_11, vl);
      vfloat32m2_t vec_J_se2_12 = __riscv_vle32_v_f32m2(p_J_se2_12, vl);
      vfloat32m2_t vec_J_se2_13 = __riscv_vle32_v_f32m2(p_J_se2_13, vl);

      vec_result_a11 = __riscv_vfsub_vv_f32m2(vec_J_se2_11, vec_result_a11, vl);
      vec_result_a12 = __riscv_vfsub_vv_f32m2(vec_J_se2_12, vec_result_a12, vl);
      vec_result_a13 = __riscv_vfsub_vv_f32m2(vec_J_se2_13, vec_result_a13, vl);

      vec_result_a11 = __riscv_vfmul_vf_f32m2(vec_result_a11, mean_inv, vl);
      vec_result_a12 = __riscv_vfmul_vf_f32m2(vec_result_a12, mean_inv, vl);
      vec_result_a13 = __riscv_vfmul_vf_f32m2(vec_result_a13, mean_inv, vl);
      
      vfloat32m2_t vec_result = __riscv_vfmul_vf_f32m2(vec_val, mean_inv, vl);

      // set data[i] = -1 when point is outlier.
      vfloat32m2_t vec_point2d_x = __riscv_vle32_v_f32m2(p_u, vl);
      // > 0
      vbool16_t mask = __riscv_vmfgt_vf_f32m2_b16(vec_point2d_x, 0, vl); // 2d点有效时，data[i] > 0;
      vec_result = __riscv_vmerge_vvm_f32m2(vec_minus_one, vec_result, mask, vl); // 否则data[i] = -1.

      vec_result_a11 = __riscv_vmerge_vvm_f32m2(vec_zero, vec_result_a11, mask, vl);
      vec_result_a12 = __riscv_vmerge_vvm_f32m2(vec_zero, vec_result_a12, mask, vl);
      vec_result_a13 = __riscv_vmerge_vvm_f32m2(vec_zero, vec_result_a13, mask, vl);

      __riscv_vse32_v_f32m2(p_J_se2_11, vec_result_a11, vl);
      __riscv_vse32_v_f32m2(p_J_se2_12, vec_result_a12, vl);
      __riscv_vse32_v_f32m2(p_J_se2_13, vec_result_a13, vl);
     
      __riscv_vse32_v_f32m2(pval, vec_result, vl);

      p_u += vl;
      pval += vl;
      // p_coefficient += vl;

      p_J_se2_11 += vl;
      p_J_se2_12 += vl;
      p_J_se2_13 += vl;
    }

    #if 0
    auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << std::fixed << "setDataJacSe2: " << diff.count() << " s\n";
    // std::cout << std::fixed << "setDataJacSe2: " << std::chrono::duration_cast<std::chrono::milliseconds>((end - start).count()) << " ms\n";
    auto diff = end - start;
    std::cout << std::setprecision(6) << std::fixed << "RVV setDataJacSe2:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    #endif

    #if 1
    clock_t clock_end = clock();
    auto clock_diff = clock_end - clock_start;
    clock_sum += clock_diff;
    #endif


    #if 0
    std::cout << std::setprecision(3) << std::fixed << "e32m2 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    #endif

  }

#else
  //
  template <typename ImgT>
  static void setDataJacSe2(const ImgT &img, const Vector2 &pos, Scalar &mean,
                            VectorP &data, MatrixP3 &J_se2) {
    //- 雅可比是残差对状态量的偏导，这里的残差构建和几何雅可比，似乎都用到了扰动模型
    //- 正向光流法，求雅可比的时候，用的是第二个图像I2处的梯度
    // 本算法用的是反向光流法：即用第一个图像I1的梯度来代替
    // r = I2 - I1
    // J = ∂r/∂se2 = - ∂I/∂xi * ∂xi/∂se2

    int num_valid_points = 0;
    Scalar sum = 0;
    Vector3 grad_sum_se2(0, 0, 0);

    Eigen::Matrix<Scalar, 2, 3> Jw_se2; // 2 * 3的矩阵, 这个属于几何雅可比
    Jw_se2.template topLeftCorner<2, 2>().setIdentity(); // 左上角2 * 2设为单位阵，即前面两列由单位阵占据

#ifdef _USE_RISCV_V
    #if 0
    auto start = std::chrono::steady_clock::now();
    #endif

    #if 1
    clock_t clock_start = clock();
    #endif

    J_se2.setZero();
    // uint8_t valid_index[PATTERN_SIZE] = { 0 };
    Scalar coefficient[PATTERN_SIZE] = { 0 };

    Scalar val[PATTERN_SIZE] = { 0 };

    Scalar grad_x[PATTERN_SIZE] = { 0 };
    Scalar grad_y[PATTERN_SIZE] = { 0 };

    Scalar Jw_se2_11[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_21[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_12[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_22[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_13[PATTERN_SIZE] = { 0 };
    Scalar Jw_se2_23[PATTERN_SIZE] = { 0 };

    //! 计算的结果似乎没有必要专门保存，直接赋值到J_se2效率更高
    // Scalar result_a11[PATTERN_SIZE] = { 0 };
    // Scalar result_a12[PATTERN_SIZE] = { 0 };
    // Scalar result_a13[PATTERN_SIZE] = { 0 };
    
    Scalar *p_grad_x = grad_x;
    Scalar *p_grad_y = grad_y;

    Scalar*p_Jw_se2_11 = Jw_se2_11;
    Scalar*p_Jw_se2_21 = Jw_se2_21;
    Scalar*p_Jw_se2_12 = Jw_se2_12;
    Scalar*p_Jw_se2_22 = Jw_se2_22;
    Scalar*p_Jw_se2_13 = Jw_se2_13;
    Scalar*p_Jw_se2_23 = Jw_se2_23;

    // Scalar*p_result_a11 = result_a11;
    // Scalar*p_result_a12 = result_a12;
    // Scalar*p_result_a13 = result_a13;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      //
      Vector2 p = pos + pattern2.col(i);
      Jw_se2(0, 2) = -pattern2(1, i);
      Jw_se2(1, 2) = pattern2(0, i);
      if (img.InBounds(p, 2)) {
        Vector3 valGrad = img.template interpGrad<Scalar>(p);
        data[i] = valGrad[0];
        // sum += valGrad[0];
        // J_se2.row(i) = valGrad.template tail<2>().transpose() * Jw_se2;
        // grad_sum_se2 += J_se2.row(i);

        // valid_index[num_valid_points] = i;

        val[i] = valGrad[0];

        grad_x[i] = valGrad[1];
        grad_y[i] = valGrad[2];

        Jw_se2_11[i] = Jw_se2(0, 0);
        Jw_se2_21[i] = Jw_se2(1, 0);
        Jw_se2_12[i] = Jw_se2(0, 1);
        Jw_se2_22[i] = Jw_se2(1, 1);
        Jw_se2_13[i] = Jw_se2(0, 2);
        Jw_se2_23[i] = Jw_se2(1, 2);

        num_valid_points++;

        coefficient[i] = 1;
      } else {
        data[i] = -1;

        val[i] = 0;

        grad_x[i] = 0;
        grad_y[i] = 0;

        Jw_se2_11[i] = 0;
        Jw_se2_21[i] = 0;
        Jw_se2_12[i] = 0;
        Jw_se2_22[i] = 0;
        Jw_se2_13[i] = 0;
        Jw_se2_23[i] = 0;

        coefficient[i] = 0;
      }
    }

    // fetch 3 columns
    Scalar *p_J_se2_11 = J_se2.data();
    Scalar *p_J_se2_12 = J_se2.data() + 52;
    Scalar *p_J_se2_13 = J_se2.data() + 104;

    int n = PATTERN_SIZE;
    for (size_t vl; n > 0; n -= vl) // , a += vl, b += vl, c += vl
    {
      vl = __riscv_vsetvl_e32m8(n);
      vfloat32m8_t vec_grad_x = __riscv_vle32_v_f32m8(p_grad_x, vl);
      vfloat32m8_t vec_grad_y = __riscv_vle32_v_f32m8(p_grad_y, vl);
#if 0
      vfloat32m8_t vec_Jw_se2_11 = __riscv_vle32_v_f32m8(p_Jw_se2_11, vl);
      vfloat32m8_t vec_Jw_se2_21 = __riscv_vle32_v_f32m8(p_Jw_se2_21, vl);
      vfloat32m8_t vec_Jw_se2_12 = __riscv_vle32_v_f32m8(p_Jw_se2_12, vl);
      vfloat32m8_t vec_Jw_se2_22 = __riscv_vle32_v_f32m8(p_Jw_se2_22, vl);
      vfloat32m8_t vec_Jw_se2_13 = __riscv_vle32_v_f32m8(p_Jw_se2_13, vl);
      vfloat32m8_t vec_Jw_se2_23 = __riscv_vle32_v_f32m8(p_Jw_se2_23, vl);

      vfloat32m8_t vec_dx_J_geo_11 = __riscv_vfmul_vv_f32m8(vec_grad_x, vec_Jw_se2_11, vl);
      vfloat32m8_t vec_dx_J_geo_12 = __riscv_vfmul_vv_f32m8(vec_grad_x, vec_Jw_se2_12, vl);
      vfloat32m8_t vec_dx_J_geo_13 = __riscv_vfmul_vv_f32m8(vec_grad_x, vec_Jw_se2_13, vl);

      vfloat32m8_t vec_dy_J_geo_21 = __riscv_vfmul_vv_f32m8(vec_grad_y, vec_Jw_se2_21, vl);
      vfloat32m8_t vec_dy_J_geo_22 = __riscv_vfmul_vv_f32m8(vec_grad_y, vec_Jw_se2_22, vl);
      vfloat32m8_t vec_dy_J_geo_23 = __riscv_vfmul_vv_f32m8(vec_grad_y, vec_Jw_se2_23, vl);

      vfloat32m8_t vec_result_a11 = __riscv_vfadd_vv_f32m8(vec_dx_J_geo_11, vec_dy_J_geo_21, vl);
      vfloat32m8_t vec_result_a12 = __riscv_vfadd_vv_f32m8(vec_dx_J_geo_12, vec_dy_J_geo_22, vl);
      vfloat32m8_t vec_result_a13 = __riscv_vfadd_vv_f32m8(vec_dx_J_geo_13, vec_dy_J_geo_23, vl);

#else
      /*
       * vd * vs1 + vs2
       vfloat32m8_t __riscv_vfmadd_vv_f32m8(vfloat32m8_t vd, vfloat32m8_t vs1,
                                     vfloat32m8_t vs2, size_t vl);
       */
      vfloat32m8_t vec_result_a11 = __riscv_vfmadd_vv_f32m8(vec_grad_x, __riscv_vle32_v_f32m8(p_Jw_se2_11, vl), __riscv_vfmul_vv_f32m8(vec_grad_y, __riscv_vle32_v_f32m8(p_Jw_se2_21, vl), vl), vl);
      vfloat32m8_t vec_result_a12 = __riscv_vfmadd_vv_f32m8(vec_grad_x, __riscv_vle32_v_f32m8(p_Jw_se2_12, vl), __riscv_vfmul_vv_f32m8(vec_grad_y, __riscv_vle32_v_f32m8(p_Jw_se2_22, vl), vl), vl);
      vfloat32m8_t vec_result_a13 = __riscv_vfmadd_vv_f32m8(vec_grad_x, __riscv_vle32_v_f32m8(p_Jw_se2_13, vl), __riscv_vfmul_vv_f32m8(vec_grad_y, __riscv_vle32_v_f32m8(p_Jw_se2_23, vl), vl), vl);
#endif
      __riscv_vse32_v_f32m8(p_J_se2_11, vec_result_a11, vl);
      __riscv_vse32_v_f32m8(p_J_se2_12, vec_result_a12, vl);
      __riscv_vse32_v_f32m8(p_J_se2_13, vec_result_a13, vl);

      p_grad_x += vl;
      p_grad_y += vl;

      p_Jw_se2_11 += vl;
      p_Jw_se2_21 += vl;

      p_Jw_se2_12 += vl;
      p_Jw_se2_22 += vl;

      p_Jw_se2_13 += vl;
      p_Jw_se2_23 += vl;

      // p_result_a11 += vl;
      // p_result_a12 += vl;
      // p_result_a13 += vl;

      p_J_se2_11 += vl;
      p_J_se2_12 += vl;
      p_J_se2_13 += vl;

    }

    // std::cout << "sum=" << sum << "  J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "1 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;

    // TODO: grad_sum_se2 += J_se2.row(i);  sum += valGrad[0];
#if 0    
    Scalar tmp_J_se2_11[PATTERN_SIZE + 1] = { 0 };
    Scalar tmp_J_se2_12[PATTERN_SIZE + 1] = { 0 };
    Scalar tmp_J_se2_13[PATTERN_SIZE + 1] = { 0 };
#if 0    
    memcpy(tmp_J_se2_11, J_se2.data(), PATTERN_SIZE * sizeof(Scalar));
    memcpy(tmp_J_se2_12, J_se2.data() + 52, PATTERN_SIZE * sizeof(Scalar));
    memcpy(tmp_J_se2_13, J_se2.data() + 104, PATTERN_SIZE * sizeof(Scalar));
#else
    // memcpy with riscv vector
    memcpy_vec(tmp_J_se2_11, J_se2.data(), PATTERN_SIZE * sizeof(Scalar));
    memcpy_vec(tmp_J_se2_12, J_se2.data() + 52, PATTERN_SIZE * sizeof(Scalar));
    memcpy_vec(tmp_J_se2_13, J_se2.data() + 104, PATTERN_SIZE * sizeof(Scalar));
#endif
    Scalar tmp_val[PATTERN_SIZE + 1] = { 0 };
#if 0    
    memcpy(tmp_val, val, PATTERN_SIZE * sizeof(Scalar));
#else    
    memcpy_vec(tmp_val, val, PATTERN_SIZE * sizeof(Scalar));
#endif    

    size_t vlmax = __riscv_vsetvlmax_e32m8();
    n = PATTERN_SIZE;
    Scalar *p_val = tmp_val;
    
    while( n > 0 )
    {
      if(n > 2)
      {
        if(n % 2 != 0)
        {
          tmp_J_se2_11[n] = 0;
          tmp_J_se2_12[n] = 0;
          tmp_J_se2_13[n] = 0;

          tmp_val[n] = 0;

          n += 1;
        }

        p_val = tmp_val;
        p_J_se2_11 = tmp_J_se2_11;
        p_J_se2_12 = tmp_J_se2_12;
        p_J_se2_13 = tmp_J_se2_13;

        int half = n / 2;
        if(half > 0 && half < vlmax)
        {
          // calculate intensity value
          vfloat32m8_t vec_val1 = __riscv_vle32_v_f32m8(p_val, half);
          p_val += half;
          vfloat32m8_t vec_val2 = __riscv_vle32_v_f32m8(p_val, half);
          // p_val += half;

          vfloat32m8_t vec_sum = __riscv_vfadd_vv_f32m8(vec_val1, vec_val2, half);
          __riscv_vse32_v_f32m8(tmp_val, vec_sum, half);

          // calculate J_se2_11
          vec_val1 = __riscv_vle32_v_f32m8(p_J_se2_11, half);
          p_J_se2_11 += half;
          vec_val2 = __riscv_vle32_v_f32m8(p_J_se2_11, half);
          // p_J_se2_11 += half;

          vec_sum = __riscv_vfadd_vv_f32m8(vec_val1, vec_val2, half);
          __riscv_vse32_v_f32m8(tmp_J_se2_11, vec_sum, half);

          // calculate J_se2_12
          vec_val1 = __riscv_vle32_v_f32m8(p_J_se2_12, half);
          p_J_se2_12 += half;
          vec_val2 = __riscv_vle32_v_f32m8(p_J_se2_12, half);
          // p_J_se2_12 += half;

          vec_sum = __riscv_vfadd_vv_f32m8(vec_val1, vec_val2, half);
          __riscv_vse32_v_f32m8(tmp_J_se2_12, vec_sum, half);

          // calculate J_se2_12
          vec_val1 = __riscv_vle32_v_f32m8(p_J_se2_13, half);
          p_J_se2_13 += half;
          vec_val2 = __riscv_vle32_v_f32m8(p_J_se2_13, half);
          // p_J_se2_13 += half;

          vec_sum = __riscv_vfadd_vv_f32m8(vec_val1, vec_val2, half);
          __riscv_vse32_v_f32m8(tmp_J_se2_13, vec_sum, half);

          n = half; // n -= half;
        }
      }
      else if(n == 2)
      {
        sum = tmp_val[0] + tmp_val[1];

        grad_sum_se2[0] = tmp_J_se2_11[0] + tmp_J_se2_11[1];
        grad_sum_se2(1) = tmp_J_se2_12[0] + tmp_J_se2_12[1];
        grad_sum_se2[2] = tmp_J_se2_13[0] + tmp_J_se2_13[1];

        break ;
      }
      else if(n == 1)
      {
        sum = tmp_val[0];

        grad_sum_se2[0] = tmp_J_se2_11[0];
        grad_sum_se2(1) = tmp_J_se2_12[0];
        grad_sum_se2[2] = tmp_J_se2_13[0];

        break ;
      }
    }
#else
    // another method is reduction
    // 初始化一些变量
    size_t vlmax = __riscv_vsetvlmax_e32m8();  // 获取最大向量长度
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);  // 初始化零向量
    vfloat32m8_t vec_one = __riscv_vfmv_v_f_f32m8(1, vlmax);  // 初始化一向量
    vfloat32m8_t vec_s = __riscv_vfmv_v_f_f32m8(0, vlmax);  // 初始化累加向量

    vfloat32m8_t vec_s1 = __riscv_vfmv_v_f_f32m8(0, vlmax);
    vfloat32m8_t vec_s2 = __riscv_vfmv_v_f_f32m8(0, vlmax);
    vfloat32m8_t vec_s3 = __riscv_vfmv_v_f_f32m8(0, vlmax);

    Scalar *p_val = val;
    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;
    // 处理整个向量数据
    n = PATTERN_SIZE;
    for (size_t vl; n > 0; n -= vl) {
        vl = __riscv_vsetvl_e32m8(n);  // 根据剩余元素数设置向量长度
        // std::cout << "vl=" << vl << std::endl;

        // 从数据中加载向量
        vfloat32m8_t vec_val = __riscv_vle32_v_f32m8(p_val, vl);

        vfloat32m8_t vec_J_se2_11 = __riscv_vle32_v_f32m8(p_J_se2_11, vl);
        vfloat32m8_t vec_J_se2_12 = __riscv_vle32_v_f32m8(p_J_se2_12, vl);
        vfloat32m8_t vec_J_se2_13 = __riscv_vle32_v_f32m8(p_J_se2_13, vl);

        vbool4_t mask = __riscv_vmfne_vf_f32m8_b4(vec_val, 0, vl);

        vbool4_t mask_J_se2_11 = __riscv_vmfne_vf_f32m8_b4(vec_J_se2_11, 0, vl);
        vbool4_t mask_J_se2_12 = __riscv_vmfne_vf_f32m8_b4(vec_J_se2_12, 0, vl);
        vbool4_t mask_J_se2_13 = __riscv_vmfne_vf_f32m8_b4(vec_J_se2_13, 0, vl);

        // 执行归约求和
        vec_s = __riscv_vfmacc_vv_f32m8_tumu(mask, vec_s, vec_val, vec_one, vl);

        vec_s1 = __riscv_vfmacc_vv_f32m8_tumu(mask_J_se2_11, vec_s1, vec_J_se2_11, vec_one, vl);
        vec_s2 = __riscv_vfmacc_vv_f32m8_tumu(mask_J_se2_12, vec_s2, vec_J_se2_12, vec_one, vl);
        vec_s3 = __riscv_vfmacc_vv_f32m8_tumu(mask_J_se2_13, vec_s3, vec_J_se2_13, vec_one, vl);

        p_val += vl;

        p_J_se2_11 += vl;
        p_J_se2_12 += vl;
        p_J_se2_13 += vl;
    }

    // 最终归约求和
    vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m8_f32m1(vec_s, vec_zero, vlmax);

    vfloat32m1_t vec_sum1 = __riscv_vfredusum_vs_f32m8_f32m1(vec_s1, vec_zero, vlmax);
    vfloat32m1_t vec_sum2 = __riscv_vfredusum_vs_f32m8_f32m1(vec_s2, vec_zero, vlmax);
    vfloat32m1_t vec_sum3 = __riscv_vfredusum_vs_f32m8_f32m1(vec_s3, vec_zero, vlmax);
    
    // 提取最终的结果
    sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    
    grad_sum_se2[0] = __riscv_vfmv_f_s_f32m1_f32(vec_sum1);
    grad_sum_se2[1] = __riscv_vfmv_f_s_f32m1_f32(vec_sum2);
    grad_sum_se2(2) = __riscv_vfmv_f_s_f32m1_f32(vec_sum3);
#endif
    // std::cout << std::setprecision(3) << std::fixed << "2 sum=" << sum << "  grad_sum_se2=" << grad_sum_se2.transpose() << std::endl;

    mean = sum / num_valid_points; // 总灰度除以有效点数，得到平均亮度值，可以消除曝光时间引起的图像光度尺度变化，但无法消除光度的偏移变化

    const Scalar mean_inv = num_valid_points / sum; // 平均亮度的逆

    const Scalar constant1 = grad_sum_se2(0) / sum;
    const Scalar constant2 = grad_sum_se2(1) / sum;
    const Scalar constant3 = grad_sum_se2(2) / sum;

    float* pval = data.data();

    Scalar *p_coefficient = coefficient;

    p_J_se2_11 = J_se2.data();
    p_J_se2_12 = J_se2.data() + 52;
    p_J_se2_13 = J_se2.data() + 104;

#if 0
    // size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vec_constant1 = __riscv_vfmv_v_f_f32m8(constant1, vlmax);
    vfloat32m8_t vec_constant2 = __riscv_vfmv_v_f_f32m8(constant2, vlmax);
    vfloat32m8_t vec_constant3 = __riscv_vfmv_v_f_f32m8(constant3, vlmax);

    vfloat32m8_t vec_mean_inv = __riscv_vfmv_v_f_f32m8(mean_inv, vlmax);
#endif

    n = PATTERN_SIZE;
    // p_result_a11 = result_a11;
    // p_result_a12 = result_a12;
    // p_result_a13 = result_a13;

    // std::cout << std::setprecision(7) << std::fixed << "22 mean_inv=" << mean_inv << "  num_valid_points=" << num_valid_points << "  vlmax=" << vlmax << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "23 data=" << data.transpose() << std::endl;

    for (size_t vl; n > 0; n -= vl)
    {
      vl = __riscv_vsetvl_e32m8(n);
      
      vfloat32m8_t vec_val = __riscv_vle32_v_f32m8(pval, vl);
      vfloat32m8_t vec_coefficient = __riscv_vle32_v_f32m8(p_coefficient, vl);
      vec_val = __riscv_vfmul_vv_f32m8(vec_val, vec_coefficient, vl);
#if 0
      vfloat32m8_t vec_result_a11 = __riscv_vfmul_vv_f32m8(vec_val, vec_constant1, vl);
      vfloat32m8_t vec_result_a12 = __riscv_vfmul_vv_f32m8(vec_val, vec_constant2, vl);
      vfloat32m8_t vec_result_a13 = __riscv_vfmul_vv_f32m8(vec_val, vec_constant3, vl);
#else
      vfloat32m8_t vec_result_a11 = __riscv_vfmul_vf_f32m8(vec_val, constant1, vl);
      vfloat32m8_t vec_result_a12 = __riscv_vfmul_vf_f32m8(vec_val, constant2, vl);
      vfloat32m8_t vec_result_a13 = __riscv_vfmul_vf_f32m8(vec_val, constant3, vl);
#endif
      vfloat32m8_t vec_J_se2_11 = __riscv_vle32_v_f32m8(p_J_se2_11, vl);
      vfloat32m8_t vec_J_se2_12 = __riscv_vle32_v_f32m8(p_J_se2_12, vl);
      vfloat32m8_t vec_J_se2_13 = __riscv_vle32_v_f32m8(p_J_se2_13, vl);

      vec_result_a11 = __riscv_vfsub_vv_f32m8(vec_J_se2_11, vec_result_a11, vl);
      vec_result_a12 = __riscv_vfsub_vv_f32m8(vec_J_se2_12, vec_result_a12, vl);
      vec_result_a13 = __riscv_vfsub_vv_f32m8(vec_J_se2_13, vec_result_a13, vl);

#if 0
      vec_result_a11 = __riscv_vfmul_vv_f32m8(vec_result_a11, vec_mean_inv, vl);
      vec_result_a12 = __riscv_vfmul_vv_f32m8(vec_result_a12, vec_mean_inv, vl);
      vec_result_a13 = __riscv_vfmul_vv_f32m8(vec_result_a13, vec_mean_inv, vl);
#else
      vec_result_a11 = __riscv_vfmul_vf_f32m8(vec_result_a11, mean_inv, vl);
      vec_result_a12 = __riscv_vfmul_vf_f32m8(vec_result_a12, mean_inv, vl);
      vec_result_a13 = __riscv_vfmul_vf_f32m8(vec_result_a13, mean_inv, vl);
#endif      

      __riscv_vse32_v_f32m8(p_J_se2_11, vec_result_a11, vl);
      __riscv_vse32_v_f32m8(p_J_se2_12, vec_result_a12, vl);
      __riscv_vse32_v_f32m8(p_J_se2_13, vec_result_a13, vl);

#if 0
      vfloat32m8_t vec_result = __riscv_vfmul_vv_f32m8(vec_val, vec_mean_inv, vl);
#else     
      vfloat32m8_t vec_result = __riscv_vfmul_vf_f32m8(vec_val, mean_inv, vl);
#endif
      __riscv_vse32_v_f32m8(pval, vec_result, vl);

      pval += vl;
      
      p_coefficient += vl;

      p_J_se2_11 += vl;
      p_J_se2_12 += vl;
      p_J_se2_13 += vl;
    }

    #if 0
    auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << std::fixed << "setDataJacSe2: " << diff.count() << " s\n";
    // std::cout << std::fixed << "setDataJacSe2: " << std::chrono::duration_cast<std::chrono::milliseconds>((end - start).count()) << " ms\n";
    auto diff = end - start;
    std::cout << std::setprecision(6) << std::fixed << "RVV setDataJacSe2:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    #endif

    #if 1
    clock_t clock_end = clock();
    auto clock_diff = clock_end - clock_start;
    clock_sum += clock_diff;
    #endif

#else
    #if 0
    auto start = std::chrono::steady_clock::now();
    #endif

    #if 1
    clock_t clock_start = clock();
    #endif

    // 对于每个pattern内部的点进行计算
    for (int i = 0; i < PATTERN_SIZE; i++) { // PATTERN_SIZE=52的时候，表示patch里面有52个点，pattern2里面是坐标的偏移量
      Vector2 p = pos + pattern2.col(i); // 位于图像的位置，点的位置加上pattern里面的偏移量，得到在patch里面的新的位姿

      // Fill jacobians with respect to SE2 warp 对Jw_se2的第2列（列下标从0开始的,也即最后一列）赋值 //- 下面两行完全是为了构建几何雅可比。
      Jw_se2(0, 2) = -pattern2(1, i); // 取pattern2的第1行，第i列。 对于Pattern51来说，pattern2表示的是2*52的矩阵
      Jw_se2(1, 2) = pattern2(0, i); // 取pattern2的第0行，第i列

      if (img.InBounds(p, 2)) { // 判断加了偏移量的点p是否在图像内，border=2
        // valGrad[0]表示图像强度，valGrad[1]表示x方向梯度，valGrad[0]表示y方向梯度
        Vector3 valGrad = img.template interpGrad<Scalar>(p); // interp是interpolation的缩写，表示利用双线性插值计算图像灰度和图像梯度 ( x方向梯度, y方向梯度 )
        data[i] = valGrad[0]; // 赋值图像灰度值
        sum += valGrad[0]; // 统计总图像强度
        // J_se2在Pattern51的情况下是52*3，每一行是1*3. //?具体含义有待补充：其实这一部分是，梯度*几何雅可比
        J_se2.row(i) = valGrad.template tail<2>().transpose() * Jw_se2; // 链式法则: 取valGrad的后2位元素，即图像梯度，列向量转置后，变成1*2，再乘以2*3 
        grad_sum_se2 += J_se2.row(i); // 所有行的梯度相加
        num_valid_points++;
      } else {
        data[i] = -1;
      }
    }

    // std::cout << std::setprecision(3) << std::fixed << "1 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "2 sum=" << sum << "  grad_sum_se2=" << grad_sum_se2.transpose() << std::endl;

    mean = sum / num_valid_points; // 总灰度除以有效点数，得到平均亮度值，可以消除曝光时间引起的图像光度尺度变化，但无法消除光度的偏移变化

    const Scalar mean_inv = num_valid_points / sum; // 平均亮度的逆

    // std::cout << std::setprecision(7) << std::fixed << "22 mean_inv=" << mean_inv << "  num_valid_points=" << num_valid_points << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "23 data=" << data.transpose() << std::endl;
    // std::cout << std::setprecision(3) << std::fixed << "24 J_se2.row(0)=" << J_se2.row(0) << std::endl;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (data[i] >= 0) { // 如果patch里面序号i对应的点的图像强度大于等于0，
        J_se2.row(i) -= grad_sum_se2.transpose() * data[i] / sum; //? //TODO: -= 和 /  谁的优先级更高
        data[i] *= mean_inv;
      } else { // 否则无效的图像强度，该行直接置为0
        J_se2.row(i).setZero();
      }
    }
    J_se2 *= mean_inv; // 至此，完成了梯度雅可比和几何雅可比的链式法则求偏导的整个过程。

    #if 0
    auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << std::fixed << "setDataJacSe2: " << diff.count() << " s\n";
    // std::cout << std::fixed << "setDataJacSe2: " << std::chrono::duration_cast<std::chrono::milliseconds>((end - start).count()) << " ms\n";
    auto diff = end - start;
    std::cout << std::setprecision(6) << std::fixed << "setDataJacSe2:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    #endif

    #if 1
    clock_t clock_end = clock();
    auto clock_diff = clock_end - clock_start;
    clock_sum += clock_diff;
    #endif

#endif

    #if 0
    std::cout << std::setprecision(3) << std::fixed << "3 J_se2.trace=" << J_se2.trace() <<"  J_se2.diagonal=" << J_se2.diagonal().transpose() << std::endl;
    #endif

  }
#endif

  void setFromImage(const Image<const uint16_t> &img, const Vector2 &pos) {
    this->pos = pos;

    MatrixP3 J_se2;

    setDataJacSe2(img, pos, mean, data, J_se2); // 求雅可比J_se2.

    Matrix3 H_se2 = J_se2.transpose() * J_se2; // H = J^T * J
    Matrix3 H_se2_inv;
    H_se2_inv.setIdentity();
    H_se2.ldlt().solveInPlace(H_se2_inv); // 求H^(-1)

    H_se2_inv_J_se2_T = H_se2_inv * J_se2.transpose(); // 求H^(-1) * J^T

    // NOTE: while it's very unlikely we get a source patch with all black
    // pixels, since points are usually selected at corners, it doesn't cost
    // much to be safe here.
    // 注意：这时得到一个所有黑色像素的patch是非常不可能的，因为通常是在角上选择点，在这里安全的开销并不多
    // 全黑的patch 不能被归一化；会导致0平均值，同时，H_se2_inv_J_se2_T会包含非数（NaN）并且数据会包含无穷（inf）

    // all-black patch cannot be normalized; will result in mean of "zero" and
    // H_se2_inv_J_se2_T will contain "NaN" and data will contain "inf"
    valid = mean > std::numeric_limits<Scalar>::epsilon() &&
            H_se2_inv_J_se2_T.array().isFinite().all() &&
            data.array().isFinite().all();
  }

  inline bool residual(const Image<const uint16_t> &img,
                       const Matrix2P &transformed_pattern,
                       VectorP &residual) const {
    Scalar sum = 0;
    int num_valid_points = 0;

#if 0
    if(1)
    {
      std::cout << "transformed_pattern=\n" << transformed_pattern << std::endl;

      const Scalar *p_point2d = transformed_pattern.data();
      std::cout << "index0123=" << p_point2d[0] << " " << p_point2d[1] \
        << " " << p_point2d[2] << " " << p_point2d[3] << std::endl;
    }
#endif    

#if !defined(_USE_RISCV_V)

    #if 1
    clock_t start1 = clock();
    #endif

    // std::cout << "residual:\n";

    // 对pattern的每一个数据进行计算 这里还没有做差，只是求取了每个pattern在像素处的值
    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (img.InBounds(transformed_pattern.col(i), 2)) {
        residual[i] = img.interp<Scalar>(transformed_pattern.col(i));
        sum += residual[i]; // 求总和
        num_valid_points++;
      } else {
        residual[i] = -1;
      }
    }
    #if 1
    clock_t end1 = clock();
    #endif

    // std::cout << "sum=" << sum << std::endl;
#else
    // use risc-v vector here.
    #if 0
    clock_t rvv_start1 = clock();
    #endif
    alignas(32) Scalar u[PATTERN_SIZE];
    alignas(32) Scalar v[PATTERN_SIZE];
    // alignas(32) Scalar result[PATTERN_SIZE];
    Scalar *result = residual.data();
    const Scalar *p_point2d = transformed_pattern.data();
    // for (int i = 0; i < PATTERN_SIZE; i++) {
    //   u[i] = transformed_pattern.col(i)[0];
    //   v[i] = transformed_pattern.col(i)[1];
    // }
    for (int i = 0, j = 0; i < PATTERN_SIZE; i++) {
      j = 2 * i;
      u[i] = p_point2d[j];
      v[i] = p_point2d[j + 1];
      // std::cout << "1 u[" << i << "]=" << u[i] << " v[" << i << "]=" << v[i] << std::endl;
    }

    // method 1: take all valid points into array alone. 
    // and only apply these good points with bilinear interpolation of rvv.


    size_t n = PATTERN_SIZE;
    size_t w = img.w;
    size_t h = img.h;
    Scalar border = 2.0;
    // Scalar offset(0);
    Scalar offset(1);

    size_t vlmax = __riscv_vsetvlmax_e32m2(); // return 16
    vfloat32m2_t vec_zero = __riscv_vfmv_v_f_f32m2(0, vlmax);
    
    // mehtod 2: use mask & merge
    for(size_t vl, i = 0; i < n; i += vl)
    {
        // 根据剩余元素数设置向量长度
        vl = __riscv_vsetvl_e32m2(n - i);

        // 从数据中加载向量
        vfloat32m2_t vec_point2d_x = __riscv_vle32_v_f32m2(u + i, vl);
        vfloat32m2_t vec_point2d_y = __riscv_vle32_v_f32m2(v + i, vl);

        // >= border
        vbool16_t mask = __riscv_vmfge_vf_f32m2_b16(vec_point2d_x, border, vl);

        // < (w - border - offset)
        mask = __riscv_vmflt_vf_f32m2_b16_m(mask, vec_point2d_x, (w - border - offset), vl);

        mask = __riscv_vmfge_vf_f32m2_b16_m(mask, vec_point2d_y, border, vl);
        mask = __riscv_vmflt_vf_f32m2_b16_m(mask, vec_point2d_y, (h - border - offset), vl);

        vec_point2d_x= __riscv_vmerge_vvm_f32m2(vec_zero, vec_point2d_x, mask, vl);
        vec_point2d_y= __riscv_vmerge_vvm_f32m2(vec_zero, vec_point2d_y, mask, vl);
        /// storage
        __riscv_vse32_v_f32m2(u + i, vec_point2d_x, vl);
        __riscv_vse32_v_f32m2(v + i, vec_point2d_y, vl);

        num_valid_points += __riscv_vcpop_m_b16(mask, vl);
        // std::cout << "vl=" << vl << " num_valid_points=" << num_valid_points << std::endl;
    }
    // std::cout << "w=" << w << " h=" << h << " num_valid_points=" << num_valid_points << std::endl;

    // for(int i = 0; i < PATTERN_SIZE; i++)
    // {
    //   std::cout << "u[" << i << "]=" << u[i] << " v[" << i << "]=" << v[i] << std::endl;
    // }

    sum = img.compute_patch_intensity(u, v, result, PATTERN_SIZE);
    // Scalar sum2 = 0;
    // for(int i = 0; i < PATTERN_SIZE; i++)
    // {
    //   sum2 += residual[i];
    // }
    // std::cout << "sum2=" << sum2 << std::endl;
    #if 0
    clock_t rvv_end1 = clock();
    #endif
#endif

    // all-black patch cannot be normalized
    if (sum < std::numeric_limits<Scalar>::epsilon()) { // 小于优化的值了 返回
      residual.setZero();
      return false;
    }

    int num_residuals = 0;

#if !defined(_USE_RISCV_V)
    #if 1
    clock_t start2 = clock();
    #endif
    // 对于pattern的每个点进行计算
    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (residual[i] >= 0 && data[i] >= 0) {
        const Scalar val = residual[i]; // 这地方相当于做类型转换
        residual[i] = num_valid_points * val / sum - data[i]; // 归一化后再相减
        num_residuals++;

      } else {
        residual[i] = 0;
      }
    }
    #if 1
    clock_t end2 = clock();

    std::cout << std::fixed << std::setprecision(3) << "redidual:" << " sum=" << sum \
      // << " num_valid_points=" << num_valid_points 
      // << " num_residuals=" << num_residuals 
      << " final residual[0]=" << residual[0] << " residual[1]= " << residual[1] \
      << " clock=" << (end2 - start2 + (end1 - start1)) << " cycles\n";
    #endif
#else
    #if 0
    clock_t rvv_start2 = clock();
    #endif
    const Scalar *p_data = data.data();
    const Scalar constant = static_cast<Scalar>(num_valid_points) / sum;

    for(size_t vl, i = 0; i < n; i += vl)
    {
      // 根据剩余元素数设置向量长度
      vl = __riscv_vsetvl_e32m2(n - i);

      // 从数据中加载向量
      vfloat32m2_t vec_point2d_x = __riscv_vle32_v_f32m2(u + i, vl);
      vfloat32m2_t vec_data = __riscv_vle32_v_f32m2(p_data + i, vl);
      vfloat32m2_t vec_residual = __riscv_vle32_v_f32m2(result + i, vl);

      // multiply
      vec_residual = __riscv_vfmul_vf_f32m2(vec_residual, constant, vl);

      // subtract
      vec_residual = __riscv_vfsub_vv_f32m2(vec_residual, vec_data, vl);

      // > 0
      vbool16_t mask = __riscv_vmfgt_vf_f32m2_b16(vec_point2d_x, 0, vl); // 2d点有效时，等价于残差也是有效的
      // >= 0
      mask = __riscv_vmfge_vf_f32m2_b16_m(mask, vec_data, 0, vl);

      vec_residual= __riscv_vmerge_vvm_f32m2(vec_zero, vec_residual, mask, vl);

      __riscv_vse32_v_f32m2(&result[i], vec_residual, vl);

      num_residuals += __riscv_vcpop_m_b16(mask, vl);
    }

    #if 0
    clock_t rvv_end2 = clock();

    std::cout << std::fixed << std::setprecision(3) << "rvv redidual:" << " sum=" << sum \
      << " num_valid_points=" << num_valid_points \
      << " num_residuals=" << num_residuals \
      << " final residual[0]=" << residual[0] << " residual[1]= " << residual[1] \
      << " clock=" << (rvv_end2 - rvv_start2 + (rvv_end1 - rvv_start1)) << " cycles\n";
    #endif

#endif

    return num_residuals > PATTERN_SIZE / 2; // 超过一半的值才是符合的
  }

  Vector2 pos = Vector2::Zero();
  VectorP data = VectorP::Zero();  // negative if the point is not valid

  // MatrixP3 J_se2;  // total jacobian with respect to se2 warp
  // Matrix3 H_se2_inv;
  Matrix3P H_se2_inv_J_se2_T = Matrix3P::Zero();

  Scalar mean = 0;

  bool valid = false;

  // static clock_t clock_sum = 0; // // 错误：不允许在类定义内部初始化非 const 静态成员 added on 2024-11-28.
  static clock_t clock_sum; // 仅仅只是声明
  static clock_t residual_clock_sum;
};

template<>
clock_t OpticalFlowPatch<float, Pattern51<float>>::clock_sum = 0; // 在类外定义并初始化

template<> clock_t OpticalFlowPatch<float, Pattern52<float>>::clock_sum = 0;
template<> clock_t OpticalFlowPatch<float, Pattern50<float>>::clock_sum = 0;
template<> clock_t OpticalFlowPatch<float, Pattern24<float>>::clock_sum = 0;

template<>
clock_t OpticalFlowPatch<float, Pattern51<float>>::residual_clock_sum = 0; // 在类外定义并初始化

template<> clock_t OpticalFlowPatch<float, Pattern52<float>>::residual_clock_sum = 0;
template<> clock_t OpticalFlowPatch<float, Pattern50<float>>::residual_clock_sum = 0;
template<> clock_t OpticalFlowPatch<float, Pattern24<float>>::residual_clock_sum = 0;

template <typename Scalar, typename Pattern>
const typename OpticalFlowPatch<Scalar, Pattern>::Matrix2P
    OpticalFlowPatch<Scalar, Pattern>::pattern2 = Pattern::pattern2;

}  // namespace basalt

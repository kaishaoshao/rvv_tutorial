
#include <iostream>
#include <cmath>

#include <riscv_vector.h>


#if 0
template <typename S>
inline S computePatchInterpGrad(const S u[], const S v[], S *interp, S *grad_x, S *grad_y, int n) const // argument list: a series of coordinates
{
    //
    const int PATCH_SIZE = n;
    S px0y0[PATCH_SIZE],  px0y1[PATCH_SIZE],  px1y0[PATCH_SIZE],  px1y1[PATCH_SIZE];//, \
        ddx[PATCH_SIZE],  ddy[PATCH_SIZE],  dx[PATCH_SIZE],  dy[PATCH_SIZE];

    S pxm1y0[PATCH_SIZE], pxm1y1[PATCH_SIZE], px2y0[PATCH_SIZE], px2y1[PATCH_SIZE];
    S px0ym1[PATCH_SIZE], px1ym1[PATCH_SIZE], px0y2[PATCH_SIZE], px1y2[PATCH_SIZE];

    // for(int i = 0; i < n; i++)
    for(int i = 0; i < PATCH_SIZE; i++)
    {
        int ix = static_cast<int>(u[i]);
        int iy = static_cast<int>(v[i]);
        px0y0[i] = (*this)(ix, iy);
        px0y1[i] = (*this)(ix, iy + 1);
        px1y0[i] = (*this)(ix + 1, iy);
        px1y1[i] = (*this)(ix + 1, iy + 1);

        // for gradient
        pxm1y0[i] = (*this)(ix - 1, iy);
        pxm1y1[i] = (*this)(ix - 1, iy + 1);

        px2y0[i] = (*this)(ix + 2, iy);
        px2y1[i] = (*this)(ix + 2, iy + 1);

        px0ym1[i] = (*this)(ix, iy - 1);
        px1ym1[i] = (*this)(ix + 1, iy - 1);

        px0y2[i] = (*this)(ix, iy + 2);
        px1y2[i] = (*this)(ix + 1, iy + 2);
    }

    // vint32m2_t __riscv_vfcvt_x_f_v_i32m2(vfloat32m2_t vs2, size_t vl); // float --> int
    // vfloat32m2_t __riscv_vfcvt_f_x_v_f32m2(vint32m2_t vs2, size_t vl); // int --> float

    vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // 初始化累加向量


    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vec_one = __riscv_vfmv_v_f_f32m2(1.0, vlmax);
    size_t vl;
    // for (size_t vl; n > 0; n -= vl)
    for (int i = 0; i < n; i += vl)
    {
        // vl = __riscv_vsetvl_e32m2(n);
        vl = __riscv_vsetvl_e32m2(n - i);

        vfloat32m2_t vec_u = __riscv_vle32_v_f32m2(u + i, vl);
        vint32m2_t vec_iu = __riscv_vfcvt_rtz_x_f_v_i32m2(vec_u, vl);
        vfloat32m2_t vec_fu = __riscv_vfcvt_f_x_v_f32m2(vec_iu, vl);

        vfloat32m2_t v_dx = __riscv_vfsub_vv_f32m2(vec_u, vec_fu, vl);

        vfloat32m2_t vec_v = __riscv_vle32_v_f32m2(v + i, vl);
        vint32m2_t vec_iv = __riscv_vfcvt_rtz_x_f_v_i32m2(vec_v, vl);
        vfloat32m2_t vec_fv = __riscv_vfcvt_f_x_v_f32m2(vec_iv, vl);

        vfloat32m2_t v_dy = __riscv_vfsub_vv_f32m2(vec_v, vec_fv, vl);

        vfloat32m2_t v_ddx = __riscv_vfsub_vv_f32m2(vec_one, v_dx, vl);
        vfloat32m2_t v_ddy = __riscv_vfsub_vv_f32m2(vec_one, v_dy, vl);

        //
        vfloat32m2_t v_px0y0 = __riscv_vle32_v_f32m2(&px0y0[i], vl);
        vfloat32m2_t v_px0y1 = __riscv_vle32_v_f32m2(&px0y1[i], vl);
        vfloat32m2_t v_px1y0 = __riscv_vle32_v_f32m2(&px1y0[i], vl);
        vfloat32m2_t v_px1y1 = __riscv_vle32_v_f32m2(&px1y1[i], vl);

        // __riscv_vfmul_vf_f32m8 可以计算矢量与标量的乘法
        vfloat32m2_t v_ddx_ddy = __riscv_vfmul_vv_f32m2(v_ddx, v_ddy, vl);
        vfloat32m2_t v_ddx_dy = __riscv_vfmul_vv_f32m2(v_ddx, v_dy, vl);
        vfloat32m2_t v_dx_ddy = __riscv_vfmul_vv_f32m2(v_dx, v_ddy, vl);
        vfloat32m2_t v_dx_dy = __riscv_vfmul_vv_f32m2(v_dx, v_dy, vl);

        // 按权重进行逐元素乘法
        vfloat32m2_t v_term0 = __riscv_vfmul_vv_f32m2(v_px0y0, v_ddx_ddy, vl);
        vfloat32m2_t v_term1 = __riscv_vfmul_vv_f32m2(v_px0y1, v_ddx_dy, vl);
        vfloat32m2_t v_term2 = __riscv_vfmul_vv_f32m2(v_px1y0, v_dx_ddy, vl);
        vfloat32m2_t v_term3 = __riscv_vfmul_vv_f32m2(v_px1y1, v_dx_dy, vl);

        // 累加所有项
        vfloat32m2_t v_result = __riscv_vfadd_vv_f32m2(v_term0, v_term1, vl);
        v_result = __riscv_vfadd_vv_f32m2(v_result, v_term2, vl);
        v_result = __riscv_vfadd_vv_f32m2(v_result, v_term3, vl);

        vec_sum = __riscv_vfredusum_vs_f32m2_f32m1(v_result, vec_sum, vl);

        // 存储结果
        __riscv_vse32_v_f32m2(&interp[i], v_result, vl);

        {
            // gradient in x
            // S res_mx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;
            vfloat32m2_t v_pxm1y0 = __riscv_vle32_v_f32m2(&pxm1y0[i], vl);
            vfloat32m2_t v_pxm1y1 = __riscv_vle32_v_f32m2(&pxm1y1[i], vl);

            vfloat32m2_t v_px2y0 = __riscv_vle32_v_f32m2(&px2y0[i], vl);
            vfloat32m2_t v_px2y1 = __riscv_vle32_v_f32m2(&px2y1[i], vl);

            v_term0 = __riscv_vfmul_vv_f32m2(v_pxm1y0, v_ddx_ddy, vl);
            v_term1 = __riscv_vfmul_vv_f32m2(v_pxm1y1, v_ddx_dy, vl);
            v_term2 = __riscv_vfmul_vv_f32m2(v_px0y0, v_dx_ddy, vl);
            v_term3 = __riscv_vfmul_vv_f32m2(v_px0y1, v_dx_dy, vl);

            // 累加所有项
            vfloat32m2_t res_mx = __riscv_vfadd_vv_f32m2(v_term0, v_term1, vl);
            res_mx = __riscv_vfadd_vv_f32m2(res_mx, v_term2, vl);
            res_mx = __riscv_vfadd_vv_f32m2(res_mx, v_term3, vl);

            // S res_px = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;
            v_term0 = __riscv_vfmul_vv_f32m2(v_px1y0, v_ddx_ddy, vl);
            v_term1 = __riscv_vfmul_vv_f32m2(v_px1y1, v_ddx_dy, vl);
            v_term2 = __riscv_vfmul_vv_f32m2(v_px2y0, v_dx_ddy, vl);
            v_term3 = __riscv_vfmul_vv_f32m2(v_px2y1, v_dx_dy, vl);

            vfloat32m2_t res_px = __riscv_vfadd_vv_f32m2(v_term0, v_term1, vl);
            res_px = __riscv_vfadd_vv_f32m2(res_px, v_term2, vl);
            res_px = __riscv_vfadd_vv_f32m2(res_px, v_term3, vl);

            // res[1] = S(0.5) * (res_px - res_mx);
            v_result = __riscv_vfsub_vv_f32m2(res_px, res_mx, vl);
            v_result = __riscv_vfmul_vf_f32m2(v_result, 0.5, vl);

            __riscv_vse32_v_f32m2(&grad_x[i], v_result, vl);


            // gradient in y
            // S res_my = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;
            vfloat32m2_t v_px0ym1 = __riscv_vle32_v_f32m2(&px0ym1[i], vl);
            vfloat32m2_t v_px1ym1 = __riscv_vle32_v_f32m2(&px1ym1[i], vl);
            v_term0 = __riscv_vfmul_vv_f32m2(v_px0ym1, v_ddx_ddy, vl);
            v_term1 = __riscv_vfmul_vv_f32m2(v_px0y0, v_ddx_dy, vl);
            v_term2 = __riscv_vfmul_vv_f32m2(v_px1ym1, v_dx_ddy, vl);
            v_term3 = __riscv_vfmul_vv_f32m2(v_px1y0, v_dx_dy, vl);

            // 累加所有项
            vfloat32m2_t res_my = __riscv_vfadd_vv_f32m2(v_term0, v_term1, vl);
            res_my = __riscv_vfadd_vv_f32m2(res_my, v_term2, vl);
            res_my = __riscv_vfadd_vv_f32m2(res_my, v_term3, vl);

            // TODO:
            // S res_py = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;
            vfloat32m2_t v_px0y2 = __riscv_vle32_v_f32m2(&px0y2[i], vl);
            vfloat32m2_t v_px1y2 = __riscv_vle32_v_f32m2(&px1y2[i], vl);

            v_term0 = __riscv_vfmul_vv_f32m2(v_px0y1, v_ddx_ddy, vl);
            v_term1 = __riscv_vfmul_vv_f32m2(v_px0y2, v_ddx_dy, vl);
            v_term2 = __riscv_vfmul_vv_f32m2(v_px1y1, v_dx_ddy, vl);
            v_term3 = __riscv_vfmul_vv_f32m2(v_px1y2, v_dx_dy, vl);

            // 累加所有项
            vfloat32m2_t res_py = __riscv_vfadd_vv_f32m2(v_term0, v_term1, vl);
            res_py = __riscv_vfadd_vv_f32m2(res_py, v_term2, vl);
            res_py = __riscv_vfadd_vv_f32m2(res_py, v_term3, vl);

            // res[2] = S(0.5) * (res_py - res_my);
            v_result = __riscv_vfsub_vv_f32m2(res_py, res_my, vl);
            v_result = __riscv_vfmul_vf_f32m2(v_result, 0.5, vl);

            __riscv_vse32_v_f32m2(&grad_y[i], v_result, vl);
        }

    }

    S sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);

    return sum;
}
#endif

/*
C++ std::floor和std:ceil简述
std::floor 和 std::ceil都是对变量进行四舍五入，只不过四舍五入的方向不同。

1: std::floor -->向下取整数

2: std::ceil -->向上取整数：

例如：5.88

std::floor(5.88) = 5;

std::ceil(5.88) = 6;
*/

int main(int argc, char** argv)
{
    // 2 1 2 2 7 3 4 4 5 8
    // float u[] = { -1.53, 0.67, 2.45, 1.61, 7.23, 2.66, 3.78, 4.11, 5.16, 7.75 };
    float u[] = { 1.53, 0.67, 2.45, 1.61, 7.23, 2.66, 3.78, 4.11, 5.16, 7.75 };
    // 1 0 2 1 7 2 3 4 5 7
    size_t n = 10;
    int iu[10];
    float fu[10];

    std::cout << "floor u[0]=" << std::floor(u[0]) << std::endl;
    std::cout << "ceil u[0]=" << std::ceil(u[0]) << std::endl;

    for(size_t vl, i = 0; i < n; i+= vl)
    {
        vl = __riscv_vsetvl_e32m2(n - i);

        vfloat32m2_t vec_u = __riscv_vle32_v_f32m2(u + i, vl);
        // vint32m2_t vec_iu = __riscv_vfcvt_x_f_v_i32m2(vec_u, vl); // 该函数对于正数是向上取整，对于负数是向下取整。

        // vint32m2_t __riscv_vfcvt_rtz_x_f_v_i32m2(vfloat32m2_t vs2, size_t vl); // 该函数对于正数向下取整
        vint32m2_t vec_iu = __riscv_vfcvt_rtz_x_f_v_i32m2(vec_u, vl);
        vfloat32m2_t vec_fu = __riscv_vfcvt_f_x_v_f32m2(vec_iu, vl);

        __riscv_vse32_v_i32m2(&iu[i], vec_iu, vl);
        __riscv_vse32_v_f32m2(&fu[i], vec_fu, vl);
    }

    std::cout << "\niu:\n";
    for(size_t i = 0; i < n; i++) std::cout << iu[i] << " "; std::cout << std::endl;

    std::cout << "\nfu:\n";
    for(size_t i = 0; i < n; i++) std::cout << fu[i] << " "; std::cout << std::endl;

    return 0;
}
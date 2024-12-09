
#include <iostream>

#include <riscv_vector.h>

// #define _METHOD1_
#define _METHOD2_

int main(int argc, char** argv)
{
    float point2d_x[10] = { 0.1, 2.3, 1.5, 0.7, 10.8, 0.3, 639.1, 640.2, 613, 7.8 };
    float new_point2d_x[10];
    
    size_t w = 640;
    size_t h = 480;
    const float border = 2.0;
    float offset = 1.0;
    size_t n = 10;
    int valid_count = 0;

    size_t vlmax = __riscv_vsetvlmax_e32m2();
    std::cout << "vlmax=" << vlmax << std::endl;
    vfloat32m2_t vec_zero = __riscv_vfmv_v_f_f32m2(0, vlmax);

#if defined(_METHOD1_)
    // for(size_t vl; n > 0; n -= vl)
    for(size_t vl, i = 0; i < n; i += vl)
    {
        // 根据剩余元素数设置向量长度
        // vl = __riscv_vsetvl_e32m2(n);
        vl = __riscv_vsetvl_e32m2(n - i);
        std::cout << "vl=" << vl << std::endl;

        // 从数据中加载向量
        vfloat32m2_t vec_point2d_x = __riscv_vle32_v_f32m2(point2d_x + i, vl);

        //
        // vbool16_t __riscv_vmfgt_vf_f32m2_b16(vfloat32m2_t vs2, float rs1, size_t vl);
        // vbool16_t __riscv_vmflt_vf_f32m2_b16(vfloat32m2_t vs2, float rs1, size_t vl);
        // vbool16_t __riscv_vmand_mm_b16(vbool16_t vs2, vbool16_t vs1, size_t vl);
        // vfloat32m2_t __riscv_vmerge_vvm_f32m2(vfloat32m2_t vs2, vfloat32m2_t vs1,vbool16_t v0, size_t vl);

        // >= // vbool16_t __riscv_vmfge_vf_f32m2_b16(vfloat32m2_t vs2, float rs1, size_t vl);
        vbool16_t mask1 = __riscv_vmfge_vf_f32m2_b16(vec_point2d_x, border, vl);
        unsigned int count1 =__riscv_vcpop_m_b16(mask1, vl);
        std::cout << "count1=" << count1 << std::endl;

        // <
        vbool16_t mask2 = __riscv_vmflt_vf_f32m2_b16(vec_point2d_x, (w - border - offset), vl);
        unsigned int count2 =__riscv_vcpop_m_b16(mask2, vl);
        std::cout << "count2=" << count2 << std::endl;

        // 按位与
        vbool16_t mask = __riscv_vmand_mm_b16(mask2, mask1, vl);

        unsigned int count =__riscv_vcpop_m_b16(mask, vl);
        std::cout << "count=" << count << std::endl;

        // // for (int i = 0; i < 16; ++i) {
        // for (int i = 0; i < vl; ++i) {
        //     printf("%d", mask[i] ? 1 : 0);
        // }
        // printf("\n");

        /*
         * vfloat32m2_t __riscv_vmerge_vvm_f32m2(vfloat32m2_t vs2, vfloat32m2_t vs1,vbool16_t v0, size_t vl);
         该指令的作用是：

         如果掩码 v0 中某一位置为 true，则该位置的元素将从 vs1 中取值。
         如果掩码 v0 中某一位置为 false，则该位置的元素将从 vs2 中取值。
         简单来说：
         通过掩码 v0，选择性地将 vs1 或 vs2 的值赋给结果向量。
         */
        vec_point2d_x= __riscv_vmerge_vvm_f32m2(vec_zero, vec_point2d_x, mask, vl);
        __riscv_vse32_v_f32m2(new_point2d_x + i, vec_point2d_x, vl);
    }
#elif defined(_METHOD2_)
    // vbool16_t __riscv_vmfge_vf_f32m2_b16_m(vbool16_t vm, vfloat32m2_t vs2, float rs1, size_t vl);

    for(size_t vl, i = 0; i < n; i += vl)
    {
        // 根据剩余元素数设置向量长度
        // vl = __riscv_vsetvl_e32m2(n);
        vl = __riscv_vsetvl_e32m2(n - i);
        std::cout << "vl=" << vl << std::endl;

        // 从数据中加载向量
        vfloat32m2_t vec_point2d_x = __riscv_vle32_v_f32m2(point2d_x + i, vl);

        // >=
        vbool16_t mask = __riscv_vmfge_vf_f32m2_b16(vec_point2d_x, border, vl);
        // unsigned int count1 = __riscv_vcpop_m_b16(mask, vl);
        // std::cout << "count1=" << count1 << std::endl;

        // <
        mask = __riscv_vmflt_vf_f32m2_b16_m(mask, vec_point2d_x, (w - border - offset), vl);
        // unsigned int count2 = __riscv_vcpop_m_b16(mask, vl);
        // std::cout << "count2=" << count2 << std::endl;

        vec_point2d_x= __riscv_vmerge_vvm_f32m2(vec_zero, vec_point2d_x, mask, vl);
        __riscv_vse32_v_f32m2(new_point2d_x + i, vec_point2d_x, vl);

        valid_count += __riscv_vcpop_m_b16(mask, vl);
    }

#endif
    std::cout << "valid_count=" << valid_count << std::endl;

    std::cout << "point2d_x: ";
    for(int i = 0; i < n; i++) std::cout << point2d_x[i] << "  "; std::cout << std::endl;

    std::cout << "new_point2d_x: ";
    for(int i = 0; i < n; i++) std::cout << new_point2d_x[i] << "  "; std::cout << std::endl;


    // TODO: 怎样输出vbool16_t的每个布尔值？？
    // vbool16_t bool_vector = {1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1}; // error.
    // size_t vl = 16;
    // std::cout << "true count:" << __riscv_vcpop_m_b16(bool_vector, vl) << std::endl;

    return 0;

}
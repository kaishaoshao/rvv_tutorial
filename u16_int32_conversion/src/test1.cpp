
#include <iostream>
#include <cmath>
#include <time.h>
#include <iomanip> // for std::setprecision
#include <riscv_vector.h>

int main(int argc, char *argv[])
{
    uint16_t u16_array[] = { 9728, 65280, 13312, 43520, 47872, 52224 };
    int int_array[6];
    uint16_t u16_array2[6];

    // vint32m2_t __riscv_vsext_vf2_i32m2(vint16m1_t vs2, size_t vl);
    // vuint32m2_t __riscv_vzext_vf2_u32m2(vuint16m1_t vs2, size_t vl);
    // vint32m2_t __riscv_vreinterpret_v_u32m2_i32m2(vuint32m2_t src);
    // vint16m2_t __riscv_vreinterpret_v_i32m2_i16m2(vint32m2_t src);
    // vuint16m2_t __riscv_vreinterpret_v_i16m2_u16m2(vint16m2_t src);

    // vint32m2_t __riscv_vwcvt_x_x_v_i32m2(vint16m1_t vs2, size_t vl);

    size_t n = 6;
    // for(size_t vl; n > 0; n -= vl)
    for(size_t vl, i = 0; i < n; i += vl)
    {
        // vl = __riscv_vsetvl_e16m2(n - i);
        // vuint16m2_t vec_u16_array = __riscv_vle16_v_u16m2(u16_array + i, vl); // u16m2
        // __riscv_vse16_v_u16m2((uint16_t *)u16_array2, vec_u16_array, vl);

        // vl = __riscv_vsetvl_e32m2(n - i);
        // vint32m2_t vec_i32_array = __riscv_vle32_v_i32m2((int *)u16_array + i, vl); // error
        // __riscv_vse32_v_i32m2(int_array, vec_i32_array, vl);

        // u16 to int32 : ok!
        vl = __riscv_vsetvl_e16m1(n - i);
        vuint16m1_t vec_u16_array = __riscv_vle16_v_u16m1(u16_array + i, vl);
        // vuint32m2_t vec_u32_array = __riscv_vzext_vf2_u32m2(vec_u16_array, vl);
        vint32m2_t vec_i32_array = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf2_u32m2(vec_u16_array, vl));
        __riscv_vse32_v_i32m2(int_array, vec_i32_array, vl);
    }

    std::cout << "u16 to int32 : \n";
    for(size_t i = 0; i < n; i++)
    std::cout << int_array[i] << "  "; std::cout << std::endl;
    // std::cout << u16_array2[i] << "  "; std::cout << std::endl;


    // int32 to u16
    int _i32_array_[] = { 9728, 65280, 13312, 43520, 47872, 52224, 56576, 60928 };
    uint16_t _u16_array_[8];
    n = 8;
    for(size_t vl, i = 0; i < n; i += vl)
    {
        vl = __riscv_vsetvl_e32m2(n - i);
        vint32m2_t vec_i32_array = __riscv_vle32_v_i32m2(_i32_array_ + i, vl);

        // vuint16m2_t vec_u16 = __riscv_vreinterpret_v_i16m2_u16m2(__riscv_vreinterpret_v_i32m2_i16m2(vec_i32_array));
        // __riscv_vse16_v_u16m2(_u16_array_, vec_u16, vl);

        // vuint16m1_t __riscv_vnclipu_wx_u16m1(vuint32m2_t vs2, size_t rs1, unsigned int vxrm, size_t vl);
        // vint16m1_t __riscv_vnclip_wx_i16m1(vint32m2_t vs2, size_t rs1, unsigned int vxrm, size_t vl);

        // vuint16m1_t __riscv_vreinterpret_v_i16m1_u16m1(vint16m1_t src);
        // vuint16m1_t vec_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vnclip_wx_i16m1(vec_i32_array, 0, 0, vl));

        vuint16m1_t vec_u16 = __riscv_vnclipu_wx_u16m1(__riscv_vreinterpret_v_i32m2_u32m2(vec_i32_array), 0, 0, vl); // ok.
        __riscv_vse16_v_u16m1(_u16_array_, vec_u16, vl);
    }

    std::cout << "int32 to u16 : \n";
    for(size_t i = 0; i < n; i++)
    std::cout << _u16_array_[i] << "  "; std::cout << std::endl;

    return 0;
}

/*
output:
root@ubuntu:~/lwx/rvv_tutorial/u16_int32_conversion/build# ./test1
u16 to int32 : 
9728  65280  13312  43520  47872  52224  
int32 to u16 : 
9728  65280  13312  43520  47872  52224  56576  60928  
root@ubuntu:~/lwx/rvv_tutorial/u16_int32_conversion/build# 

*/

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


    // test multiply & add
    {
        constexpr int kernel[5] = {1, 4, 6, 4, 1};
        int row_m2[] = { 1, 3, 5, 7, 9 };   // 1  3   5   7   9
        int row_m1[] = { 1, 3, 5, 7, 9 };   // 4  12  20  28  36
        int row[] = { 1, 3, 5, 7, 9 };      // 6  18  30  42  54
        int row_p1[] = { 1, 3, 5, 7, 9 };   // 4  12  20  28  36
        int row_p2[] = { 1, 3, 5, 7, 9 };   // 1  3   5   7   9
                                            // 16 48  80  112 144

        int result_a[5];
        size_t n = 5;
#if 1
        // vint32m2_t __riscv_vle32_v_i32m2(const int32_t *rs1, size_t vl);
        // vint32m2_t __riscv_vmul_vx_i32m2(vint32m2_t vs2, int32_t rs1, size_t vl);
        // vint32m2_t __riscv_vmadd_vx_i32m2(vint32m2_t vd, int32_t rs1, vint32m2_t vs2, size_t vl);
        int result_tmp[5];
        for (size_t vl, i = 0; i < n; i += vl)
        {
          vl = __riscv_vsetvl_e32m2(n - i);

          vint32m2_t vec_row_m2 = __riscv_vle32_v_i32m2(row_m2 + i, vl);
          vint32m2_t vec_row_m1 = __riscv_vle32_v_i32m2(row_m1 + i, vl);
          vint32m2_t vec_row = __riscv_vle32_v_i32m2(row + i, vl);
          vint32m2_t vec_row_p1 = __riscv_vle32_v_i32m2(row_p1 + i, vl);
          vint32m2_t vec_row_p2 = __riscv_vle32_v_i32m2(row_p2 + i, vl);
#if 0
          vint32m2_t result = __riscv_vmul_vx_i32m2(vec_row_m2, kernel[0], vl);
          __riscv_vse32_v_i32m2(result_tmp + i, result, vl);
          std::cout << "s1: ";
          for(size_t i = 0; i < vl; i++)
          std::cout << result_tmp[i] << "  "; std::cout << std::endl;


          result = __riscv_vmadd_vx_i32m2(result, kernel[1], vec_row_m1, vl);
          __riscv_vse32_v_i32m2(result_tmp + i, result, vl);
          std::cout << "s2: ";
          for(size_t i = 0; i < vl; i++)
          std::cout << result_tmp[i] << "  "; std::cout << std::endl;

        //   s2: 5  15  25  35  45  
        //   s3: 31  93  155  217  279  //err. 11 33 55 77 99

          result = __riscv_vmadd_vx_i32m2(result, kernel[2], vec_row, vl); // 5  15  25  35  45  +  6  18  30  42  54
          __riscv_vse32_v_i32m2(result_tmp + i, result, vl);
          std::cout << "s3: ";
          for(size_t i = 0; i < vl; i++)
          std::cout << result_tmp[i] << "  "; std::cout << std::endl;

          result = __riscv_vmadd_vx_i32m2(result, kernel[3], vec_row_p1, vl);
          __riscv_vse32_v_i32m2(result_tmp + i, result, vl);
          std::cout << "s4: ";
          for(size_t i = 0; i < vl; i++)
          std::cout << result_tmp[i] << "  "; std::cout << std::endl;

          result = __riscv_vmadd_vx_i32m2(result, kernel[4], vec_row_p2, vl);
          __riscv_vse32_v_i32m2(result_tmp + i, result, vl);
          std::cout << "s5: ";
          for(size_t i = 0; i < vl; i++)
          std::cout << result_tmp[i] << "  "; std::cout << std::endl;
#else
          vint32m2_t result = __riscv_vmul_vx_i32m2(vec_row_m2, kernel[0], vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_m1, kernel[1], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row, kernel[2], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_p1, kernel[3], result, vl);
          result = __riscv_vmadd_vx_i32m2(vec_row_p2, kernel[4], result, vl);
#endif
          __riscv_vse32_v_i32m2(result_a + i, result, vl);
        }

        std::cout << "vmadd : \n";
        for(size_t i = 0; i < n; i++)
        std::cout << result_a[i] << "  "; std::cout << std::endl;
#else
        for (int r = 0; r < n; r++) {
          int val_int = kernel[0] * row_m2[r] + kernel[1] * row_m1[r] +
                        kernel[2] * row[r] + kernel[3] * row_p1[r] +
                        kernel[4] * row_p2[r];
        //   T val = ((val_int + (1 << 7)) >> 8);
          result_a[r] = val_int; //val;
        }

        std::cout << "general compute : \n";
        for(size_t i = 0; i < n; i++)
        std::cout << result_a[i] << "  "; std::cout << std::endl;
#endif

    }

    return 0;
}

/*
output:
root@ubuntu:~/lwx/rvv_tutorial/u16_int32_conversion/build# ./test1
u16 to int32 : 
9728  65280  13312  43520  47872  52224
int32 to u16 : 
9728  65280  13312  43520  47872  52224  56576  60928
s1: 1  3  5  7  9  
s2: 5  15  25  35  45  
s3: 31  93  155  217  279  // 11 33 55 77 99
s4: 125  375  625  875  1125  
s5: 126  378  630  882  1134 
vmadd : 
126  378  630  882  1134 
general compute : 
16  48  80  112  144 
root@ubuntu:~/lwx/rvv_tutorial/u16_int32_conversion/build# 

*/


#include <iostream>
#include <cmath>
#include <time.h>
#include <iomanip> // for std::setprecision
#include <riscv_vector.h>

/*
loat16（半精度浮点数）是 IEEE 754 标准中的一种浮点数格式，使用 16 位（2 字节）存储。它比 float32（单精度浮点数）占用更少的内存，但精度和范围较小。其数据范围如下：

符号位（1 位）：表示正负
指数位（5 位）：表示指数的偏移量
尾数位（10 位）：表示有效数字部分（也叫做“精度”）
具体数据范围是：
最大正值约：2^16-1 = 65535
因为指数为范围是0~31，指数最大为16，表示最大可以有16个二进制位。因此最大数是2^16 -1.

注意事项：
精度：float16 只有 10 位有效数字，因此精度较低，适合存储大范围的近似值，而不适用于需要高精度的计算。
子正常数（Subnormal Numbers）：当数字小于最小正值时，float16 仍然可以表示这些数字，但它们的精度会进一步降低。
float16 经常用于机器学习、图形处理和某些嵌入式系统中，尤其是在需要优化内存使用和计算速度时。
*/

/*
_Float16 类型将整数值 5050 转换为 5048 的现象，主要是由于浮点数在计算机中的存储方式和精度限制导致的。以下是几个关键点来解释这一现象：

IEEE 754 标准：计算机中的浮点数遵循 IEEE 754 标准，该标准规定了浮点数的存储方式，包括符号位、指数位和尾数位。对于 _Float16 类型，即半精度浮点数，它只有16位，其中1位符号位，5位指数位和10位尾数位
。

精度限制：由于尾数位只有10位，_Float16 类型的浮点数在表示整数时只能精确表示到 2^11 - 1 个不同的整数值，即从 0 到 2047。对于超出这个范围的整数，比如 5050，就无法被精确表示，因此会发生精度损失
。

舍入误差：当一个整数无法被浮点类型精确表示时，它会被舍入到最接近的可表示值。在 _Float16 类型中，5050 被舍入到了最接近的可表示的浮点数值，即 5048
。

二进制表示：十进制数在转换为二进制时可能会有无限循环小数，而浮点数的尾数部分是有限的，因此在转换过程中必须截断这些无限循环的小数部分，这也会导致精度损失
。

浮点数转换：在将整数转换为浮点数时，如果整数的大小超出了浮点类型能精确表示的范围，就会发生舍入。在 _Float16 类型中，由于其精度较低，许多稍大的整数都无法被精确表示，从而导致转换后的值与原值不同
。
*/

// 半精度浮点数
// clang++ -o test1 test1.cpp -march=rv64gcv_zfh_zvfh -menable-experimental-extensions -D__fp16=_Float16

// clang++ -o test1_clang test1.cpp -march=rv64gcv_zfh_zvfh1p0 -menable-experimental-extensions -D__fp16=_Float16

int main(int argc, char* argv[])
{
    // std::cout << argc << std::endl;

    // _Float16 number = static_cast<_Float16>(5050.12345678);
    // _Float16 number = static_cast<_Float16>(1013.456789); // 精度损失较大，只能表示1013.5
    
    // 以下正解：
    // 由于尾数位只有10位，加上默认的整数位1，总共可以表示最大数为11个bit位的数，即2^11 - 1 = 2047.
    // 0 ~ 2047之间的整数完全不会受到截断影响
    //大于2047的整数，有可能受影响，因为如果截断的部分刚好都是0，则没有影响
    // _Float16 number = static_cast<_Float16>(2048.0); // 可以输出2048 截断了，但是不受影响
    // _Float16 number = static_cast<_Float16>(2049.0); // 还是输出2048， 截断了，受影响
    // _Float16 number = static_cast<_Float16>(2050.0); // 可以输出2050，截断了，但是不受影响
    // _Float16 number = static_cast<_Float16>(0.123456789); // 输出0.123474
    // _Float16 number = static_cast<_Float16>(0.12345); // 输出0.123474
    _Float16 number = static_cast<_Float16>(0.1234); // 输出0.123413
    std::cout << "number=" << static_cast<float>(number) << std::endl;
    
    if(argc != 2)
    {
        std::cout << "please input size. format: ./test1 100" << std::endl;
        return 0;
    }
    // vfloat16m2_t __riscv_vle16_v_f16m2(const _Float16 *rs1, size_t vl);

    // _Float16 
    // constexpr int _SIZE_ = 99;//100; //10000; // 5000 // 100
    int _SIZE_ = 100;
    if(argv[1])
    {
        _SIZE_ = atol(argv[1]);
        std::cout << "size=" << _SIZE_ << std::endl;
    }
    // __attribute__((aligned(32))) float arr[_SIZE_];
    alignas(16) _Float16 arr[_SIZE_];		// 16-byte aligned
    for (int i = 0; i < _SIZE_; i++) {
        arr[i] = (_Float16)(i + 1);  // 初始化数组为 1.0, 2.0, 3.0, ..., 100.0
    }

    clock_t t0, t1;


    t0 = clock();


    // vfloat16m1_t __riscv_vfmv_v_f_f16m1(_Float16 rs1, size_t vl);
    vfloat16m1_t vec_sum = __riscv_vfmv_v_f_f16m1(0.0f, 1);  // 初始化累加向量

    size_t n = _SIZE_;
    for (size_t vl, i = 0; i < n; i += vl)
    {
        // 根据剩余元素数设置向量长度
        vl = __riscv_vsetvl_e16m2(n - i);
        // 从数据中加载向量
        vfloat16m2_t vec_val = __riscv_vle16_v_f16m2(arr + i, vl);

        // 执行归约操作reduction
        // vfloat16m1_t __riscv_vfredusum_vs_f16m2_f16m1(vfloat16m2_t vs2, vfloat16m1_t vs1, size_t vl);
        vec_sum = __riscv_vfredusum_vs_f16m2_f16m1(vec_val, vec_sum, vl);

        // float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        // std::cout << "sum=" << sum << std::endl;

    }

    // _Float16 __riscv_vfmv_f_s_f16m1_f16(vfloat16m1_t vs1);
    _Float16 sum = __riscv_vfmv_f_s_f16m1_f16(vec_sum);
    t1 = clock();

    // std::cout << "1.0 + 2.0 + ... + 100=" << sum << " cost=" << (t1 - t0) << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "1.0 + 2.0 + ... + " << _SIZE_ << "=" << static_cast<float>(sum) << " cost=" << (t1 - t0) << " cycles" << std::endl;

    t0 = clock();
    _Float16 sum2 = 0;
    for(int i = 0; i < _SIZE_; i++) sum2 += arr[i];
    t1 = clock();
    std::cout << std::fixed << std::setprecision(2) << "sum2=" << static_cast<float>(sum2) << " cost=" << (t1 - t0) << " cycles" << std::endl;


    return 0;
}
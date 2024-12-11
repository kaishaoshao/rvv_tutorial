

#include <iostream>
#include <cmath>
#include <time.h>
#include <iomanip> // for std::setprecision
#include <riscv_vector.h>

int main(int argc, char* argv[])
{
    // vfloat16m2_t __riscv_vle16_v_f16m2(const _Float16 *rs1, size_t vl);

    // _Float16 
    constexpr int _SIZE_ = 50; //10000; // 5000 // 100
    // __attribute__((aligned(32))) float arr[_SIZE_];
    alignas(16) _Float16 arr[_SIZE_];		// 16-byte aligned
    for (int i = 0; i < _SIZE_; i++) {
        arr[i] = (_Float16)(i + 1);  // 初始化数组为 1.0, 2.0, 3.0, ..., 100.0
    }

    clock_t t0, t1;

    vfloat32m1_t param2;
    vfloat64m1_t param3;
    vfloat16m2_t param1;

    t0 = clock();

    // vfloat16m1_t param1;
    // vfloat32m1_t param2;
    // vfloat64m1_t param3;

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
    std::cout << std::fixed << std::setprecision(2) << "1.0 + 2.0 + ... + " << _SIZE_ << "=" << sum << " cost=" << (t1 - t0) << " cycles" << std::endl;
/**/
    t0 = clock();
    _Float16 sum2 = 0;
    for(int i = 0; i < _SIZE_; i++) sum2 += arr[i];
    t1 = clock();
    std::cout << std::fixed << std::setprecision(2) << "sum2=" << static_cast<float>(sum2) << " cost=" << (t1 - t0) << " cycles" << std::endl;

    return 0;
}
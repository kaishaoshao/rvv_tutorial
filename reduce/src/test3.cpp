
#include <iostream>
#include <chrono>
#include <time.h>
#include <riscv_vector.h>
#include <iomanip> // for std::setprecision

/*

root@baton:~/lwx/install/stereo3/lib/stereo3# ./test3
sum=136
sum=528
sum=1176
sum=2080
sum=3240
sum=4656
sum=5050
1.0 + 2.0 + ... + 100=5050

*/

int main(int argc, char *argv[])
{
    // TODO: size为1万时，计算结果居然不正确。
    constexpr int _SIZE_ = 10000; // 5000 // 100
    __attribute__((aligned(32))) float arr[_SIZE_];
    for (int i = 0; i < _SIZE_; i++) {
        arr[i] = (float)(i + 1);  // 初始化数组为 1.0, 2.0, 3.0, ..., 100.0
    }

    clock_t t0 = clock();

    // vfloat32m1_t __riscv_vfredusum_vs_f32m2_f32m1(vfloat32m2_t vs2, vfloat32m1_t vs1, size_t vl);

    // size_t vlmax = __riscv_vsetvlmax_e32m1();  // 获取最大向量长度
    // vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0, vlmax);  // 初始化累加向量
    vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);  // 初始化累加向量

    size_t n = _SIZE_;
#if 0
    // method 1
    float *p_val = arr;
    for (size_t vl; n > 0; n -= vl)
    {
        // 根据剩余元素数设置向量长度
        vl = __riscv_vsetvl_e32m2(n);
        // 从数据中加载向量
        vfloat32m2_t vec_val = __riscv_vle32_v_f32m2(p_val, vl);

        vec_sum = __riscv_vfredusum_vs_f32m2_f32m1(vec_val, vec_sum, vl);

        p_val += vl;

        float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        std::cout << "sum=" << sum << std::endl;
    }

#elif 1
    // method 2
    for (size_t vl, i = 0; i < n; i += vl)
    {
        // 根据剩余元素数设置向量长度
        vl = __riscv_vsetvl_e32m2(n - i);
        // 从数据中加载向量
        vfloat32m2_t vec_val = __riscv_vle32_v_f32m2(arr + i, vl);

        // 执行归约操作reduction
        vec_sum = __riscv_vfredusum_vs_f32m2_f32m1(vec_val, vec_sum, vl);

        // float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        // std::cout << "sum=" << sum << std::endl;

    }

#else

    // method 3
    for (size_t vl, i = 0; i < n; i += vl)
    {
        // 根据剩余元素数设置向量长度
        vl = __riscv_vsetvl_e32m8(n - i);
        // 从数据中加载向量
        vfloat32m8_t vec_val = __riscv_vle32_v_f32m8(arr + i, vl);

        // 执行归约操作reduction
        vec_sum = __riscv_vfredusum_vs_f32m8_f32m1(vec_val, vec_sum, vl);

        // float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
        // std::cout << "sum=" << sum << std::endl;

    }

#endif

    float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    clock_t t1 = clock();

    // std::cout << "1.0 + 2.0 + ... + 100=" << sum << " cost=" << (t1 - t0) << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "1.0 + 2.0 + ... + " << _SIZE_ << "=" << sum << " cost=" << (t1 - t0) << std::endl;

    t0 = clock();
    double sum2 = 0;
    for(int i = 0; i < _SIZE_; i++) sum2 += arr[i];
    t1 = clock();
    std::cout << std::fixed << std::setprecision(2) << "sum2=" << sum2 << " cost=" << (t1 - t0) << std::endl;

    return 0;
}
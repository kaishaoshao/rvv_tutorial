
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

// clock()函数在windows上返回毫秒，在linux上返回微秒，毫秒级精度

// alignas是设置对齐方式，alignof是查询对齐方式。
// alignof是一个操作符，用于查询类型或变量的对齐要求。它返回一个std::size_t类型的值，表示类型或变量的对齐字节数
// alignas是一个对齐说明符，用于指定变量或类型的最小对齐要求。alignas可以用于变量声明或类型定义中，以确保所声明的变量或类型实例具有特定的对齐。对齐值必须是 2 的幂

/*
alignas(n)用来指定对象的对齐字节数。效果和__attribute__((aligned(n)))一样

问答环节：

问题1） 什么是对齐。

举例说明，某个int类型的对象，要求其存储地址的特征是4的整倍数。例如0x0000CC04。我们把地址值0x0000CC04除以4，余数0，那么这个对象的地址就是对齐的。

问题2） 为什么要对齐。

举例说明，对于int数据，硬件只能在4的倍数的地址读写，假设某int对象的地址是0x0000CC06，则硬件先读取0x0000CC04开始的4个字节，

取其0x0000CC06, 0x0000CC07。硬件然后读取0x0000CC08开始的4个字节，取其0x0000CC08, 0x0000CC09。将两次读取的有用信息拼接即可。

显然，效率不高。更严重的，硬件会报错，程序执行不下去。

问题3） x86体系下，用#pragma pack(1) 改变结构体中int成员的对齐属性，也没报错呀

只能说x86容忍性高，程序正常跑，不见得效率没有降低。
*/

struct p
{
    int a;
    char b;
    short c;
 
}__attribute__((aligned(4))) pp;

struct x
{
    int a;
    char b;
    struct p px;
    short c;
}__attribute__((aligned(8))) xx;

// sizeof(xx)=24


struct MyStruct {
    char c;
    int i;
};

struct MyStruct2 {
    char c[20];
};

struct alignas(16) AlignedStruct {
    int i;
};

struct alignas(16) AlignedStruct2 {
    char c[20];
};

int main2();

int main(int argc, char *argv[])
{
    std::cout << "Alignment of char: " << alignof(char) << std::endl;
    std::cout << "Alignment of int: " << alignof(int) << std::endl;

    std::cout << "sizeof of MyStruct: " << sizeof(MyStruct) << std::endl;
    std::cout << "Alignment of MyStruct: " << alignof(MyStruct) << std::endl;

    std::cout << "sizeof of MyStruct: " << sizeof(MyStruct2) << std::endl;
    std::cout << "Alignment of MyStruct: " << alignof(MyStruct2) << std::endl;

    AlignedStruct a;
    std::cout << "Alignment of AlignedStruct: " << alignof(a) << std::endl;

    std::cout << "sizeof of AlignedStruct2: " << sizeof(AlignedStruct2) << std::endl;
    std::cout << "Alignment of AlignedStruct2: " << alignof(AlignedStruct2) << std::endl;

    alignas(16) char sz[20];
    std::cout << "sizeof(sz)=" << sizeof(sz) << std::endl;
    std::cout << "alignof(sz)=" << alignof(sz) << std::endl;

    alignas(16) float val[52];
    std::cout << "sizeof(val)=" << sizeof(val) << std::endl;
    std::cout << "alignof(val)=" << alignof(val) << std::endl;

    alignas(32) float val2[10];
    std::cout << "sizeof(val2)=" << sizeof(val2) << std::endl;
    std::cout << "alignof(val2)=" << alignof(val2) << std::endl;

    main2();
    //-----------------------------------------------------------------------------------------

    // TODO: size为1万时，计算结果居然不正确。
    constexpr int _SIZE_ = 10000; // 5000 // 100
    // __attribute__((aligned(32))) float arr[_SIZE_];
    alignas(16) float arr[_SIZE_];		// 16-byte aligned
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
    std::cout << std::fixed << std::setprecision(2) << "1.0 + 2.0 + ... + " << _SIZE_ << "=" << sum << " cost=" << (t1 - t0) << " cycles" << std::endl;

    t0 = clock();
    double sum2 = 0;
    for(int i = 0; i < _SIZE_; i++) sum2 += arr[i];
    t1 = clock();
    std::cout << std::fixed << std::setprecision(2) << "sum2=" << sum2 << " cost=" << (t1 - t0) << " cycles" << std::endl;

    return 0;
}

 
// 每个 struct_float 类型的对象将会按照 alignof(float) 的边界对齐（通常是 4）：
struct alignas(float) struct_float
{
    // 你的定义在这里
};
 
// 每个 sse_t 类型的对象将会按照 32 字节的边界对齐：
struct alignas(32) sse_t
{
    float sse_data[4];
};
 
// 数组 cacheline 将会按照 64 字节的边界对齐：
using cacheline_t = alignas(64) char[64];
cacheline_t cacheline;
 
int main2()
{
    struct default_aligned
    {
        float data[4];
    } a, b, c;
    sse_t x, y, z;
 
    std::cout
        << "alignof(struct_float) = " << alignof(struct_float) << '\n'
        << "sizeof(sse_t) = " << sizeof(sse_t) << '\n'
        << "alignof(sse_t) = " << alignof(sse_t) << '\n'
        << "alignof(cacheline_t) = " << alignof(cacheline_t) << '\n'
        << "alignof(cacheline) = " << alignof(decltype(cacheline)) << '\n'
        << std::hex << std::showbase
        << "&a: " << &a << "\n"
           "&b: " << &b << "\n"
           "&c: " << &c << "\n"
           "&x: " << &x << "\n"
           "&y: " << &y << "\n"
           "&z: " << &z << '\n';

    return 0;
}


#include <iostream>
#include <chrono>
#include <iomanip>
#include <riscv_vector.h>

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

void vector_sum(double *data, double *result, int n) {
    // 初始化一些变量
    size_t vlmax = __riscv_vsetvlmax_e64m1();  // 获取最大向量长度
    vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0, vlmax);  // 初始化零向量
    vfloat64m1_t vec_one = __riscv_vfmv_v_f_f64m1(1, vlmax);  // 初始化零向量
    vfloat64m1_t vec_s = __riscv_vfmv_v_f_f64m1(0, vlmax);  // 初始化累加向量

    // 处理整个向量数据
    for (size_t vl; n > 0; n -= vl, data += vl) {
        vl = __riscv_vsetvl_e64m1(n);  // 根据剩余元素数设置向量长度
        // std::cout << "vl=" << vl << std::endl;

        // 从数据中加载向量
        vfloat64m1_t vec_data = __riscv_vle64_v_f64m1(data, vl);

        vbool64_t mask = __riscv_vmfne_vf_f64m1_b64(vec_data, 0, vl);

        // 执行归约求和
        vec_s = __riscv_vfmacc_vv_f64m1_tumu(mask, vec_s, vec_data, vec_one, vl);
    }

    // 最终归约求和
    vfloat64m1_t vec_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_s, vec_zero, vlmax);
    
    // 提取最终的结果
    double sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
    
    // 将结果存储到输出参数中
    *result = sum;
}

int main2() {
    // double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};  // 示例数据
    // double result;
    // vector_sum(data, &result, 5);
    double data[100];
    for(int i = 0; i < 100; i++) data[i] = static_cast<double>(i + 1);
    double result;
    auto start = std::chrono::steady_clock::now();
    vector_sum(data, &result, 100);
    auto end = std::chrono::steady_clock::now(); // 2024-11-15
    auto diff = end - start;
    std::cout << std::setprecision(7) << std::fixed << "1 cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;

    double result2 = 0;
    start = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++) result2 += data[i];
    end = std::chrono::steady_clock::now(); // 2024-11-15
    diff = end - start;
    std::cout << std::setprecision(7) << std::fixed << "2 cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    
    // 打印结果
    printf("1 Sum: %f\n", result);  // 输出：Sum: 15.000000
    printf("2 Sum: %f\n", result2);  // 输出：Sum: 15.000000

    return 0;
}


int main(int argc, char *argv[])
{
    main2();
    return 0;

    constexpr int CNT = 10;
    float array[CNT] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    float destination[CNT + 1] = { 0 };
    memcpy_vec(destination, array, CNT * sizeof(float));
    for(int i =0; i < (CNT + 1); i++) std::cout << destination[i] << ", "; std::cout << std::endl;

    // double result;
    // vector_sum(data, &result, 5);
    
    // // 打印结果
    // printf("Sum: %f\n", result);

/*    
    constexpr int CNT = 10;
    float array[CNT] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    float *p_array = array;

    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vec_zero = __riscv_vfmv_v_f_f32m8(0, vlmax);

    int n = CNT;
    for (size_t vl; n > 0; n -= vl) // , a += vl, b += vl, c += vl
    {
      vl = __riscv_vsetvl_e32m8(n);
      std::cout << "vl=" << vl << std::endl;
      vfloat32m8_t vec_s = __riscv_vle32_v_f32m8(p_array, vl);

      vfloat64m1_t vec_sum;
      vec_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_s, vec_zero, vlmax);
      double sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
    }
*/

    return 0;
}
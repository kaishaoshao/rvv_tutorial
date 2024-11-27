#include <riscv_vector.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
// add
#include <time.h>
clock_t t0, t1;
// the end.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

// #define _E32M8_
#define _E32M2_

#define N 31
// #define N 64
// #define N 128
// #define N 512

float input[N] = {-0.4325648115282207, -1.6655843782380970, 0.1253323064748307,
                  0.2876764203585489,  -1.1464713506814637, 1.1909154656429988,
                  1.1891642016521031,  -0.0376332765933176, 0.3272923614086541,
                  0.1746391428209245,  -0.1867085776814394, 0.7257905482933027,
                  -0.5883165430141887, 2.1831858181971011,  -0.1363958830865957,
                  0.1139313135208096,  1.0667682113591888,  0.0592814605236053,
                  -0.0956484054836690, -0.8323494636500225, 0.2944108163926404,
                  -1.3361818579378040, 0.7143245518189522,  1.6235620644462707,
                  -0.6917757017022868, 0.8579966728282626,  1.2540014216025324,
                  -1.5937295764474768, -1.4409644319010200, 0.5711476236581780,
                  -0.3998855777153632};

float output_golden[N] = {
    1.7491401329284098,  0.1325982188803279,  0.3252281811989881,
    -0.7938091410349637, 0.3149236145048914,  -0.5272704888029532,
    0.9322666565031119,  1.1646643544607362,  -2.0456694357357357,
    -0.6443728590041911, 1.7410657940825480,  0.4867684246821860,
    1.0488288293660140,  1.4885752747099299,  1.2705014969484090,
    -1.8561241921210170, 2.1343209047321410,  1.4358467535865909,
    -0.9173023332875400, -1.1060770780029008, 0.8105708062681296,
    0.6985430696369063,  -0.4015827425012831, 1.2687512030669628,
    -0.7836083053674872, 0.2132664971465569,  0.7878984786088954,
    0.8966819356782295,  -0.1869172943544062, 1.0131816724341454,
    0.2484350696132857};

float output[N] = {
    1.7491401329284098,  0.1325982188803279,  0.3252281811989881,
    -0.7938091410349637, 0.3149236145048914,  -0.5272704888029532,
    0.9322666565031119,  1.1646643544607362,  -2.0456694357357357,
    -0.6443728590041911, 1.7410657940825480,  0.4867684246821860,
    1.0488288293660140,  1.4885752747099299,  1.2705014969484090,
    -1.8561241921210170, 2.1343209047321410,  1.4358467535865909,
    -0.9173023332875400, -1.1060770780029008, 0.8105708062681296,
    0.6985430696369063,  -0.4015827425012831, 1.2687512030669628,
    -0.7836083053674872, 0.2132664971465569,  0.7878984786088954,
    0.8966819356782295,  -0.1869172943544062, 1.0131816724341454,
    0.2484350696132857};

void saxpy_golden(size_t n, const float a, const float *x, float *y) {
  for (size_t i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// reference https://github.com/riscv/riscv-v-spec/blob/master/example/saxpy.s
void saxpy_vec(size_t n, const float a, const float *x, float *y) {
  for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
    vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
    vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
    __riscv_vse32_v_f32m8(y, __riscv_vfmacc_vf_f32m8(vy, a, vx, vl), vl);
  }
}

void saxpy_vec2(size_t n, const float a, const float *x, float *y) {
  for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
    vl = __riscv_vsetvl_e32m2(n);
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(x, vl);
    vfloat32m2_t vy = __riscv_vle32_v_f32m2(y, vl);
    __riscv_vse32_v_f32m2(y, __riscv_vfmacc_vf_f32m2(vy, a, vx, vl), vl);
  }
}

int fp_eq(float reference, float actual, float relErr)
{
  // if near zero, do absolute error instead.
  float absErr = relErr * ((fabsf(reference) > relErr) ? fabsf(reference) : relErr);
  return fabsf(actual - reference) < absErr;
}

int main2();
int main3();
int main4();
int main5();

void command(void *pParam)
{
  std::cout << "command\n";
  auto start1 = std::chrono::high_resolution_clock::now();
  clock_t start2 = clock();
  auto start3 = std::chrono::steady_clock::now();
  // saxpy_golden(N, 55.66, input, output_golden);
  saxpy_golden(N, 55.66, input, output_golden);
  auto end3 = std::chrono::steady_clock::now();
  clock_t end2 = clock();
  auto end1 = std::chrono::high_resolution_clock::now();
  
  auto diff1 = end1 - start1;
  double duration2 = (double)(end2 - start2) * 1000 / CLOCKS_PER_SEC;
  auto diff3 = end3 - start3;
  // std::cout << std::setprecision(7) << std::fixed << "1 saxpy_golden cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  std::cout << "saxpy_golden cost:\n";
  std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  std::cout << "clock():" << duration2 << " ms" << std::endl;
  std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;

  // clock_t asm_start = clock();
  // asm volatile ("vsetvli t0, a0, e8, m8, ta, ma");
  // clock_t asm_end = clock();
  // std::cout << "asm cost : " << (double)(asm_end - asm_start) * 1000 / CLOCKS_PER_SEC << std::endl;

  start1 = std::chrono::high_resolution_clock::now();
  start2 = clock();
  start3 = std::chrono::steady_clock::now();
  asm volatile ("vsetvli t0, a0, e8, m8, ta, ma");
  // saxpy_vec(N, 55.66, input, output);
  saxpy_vec(N, 55.66, input, output);
  end3 = std::chrono::steady_clock::now();
  end2 = clock();
  end1 = std::chrono::high_resolution_clock::now();


  diff1 = end1 - start1;
  duration2 = (double)(end2 - start2) * 1000 / CLOCKS_PER_SEC;
  diff3 = end3 - start3;
  // std::cout << std::setprecision(7) << std::fixed << "2 saxpy_vec cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  std::cout << "--------------------------------\nsaxpy_vec cost:\n";
  std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  std::cout << "clock():" << duration2 << " ms" << std::endl;
  std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;

  int pass = 1;
  for (int i = 0; i < N; i++) {
    if (!fp_eq(output_golden[i], output[i], 1e-6)) {
      printf("fail, %f=!%f\n", output_golden[i], output[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("pass\n");
}

int main1() {
  /*
  saxpy_golden(N, 55.66, input, output_golden);
  saxpy_vec(N, 55.66, input, output);
  int pass = 1;
  for (int i = 0; i < N; i++) {
    if (!fp_eq(output_golden[i], output[i], 1e-6)) {
      printf("fail, %f=!%f\n", output_golden[i], output[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("pass\n");
  return (pass == 0);
  */

  // clock_t asm_start = clock();
  asm volatile ("vsetvli t0, a0, e8, m8, ta, ma");
  // clock_t asm_end = clock();
  // std::cout << "asm cost : " << (double)(asm_end - asm_start) * 1000 / CLOCKS_PER_SEC << std::endl;

  std::thread keyboard_command_process;
  keyboard_command_process = std::thread(command, nullptr);
  keyboard_command_process.join();

  return 1;
}

int main(int argc, char *argv[])
{
  std::cout << "--------------N=" << N << "--------------" << std::endl;
  // auto start1 = std::chrono::high_resolution_clock::now();
  clock_t start2 = clock();
  // auto start3 = std::chrono::steady_clock::now();
  saxpy_golden(N, 55.66, input, output_golden);
  // auto end3 = std::chrono::steady_clock::now();
  clock_t end2 = clock();
  // auto end1 = std::chrono::high_resolution_clock::now();
  
  // auto diff1 = end1 - start1;
  double duration2 = static_cast<double>((end2 - start2) * 1000) / CLOCKS_PER_SEC;
  // auto diff3 = end3 - start3;
  // // std::cout << std::setprecision(7) << std::fixed << "1 saxpy_golden cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  // std::cout << "saxpy_golden cost:\n";
  // std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  // std::cout << "clock():" << duration2 << " ms" << std::endl;
  // std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;

  // clock_t asm_start = clock();
  // asm volatile ("vsetvli t0, a0, e8, m8, ta, ma");
  // clock_t asm_end = clock();
  // std::cout << "asm cost : " << (double)(asm_end - asm_start) * 1000 / CLOCKS_PER_SEC << std::endl;

  std::cout << "saxpy_golden cost: " << duration2 << " ms" << std::endl;
#ifdef _E32M8_
  // start1 = std::chrono::high_resolution_clock::now();
  start2 = clock();
  // start3 = std::chrono::steady_clock::now();
  saxpy_vec(N, 55.66, input, output);
  // end3 = std::chrono::steady_clock::now();
  end2 = clock();
  // end1 = std::chrono::high_resolution_clock::now();

  // diff1 = end1 - start1;
  duration2 = static_cast<double>(end2 - start2) / CLOCKS_PER_SEC * 1000;
  // diff3 = end3 - start3;
  // // std::cout << std::setprecision(7) << std::fixed << "2 saxpy_vec cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  // std::cout << "--------------------------------\nsaxpy_vec cost:\n";
  // std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  // std::cout << "clock():" << duration2 << " ms" << std::endl;
  // std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;
  std::cout << "e32m8 - saxpy_vec cost: " << duration2 << " ms" << std::endl;
#else
  start2 = clock();
  saxpy_vec2(N, 55.66, input, output);
  end2 = clock();
  duration2 = static_cast<double>(end2 - start2) / CLOCKS_PER_SEC * 1000;
  std::cout << "e32m2 - saxpy_vec cost: " << duration2 << " ms" << std::endl;
#endif
  int pass = 1;
  for (int i = 0; i < N; i++) {
    if (!fp_eq(output_golden[i], output[i], 1e-6)) {
      printf("fail, %f=!%f\n", output_golden[i], output[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("pass\n");

  std::cout << "--------------------------------\n\n";  

  main5();
  return 0;
}

int main4() {
  // while(1) ;
  auto start1 = std::chrono::high_resolution_clock::now();
  clock_t start2 = clock();
  auto start3 = std::chrono::steady_clock::now();
  // saxpy_golden(N, 55.66, input, output_golden);
  saxpy_golden(N, 55.66, input, output_golden);
  auto end3 = std::chrono::steady_clock::now();
  clock_t end2 = clock();
  auto end1 = std::chrono::high_resolution_clock::now();
  
  auto diff1 = end1 - start1;
  double duration2 = (double)(end2 - start2) * 1000 / CLOCKS_PER_SEC;
  auto diff3 = end3 - start3;
  // std::cout << std::setprecision(7) << std::fixed << "1 saxpy_golden cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  std::cout << "saxpy_golden cost:\n";
  std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  std::cout << "clock():" << duration2 << " ms" << std::endl;
  std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;

  // clock_t asm_start = clock();
  // asm volatile ("vsetvli t0, a0, e8, m8, ta, ma");
  // clock_t asm_end = clock();
  // std::cout << "asm cost : " << (double)(asm_end - asm_start) * 1000 / CLOCKS_PER_SEC << std::endl;

  start1 = std::chrono::high_resolution_clock::now();
  start2 = clock();
  start3 = std::chrono::steady_clock::now();
  // saxpy_vec(N, 55.66, input, output);
  saxpy_vec(N, 55.66, input, output);
  end3 = std::chrono::steady_clock::now();
  end2 = clock();
  end1 = std::chrono::high_resolution_clock::now();


  diff1 = end1 - start1;
  duration2 = (double)(end2 - start2) * 1000 / CLOCKS_PER_SEC;
  diff3 = end3 - start3;
  // std::cout << std::setprecision(7) << std::fixed << "2 saxpy_vec cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  std::cout << "--------------------------------\nsaxpy_vec cost:\n";
  std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  std::cout << "clock():" << duration2 << " ms" << std::endl;
  std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;

  int pass = 1;
  for (int i = 0; i < N; i++) {
    if (!fp_eq(output_golden[i], output[i], 1e-6)) {
      printf("fail, %f=!%f\n", output_golden[i], output[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("pass\n");

  // main2();
  main3();

  return (pass == 0);
}

int main2() {
  t0 = clock();
  for (int i = 0; i < 5000000; i++)
    saxpy_golden(N, 55.66, input, output_golden);
  t1 = clock();
  printf("Execution time of saxpy_golden = %0.7f s\n\n", (float)(t1-t0)/CLOCKS_PER_SEC);

  t0 = clock();
  for (int i = 0; i < 5000000; i++)
    saxpy_vec(N, 55.66, input, output);
  t1 = clock();
  printf("Execution time of saxpy_vector = %0.7f s\n\n", (float)(t1-t0)/CLOCKS_PER_SEC);

  int pass = 1;
  for (int i = 0; i < N; i++) {
    if (!fp_eq(output_golden[i], output[i], 1e-6)) {
      printf("fail, %f=!%f\n", output_golden[i], output[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("pass\n");
  return (pass == 0);
}

int main3() {
  std::cout << "\n\n loop 5000000\n";
  auto start1 = std::chrono::high_resolution_clock::now();
  clock_t start2 = clock();
  auto start3 = std::chrono::steady_clock::now();
  for (int i = 0; i < 5000000; i++)
  saxpy_golden(N, 55.66, input, output_golden);
  auto end3 = std::chrono::steady_clock::now();
  clock_t end2 = clock();
  auto end1 = std::chrono::high_resolution_clock::now();
  
  auto diff1 = end1 - start1;
  double duration2 = (double)(end2 - start2) * 1000 / CLOCKS_PER_SEC;
  auto diff3 = end3 - start3;
  // std::cout << std::setprecision(7) << std::fixed << "1 saxpy_golden cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  std::cout << "saxpy_golden cost:\n";
  std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  std::cout << "clock():" << duration2 << " ms" << std::endl;
  std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;

  start1 = std::chrono::high_resolution_clock::now();
  start2 = clock();
  start3 = std::chrono::steady_clock::now();
  for (int i = 0; i < 5000000; i++)
  saxpy_vec(N, 55.66, input, output);
  end3 = std::chrono::steady_clock::now();
  end2 = clock();
  end1 = std::chrono::high_resolution_clock::now();


  diff1 = end1 - start1;
  duration2 = (double)(end2 - start2) * 1000 / CLOCKS_PER_SEC;
  diff3 = end3 - start3;
  // std::cout << std::setprecision(7) << std::fixed << "2 saxpy_vec cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  std::cout << "--------------------------------\nsaxpy_vec cost:\n";
  std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  std::cout << "clock():" << duration2 << " ms" << std::endl;
  std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;

  int pass = 1;
  for (int i = 0; i < N; i++) {
    if (!fp_eq(output_golden[i], output[i], 1e-6)) {
      printf("fail, %f=!%f\n", output_golden[i], output[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("pass\n");

  // main2();

  return (pass == 0);
}

int main5() {
  std::cout << "\n5000000 loops\n";
  // auto start1 = std::chrono::high_resolution_clock::now();
  clock_t start2 = clock();
  // auto start3 = std::chrono::steady_clock::now();
  for (int i = 0; i < 5000000; i++)
  saxpy_golden(N, 55.66, input, output_golden);
  // auto end3 = std::chrono::steady_clock::now();
  clock_t end2 = clock();
  // auto end1 = std::chrono::high_resolution_clock::now();
  
  // auto diff1 = end1 - start1;
  double duration2 = (double)(end2 - start2) * 1000 / CLOCKS_PER_SEC;
  // auto diff3 = end3 - start3;
  // // std::cout << std::setprecision(7) << std::fixed << "1 saxpy_golden cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  // std::cout << "saxpy_golden cost:\n";
  // std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  // std::cout << "clock():" << duration2 << " ms" << std::endl;
  // std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;

  std::cout << "saxpy_golden cost: " << duration2 << " ms" << std::endl;

#ifdef _E32M8_
  // start1 = std::chrono::high_resolution_clock::now();
  start2 = clock();
  // start3 = std::chrono::steady_clock::now();
  for (int i = 0; i < 5000000; i++)
  saxpy_vec(N, 55.66, input, output);
  // end3 = std::chrono::steady_clock::now();
  end2 = clock();
  // end1 = std::chrono::high_resolution_clock::now();

  // diff1 = end1 - start1;
  duration2 = (double)(end2 - start2) * 1000 / CLOCKS_PER_SEC;
  // diff3 = end3 - start3;
  // // std::cout << std::setprecision(7) << std::fixed << "2 saxpy_vec cost:" << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
  // std::cout << "--------------------------------\nsaxpy_vec cost:\n";
  // std::cout << "chrono::high_resolution_clock:" << std::chrono::duration <double, std::milli> (diff1).count() << " ms" << std::endl;
  // std::cout << "clock():" << duration2 << " ms" << std::endl;
  // std::cout << "chrono::steady_clock:" << std::chrono::duration <double, std::milli> (diff3).count() << " ms" << std::endl;

  std::cout << "e32m8 - saxpy_vec cost: " << duration2 << " ms" << std::endl;
#else
  start2 = clock();
  for (int i = 0; i < 5000000; i++)
  saxpy_vec2(N, 55.66, input, output);
  end2 = clock();
  duration2 = static_cast<double>(end2 - start2) / CLOCKS_PER_SEC * 1000;
  std::cout << "e32m2 - saxpy_vec cost: " << duration2 << " ms" << std::endl;
#endif
  int pass = 1;
  for (int i = 0; i < N; i++) {
    if (!fp_eq(output_golden[i], output[i], 1e-6)) {
      printf("fail, %f=!%f\n", output_golden[i], output[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("pass\n");

  std::cout << "--------------------------------\n\n";

  return (pass == 0);
}
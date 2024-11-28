
/*
玄铁 C906 RVV 0.7.1 向量扩展指令测试:

milkv-duo256 采用的 sg2002 芯片，使用的是玄铁 C906，支持 Vector 0.7.1 的向量扩展。

rvv_saxpy.c 将通过 vector 与普通的浮点计算进行对比，看是否能够得到相同的结果。

同时我在其中加入了时间统计，加大了循环计算次数，通过对比时间，看是否能够提升性能。

编译
$ riscv64-unknown-elf-gcc -march=rv64gcv0p7_zfh_xtheadc -static -Os ./rvv_saxpy.c -o ./rvv_saxpy_test
必须要加上 -march=rv64gcv0p7_zfh_xtheadc 选项，建议加上 -Os 选项，进行代码优化，否则无法确认 vector 运行优化速度。

反汇编
可通过以下执行确认是否能够正确生成 vector 指令。

$ riscv64-unknown-linux-musl-objdump -S -D rvv_saxpy_test > rvv_saxpy_test.asm
在 rvv_saxpy_test.asm 中搜索 vsetvli 指令，确认是否能够生成 vector 指令。

10460:   00b07057            vsetvli zero,zero,e32,m8,d1
   10464:   0207f427            vse.v   v8,(a5)
   10468:   81ff7057            vsetvl  zero,t5,t6
运行
[root@sg200x]~# ./rvv_saxpy_test 
Execution time of saxpy_golden = 1.5525820 s

Execution time of saxpy_vector = 0.3305370 s

passed
在多次循环下，vector 优化后的代码运行时间只有普通代码的 1/5 左右，性能提升非常明显。
*/



#include <riscv_vector.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
clock_t t0, t1;

#define N 31

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

int fp_eq(float reference, float actual, float relErr)
{
  // if near zero, do absolute error instead.
  float absErr = relErr * ((fabsf(reference) > relErr) ? fabsf(reference) : relErr);
  return fabsf(actual - reference) < absErr;
}


int main() {
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
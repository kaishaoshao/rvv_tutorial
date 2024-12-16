
#include <iostream>
#include <cmath>
#include <time.h>
#include <iomanip> // for std::setprecision
#include <riscv_vector.h>

int main(int argc, char *argv[])
{
    uint16 u16_array[] = { 9728, 65280, 13312, 43520, 47872, 52224 };
    int int_array[6];

    vuint16m2_t vec_u16_array = __riscv_vle16_v_u16m2(u16_array, vl);

    __riscv_vse16_v_u16m2(int_array, result, vl);

    for(size_t i = 0; i < 6; i++)
    std::cout << int_array[i]; std::cout <<endl;

    return 0;
}
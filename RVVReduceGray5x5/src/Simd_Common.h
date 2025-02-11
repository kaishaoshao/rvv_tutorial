
#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <limits.h>

// #define SIMD_SSE41_ENABLE
// #define SIMD_NEON_ENABLE  
#define SIMD_RVV_ENABLE // for risc-v vector

#if defined(SIMD_SSE41_ENABLE)
#include <nmmintrin.h>
#define SIMD_ALIGN 16

#elif defined(SIMD_RVV_ENABLE)
#include <riscv_vector.h>
#define SIMD_ALIGN 32

#else
#define SIMD_ALIGN 16

#endif

#if defined(_MSC_VER) || defined(__CODEGEARC__)

#define SIMD_INLINE __forceinline
#define SIMD_ALIGNED(x) __declspec(align(x))

#elif defined(__GNUC__)

#define SIMD_INLINE inline __attribute__ ((always_inline))
#define SIMD_ALIGNED(x) __attribute__ ((aligned(x)))

#else

#error This platform is unsupported!

#endif

namespace wx::Simd {
    // simd const
#ifdef SIMD_RVV_ENABLE    
    namespace Rvv
    {
        // sizeof(__m128)=16 sizeof(__m128i)=16 sizeof(float)=4 sizeof(__m128) / sizeof(float)=4
        // const size_t F = sizeof(__m128) / sizeof(float);
        const size_t F = __riscv_vsetvlmax_e32m1();
        const size_t DF = 2 * F;
        const size_t QF = 4 * F;
        const size_t HF = F / 2;

        // const size_t A = sizeof(__m128i);
        const size_t A = 16;//__riscv_vsetvlmax_e32m1() * sizeof(int);
        const size_t DA = 2 * A;
        const size_t QA = 4 * A;
        const size_t OA = 8 * A;
        const size_t HA = A / 2;

        void ReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation); // 2025-2-11.
    }
#endif    
    //
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        const size_t F = sizeof(__m128) / sizeof(float);
        const size_t DF = 2 * F;
        const size_t QF = 4 * F;
        const size_t HF = F / 2;

        const size_t A = sizeof(__m128i);
        const size_t DA = 2 * A;
        const size_t QA = 4 * A;
        const size_t OA = 8 * A;
        const size_t HA = A / 2;

        const __m128i K_ZERO = SIMD_MM_SET1_EPI8(0);
        /*const __m128i K_INV_ZERO = SIMD_MM_SET1_EPI8(0xFF);

        const __m128i K16_0001 = SIMD_MM_SET1_EPI16(0x0001);
        const __m128i K16_0002 = SIMD_MM_SET1_EPI16(0x0002);
        const __m128i K16_0003 = SIMD_MM_SET1_EPI16(0x0003);*/
        const __m128i K16_0004 = SIMD_MM_SET1_EPI16(0x0004);
        const __m128i K16_0005 = SIMD_MM_SET1_EPI16(0x0005);
        const __m128i K16_0006 = SIMD_MM_SET1_EPI16(0x0006);
        const __m128i K16_0008 = SIMD_MM_SET1_EPI16(0x0008);
        const __m128i K16_0020 = SIMD_MM_SET1_EPI16(0x0020);
        const __m128i K16_0080 = SIMD_MM_SET1_EPI16(0x0080);
        const __m128i K16_00FF = SIMD_MM_SET1_EPI16(0x00FF);
        const __m128i K16_0101 = SIMD_MM_SET1_EPI16(0x0101);
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        const size_t A = sizeof(uint8x16_t);
        const size_t DA = 2 * A;
        const size_t QA = 4 * A;
        const size_t OA = 8 * A;
        const size_t HA = A / 2;

        const size_t F = sizeof(float32x4_t) / sizeof(float);
        const size_t DF = 2 * F;
        const size_t QF = 4 * F;
        const size_t HF = F / 2;

        const uint8x16_t K8_00 = SIMD_VEC_SET1_EPI8(0x00);
        const uint8x16_t K8_01 = SIMD_VEC_SET1_EPI8(0x01);
        const uint8x16_t K8_02 = SIMD_VEC_SET1_EPI8(0x02);
        const uint8x16_t K8_03 = SIMD_VEC_SET1_EPI8(0x03);
        const uint8x16_t K8_04 = SIMD_VEC_SET1_EPI8(0x04);
        const uint8x16_t K8_07 = SIMD_VEC_SET1_EPI8(0x07);
        const uint8x16_t K8_08 = SIMD_VEC_SET1_EPI8(0x08);
        const uint8x16_t K8_10 = SIMD_VEC_SET1_EPI8(0x10);
        const uint8x16_t K8_20 = SIMD_VEC_SET1_EPI8(0x20);
        const uint8x16_t K8_40 = SIMD_VEC_SET1_EPI8(0x40);
        const uint8x16_t K8_80 = SIMD_VEC_SET1_EPI8(0x80);
        const uint8x16_t K8_FF = SIMD_VEC_SET1_EPI8(0xFF);

        const uint16x8_t K16_0000 = SIMD_VEC_SET1_EPI16(0x0000);
        const uint16x8_t K16_0001 = SIMD_VEC_SET1_EPI16(0x0001);
        const uint16x8_t K16_0002 = SIMD_VEC_SET1_EPI16(0x0002);
        const uint16x8_t K16_0003 = SIMD_VEC_SET1_EPI16(0x0003);
        const uint16x8_t K16_0004 = SIMD_VEC_SET1_EPI16(0x0004);
        const uint16x8_t K16_0005 = SIMD_VEC_SET1_EPI16(0x0005);
        const uint16x8_t K16_0006 = SIMD_VEC_SET1_EPI16(0x0006);
        const uint16x8_t K16_0008 = SIMD_VEC_SET1_EPI16(0x0008);
        const uint16x8_t K16_0010 = SIMD_VEC_SET1_EPI16(0x0010);
        const uint16x8_t K16_0020 = SIMD_VEC_SET1_EPI16(0x0020);
        const uint16x8_t K16_0080 = SIMD_VEC_SET1_EPI16(0x0080);
        const uint16x8_t K16_00FF = SIMD_VEC_SET1_EPI16(0x00FF);
        const uint16x8_t K16_0101 = SIMD_VEC_SET1_EPI16(0x0101);
        const uint16x8_t K16_0800 = SIMD_VEC_SET1_EPI16(0x0800);
        const uint16x8_t K16_FF00 = SIMD_VEC_SET1_EPI16(0xFF00);

        const uint32x4_t K32_00000000 = SIMD_VEC_SET1_EPI32(0x00000000);
        const uint32x4_t K32_00000001 = SIMD_VEC_SET1_EPI32(0x00000001);
        const uint32x4_t K32_00000002 = SIMD_VEC_SET1_EPI32(0x00000002);
        const uint32x4_t K32_00000003 = SIMD_VEC_SET1_EPI32(0x00000003);
        const uint32x4_t K32_00000004 = SIMD_VEC_SET1_EPI32(0x00000004);
        const uint32x4_t K32_00000005 = SIMD_VEC_SET1_EPI32(0x00000005);
        const uint32x4_t K32_00000008 = SIMD_VEC_SET1_EPI32(0x00000008);
        const uint32x4_t K32_00000010 = SIMD_VEC_SET1_EPI32(0x00000010);
        const uint32x4_t K32_000000FF = SIMD_VEC_SET1_EPI32(0x000000FF);
        const uint32x4_t K32_0000FFFF = SIMD_VEC_SET1_EPI32(0x0000FFFF);
        const uint32x4_t K32_00010000 = SIMD_VEC_SET1_EPI32(0x00010000);
        const uint32x4_t K32_00FF0000 = SIMD_VEC_SET1_EPI32(0x00FF0000);
        const uint32x4_t K32_01000000 = SIMD_VEC_SET1_EPI32(0x01000000);
        const uint32x4_t K32_08080800 = SIMD_VEC_SET1_EPI32(0x08080800);
        const uint32x4_t K32_FF000000 = SIMD_VEC_SET1_EPI32(0xFF000000);
        const uint32x4_t K32_FFFFFF00 = SIMD_VEC_SET1_EPI32(0xFFFFFF00);
        const uint32x4_t K32_FFFFFFFF = SIMD_VEC_SET1_EPI32(0xFFFFFFFF);
        const uint32x4_t K32_0123 = SIMD_VEC_SETR_EPI32(0, 1, 2, 3);

        const uint64x2_t K64_0000000000000000 = SIMD_VEC_SET1_EPI64(0x0000000000000000);

        const uint16x4_t K16_BLUE_TO_GRAY_WEIGHT = SIMD_VEC_SET1_PI16(Base::BLUE_TO_GRAY_WEIGHT);
        const uint16x4_t K16_GREEN_TO_GRAY_WEIGHT = SIMD_VEC_SET1_PI16(Base::GREEN_TO_GRAY_WEIGHT);
        const uint16x4_t K16_RED_TO_GRAY_WEIGHT = SIMD_VEC_SET1_PI16(Base::RED_TO_GRAY_WEIGHT);
        const uint32x4_t K32_BGR_TO_GRAY_ROUND_TERM = SIMD_VEC_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);

        const int16x8_t K16_Y_ADJUST = SIMD_VEC_SET1_EPI16(Base::Y_ADJUST);
        const int16x8_t K16_UV_ADJUST = SIMD_VEC_SET1_EPI16(Base::UV_ADJUST);

        const int16x4_t K16_BLUE_TO_Y_WEIGHT = SIMD_VEC_SET1_PI16(Base::BLUE_TO_Y_WEIGHT);
        const int16x4_t K16_GREEN_TO_Y_WEIGHT = SIMD_VEC_SET1_PI16(Base::GREEN_TO_Y_WEIGHT);
        const int16x4_t K16_RED_TO_Y_WEIGHT = SIMD_VEC_SET1_PI16(Base::RED_TO_Y_WEIGHT);

        const int16x4_t K16_BLUE_TO_U_WEIGHT = SIMD_VEC_SET1_PI16(Base::BLUE_TO_U_WEIGHT);
        const int16x4_t K16_GREEN_TO_U_WEIGHT = SIMD_VEC_SET1_PI16(Base::GREEN_TO_U_WEIGHT);
        const int16x4_t K16_RED_TO_U_WEIGHT = SIMD_VEC_SET1_PI16(Base::RED_TO_U_WEIGHT);

        const int16x4_t K16_BLUE_TO_V_WEIGHT = SIMD_VEC_SET1_PI16(Base::BLUE_TO_V_WEIGHT);
        const int16x4_t K16_GREEN_TO_V_WEIGHT = SIMD_VEC_SET1_PI16(Base::GREEN_TO_V_WEIGHT);
        const int16x4_t K16_RED_TO_V_WEIGHT = SIMD_VEC_SET1_PI16(Base::RED_TO_V_WEIGHT);

        const int32x4_t K32_BGR_TO_YUV_ROUND_TERM = SIMD_VEC_SET1_EPI32(Base::BGR_TO_YUV_ROUND_TERM);

        const int16x4_t K16_Y_TO_RGB_WEIGHT = SIMD_VEC_SET1_PI16(Base::Y_TO_RGB_WEIGHT);

        const int16x4_t K16_U_TO_BLUE_WEIGHT = SIMD_VEC_SET1_PI16(Base::U_TO_BLUE_WEIGHT);
        const int16x4_t K16_U_TO_GREEN_WEIGHT = SIMD_VEC_SET1_PI16(Base::U_TO_GREEN_WEIGHT);

        const int16x4_t K16_V_TO_GREEN_WEIGHT = SIMD_VEC_SET1_PI16(Base::V_TO_GREEN_WEIGHT);
        const int16x4_t K16_V_TO_RED_WEIGHT = SIMD_VEC_SET1_PI16(Base::V_TO_RED_WEIGHT);

        const int32x4_t K32_YUV_TO_BGR_ROUND_TERM = SIMD_VEC_SET1_EPI32(Base::YUV_TO_BGR_ROUND_TERM);
    }
#endif


    // simd math
    template <class T> SIMD_INLINE void Swap(T & a, T & b)
    {
        T t = a;
        a = b;
        b = t;
    }

    // simd memory
    SIMD_INLINE size_t AlignHi(size_t size, size_t align) // 不足对齐字节数的部分，填充补齐
    {
        return (size + align - 1) & ~(align - 1);
    }

    SIMD_INLINE void * AlignHi(const void * ptr, size_t align)
    {
        return (void *)((((size_t)ptr) + align - 1) & ~(align - 1));
    }

    SIMD_INLINE size_t AlignLo(size_t size, size_t align) // 不足对齐字节数的部分，舍弃
    {
        return size & ~(align - 1); // 如果align=16, 该计算相当于把低4位完全抹掉，使得size能被16整除。
    }

    SIMD_INLINE void * AlignLo(const void * ptr, size_t align)
    {
        return (void *)(((size_t)ptr) & ~(align - 1));
    }

    SIMD_INLINE void* Allocate(size_t size, size_t align = SIMD_ALIGN)
    {
#ifdef SIMD_NO_MANS_LAND
        size += 2 * SIMD_NO_MANS_LAND;
#endif
        void* ptr = NULL;
#if defined(_MSC_VER) 
        ptr = _aligned_malloc(size, align);
#elif defined(__MINGW32__) || defined(__MINGW64__)
        ptr = __mingw_aligned_malloc(size, align);
#elif defined(__GNUC__)
        align = AlignHi(align, sizeof(void*));
        size = AlignHi(size, align);
        int result = ::posix_memalign(&ptr, align, size);
        if (result != 0)
            ptr = NULL;
#else
        ptr = malloc(size);
#endif
#ifdef SIMD_ALLOCATE_ERROR_MESSAGE
        if (ptr == NULL)
            std::cout << "The function posix_memalign can't allocate " << size << " bytes with align " << align << " !" << std::endl << std::flush;
#endif
#ifdef SIMD_ALLOCATE_ASSERT
        assert(ptr);
#endif
#ifdef SIMD_NO_MANS_LAND
        if (ptr)
        {
#if !defined(NDEBUG) && SIMD_NO_MANS_LAND >= 16
            * (size_t*)ptr = size - 2 * SIMD_NO_MANS_LAND;
            memset((char*)ptr + sizeof(size_t), NO_MANS_LAND_WATERMARK, SIMD_NO_MANS_LAND - sizeof(size_t));
            memset((char*)ptr + size - SIMD_NO_MANS_LAND, NO_MANS_LAND_WATERMARK, SIMD_NO_MANS_LAND);
#endif
            ptr = (char*)ptr + SIMD_NO_MANS_LAND;
        }
#endif
        return ptr;
    }

    SIMD_INLINE void Free(void * ptr)
    {
#ifdef SIMD_NO_MANS_LAND
        if (ptr)
        {
            ptr = (char*)ptr - SIMD_NO_MANS_LAND;
#if !defined(NDEBUG) && SIMD_NO_MANS_LAND >= 16
            size_t size = *(size_t*)ptr;
            char* nose = (char*)ptr + sizeof(size_t), *tail = (char*)ptr + SIMD_NO_MANS_LAND + size;
            for (size_t i = 0, n = SIMD_NO_MANS_LAND - sizeof(size_t); i < n; ++i)
                assert(nose[i] == NO_MANS_LAND_WATERMARK);
            for (size_t i = 0, n = SIMD_NO_MANS_LAND; i < n; ++i)
                assert(tail[i] == NO_MANS_LAND_WATERMARK);
#endif  
        }
#endif
#if defined(_MSC_VER) 
        _aligned_free(ptr);
#elif defined(__MINGW32__) || defined(__MINGW64__)
        return __mingw_aligned_free(ptr);
#else
        free(ptr);
#endif
    }


}
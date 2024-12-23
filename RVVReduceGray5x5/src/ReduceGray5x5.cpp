#include <assert.h>

#include <iostream>
#include <cmath>
#include <time.h>
#include <iomanip> // for std::setprecision


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// #include <riscv_vector.h>

#define SIMD_SSE41_ENABLE


#ifdef SIMD_SSE41_ENABLE
#include <nmmintrin.h>
#define SIMD_ALIGN 16
#endif

#if defined(_MSC_VER) || defined(__CODEGEARC__)

#define SIMD_INLINE __forceinline

#elif defined(__GNUC__)

#define SIMD_INLINE inline __attribute__ ((always_inline))
#define SIMD_ALIGNED(x) __attribute__ ((aligned(x)))

#else

#error This platform is unsupported!

#endif

namespace Simd
{
    template <class T> SIMD_INLINE void Swap(T & a, T & b)
    {
        T t = a;
        a = b;
        b = t;
    }

    const size_t A = sizeof(__m128i);

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

#define SIMD_CHAR_AS_LONGLONG(a) (((long long)a) & 0xFF)

#define SIMD_SHORT_AS_LONGLONG(a) (((long long)a) & 0xFFFF)

#define SIMD_LL_SET1_EPI8(a) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(a) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 16) | (SIMD_CHAR_AS_LONGLONG(a) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 32) | (SIMD_CHAR_AS_LONGLONG(a) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 48) | (SIMD_CHAR_AS_LONGLONG(a) << 56)

#define SIMD_LL_SET1_EPI16(a) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(a) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(a) << 32) | (SIMD_SHORT_AS_LONGLONG(a) << 48)

#define SIMD_MM_SET1_EPI8(a) \
    {SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a)}

#define SIMD_MM_SET1_EPI16(a) \
    {SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a)}

}



//------------------------------------------------------------------------------------------------

namespace Simd
{
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

        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t)*(5 * width + A));
                    in0 = (uint16_t*)_p; // 第0行
                    in1 = in0 + width; // 第1行
                    out0 = in1 + width; // 第2行
                    out1 = out0 + width; // 第3行
                    dst = out1 + width + HA; // 第4行，第HA序号开始
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * in0;
                uint16_t * in1;
                uint16_t * out0;
                uint16_t * out1;
                uint16_t * dst;
            private:
                void *_p;
            };
        }

        template <bool align> SIMD_INLINE __m128 Load(const float * p);

        template <> SIMD_INLINE __m128 Load<false>(const float * p)
        {
            return _mm_loadu_ps(p);
        }

        template <> SIMD_INLINE __m128 Load<true>(const float * p)
        {
            return _mm_load_ps(p);
        }

        template <bool align> SIMD_INLINE __m128i Load(const __m128i * p);

        template <> SIMD_INLINE __m128i Load<false>(const __m128i * p)
        {
            return _mm_loadu_si128(p);
        }

        template <> SIMD_INLINE __m128i Load<true>(const __m128i * p)
        {
            return _mm_load_si128(p);
        }

        template <bool align> SIMD_INLINE void Store(float* p, __m128 a);

        template <> SIMD_INLINE void Store<false>(float* p, __m128 a)
        {
            _mm_storeu_ps(p, a);
        }

        template <> SIMD_INLINE void Store<true>(float* p, __m128 a)
        {
            _mm_store_ps(p, a);
        }

        SIMD_INLINE void Store(float* ptr, __m128 val, size_t size)
        {
            SIMD_ALIGNED(16) float buf[F];
            _mm_store_ps(buf, val);
            for (size_t i = 0; i < size; ++i)
                ptr[i] = buf[i];
        }

        template <bool align> SIMD_INLINE void Store(__m128i * p, __m128i a);

        template <> SIMD_INLINE void Store<false>(__m128i * p, __m128i a)
        {
            // Store 128-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
            _mm_storeu_si128(p, a);
        }

        template <> SIMD_INLINE void Store<true>(__m128i * p, __m128i a)
        {
            // p 指向目标内存位置的指针，数据将被存储到这个地址。此地址必须是 16 字节对齐的（即内存地址应为 16 的倍数）
            // void _mm_store_si128 (__m128i* mem_addr, __m128i a)
            // _mm_store_si128 会将 a 向量中的 128 位数据（即 16 个字节）存储到由 mem_address 指定的内存位置。
            _mm_store_si128(p, a);
        }

        template <bool compensation> SIMD_INLINE __m128i DivideBy256(__m128i value);

        template <> SIMD_INLINE __m128i DivideBy256<true>(__m128i value)
        {
            return _mm_srli_epi16(_mm_add_epi16(value, K16_0080), 8);
        }

        template <> SIMD_INLINE __m128i DivideBy256<false>(__m128i value)
        {
            return _mm_srli_epi16(value, 8);
        }

        // @brief 每次处理8个uint8_t数据,扩展为uint16_t的数据，加载到__m128i类型的向量
        SIMD_INLINE __m128i LoadUnpacked(const void * src)
        {
            // _mm_loadl_epi64: 从内存中加载一个64位整数到SIMD寄存器的低64位，同时将高64位设置为0。
            // 具体来说，这个函数会读取传入指针指向的内存地址中的64位数据，并将其放置在结果向量的低64位，而向量中的高64位则会被清零

            // _mm_unpacklo_epi8(a, b) 会从两个 128 位的向量 a 和 b 中的低 64 位（即每个向量的前 8 个字节）提取数据，
            // 并交替地将它们放入一个新的 128 位的向量中

            // K_ZERO是一个__m128i类型的向量: 0x 00 00 00 00 00 00 00 00, 0x 00 00 00 00 00 00 00 00
            // 总结：_mm_unpacklo_epi8的作用是将uint8_t的数据扩展为uint16_t.
            // 对于SSE和NEON，将8个uint8_t的数据加载到一个向量寄存器，然后扩展为一个包含8个uint16_t的向量。
            // 每个数据的高位填充为0x00
            return _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)src), K_ZERO);
        }

        template<bool align> SIMD_INLINE void FirstRow5x5(__m128i src, Buffer & buffer, size_t offset)
        {
            Store<align>((__m128i*)(buffer.in0 + offset), src);
            // _mm_mullo_epi16 对两个 128 位的整数向量 a 和 b 中的每对 16 位整数执行乘法操作，并将结果保存在一个新的 128 位整数向量中。
            // 两个 16 位整数相乘后，结果的低 16 位会被保存，超出 16 位的部分会被丢弃
            // K16_0005 = 0x0005 0x0005 0x0005 0x0005 0x0005 0x0005 0x0005 0x0005表示的是8个16bit(0x0005)的128bit向量
            Store<align>((__m128i*)(buffer.in1 + offset), _mm_mullo_epi16(src, K16_0005));
        }

        // @brief 总共处理16个数据，分两次完全一样的操作进行，每次计算8个数据。
        // 具体为：8个字节的数据跟0交替混编后（扩展为8个16bit的uint16_t类型的数据）存入in0，8个uint16_t的数据乘以5后存入in1
        // @param src 图像数据
        // @param buffer 新分配的内存，用于存储中间计算结果
        // @param offset 偏移量对于sse和neon为16
        // @return void
        template<bool align> SIMD_INLINE void FirstRow5x5(const uint8_t * src, Buffer & buffer, size_t offset)
        {
            // step1: 从src + offset开始的内存中加载8个字节的数据，这8个字节的图像数据和8个字节的0交替插入(其实就是 uint8_t 转 uint16_t)，
            // 放入一个新的 128 位的向量中；
            // step2: 将新的 128 位的向量中的数据存入buffer的第0行in0中；
            // step3: 将新的128bit向量，取出每个16bit的数据乘以0x0005,截断取低16位存储到buffer的第1行in1中。
            FirstRow5x5<align>(LoadUnpacked(src + offset), buffer, offset);
            // step4: 接下来偏移量再增加8，然后把上面的step1~step3再来一次 (因为向量一次只能处理128bit的数据，
            // 而数据扩展为2字节的uint16_t之后，只能每次处理8个数据，因此这里需要分两次操作，同时循环展开可以同时调用2个ALU)
            offset += HA;
            FirstRow5x5<align>(LoadUnpacked(src + offset), buffer, offset);
        }

        template<bool align> SIMD_INLINE void MainRowY5x5(__m128i odd, __m128i even, Buffer & buffer, size_t offset)
        {
            __m128i cp = _mm_mullo_epi16(odd, K16_0004);
            __m128i c0 = Load<align>((__m128i*)(buffer.in0 + offset));
            __m128i c1 = Load<align>((__m128i*)(buffer.in1 + offset));
            Store<align>((__m128i*)(buffer.dst + offset), _mm_add_epi16(even, _mm_add_epi16(c1, _mm_add_epi16(cp, _mm_mullo_epi16(c0, K16_0006)))));
            Store<align>((__m128i*)(buffer.out1 + offset), _mm_add_epi16(c0, cp));
            Store<align>((__m128i*)(buffer.out0 + offset), even);
        }

        template<bool align> SIMD_INLINE void MainRowY5x5(const uint8_t *odd, const uint8_t *even, Buffer & buffer, size_t offset)
        {
            // LoadUnpacked的作用：8个uint8_t的数据，扩展为uint16_t的数据后加载到__m128i类型的向量
            MainRowY5x5<align>(LoadUnpacked(odd + offset), LoadUnpacked(even + offset), buffer, offset);
            offset += HA;
            MainRowY5x5<align>(LoadUnpacked(odd + offset), LoadUnpacked(even + offset), buffer, offset);
        }

        template <bool align, bool compensation> SIMD_INLINE __m128i MainRowX5x5(uint16_t * dst)
        {
            __m128i t0 = _mm_loadu_si128((__m128i*)(dst - 2));
            __m128i t1 = _mm_loadu_si128((__m128i*)(dst - 1));
            __m128i t2 = Load<align>((__m128i*)dst);
            __m128i t3 = _mm_loadu_si128((__m128i*)(dst + 1));
            __m128i t4 = _mm_loadu_si128((__m128i*)(dst + 2));
            t2 = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(t2, K16_0006), _mm_mullo_epi16(_mm_add_epi16(t1, t3), K16_0004)), _mm_add_epi16(t0, t4));
            return DivideBy256<compensation>(t2);
        }

        template <bool align, bool compensation> SIMD_INLINE void MainRowX5x5(Buffer & buffer, size_t offset, uint8_t *dst)
        {
            __m128i t0 = MainRowX5x5<align, compensation>(buffer.dst + offset);
            __m128i t1 = MainRowX5x5<align, compensation>(buffer.dst + offset + HA);
            // K16_00FF = 0x00FF 0x00FF 0x00FF 0x00FF 0x00FF 0x00FF 0x00FF 0x00FF表示的是8个16bit(0x00FF)的128bit向量
            // _mm_and_si128 对两个128位的寄存器进行按位与操作dst[127:0] := (a[127:0] AND b[127:0])
            t0 = _mm_packus_epi16(_mm_and_si128(_mm_packus_epi16(t0, t1), K16_00FF), K_ZERO);
            _mm_storel_epi64((__m128i*)dst, t0);
        }

        template <bool compensation> void ReduceGray5x5(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= A);

            size_t alignedWidth = Simd::AlignLo(srcWidth, A);
            size_t bufferDstTail = Simd::AlignHi(srcWidth - A, 2);

            Buffer buffer(Simd::AlignHi(srcWidth, A));

            for (size_t col = 0; col < alignedWidth; col += A)
                FirstRow5x5<true>(src, buffer, col);
            if (alignedWidth != srcWidth) // 针对宽度未对齐的情况，把最后A个元素再按照非对齐方式存储计算一次
                FirstRow5x5<false>(src, buffer, srcWidth - A);
            src += srcStride;

            for (size_t row = 1; row <= srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t *odd = src - (row < srcHeight ? 0 : srcStride);
                const uint8_t *even = odd + (row < srcHeight - 1 ? srcStride : 0);

                for (size_t col = 0; col < alignedWidth; col += A)
                    MainRowY5x5<true>(odd, even, buffer, col);
                if (alignedWidth != srcWidth)
                    MainRowY5x5<false>(odd, even, buffer, srcWidth - A);

                Swap(buffer.in0, buffer.out0);
                Swap(buffer.in1, buffer.out1);

                buffer.dst[-2] = buffer.dst[0];
                buffer.dst[-1] = buffer.dst[0];
                buffer.dst[srcWidth] = buffer.dst[srcWidth - 1];
                buffer.dst[srcWidth + 1] = buffer.dst[srcWidth - 1];

                for (size_t srcCol = 0, dstCol = 0; srcCol < alignedWidth; srcCol += A, dstCol += HA)
                    MainRowX5x5<true, compensation>(buffer, srcCol, dst + dstCol);
                if (alignedWidth != srcWidth)
                    MainRowX5x5<false, compensation>(buffer, bufferDstTail, dst + dstWidth - HA);
            }
        }

        void ReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray5x5<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray5x5<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif
}

/*
root@ubuntu:~/lwx_dataset/compile_humble/root/lwx/rvv_tutorial/RVVReduceGray5x5/build# ./test1
A=16
alignedWidth=640
bufferDstTail=624
test1=16
test2=0
test3=16
test4=16
test5=32
test6=32
*/

int main(int argc, char *argv[])
{
    size_t srcWidth = 640;
    size_t srcHeight = 480;
/*    
    std::cout << "A=" << Simd::A << std::endl;

    size_t alignedWidth = Simd::AlignLo(srcWidth, Simd::A);
    std::cout << "alignedWidth=" << alignedWidth << std::endl;

    size_t bufferDstTail = Simd::AlignHi(srcWidth - Simd::A, 2);
    std::cout << "bufferDstTail=" << bufferDstTail << std::endl;
    
    std::cout << "test1=" <<  Simd::AlignLo(17, Simd::A) << std::endl;
    std::cout << "test2=" <<  Simd::AlignLo(15, Simd::A) << std::endl;
    std::cout << "test3=" <<  Simd::AlignHi(15, Simd::A) << std::endl;

    std::cout << "test4=" <<  Simd::AlignLo(31, Simd::A) << std::endl;
    std::cout << "test5=" <<  Simd::AlignHi(31, Simd::A) << std::endl;
    std::cout << "test6=" <<  Simd::AlignHi(17, Simd::A) << std::endl;
*/

#if 0 // for test. on 2024-12-18.
    {
      // for test.
      char sz[100] = { 0 };
      sprintf(sz, "%lld", t_ns);
      std::string strStamp = sz;
      std::string image_path = strStamp + ".png";
      std::cout << "image_path=" << image_path << std::endl;
      imwrite(image_path.c_str(), image0);
    }
#endif

    // cv::IMREAD_GRAYSCALE
    // cv::Mat image_color = cv::imread("house.jpg", cv::IMREAD_COLOR);
    cv::Mat image_gray = cv::imread("../assets/1726299898318057216.png", cv::IMREAD_GRAYSCALE);

    if (image_gray.empty() ) {
		std::cout << "can't read image!!" << std::endl;
		return -1;
	}
/*
    cv::imshow("raw_img", image_gray);

    // delay：等待时间（单位为毫秒）。如果是 <=0，表示无限等待，直到用户按下某个键。如果大于 0，表示等待指定的毫秒数后自动返回。
    // 如果是正整数，表示等待该时间后自动返回，若按下键则返回按键的ASCII码。
    // 如果是 0，则表示程序会无限等待直到用户按下一个键。
    // 返回值：
    // 返回按下键的 ASCII 码值（或Unicode值），如果没有按键，返回 -1。

    // cv::waitKey(0); // waits for a key event infinitely (when delay≤0 ) or for delay milliseconds
    cv::waitKey(100); // wait 100 ms
*/

    clock_t clock_start = clock();
    cv::Mat mipmap(cv::Size(srcWidth + srcWidth / 2, srcHeight), CV_8UC1);
    image_gray.copyTo(mipmap(cv::Rect(0, 0, srcWidth, srcHeight)));

    int stride = mipmap.step[0]; // 输出图像第0行的总字节数。本例即为960.
    std::cout << "stride=" << stride << std::endl;
    int srcW = srcWidth;
    int srcH = srcHeight;
    int dstW = srcW / 2;
    int dstH = srcH / 2;
    uint8_t* src_begin = mipmap.data;
    uint8_t* dst_begin = src_begin + srcW;
    Simd::Sse41::ReduceGray5x5(src_begin, srcW, srcH, stride, dst_begin, dstW, dstH, stride, 1);
    for(int Level=1; Level<4; Level++)
    {
        srcW = dstW;
        srcH = dstH;
        dstW /= 2;
        dstH /= 2;
        src_begin = dst_begin;
        dst_begin += srcH*stride;
        Simd::Sse41::ReduceGray5x5(src_begin, srcW, srcH, stride, dst_begin, dstW, dstH, stride, 1);
    }

    clock_t clock_end = clock();
    std::cout << std::fixed << std::setprecision(3) << "risc-v vector image pyramid:" << " clock=" << (clock_end - clock_start) << " cycles\n";

    cv::imshow("mipmap", mipmap);
    cv::waitKey(0);


    return 0;
}
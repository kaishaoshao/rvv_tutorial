
#include "wx_lkpyramid.h"

#include <tbb/parallel_for.h>

#include <opencv2/core/utility.hpp>

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))
using namespace cv;

namespace wx::liu
{


static void calcScharrDeriv(const cv::Mat& src, cv::Mat& dst)
{
    using namespace cv;
    // using cv::detail::deriv_type;
    int rows = src.rows, cols = src.cols, cn = src.channels(), depth = src.depth();
    CV_Assert(depth == CV_8U);
    dst.create(rows, cols, CV_MAKETYPE(DataType<deriv_type>::depth, cn*2));
#if 0    
    parallel_for_(Range(0, rows), ScharrDerivInvoker(src, dst), cv::getNumThreads());
#else    
    tbb::parallel_for(tbb::blocked_range<size_t>(0, rows), ScharrDerivInvoker(src, dst));
#endif    
}

// void ScharrDerivInvoker::operator()(const Range& range) const
void ScharrDerivInvoker::operator()(const tbb::blocked_range<size_t>& range) const
{
    using namespace cv;
    // using cv::detail::deriv_type;
    int rows = src.rows, cols = src.cols, cn = src.channels(), colsn = cols*cn;

    int x, y, delta = (int)alignSize((cols + 2)*cn, 16);
    cv::AutoBuffer<deriv_type> _tempBuf(delta*2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf.data() + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);

#if CV_SIMD128
    v_int16x8 c3 = v_setall_s16(3), c10 = v_setall_s16(10);
#endif

    // for( y = range.start; y < range.end; y++ )
    for( y = range.begin(); y != range.end(); y++ )
    {
        const uchar* srow0 = src.ptr<uchar>(y > 0 ? y-1 : rows > 1 ? 1 : 0);
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow2 = src.ptr<uchar>(y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
        deriv_type* drow = (deriv_type *)dst.ptr<deriv_type>(y);

        // do vertical convolution
        x = 0;
#if CV_SIMD128
        {
            for( ; x <= colsn - 8; x += 8 )
            {
                v_int16x8 s0 = v_reinterpret_as_s16(v_load_expand(srow0 + x));
                v_int16x8 s1 = v_reinterpret_as_s16(v_load_expand(srow1 + x));
                v_int16x8 s2 = v_reinterpret_as_s16(v_load_expand(srow2 + x));

                v_int16x8 t1 = s2 - s0;
                v_int16x8 t0 = v_mul_wrap(s0 + s2, c3) + v_mul_wrap(s1, c10);

                v_store(trow0 + x, t0);
                v_store(trow1 + x, t1);
            }
        }
#endif

        for( ; x < colsn; x++ )
        {
            int t0 = (srow0[x] + srow2[x])*3 + srow1[x]*10;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        // make border
        int x0 = (cols > 1 ? 1 : 0)*cn, x1 = (cols > 1 ? cols-2 : 0)*cn;
        for( int k = 0; k < cn; k++ )
        {
            trow0[-cn + k] = trow0[x0 + k]; trow0[colsn + k] = trow0[x1 + k];
            trow1[-cn + k] = trow1[x0 + k]; trow1[colsn + k] = trow1[x1 + k];
        }

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;
#if CV_SIMD128
        {
            for( ; x <= colsn - 8; x += 8 )
            {
                v_int16x8 s0 = v_load(trow0 + x - cn);
                v_int16x8 s1 = v_load(trow0 + x + cn);
                v_int16x8 s2 = v_load(trow1 + x - cn);
                v_int16x8 s3 = v_load(trow1 + x);
                v_int16x8 s4 = v_load(trow1 + x + cn);

                v_int16x8 t0 = s1 - s0;
                v_int16x8 t1 = v_mul_wrap(s2 + s4, c3) + v_mul_wrap(s3, c10);

                v_store_interleave((drow + x*2), t0, t1);
            }
        }
#endif
        for( ; x < colsn; x++ )
        {
            deriv_type t0 = (deriv_type)(trow0[x+cn] - trow0[x-cn]);
            deriv_type t1 = (deriv_type)((trow1[x+cn] + trow1[x-cn])*3 + trow1[x]*10);
            drow[x*2] = t0; drow[x*2+1] = t1;
        }
    }
}

LKTrackerInvoker::LKTrackerInvoker(
    const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
    const Point2f* _prevPts, Point2f* _nextPts,
    uchar* _status, float* _err,
    Size _winSize, TermCriteria _criteria,
    int _level, int _maxLevel, int _flags, float _minEigThreshold )
{
    prevImg = &_prevImg;
    prevDeriv = &_prevDeriv;
    nextImg = &_nextImg;
    prevPts = _prevPts;
    nextPts = _nextPts;
    status = _status;
    err = _err;
    winSize = _winSize;
    criteria = _criteria;
    level = _level;
    maxLevel = _maxLevel;
    flags = _flags;
    minEigThreshold = _minEigThreshold;
}

#if defined __arm__ && !CV_NEON
typedef int64 acctype;
typedef int itemtype;
#else
typedef float acctype;
typedef float itemtype;
#endif

/*
Lucas-Kanade光流法（Lucas-Kanade Optical Flow） 是一种基于局部图像窗口的光流估计方法
Lucas-Kanade方法通过假设在小的时间间隔内，图像中的局部区域（通常是一个小窗口）内的所有像素点的运动是相同的，
因此通过这种假设来估计图像中每个像素的运动。具体来说，它通过两帧图像（当前帧和前一帧）之间的亮度变化来估计像素的运动，
推算出从前一帧到当前帧的光流。这个过程就是正向光流的计算。
*/
// @brief: tbb多线程并行调用仿函数来对多个点进行光流跟踪
// Lucas-Kanade光流追踪算法的实现，用于计算连续帧之间特征点的位移。
// 在每次迭代中，它通过计算图像的局部梯度和协方差矩阵来估算特征点的运动，并通过最小化误差来更新光流估计。
// void LKTrackerInvoker::operator()(const Range& range) const
void LKTrackerInvoker::operator()(const tbb::blocked_range<size_t>& range) const
{
/*
    CV_INSTRUMENT_REGION(); //? 用于性能分析和标记代码执行区域
*/
    // 计算半窗口大小 (winSize.width 和 winSize.height 是窗口的宽高)
    Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
    // 获取前一帧图像（prevImg）、当前帧图像（nextImg）和前一帧图像的梯度（prevDeriv）
    const Mat& I = *prevImg;
    const Mat& J = *nextImg;
    const Mat& derivI = *prevDeriv;

    // cn是图像的通道数，cn2是通道数的两倍（用于梯度）
    int j, cn = I.channels(), cn2 = cn*2;
    cv::AutoBuffer<deriv_type> _buf(winSize.area()*(cn + cn2)); // 创建一个缓冲区用于存储图像窗口和梯度信息
    int derivDepth = cv::DataType<deriv_type>::depth; // 获取梯度数据的深度: CV_16S,表示每个数据是16-bit signed类型

    // 创建用于存储窗口内图像数据的矩阵（包括图像和梯度）：创建图像窗口的缓冲区和梯度窗口的缓冲区
    // 对于默认参数来说IWinBuf \ derivIWinBuf是 21*21大小的Mat.
    Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), _buf.data());
    Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2), _buf.data() + winSize.area()*cn);

    // 遍历指定范围内的特征点：遍历range范围内的点的序号，对每一个点进行光流跟踪。
    // for( int ptidx = range.start; ptidx < range.end; ptidx++ )
    for( int ptidx = range.begin(); ptidx != range.end(); ptidx++ )
    {
        // std::cout << "level=" << level << " ptidx=" << ptidx << std::endl;
        // 获取当前特征点的位置，考虑不同层级的缩放 // 计算每个特征点在当前金字塔层的坐标 // 点坐标缩小到对应层
        Point2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
        Point2f nextPt;
        if( level == maxLevel )
        {
            if( flags & OPTFLOW_USE_INITIAL_FLOW )
                nextPt = nextPts[ptidx]*(float)(1./(1 << level));
            else // 如果没有使用初始光流，则将当前点位置设为上一帧的点位置
                nextPt = prevPt; //对于最高层来说，把前一帧的点作为当前帧跟踪点的坐标初值进行赋值
        }
        else // 如果不是最后一层，将特征点位置缩放到当前层级
            nextPt = nextPts[ptidx]*2.f; // 对于其它层， 直接把点坐标乘以2.0，作为初值
        nextPts[ptidx] = nextPt; // 给当前帧追踪点数组对应序号的点赋初值

        Point2i iprevPt, inextPt;
        // 对特征点进行半窗口偏移: 减去winSize的一半
        prevPt -= halfWin;
        // 向下取整
        iprevPt.x = cvFloor(prevPt.x);
        iprevPt.y = cvFloor(prevPt.y);

        // 判断特征点是否越界（超出了图像范围）
        if( iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
            iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows )
        {
            std::cout << "[wx] iprevPt out of boundary.\n";
            // 如果点的坐标超出界限，并且是最底层，认为该点跟踪失败，skip.
            if( level == 0 )
            {
                if( status )
                    status[ptidx] = false;
                if( err )
                    err[ptidx] = 0;
            }
            continue;
        }

        // 计算窗口内像素的插值权重
        float a = prevPt.x - iprevPt.x;
        float b = prevPt.y - iprevPt.y;
        const int W_BITS = 14, W_BITS1 = 14; // 定义权重的位数
        const float FLT_SCALE = 1.f/(1 << 20); // 定义缩放因子，用于提高计算精度
        // const int W_BITS = 0, W_BITS1 = 0; // 定义权重的位数
        // const float FLT_SCALE = 1.; // also ok
        // 计算窗口权重
        int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
        int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
        int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
        int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        // 定义步长和加权矩阵的其他变量
        // step：是cv::Mat类的一个属性，等同于step[0]
        // step[0]: 图像一行元素的字节数，图1中，step[0]就是任意一行，比如row0，所有元素的字节数
        // elemSize1()：图像中一个元素中的一个通道的字节数
        // 梯度图derivI的第一行元素字节数 除以一个通道的字节数
        // 即: w * channels * sizeof(data_type) / sizeof(data_type)
        int dstep = (int)(derivI.step/derivI.elemSize1()); // 获取梯度图像的步长
        int stepI = (int)(I.step/I.elemSize1()); // 获取前一帧图像的步长
        int stepJ = (int)(J.step/J.elemSize1()); // 获取当前帧图像的步长
        // 初始化协方差矩阵的元素
        acctype iA11 = 0, iA12 = 0, iA22 = 0;
        float A11, A12, A22;

#if CV_SIMD128 && !CV_NEON
        // SIMD优化代码（仅适用于x86架构），加速图像处理
        // 数据类型，例如cv::v_int8x16 表示int8_t的基本数据，16个。即类型在前，数量在后。
        v_int16x8 qw0((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
        v_int16x8 qw1((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
        v_int32x4 qdelta_d = v_setall_s32(1 << (W_BITS1-1));
        v_int32x4 qdelta = v_setall_s32(1 << (W_BITS1-5-1));
        v_float32x4 qA11 = v_setzero_f32(), qA12 = v_setzero_f32(), qA22 = v_setzero_f32();
#endif

#if CV_NEON

        // SIMD优化代码（适用于ARM架构）
        float CV_DECL_ALIGNED(16) nA11[] = { 0, 0, 0, 0 }, nA12[] = { 0, 0, 0, 0 }, nA22[] = { 0, 0, 0, 0 };
        const int shifter1 = -(W_BITS - 5); //negative so it shifts right
        const int shifter2 = -(W_BITS);

        const int16x4_t d26 = vdup_n_s16((int16_t)iw00);
        const int16x4_t d27 = vdup_n_s16((int16_t)iw01);
        const int16x4_t d28 = vdup_n_s16((int16_t)iw10);
        const int16x4_t d29 = vdup_n_s16((int16_t)iw11);
        const int32x4_t q11 = vdupq_n_s32((int32_t)shifter1);
        const int32x4_t q12 = vdupq_n_s32((int32_t)shifter2);

#endif

        // 从前一帧图像中提取特征点所在的窗口，并计算该窗口的梯度的协方差矩阵
        // 该小窗口是以prevPt为中心构建的winSize大小的窗口
        // 计算该小窗口内每个点的加权像素值和梯度值
        // extract the patch from the first image, compute covariation matrix of derivatives
        int x, y;
        // 按行遍历该小窗口的灰度值和梯度值
        for( y = 0; y < winSize.height; y++ )
        {
            // 获取小窗口的当前行（第y行）的图像强度值指针和梯度值指针
            const uchar* src = I.ptr() + (y + iprevPt.y)*stepI + iprevPt.x*cn;
            const deriv_type* dsrc = derivI.ptr<deriv_type>() + (y + iprevPt.y)*dstep + iprevPt.x*cn2;

            // Iptr 和 dIptr 是用于存储当前行的加权像素值和梯度信息的指针，分别指向 IWinBuf 和 derivIWinBuf（这两个矩阵用于存储图像窗口和梯度窗口的加权值）。
            deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y); // winSize大小的窗口的图像数据指针
            deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y); // winSize大小的窗口的梯度数据指针

            x = 0;

#if CV_SIMD128 && !CV_NEON
            for( ; x <= winSize.width*cn - 8; x += 8, dsrc += 8*2, dIptr += 8*2 )
            {
                v_int32x4 t0, t1;
                v_int16x8 v00, v01, v10, v11, t00, t01, t10, t11;

                v00 = v_reinterpret_as_s16(v_load_expand(src + x));
                v01 = v_reinterpret_as_s16(v_load_expand(src + x + cn));
                v10 = v_reinterpret_as_s16(v_load_expand(src + x + stepI));
                v11 = v_reinterpret_as_s16(v_load_expand(src + x + stepI + cn));

                v_zip(v00, v01, t00, t01);
                v_zip(v10, v11, t10, t11);

                t0 = v_dotprod(t00, qw0, qdelta) + v_dotprod(t10, qw1);
                t1 = v_dotprod(t01, qw0, qdelta) + v_dotprod(t11, qw1);
                t0 = t0 >> (W_BITS1-5);
                t1 = t1 >> (W_BITS1-5);
                v_store(Iptr + x, v_pack(t0, t1));

                v00 = v_reinterpret_as_s16(v_load(dsrc));
                v01 = v_reinterpret_as_s16(v_load(dsrc + cn2));
                v10 = v_reinterpret_as_s16(v_load(dsrc + dstep));
                v11 = v_reinterpret_as_s16(v_load(dsrc + dstep + cn2));

                v_zip(v00, v01, t00, t01);
                v_zip(v10, v11, t10, t11);

                t0 = v_dotprod(t00, qw0, qdelta_d) + v_dotprod(t10, qw1);
                t1 = v_dotprod(t01, qw0, qdelta_d) + v_dotprod(t11, qw1);
                t0 = t0 >> W_BITS1;
                t1 = t1 >> W_BITS1;
                v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
                v_store(dIptr, v00);

                v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(v00))));
                v_expand(v00, t1, t0);

                v_float32x4 fy = v_cvt_f32(t0);
                v_float32x4 fx = v_cvt_f32(t1);

                qA22 = v_muladd(fy, fy, qA22);
                qA12 = v_muladd(fx, fy, qA12);
                qA11 = v_muladd(fx, fx, qA11);

                v00 = v_reinterpret_as_s16(v_load(dsrc + 4*2));
                v01 = v_reinterpret_as_s16(v_load(dsrc + 4*2 + cn2));
                v10 = v_reinterpret_as_s16(v_load(dsrc + 4*2 + dstep));
                v11 = v_reinterpret_as_s16(v_load(dsrc + 4*2 + dstep + cn2));

                v_zip(v00, v01, t00, t01);
                v_zip(v10, v11, t10, t11);

                t0 = v_dotprod(t00, qw0, qdelta_d) + v_dotprod(t10, qw1);
                t1 = v_dotprod(t01, qw0, qdelta_d) + v_dotprod(t11, qw1);
                t0 = t0 >> W_BITS1;
                t1 = t1 >> W_BITS1;
                v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
                v_store(dIptr + 4*2, v00);

                v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(v00))));
                v_expand(v00, t1, t0);

                fy = v_cvt_f32(t0);
                fx = v_cvt_f32(t1);

                qA22 = v_muladd(fy, fy, qA22);
                qA12 = v_muladd(fx, fy, qA12);
                qA11 = v_muladd(fx, fx, qA11);
            }
#endif

#if CV_NEON
            for( ; x <= winSize.width*cn - 4; x += 4, dsrc += 4*2, dIptr += 4*2 )
            {

                uint8x8_t d0 = vld1_u8(&src[x]);
                uint8x8_t d2 = vld1_u8(&src[x+cn]);
                uint16x8_t q0 = vmovl_u8(d0);
                uint16x8_t q1 = vmovl_u8(d2);

                int32x4_t q5 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26);
                int32x4_t q6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27);

                uint8x8_t d4 = vld1_u8(&src[x + stepI]);
                uint8x8_t d6 = vld1_u8(&src[x + stepI + cn]);
                uint16x8_t q2 = vmovl_u8(d4);
                uint16x8_t q3 = vmovl_u8(d6);

                int32x4_t q7 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28);
                int32x4_t q8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29);

                q5 = vaddq_s32(q5, q6);
                q7 = vaddq_s32(q7, q8);
                q5 = vaddq_s32(q5, q7);

                int16x4x2_t d0d1 = vld2_s16(dsrc);
                int16x4x2_t d2d3 = vld2_s16(&dsrc[cn2]);

                q5 = vqrshlq_s32(q5, q11);

                int32x4_t q4 = vmull_s16(d0d1.val[0], d26);
                q6 = vmull_s16(d0d1.val[1], d26);

                int16x4_t nd0 = vmovn_s32(q5);

                q7 = vmull_s16(d2d3.val[0], d27);
                q8 = vmull_s16(d2d3.val[1], d27);

                vst1_s16(&Iptr[x], nd0);

                int16x4x2_t d4d5 = vld2_s16(&dsrc[dstep]);
                int16x4x2_t d6d7 = vld2_s16(&dsrc[dstep+cn2]);

                q4 = vaddq_s32(q4, q7);
                q6 = vaddq_s32(q6, q8);

                q7 = vmull_s16(d4d5.val[0], d28);
                int32x4_t q14 = vmull_s16(d4d5.val[1], d28);
                q8 = vmull_s16(d6d7.val[0], d29);
                int32x4_t q15 = vmull_s16(d6d7.val[1], d29);

                q7 = vaddq_s32(q7, q8);
                q14 = vaddq_s32(q14, q15);

                q4 = vaddq_s32(q4, q7);
                q6 = vaddq_s32(q6, q14);

                float32x4_t nq0 = vld1q_f32(nA11);
                float32x4_t nq1 = vld1q_f32(nA12);
                float32x4_t nq2 = vld1q_f32(nA22);

                q4 = vqrshlq_s32(q4, q12);
                q6 = vqrshlq_s32(q6, q12);

                q7 = vmulq_s32(q4, q4);
                q8 = vmulq_s32(q4, q6);
                q15 = vmulq_s32(q6, q6);

                nq0 = vaddq_f32(nq0, vcvtq_f32_s32(q7));
                nq1 = vaddq_f32(nq1, vcvtq_f32_s32(q8));
                nq2 = vaddq_f32(nq2, vcvtq_f32_s32(q15));

                vst1q_f32(nA11, nq0);
                vst1q_f32(nA12, nq1);
                vst1q_f32(nA22, nq2);

                int16x4_t d8 = vmovn_s32(q4);
                int16x4_t d12 = vmovn_s32(q6);

                int16x4x2_t d8d12;
                d8d12.val[0] = d8; d8d12.val[1] = d12;
                vst2_s16(dIptr, d8d12);
            }
#endif

            //然后对于每一行 按列遍历窗口中的每个像素，计算其强度和梯度加权值并更新协方差矩阵
            // dsrc += 2 和 dIptr += 2 是为了每次跳过两个元素，分别对应图像梯度在 x 和 y 方向上的变化。
            for( ; x < winSize.width*cn; x++, dsrc += 2, dIptr += 2 )
            {
                // 双线性插值计算图像灰度值、x方向梯度、y方向梯度
                int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                      src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1-5);
                                    //   src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, 0);
                int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                       dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
                int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                                       dsrc[dstep+cn2+1]*iw11, W_BITS1);

                // 将计算出的加权像素值和梯度值存入 IWinBuf 和 derivIWinBuf。
                Iptr[x] = (short)ival; // 存储强度加权值
                dIptr[0] = (short)ixval; // 存储x方向梯度加权值
                dIptr[1] = (short)iyval; // 存储y方向梯度加权值

                // 这部分计算了梯度协方差矩阵的元素：
                // A11 是 x 方向的梯度平方的总和。
                // A12 是 x 和 y 方向的梯度乘积的总和。
                // A22 是 y 方向的梯度平方的总和。
                iA11 += (itemtype)(ixval*ixval); // 计算协方差矩阵元素
                iA12 += (itemtype)(ixval*iyval);
                iA22 += (itemtype)(iyval*iyval);
            }
        }

#if CV_SIMD128 && !CV_NEON
        iA11 += v_reduce_sum(qA11);
        iA12 += v_reduce_sum(qA12);
        iA22 += v_reduce_sum(qA22);
#endif

#if CV_NEON
        iA11 += nA11[0] + nA11[1] + nA11[2] + nA11[3];
        iA12 += nA12[0] + nA12[1] + nA12[2] + nA12[3];
        iA22 += nA22[0] + nA22[1] + nA22[2] + nA22[3];
#endif

        // 将计算得到的协方差矩阵进行缩放
        A11 = iA11*FLT_SCALE;
        A12 = iA12*FLT_SCALE;
        A22 = iA22*FLT_SCALE;

        // 关于奇异矩阵的补充说明：
        // 满秩矩阵对应非奇异矩阵，非零行列式是矩阵可逆的充分必要条件。
        // 可逆矩阵就是非奇异矩阵，非奇异矩阵也是可逆矩阵。 如果A为奇异矩阵，则AX=0有无穷解，AX=b有无穷解或者无解。
        // 如果A为非奇异矩阵，则AX=0有且只有唯一零解，AX=b有唯一解。

        // 计算协方差矩阵的行列式和最小特征值
        float D = A11*A22 - A12*A12;
        // 计算特征值的方程：det(A−λI)=0
        // 对于2阶矩阵|a b|来说：得到一个关于λ的二次方程λ^2-(a+d)λ+(ad−bc)=0
        //           |c d|
        // 求解λ，有两个解，由最小的λ即可得到下式：
        // λ_{min} = ((a + d) - sqrt((a + d)^2 - 4(ad -bc))) / 2
        float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                        4.f*A12*A12))/(2*winSize.width*winSize.height);

        // 如果需要，计算并保存最小特征值
        if( err && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0 )
            err[ptidx] = (float)minEig;

        // 具体步骤是：
        // 1、计算最小特征值： 该矩阵的最小特征值（minEig）被计算出来。
        // 2、归一化： 这个最小特征值会被窗口中的像素数目除以，得到一个归一化的最小特征值。
        // 3、阈值判断： 如果这个归一化的最小特征值小于给定的阈值（minEigThreshold），则该特征点会被过滤掉，不再参与光流计算。
        // 作用：
        // 1、过滤不可靠的特征点： 当光流计算时，某些区域的特征点可能由于图像的纹理较少、对比度较低或噪声较大，导致计算出的最小特征值非常小，表明这些区域的光流计算不可靠。
        // 通过设置一个阈值 minEigThreshold，算法可以过滤掉这些“坏”特征点。
        // 2、提高性能： 通过去除不可靠的特征点，算法可以集中计算更稳定、更可靠的特征点，从而提升整体的计算效率和精度。
        // 如果最小特征值小于阈值或者行列式接近零，认为光流计算不可靠，跳过该点
        if( minEig < minEigThreshold || D < FLT_EPSILON )
        {
            std::cout << "[wx] minEig < minEigThreshold || D < FLT_EPSILON.\n";
            if( level == 0 && status )
                status[ptidx] = false;
            // 如果一个矩阵的行列式或者其最小特征值接近于0，这表明矩阵接近于奇异（即，不可逆）
            // 奇异矩阵至少有一个特征值为零。这是因为矩阵的行列式是其所有特征值的乘积，如果行列式为零，至少有一个特征值必须为零。
            continue;
        }

        D = 1.f/D;

        // 计算特征点在当前帧中的位移
        nextPt -= halfWin;
        Point2f prevDelta;

        /*
        在 OpenCV 中，TermCriteria 是一个用于定义迭代停止条件的类，它通常用于迭代算法，比如 K-means 聚类或光流计算等。构造函数中的参数决定了迭代停止的标准。
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01) 的具体含义如下：
        TermCriteria::COUNT：表示最大迭代次数的限制。此处的 30 表示算法最多允许迭代 30 次。
        TermCriteria::EPS：表示精度的限制，即当结果变化的幅度小于这个值时，算法停止迭代。此处的 0.01 表示当算法的结果（如误差、位置等）变化小于 0.01 时，停止迭代。
        COUNT + EPS：这表示停止条件是基于迭代次数或精度中的一个或两个条件都满足时停止迭代。如果满足迭代次数达到 30 次，或者精度（变化小于 0.01）达到要求，就会停止迭代。
        总结：
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01) 表示在最多迭代 30 次，或者当算法的结果变化小于 0.01 时就停止迭代。
        */
        // 迭代计算特征点的平移，直到满足收敛条件:迭代直到光流收敛（即位移增量小于某个阈值），或者达到最大迭代次数。
        for( j = 0; j < criteria.maxCount; j++ ) // 遍历迭代次数
        {
            inextPt.x = cvFloor(nextPt.x);
            inextPt.y = cvFloor(nextPt.y);

            // 如果特征点超出图像边界，则跳出迭代
            if( inextPt.x < -winSize.width || inextPt.x >= J.cols ||
               inextPt.y < -winSize.height || inextPt.y >= J.rows )
            {
                std::cout << "[wx] inextPt out of boundary.\n";
                if( level == 0 && status )
                    status[ptidx] = false;
                break;
            }

            // 计算当前特征点的权重
            a = nextPt.x - inextPt.x;
            b = nextPt.y - inextPt.y;
            iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            // 计算当前特征点的光流位移
            acctype ib1 = 0, ib2 = 0;
            float b1, b2;
#if CV_SIMD128 && !CV_NEON
            qw0 = v_int16x8((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
            qw1 = v_int16x8((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
            v_float32x4 qb0 = v_setzero_f32(), qb1 = v_setzero_f32();
#endif

#if CV_NEON
            float CV_DECL_ALIGNED(16) nB1[] = { 0,0,0,0 }, nB2[] = { 0,0,0,0 };

            const int16x4_t d26_2 = vdup_n_s16((int16_t)iw00);
            const int16x4_t d27_2 = vdup_n_s16((int16_t)iw01);
            const int16x4_t d28_2 = vdup_n_s16((int16_t)iw10);
            const int16x4_t d29_2 = vdup_n_s16((int16_t)iw11);

#endif

            // 按行遍历小窗口
            for( y = 0; y < winSize.height; y++ )
            {
                // Jptr 和 Iptr 分别指向当前帧和前一帧的图像数据，dIptr 是前一帧的梯度信息。
                // 在当前帧J上获取小窗口的第y行像素值（或说灰度值或说强度值）指针
                const uchar* Jptr = J.ptr() + (y + inextPt.y)*stepJ + inextPt.x*cn;

                // 获取对应的I帧上的小窗口的加权灰度值和梯度值指针
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                const deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

                x = 0;

#if CV_SIMD128 && !CV_NEON
                for( ; x <= winSize.width*cn - 8; x += 8, dIptr += 8*2 )
                {
                    v_int16x8 diff0 = v_reinterpret_as_s16(v_load(Iptr + x)), diff1, diff2;
                    v_int16x8 v00 = v_reinterpret_as_s16(v_load_expand(Jptr + x));
                    v_int16x8 v01 = v_reinterpret_as_s16(v_load_expand(Jptr + x + cn));
                    v_int16x8 v10 = v_reinterpret_as_s16(v_load_expand(Jptr + x + stepJ));
                    v_int16x8 v11 = v_reinterpret_as_s16(v_load_expand(Jptr + x + stepJ + cn));

                    v_int32x4 t0, t1;
                    v_int16x8 t00, t01, t10, t11;
                    v_zip(v00, v01, t00, t01);
                    v_zip(v10, v11, t10, t11);

                    t0 = v_dotprod(t00, qw0, qdelta) + v_dotprod(t10, qw1);
                    t1 = v_dotprod(t01, qw0, qdelta) + v_dotprod(t11, qw1);
                    t0 = t0 >> (W_BITS1-5);
                    t1 = t1 >> (W_BITS1-5);
                    diff0 = v_pack(t0, t1) - diff0;
                    v_zip(diff0, diff0, diff2, diff1); // It0 It0 It1 It1 ...
                    v00 = v_reinterpret_as_s16(v_load(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                    v01 = v_reinterpret_as_s16(v_load(dIptr + 8));
                    v_zip(v00, v01, v10, v11);
                    v_zip(diff2, diff1, v00, v01);
                    qb0 += v_cvt_f32(v_dotprod(v00, v10));
                    qb1 += v_cvt_f32(v_dotprod(v01, v11));
                }
#endif

#if CV_NEON
                for( ; x <= winSize.width*cn - 8; x += 8, dIptr += 8*2 )
                {

                    uint8x8_t d0 = vld1_u8(&Jptr[x]);
                    uint8x8_t d2 = vld1_u8(&Jptr[x+cn]);
                    uint8x8_t d4 = vld1_u8(&Jptr[x+stepJ]);
                    uint8x8_t d6 = vld1_u8(&Jptr[x+stepJ+cn]);

                    uint16x8_t q0 = vmovl_u8(d0);
                    uint16x8_t q1 = vmovl_u8(d2);
                    uint16x8_t q2 = vmovl_u8(d4);
                    uint16x8_t q3 = vmovl_u8(d6);

                    int32x4_t nq4 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26_2);
                    int32x4_t nq5 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q0)), d26_2);

                    int32x4_t nq6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27_2);
                    int32x4_t nq7 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q1)), d27_2);

                    int32x4_t nq8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28_2);
                    int32x4_t nq9 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q2)), d28_2);

                    int32x4_t nq10 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29_2);
                    int32x4_t nq11 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q3)), d29_2);

                    nq4 = vaddq_s32(nq4, nq6);
                    nq5 = vaddq_s32(nq5, nq7);
                    nq8 = vaddq_s32(nq8, nq10);
                    nq9 = vaddq_s32(nq9, nq11);

                    int16x8_t q6 = vld1q_s16(&Iptr[x]);

                    nq4 = vaddq_s32(nq4, nq8);
                    nq5 = vaddq_s32(nq5, nq9);

                    nq8 = vmovl_s16(vget_high_s16(q6));
                    nq6 = vmovl_s16(vget_low_s16(q6));

                    nq4 = vqrshlq_s32(nq4, q11);
                    nq5 = vqrshlq_s32(nq5, q11);

                    int16x8x2_t q0q1 = vld2q_s16(dIptr);
                    float32x4_t nB1v = vld1q_f32(nB1);
                    float32x4_t nB2v = vld1q_f32(nB2);

                    nq4 = vsubq_s32(nq4, nq6);
                    nq5 = vsubq_s32(nq5, nq8);

                    int32x4_t nq2 = vmovl_s16(vget_low_s16(q0q1.val[0]));
                    int32x4_t nq3 = vmovl_s16(vget_high_s16(q0q1.val[0]));

                    nq7 = vmovl_s16(vget_low_s16(q0q1.val[1]));
                    nq8 = vmovl_s16(vget_high_s16(q0q1.val[1]));

                    nq9 = vmulq_s32(nq4, nq2);
                    nq10 = vmulq_s32(nq5, nq3);

                    nq4 = vmulq_s32(nq4, nq7);
                    nq5 = vmulq_s32(nq5, nq8);

                    nq9 = vaddq_s32(nq9, nq10);
                    nq4 = vaddq_s32(nq4, nq5);

                    nB1v = vaddq_f32(nB1v, vcvtq_f32_s32(nq9));
                    nB2v = vaddq_f32(nB2v, vcvtq_f32_s32(nq4));

                    vst1q_f32(nB1, nB1v);
                    vst1q_f32(nB2, nB2v);
                }
#endif

                for( ; x < winSize.width*cn; x++, dIptr += 2 )
                {
                    /*
                     正向光流与反向光流的区别：
                    正向光流： 给定前一帧图像，计算特征点或像素在当前帧的位置。
                    反向光流： 给定当前帧图像，计算特征点或像素在前一帧的位置。
                    在实际应用中，反向光流常常用于验证光流的质量，或者作为正向光流计算的对照。
                    比如，计算正向光流后，我们可以使用反向光流来确认正向光流的估算是否可靠。具体来说，反向光流的主要用途有以下几点：

                    验证： 正向光流和反向光流应该相互一致。也就是说，如果一个点从上一帧到当前帧的光流是正确的，
                    那么使用反向光流再从当前帧回到上一帧时，应该能够恢复到原来的位置。如果这种一致性差，说明光流计算可能存在误差。
                    增加鲁棒性： 在计算中使用正向光流和反向光流的结合，能够减少由图像噪声和光照变化等因素引起的误差。
                    处理运动模糊： 在一些特定场景下，反向光流可以帮助缓解正向光流在模糊图像中的估计问题。
                     * 反向光流，即从当前帧追踪到上一帧
                     * 补充说明：似乎这里用的是反向光流法（即正向光流里面关于计算雅可比用的是反向光流的思路，称之为反向光流法）
                     SLAM十四讲P214:
                     在反向光流中，I1(x,y)的梯度是保持不变的，当雅可比不变时，H矩阵不变，每次迭代只需计算残差
                     */
                    // 计算光度残差：双线性插值加权后的灰度值减去第i帧对应的灰度值
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                                        //   0) - Iptr[x];
                    ib1 += (itemtype)(diff*dIptr[0]); // r * dx的累加
                    ib2 += (itemtype)(diff*dIptr[1]); // r * dy的累加
                }
            }

#if CV_SIMD128 && !CV_NEON
            v_float32x4 qf0, qf1;
            v_recombine(v_interleave_pairs(qb0 + qb1), v_setzero_f32(), qf0, qf1);
            ib1 += v_reduce_sum(qf0);
            ib2 += v_reduce_sum(qf1);
#endif

#if CV_NEON

            ib1 += (float)(nB1[0] + nB1[1] + nB1[2] + nB1[3]);
            ib2 += (float)(nB2[0] + nB2[1] + nB2[2] + nB2[3]);
#endif

            b1 = ib1*FLT_SCALE;
            b2 = ib2*FLT_SCALE;

            // 计算光流的位移量
            // 设: A=J^T*J, b =-J^T*r, J= (I_x, I_y)
            // 根据Aδx =-b,可求得如下增量δ：
            Point2f delta( (float)((A12*b2 - A22*b1) * D),
                          (float)((A12*b1 - A11*b2) * D));
            //delta = -delta;

            nextPt += delta; // 更新 nextPt，即特征点在当前帧中的新位置。
            nextPts[ptidx] = nextPt + halfWin; // 更新 nextPts[ptidx]，记录当前特征点的新位置

            // 如果光流的位移小于阈值，则认为已经收敛
            if( delta.ddot(delta) <= criteria.epsilon ) // 位移增量的2范数的平方小于阈值认为收敛
                {/*std::cout << "1 iter=" << j << " delta=(" << delta.x << ", " << delta.y << ")\n";*/break;}

            // 如果两次迭代的位移差异非常小，认为已经收敛
            if( j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
               std::abs(delta.y + prevDelta.y) < 0.01 )
            {
                // 如果迭代过程收敛，微调特征点的位置（减去一半的位移量）并跳出循环
                nextPts[ptidx] -= delta*0.5f;
                std::cout << "2 iter=" << j << " delta=(" << delta.x << ", " << delta.y << ")\n";
                break;
            }
            prevDelta = delta; // 更新 prevDelta 为当前的 delta，为下一次迭代做准备。
        }
        std::cout << "[wx] iter=" << j << std::endl;

        // 如果光流追踪成功status[ptidx] == true, 并且是金字塔最底层, 计算特征点的误差值
        CV_Assert(status != NULL);
        // status[ptidx]初值为true
        // flags & OPTFLOW_LK_GET_MIN_EIGENVALS == 0: 检查是否设置了计算最小特征值的标志。如果没有设置这个标志，才进行误差计算。
        if( status[ptidx] && err && level == 0 && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) == 0 )
        {
            // 计算当前特征点的误差值
            Point2f nextPoint = nextPts[ptidx] - halfWin;
            Point inextPoint;

            inextPoint.x = cvFloor(nextPoint.x);
            inextPoint.y = cvFloor(nextPoint.y);

            if( inextPoint.x < -winSize.width || inextPoint.x >= J.cols ||
                inextPoint.y < -winSize.height || inextPoint.y >= J.rows )
            {
                // 如果该特征点的整数坐标超出了图像的有效区域（即超出了图像的边界）
                if( status )
                    status[ptidx] = false; // 如果追踪得到的坐标超出当前帧图像范围，认为追踪失败。
                continue;
            }

            // 计算当前帧图像的误差
            float aa = nextPoint.x - inextPoint.x;
            float bb = nextPoint.y - inextPoint.y;
            iw00 = cvRound((1.f - aa)*(1.f - bb)*(1 << W_BITS));
            iw01 = cvRound(aa*(1.f - bb)*(1 << W_BITS));
            iw10 = cvRound((1.f - aa)*bb*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            float errval = 0.f;

            // 计算误差值
            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* Jptr = J.ptr() + (y + inextPoint.y)*stepJ + inextPoint.x*cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);

                for( x = 0; x < winSize.width*cn; x++ )
                {
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    errval += std::abs((float)diff);
                }
            }
            // 关于32的理解：
            // 由于计算权重时放大了1 << W_BITS即2^14倍，计算残差时调用CV_DESCALE宏缩小了1 >> (W_BITS1-5),
            // 因此这里需要再除以32，即残差结果归一化后，再缩小1 >> 5.即可以得到没有缩放的光度残差。
            // 误差值（err[ptidx]）被归一化为单位像素的平均误差，用户可以用于评估该特征点匹配的质量。
            err[ptidx] = errval * 1.f/(32*winSize.width*cn*winSize.height); // 保存平均光度误差
        }
    }
}

void LKOpticalFlow::calc( InputArray _prevImg, InputArray _nextImg,
                           InputArray _prevPts, InputOutputArray _nextPts,
                           OutputArray _status, OutputArray _err)
{
/*    
    CV_INSTRUMENT_REGION(); // 用于性能分析的宏

    // 如果启用了OpenCL且输入图像是UMat类型，尝试使用OpenCL实现
    CV_OCL_RUN(ocl::isOpenCLActivated() &&
               (_prevImg.isUMat() || _nextImg.isUMat()) &&
               ocl::Image2D::isFormatSupported(CV_32F, 1, false),
               ocl_calcOpticalFlowPyrLK(_prevImg, _nextImg, _prevPts, _nextPts, _status, _err))

    // Disabled due to bad accuracy 禁用OpenVX实现（由于准确度问题）
    CV_OVX_RUN(false,
               openvx_pyrlk(_prevImg, _nextImg, _prevPts, _nextPts, _status, _err))
*/
    // 将输入的前一帧特征点转换为 Mat 类型 ( vector<cv::Point2f> 转为InputArray 进而转为cv::Mat )
    Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = cv::DataType<short>::depth; // 导数图像（梯度图像）的数据类型深度

    // 检查最大金字塔层数和窗口大小的合法性
    CV_Assert( maxLevel >= 0 && winSize.width > 2 && winSize.height > 2 );

    int level=0, i, npoints;
    // 检查前一帧特征点的数量，并确保为有效的二维点
    CV_Assert( (npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0 );

    // 如果没有特征点，则释放相关输出并返回
    if( npoints == 0 )
    {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }

    /*
    
void cv::_OutputArray::create	(	Size 	sz,
int 	type,
int 	i = -1,
bool 	allowTransposed = false,
int 	fixedDepthMask = 0 
)		const

参数解释
Size sz
sz 参数指定了输出数组的尺寸，即宽度和高度。它是一个 Size 类型的对象，通常通过 cv::Size(width, height) 来创建。

int type
type 参数指定输出数组的元素类型（即数据的深度和通道数）。例如，CV_8UC3 表示一个 8 位无符号整数的 3 通道图像，CV_32F 表示单通道的 32 位浮点数图像。

int i = -1
i 参数用于指定如果存在多个输出数组时选择特定的索引。默认值为 -1，表示创建一个单一的输出数组。如果你使用了多输出数组的情况，可能会传入一个特定的索引。

bool allowTransposed = false
allowTransposed 参数指示是否允许输出数组的转置。默认值为 false，表示不允许转置。如果设为 true，OpenCV 可能在某些操作中返回转置矩阵，这取决于操作的要求和效率优化。

int fixedDepthMask = 0
fixedDepthMask 参数用于指定是否强制输出数组的深度类型。在大多数情况下，你可以忽略它，默认值 0 表示没有特定的深度要求。如果你需要特定的深度类型，可以使用此参数来限制输出的深度。
     */

    // 如果没有使用初始光流，则创建 _nextPts 用于存储估算的下一个特征点位置
    if( !(flags & cv::OPTFLOW_USE_INITIAL_FLOW) )
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    // 获取 _nextPts 的 Mat 形式，并确保其大小与 prevPts 相同
    Mat nextPtsMat = _nextPts.getMat();
    CV_Assert( nextPtsMat.checkVector(2, CV_32F, true) == npoints );

    // 获取前一帧和当前帧的特征点指针
    const Point2f* prevPts = prevPtsMat.ptr<Point2f>();
    Point2f* nextPts = nextPtsMat.ptr<Point2f>();

    // 创建状态数组，并初始化为所有特征点追踪成功
    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert( statusMat.isContinuous() );
    uchar* status = statusMat.ptr(); // 用于存储每个特征点是否追踪成功
    float* err = 0;

    for( i = 0; i < npoints; i++ )
        status[i] = true;

    // 如果需要计算误差，则创建 _err 并初始化
    if( _err.needed() )
    {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert( errMat.isContinuous() );
        err = errMat.ptr<float>();
    }

    std::vector<Mat> prevPyr, nextPyr; // 用于存储前后两帧的金字塔图像
    int levels1 = -1;
    int lvlStep1 = 1; // 用于前一帧金字塔的层级步长
    int levels2 = -1;
    int lvlStep2 = 1; // 用于当前帧金字塔的层级步长

    // 如果输入图像为金字塔（Mat 类型数组），提取金字塔并设置层数
    if(_prevImg.kind() == _InputArray::STD_VECTOR_MAT)
    {
        _prevImg.getMatVector(prevPyr);

        levels1 = int(prevPyr.size()) - 1;
        CV_Assert(levels1 >= 0);

        // 根据条件调整前一帧金字塔的步长
        // 如果prevPyr[0].channels() * 2 == prevPyr[1].channels()成立说明金字塔容器还包含了梯度图
        // 对于灰度图来说, 梯度图的channels==2
        if (levels1 % 2 == 1 && prevPyr[0].channels() * 2 == prevPyr[1].channels() && prevPyr[1].depth() == derivDepth)
        {
            lvlStep1 = 2;
            levels1 /= 2;
        }

        // ensure that pyramid has required padding
        if(levels1 > 0)
        {
            Size fullSize;
            Point ofs;
            prevPyr[lvlStep1].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize.width && ofs.y >= winSize.height
                && ofs.x + prevPyr[lvlStep1].cols + winSize.width <= fullSize.width
                && ofs.y + prevPyr[lvlStep1].rows + winSize.height <= fullSize.height);
        }

        if(levels1 < maxLevel)
            maxLevel = levels1;
    }

    if(_nextImg.kind() == _InputArray::STD_VECTOR_MAT)
    {
        _nextImg.getMatVector(nextPyr);

        levels2 = int(nextPyr.size()) - 1;
        CV_Assert(levels2 >= 0);

        if (levels2 % 2 == 1 && nextPyr[0].channels() * 2 == nextPyr[1].channels() && nextPyr[1].depth() == derivDepth)
        {
            lvlStep2 = 2;
            levels2 /= 2;
        }

        // ensure that pyramid has required padding
        if(levels2 > 0)
        {
            Size fullSize;
            Point ofs;
            nextPyr[lvlStep2].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize.width && ofs.y >= winSize.height
                && ofs.x + nextPyr[lvlStep2].cols + winSize.width <= fullSize.width
                && ofs.y + nextPyr[lvlStep2].rows + winSize.height <= fullSize.height);
        }

        if(levels2 < maxLevel)
            maxLevel = levels2;
    }

    // 如果没有金字塔，则通过 `buildOpticalFlowPyramid` 构建金字塔
    if (levels1 < 0)
        maxLevel = buildOpticalFlowPyramid(_prevImg, prevPyr, winSize, maxLevel, false);

    if (levels2 < 0)
        maxLevel = buildOpticalFlowPyramid(_nextImg, nextPyr, winSize, maxLevel, false);
    
    /*
    TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01) 的具体含义如下：
    TermCriteria::COUNT：表示最大迭代次数的限制。此处的 30 表示算法最多允许迭代 30 次。
    TermCriteria::EPS：表示精度的限制，即当结果变化的幅度小于这个值时，算法停止迭代。此处的 0.01 表示当算法的结果（如误差、位置等）变化小于 0.01 时，停止迭代。
    COUNT + EPS：这表示停止条件是基于迭代次数或精度中的一个或两个条件都满足时停止迭代。如果满足迭代次数达到 30 次，或者精度（变化小于 0.01）达到要求，就会停止迭代。
    总结：
    TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01) 表示在最多迭代 30 次，或者当算法的结果变化小于 0.01 时就停止迭代。
    */
    // 设置终止条件: 最大迭代次数 & 精度的限制
    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;

    // dI/dx ~ Ix, dI/dy ~ Iy // 计算图像的导数或者说梯度（用于后续的光流计算）
    Mat derivIBuf;
    if(lvlStep1 == 1)
        derivIBuf.create(prevPyr[0].rows + winSize.height*2, prevPyr[0].cols + winSize.width*2, CV_MAKETYPE(derivDepth, prevPyr[0].channels() * 2));

    // 从最高金字塔层到最低金字塔层进行迭代
    for( level = maxLevel; level >= 0; level-- )
    {
        Mat derivI;
        if(lvlStep1 == 1)
        {
            // 计算图像梯度
            Size imgSize = prevPyr[level * lvlStep1].size();
            Mat _derivI( imgSize.height + winSize.height*2,
                imgSize.width + winSize.width*2, derivIBuf.type(), derivIBuf.ptr() );
            derivI = _derivI(cv::Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
            calcScharrDeriv(prevPyr[level * lvlStep1], derivI); // 计算图像的Scharr导数
            cv::copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED); // 扩展边界
        }
        else
            derivI = prevPyr[level * lvlStep1 + 1];

        // 确保前后帧金字塔的尺寸和类型一致
        CV_Assert(prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size());
        CV_Assert(prevPyr[level * lvlStep1].type() == nextPyr[level * lvlStep2].type());

        // 使用并行计算加速光流追踪
        typedef wx::liu::LKTrackerInvoker LKTrackerInvoker;
    #if 0    
        parallel_for_(Range(0, npoints), LKTrackerInvoker(prevPyr[level * lvlStep1], derivI,
                                                          nextPyr[level * lvlStep2], prevPts, nextPts,
                                                          status, err,
                                                          winSize, criteria, level, maxLevel,
                                                          flags, (float)minEigThreshold));
    #else
        //
        // tbb::blocked_range<size_t> range(0, npoints);
        tbb::blocked_range<size_t> range(0, npoints, npoints);
        // tbb::parallel_for(range, compute_func);
        tbb::parallel_for(range, LKTrackerInvoker(prevPyr[level * lvlStep1], derivI,
                                                          nextPyr[level * lvlStep2], prevPts, nextPts,
                                                          status, err,
                                                          winSize, criteria, level, maxLevel,
                                                          flags, (float)minEigThreshold));
    #endif                                                      

    }
}

void calcOpticalFlowPyrLK( InputArray _prevImg, InputArray _nextImg,
                           InputArray _prevPts, InputOutputArray _nextPts,
                           OutputArray _status, OutputArray _err,
                           Size winSize, int maxLevel,
                           TermCriteria criteria,
                           int flags, double minEigThreshold )
{
    //
    LKOpticalFlow::Ptr optflow = std::make_shared<LKOpticalFlow>(winSize,maxLevel,criteria,flags,minEigThreshold);
    optflow->calc(_prevImg,_nextImg,_prevPts,_nextPts,_status,_err);
}                           

} // namespace wx::liu
/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <thread>

#include <sophus/se2.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optical_flow/patch.h>

#include <basalt/image/image_pyr.h>
#include <basalt/utils/keypoints.h>

// 2023-11-21
#include <condition_variable>
extern std::condition_variable vio_cv;
// the end.

// #define _GOOD_FEATURE_TO_TRACK_ 1

// #include <opencv2/opencv.hpp> // 2024-5-22

#include <atomic>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../src/wx_yaml_io.h"
using namespace wx;
extern TYamlIO* g_yaml_ptr;

#include "DS_cam.hpp"

#include <unordered_map>

// #define _CV_OF_PYR_LK_

#define _USE_S3_PYRAMID_IMG_

// #define _PATTERN_RECT_WIN_

using std::vector;
using std::pair;
using std::make_pair;

namespace wx::liu {

using namespace basalt;

using Vector2f = Eigen::Vector2f;
using Vector2i = Eigen::Vector2i;
using namespace cv;
typedef short deriv_type;

inline int wxRound(float value)
{
  return (int)lrintf(value);
}

struct WXScharrDerivInvoker
{
    WXScharrDerivInvoker(const cv::Mat& _src, const cv::Mat& _dst)
        : src(_src), dst(_dst)
    { }

    void operator()(const tbb::blocked_range<size_t>& range) const ;

    const cv::Mat& src;
    const cv::Mat& dst;
};

template <typename Scalar, typename Pattern>
struct WXTrackerInvoker
{
  static constexpr int PATTERN_SIZE = Pattern::PATTERN_SIZE;
  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 1> VectorP;
  typedef Eigen::Matrix<short, PATTERN_SIZE, 1> VectorSP;

  static const Matrix2P pattern2;

  WXTrackerInvoker( const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                    const Vector2f* _prevPts, Vector2f* _nextPts,
                    uchar* _status, int _level, int _maxLevel);

  void operator()(const tbb::blocked_range<size_t>& range) const;

  const Mat* prevImg;
  const Mat* nextImg;
  const Mat* prevDeriv;
  const Vector2f* prevPts;
  Vector2f* nextPts;
  uchar* status;
  int level;
  int maxLevel;

};


template <typename Scalar, typename Pattern>
const typename WXTrackerInvoker<Scalar, Pattern>::Matrix2P
    WXTrackerInvoker<Scalar, Pattern>::pattern2 = Pattern::pattern2;
/**/

template <typename Scalar, typename Pattern>
WXTrackerInvoker<Scalar, Pattern>::WXTrackerInvoker(
    const Mat& _prevImg, const Mat& _prevDeriv, const cv::Mat& _nextImg,
    const Vector2f* _prevPts, Vector2f* _nextPts,
    uchar* _status, int _level, int _maxLevel)
{
    prevImg = &_prevImg;
    prevDeriv = &_prevDeriv;
    nextImg = &_nextImg;
    prevPts = _prevPts;
    nextPts = _nextPts;
    status = _status;

    level = _level;
    maxLevel = _maxLevel;
}

#if 0
template <typename Scalar, typename Pattern>
void WXTrackerInvoker<Scalar, Pattern>::operator()(const tbb::blocked_range<size_t>& range) const
{
/*
    CV_INSTRUMENT_REGION(); //? 用于性能分析和标记代码执行区域
*/
    cv::Size winSize(21, 21);

    // 计算半窗口大小 (winSize.width 和 winSize.height 是窗口的宽高)
    Vector2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
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
        Vector2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
        Vector2f nextPt;
        if( level == maxLevel )
        {
            // if( flags & OPTFLOW_USE_INITIAL_FLOW )
                // nextPt = nextPts[ptidx]*(float)(1./(1 << level));
            // else // 如果没有使用初始光流，则将当前点位置设为上一帧的点位置
                nextPt = prevPt; //对于最高层来说，把前一帧的点作为当前帧跟踪点的坐标初值进行赋值
        }
        else // 如果不是最后一层，将特征点位置缩放到当前层级
            nextPt = nextPts[ptidx]*2.f; // 对于其它层， 直接把点坐标乘以2.0，作为初值
        nextPts[ptidx] = nextPt; // 给当前帧追踪点数组对应序号的点赋初值

        Vector2i iprevPt, inextPt;
        // 对特征点进行半窗口偏移: 减去winSize的一半
        prevPt -= halfWin;
        // 向下取整
        iprevPt.x() = cvFloor(prevPt.x());
        iprevPt.y() = cvFloor(prevPt.y());

        // 判断特征点是否越界（超出了图像范围）
        if( iprevPt.x() < -winSize.width || iprevPt.x() >= derivI.cols ||
            iprevPt.y() < -winSize.height || iprevPt.y() >= derivI.rows )
        {
            std::cout << "[wx] iprevPt out of boundary.\n";
            // 如果点的坐标超出界限，并且是最底层，认为该点跟踪失败，skip.
            if( level == 0 )
            {
                if( status )
                    status[ptidx] = false;
                // if( err )
                    // err[ptidx] = 0;
            }
            continue;
        }

        // 计算窗口内像素的插值权重
        float a = prevPt.x() - iprevPt.x();
        float b = prevPt.y() - iprevPt.y();
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

        // 从前一帧图像中提取特征点所在的窗口，并计算该窗口的梯度的协方差矩阵
        // 该小窗口是以prevPt为中心构建的winSize大小的窗口
        // 计算该小窗口内每个点的加权像素值和梯度值
        // extract the patch from the first image, compute covariation matrix of derivatives
        int x, y;
        // 按行遍历该小窗口的灰度值和梯度值
        for( y = 0; y < winSize.height; y++ )
        {
            // 获取小窗口的当前行（第y行）的图像强度值指针和梯度值指针
            const uchar* src = I.ptr() + (y + iprevPt.y())*stepI + iprevPt.x()*cn;
            const deriv_type* dsrc = derivI.ptr<deriv_type>() + (y + iprevPt.y())*dstep + iprevPt.x()*cn2;

            // Iptr 和 dIptr 是用于存储当前行的加权像素值和梯度信息的指针，分别指向 IWinBuf 和 derivIWinBuf（这两个矩阵用于存储图像窗口和梯度窗口的加权值）。
            deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y); // winSize大小的窗口的图像数据指针
            deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y); // winSize大小的窗口的梯度数据指针

            x = 0;

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
        // if( err && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0 )
            // err[ptidx] = (float)minEig;

        // 具体步骤是：
        // 1、计算最小特征值： 该矩阵的最小特征值（minEig）被计算出来。
        // 2、归一化： 这个最小特征值会被窗口中的像素数目除以，得到一个归一化的最小特征值。
        // 3、阈值判断： 如果这个归一化的最小特征值小于给定的阈值（minEigThreshold），则该特征点会被过滤掉，不再参与光流计算。
        // 作用：
        // 1、过滤不可靠的特征点： 当光流计算时，某些区域的特征点可能由于图像的纹理较少、对比度较低或噪声较大，导致计算出的最小特征值非常小，表明这些区域的光流计算不可靠。
        // 通过设置一个阈值 minEigThreshold，算法可以过滤掉这些“坏”特征点。
        // 2、提高性能： 通过去除不可靠的特征点，算法可以集中计算更稳定、更可靠的特征点，从而提升整体的计算效率和精度。
        // 如果最小特征值小于阈值或者行列式接近零，认为光流计算不可靠，跳过该点
        if( minEig < 1e-4 || D < FLT_EPSILON )
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
        Vector2f prevDelta;

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
        for( j = 0; j < 30; j++ ) // 遍历迭代次数
        {
            inextPt.x() = cvFloor(nextPt.x());
            inextPt.y() = cvFloor(nextPt.y());

            // 如果特征点超出图像边界，则跳出迭代
            if( inextPt.x() < -winSize.width || inextPt.x() >= J.cols ||
               inextPt.y() < -winSize.height || inextPt.y() >= J.rows )
            {
                std::cout << "[wx] inextPt out of boundary.\n";
                if( level == 0 && status )
                    status[ptidx] = false;
                break;
            }

            // 计算当前特征点的权重
            a = nextPt.x() - inextPt.x();
            b = nextPt.y() - inextPt.y();
            iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            // 计算当前特征点的光流位移
            acctype ib1 = 0, ib2 = 0;
            float b1, b2;

            // 按行遍历小窗口
            for( y = 0; y < winSize.height; y++ )
            {
                // Jptr 和 Iptr 分别指向当前帧和前一帧的图像数据，dIptr 是前一帧的梯度信息。
                // 在当前帧J上获取小窗口的第y行像素值（或说灰度值或说强度值）指针
                const uchar* Jptr = J.ptr() + (y + inextPt.y())*stepJ + inextPt.x()*cn;

                // 获取对应的I帧上的小窗口的加权灰度值和梯度值指针
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                const deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

                x = 0;

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

            b1 = ib1*FLT_SCALE;
            b2 = ib2*FLT_SCALE;

            // 计算光流的位移量
            // 设: A=J^T*J, b =-J^T*r, J= (I_x, I_y)
            // 根据Aδx =-b,可求得如下增量δ：
            Vector2f delta( (float)((A12*b2 - A22*b1) * D),
                          (float)((A12*b1 - A11*b2) * D));
            //delta = -delta;

            nextPt += delta; // 更新 nextPt，即特征点在当前帧中的新位置。
            nextPts[ptidx] = nextPt + halfWin; // 更新 nextPts[ptidx]，记录当前特征点的新位置

            // Point2f delta1(delta[0], delta(1));
            // 如果光流的位移小于阈值，则认为已经收敛
            // if( delta.ddot(delta) <= 0.01 ) // 位移增量的2范数的平方小于阈值认为收敛
            // if( delta1.ddot(delta1) <= 0.01 ) // 位移增量的2范数的平方小于阈值认为收敛
            if( delta.squaredNorm() <= 1e-4 ) // 位移增量的2范数的平方小于阈值认为收敛
                {/*std::cout << "1 iter=" << j << " delta=(" << delta.x << ", " << delta.y << ")\n";*/break;}

            // 如果两次迭代的位移差异非常小，认为已经收敛
            if( j > 0 && std::abs(delta.x() + prevDelta.x()) < 0.01 &&
               std::abs(delta.y() + prevDelta.y()) < 0.01 )
            {
                // 如果迭代过程收敛，微调特征点的位置（减去一半的位移量）并跳出循环
                nextPts[ptidx] -= delta*0.5f;
                std::cout << "2 iter=" << j << " delta=(" << delta.x() << ", " << delta.y() << ")\n";
                break;
            }
            prevDelta = delta; // 更新 prevDelta 为当前的 delta，为下一次迭代做准备。
        }
        std::cout << "[wx] iter=" << j << std::endl;

    }
}

#elif 1 // defined(_PATTERN_RECT_WIN_)

template <typename Scalar, typename Pattern>
void WXTrackerInvoker<Scalar, Pattern>::operator()(const tbb::blocked_range<size_t>& range) const
{
#if 0
  // for test
  {
    std::cout << "PATTERN_SIZE=" << PATTERN_SIZE << std::endl;
    for (int i = 0; i < PATTERN_SIZE; i++) 
    {
      if(i%21 == 0) std::cout << std::endl;
      // Vector2i p = iprevPt + pattern2.col(i).template cast<int>();
      // std::cout << pattern2.col(i).template cast<int>().transpose();

      auto offset = pattern2.col(i).template cast<int>();
      std::cout << "{" << offset[0] << ", " << offset(1) << "}, ";
    }
    std::cout << std::endl;
    std::cout << "pattern2.rows=" << pattern2.rows() << " pattern2.cols=" << pattern2.cols() << std::endl;
    return ;
  }
#endif

  cv::Size winSize(21, 21);
#if defined(_PATTERN_RECT_WIN_)  
  // Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f); 
  Vector2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
#endif  

  const Mat& I = *prevImg;
  const Mat& J = *nextImg;
  const Mat& derivI = *prevDeriv;

  int j, cn = I.channels(), cn2 = cn*2;
  int w = derivI.cols, h = derivI.rows;
#if defined(_PATTERN_RECT_WIN_)  
  cv::AutoBuffer<deriv_type> _buf(winSize.area()*(cn + cn2));
  int derivDepth = DataType<deriv_type>::depth;

  Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), _buf.data());
  Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2), _buf.data() + winSize.area()*cn);
#endif

  for (size_t ptidx = range.begin(); ptidx != range.end(); ++ptidx) 
  {
    Vector2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
    Vector2f nextPt;
    if( level == maxLevel )
    {
      nextPt = prevPt;
    }
    else
      nextPt = nextPts[ptidx]*2.f;
    nextPts[ptidx] = nextPt;

    Vector2i iprevPt, inextPt;
  #if defined(_PATTERN_RECT_WIN_)
    prevPt -= halfWin;
  #endif  
    iprevPt.x() = cvFloor(prevPt.x());
    iprevPt.y() = cvFloor(prevPt.y());

    if( iprevPt.x() < -winSize.width || iprevPt.x() >= derivI.cols ||
        iprevPt.y() < -winSize.height || iprevPt.y() >= derivI.rows )
    {
        if( level == 0 )
        {
            if( status )
                status[ptidx] = false;
            // if( err )
            //     err[ptidx] = 0;
        }
        continue;
    }

    float a = prevPt.x() - iprevPt.x();
    float b = prevPt.y() - iprevPt.y();
    const int W_BITS = 14, W_BITS1 = 14;
    const float FLT_SCALE = 1.f/(1 << 20);
    int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
    int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
    int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
    int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

    int dstep = (int)(derivI.step/derivI.elemSize1());
    int stepI = (int)(I.step/I.elemSize1());
    int stepJ = (int)(J.step/J.elemSize1());
    acctype iA11 = 0, iA12 = 0, iA22 = 0;
  #if !defined(_PATTERN_RECT_WIN_)  
    VectorSP IWinBuf = VectorSP::Zero();
    VectorSP dIxWinBuf = VectorSP::Zero();
    VectorSP dIyWinBuf = VectorSP::Zero();
  #endif  
    Scalar A11, A12, A22;

    int x, y;
  #if defined(_PATTERN_RECT_WIN_)
    for( y = 0; y < winSize.height; y++ )
    {
      const uchar* src = I.ptr() + (y + iprevPt.y())*stepI + iprevPt.x()*cn;
      const deriv_type* dsrc = derivI.ptr<deriv_type>() + (y + iprevPt.y())*dstep + iprevPt.x()*cn2;

      deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
      deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

      x = 0;

      for( ; x < winSize.width*cn; x++, dsrc += 2, dIptr += 2 )
      {
          int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1-5);
          int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                  dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
          int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                                  dsrc[dstep+cn2+1]*iw11, W_BITS1);

          Iptr[x] = (short)ival;
          dIptr[0] = (short)ixval;
          dIptr[1] = (short)iyval;

          iA11 += (itemtype)(ixval*ixval);
          iA12 += (itemtype)(ixval*iyval);
          iA22 += (itemtype)(iyval*iyval);
      }
    }
  #else
    for (int i = 0; i < PATTERN_SIZE; i++) 
    {
      Vector2i p = iprevPt + pattern2.col(i).template cast<int>(); // 位于图像的位置，点的位置加上pattern里面的偏移量，得到在patch里面的每一个位置

      
      /*if(!(2 <= p.x() && p.x() < (w - 2 - 1) && 2 <= p.y() && p.y() < (h - 2 - 1)))
      {
        IWinBuf[i] = -1;
        continue ;
      }*/
    #if 1
      const uchar* src = I.ptr() + p.y()*stepI + p.x()*cn;
      const deriv_type* dsrc = derivI.ptr<deriv_type>() + p.y()*dstep + p.x()*cn2;                                         

      x = 0;
      int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                            src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1-5);
                            // src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1);
      int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                             dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
      int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                             dsrc[dstep+cn2+1]*iw11, W_BITS1);

    #else
      x = p.x(); y = p.y();
      int ival = CV_DESCALE(I.at<uchar>(y, x)*iw00 + I.at<uchar>(y, x + 1)*iw01 +
                            I.at<uchar>(y + 1, x)*iw10 + I.at<uchar>(y + 1, x + 1)*iw11, W_BITS1-5);
      int ixval = CV_DESCALE(derivI.at<cv::Vec2b>(y, x)[0]*iw00 + derivI.at<cv::Vec2b>(y, x + 1)[0]*iw01 +
                             derivI.at<cv::Vec2b>(y + 1, x)[0]*iw10 + derivI.at<cv::Vec2b>(y + 1, x + 1)[0]*iw11, W_BITS1);
      int iyval = CV_DESCALE(derivI.at<cv::Vec2b>(y, x)[1]*iw00 + derivI.at<cv::Vec2b>(y, x + 1)[1]*iw01 +
                             derivI.at<cv::Vec2b>(y + 1, x)[1]*iw10 + derivI.at<cv::Vec2b>(y + 1, x + 1)[1]*iw11, W_BITS1);
    #endif

      IWinBuf[i] = (short) ival; // 赋值图像灰度值
      dIxWinBuf[i] = (short) ixval;
      dIyWinBuf[i] = (short) iyval;

      iA11 += (itemtype)(ixval*ixval);
      iA12 += (itemtype)(ixval*iyval);
      iA22 += (itemtype)(iyval*iyval);

    }
  #endif  

    A11 = iA11*FLT_SCALE;
    A12 = iA12*FLT_SCALE;
    A22 = iA22*FLT_SCALE;

    float D = A11*A22 - A12*A12;
  #if defined(_PATTERN_RECT_WIN_)
    float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                    4.f*A12*A12))/(2*winSize.width*winSize.height);
  #else                  
    float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                    4.f*A12*A12))/(2*PATTERN_SIZE);
  #endif

    if( minEig < 1e-4 || D < FLT_EPSILON )
    {
      if( level == 0 && status )
          status[ptidx] = false;
      continue;
    }

    D = 1.f/D;
  #if defined(_PATTERN_RECT_WIN_)
    nextPt -= halfWin;
  #endif
    Vector2f prevDelta;

    for( j = 0; j < 30; j++ )
    {
      inextPt.x() = cvFloor(nextPt.x());
      inextPt.y() = cvFloor(nextPt.y());

      if( inextPt.x() < -winSize.width || inextPt.x() >= J.cols ||
          inextPt.y() < -winSize.height || inextPt.y() >= J.rows )
      {
          if( level == 0 && status )
              status[ptidx] = false;
          break;
      }

      a = nextPt.x() - inextPt.x();
      b = nextPt.y() - inextPt.y();
      iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
      iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
      iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
      iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
      acctype ib1 = 0, ib2 = 0;
      float b1, b2;
    #if defined(_PATTERN_RECT_WIN_)
      for( y = 0; y < winSize.height; y++ )
      {
          const uchar* Jptr = J.ptr() + (y + inextPt.y())*stepJ + inextPt.x()*cn;
          const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
          const deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

          x = 0;

          for( ; x < winSize.width*cn; x++, dIptr += 2 )
          {
              int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                    Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                    W_BITS1-5) - Iptr[x];
              ib1 += (itemtype)(diff*dIptr[0]);
              ib2 += (itemtype)(diff*dIptr[1]);
          }
      }
    #else
      for (int i = 0; i < PATTERN_SIZE; i++) 
      {
        Vector2i p = inextPt + pattern2.col(i).template cast<int>();

        /*if(!(2 <= p.x() && p.x() < (w - 2 - 1) && 2 <= p.y() && p.y() < (h - 2 - 1))  || IWinBuf[i] < 0)
        {
          continue ;
        }*/

        const uchar* Jptr = J.ptr() + p.y()*stepJ + p.x()*cn;
      
      #if 1
        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                              Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                              W_BITS1-5) - IWinBuf[i];
                              // W_BITS1) - IWinBuf[i];
      #else
        x = p.x(); y = p.y();
        int diff = CV_DESCALE(J.at<uchar>(y, x)*iw00 + J.at<uchar>(y, x + 1)*iw01 +
                            J.at<uchar>(y + 1, x)*iw10 + J.at<uchar>(y + 1, x + 1)*iw11, W_BITS1-5) - IWinBuf[i];
      #endif

        ib1 += (itemtype)(diff*dIxWinBuf[i]);
        ib2 += (itemtype)(diff*dIyWinBuf[i]);

      }
    #endif

      b1 = ib1*FLT_SCALE;
      b2 = ib2*FLT_SCALE;

      Vector2f delta( (float)((A12*b2 - A22*b1) * D),
                      (float)((A12*b1 - A11*b2) * D));

      nextPt += delta;
    #if defined(_PATTERN_RECT_WIN_)
      nextPts[ptidx] = nextPt + halfWin;
    #else    
      nextPts[ptidx] = nextPt;
    #endif  

      if( delta.squaredNorm() <= 1e-4 ) // 位移增量的2范数的平方小于阈值认为收敛
        break;

      // 如果两次迭代的位移差异非常小，认为已经收敛
      if( j > 0 && std::abs(delta.x() + prevDelta.x()) < 0.01 &&
        std::abs(delta.y() + prevDelta.y()) < 0.01 )
      {
        // 如果迭代过程收敛，微调特征点的位置（减去一半的位移量）并跳出循环
      #if defined(_PATTERN_RECT_WIN_) 
        nextPts[ptidx] -= delta*0.5f;
      #endif  
        break;
      }
      prevDelta = delta; // 更新 prevDelta 为当前的 delta，为下一次迭代做准备。  
    } 

  }

}

#else
template <typename Scalar, typename Pattern>
void WXTrackerInvoker<Scalar, Pattern>::operator()(const tbb::blocked_range<size_t>& range) const
{
  const Mat& I = *prevImg;
  const Mat& J = *nextImg;
  const Mat& derivI = *prevDeriv;
  int j, cn = I.channels(), cn2 = cn*2;
  int w = derivI.cols, h = derivI.rows;
  cv::Size winSize(21, 21);

  for (size_t ptidx = range.begin(); ptidx != range.end(); ++ptidx) 
  {
    Vector2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
    Vector2f nextPt;
    if( level == maxLevel )
    {
      nextPt = prevPt;
    }
    else
      nextPt = nextPts[ptidx]*2.f;
    nextPts[ptidx] = nextPt;

    Vector2i iprevPt, inextPt;
    iprevPt.x() = int(prevPt.x());
    iprevPt.y() = int(prevPt.y());

    if( iprevPt.x() < -winSize.width || iprevPt.x() >= derivI.cols ||
        iprevPt.y() < -winSize.height || iprevPt.y() >= derivI.rows )
    {
        if( level == 0 )
        {
            if( status )
                status[ptidx] = false;
            // if( err )
            //     err[ptidx] = 0;
        }
        continue;
    }

    float a = prevPt.x() - iprevPt.x();
    float b = prevPt.y() - iprevPt.y();
    const int W_BITS = 14, W_BITS1 = 14;
    const float FLT_SCALE = 1.f/(1 << 20);
    int iw00 = wxRound((1.f - a)*(1.f - b)*(1 << W_BITS));
    int iw01 = wxRound(a*(1.f - b)*(1 << W_BITS));
    int iw10 = wxRound((1.f - a)*b*(1 << W_BITS));
    int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

    int dstep = (int)(derivI.step/derivI.elemSize1());
    int stepI = (int)(I.step/I.elemSize1());
    int stepJ = (int)(J.step/J.elemSize1());
    acctype iA11 = 0, iA12 = 0, iA22 = 0;
    VectorSP IWinBuf = VectorSP::Zero();
    VectorSP dIxWinBuf = VectorSP::Zero();
    VectorSP dIyWinBuf = VectorSP::Zero();
    Scalar A11, A12, A22;

    int x;
    for (int i = 0; i < PATTERN_SIZE; i++) {
      Vector2i p = iprevPt + pattern2.col(i).template cast<int>(); // 位于图像的位置，点的位置加上pattern里面的偏移量，得到在patch里面的每一个位置

      if(!(2 <= p.x() && p.x() < (w - 2 - 1) && 2 <= p.y() && p.y() < (h - 2 - 1)))
      {
        IWinBuf[i] = -1;
        continue ;
      }

      const uchar* src = I.ptr() + p.y()*stepI + p.x()*cn;
      const deriv_type* dsrc = derivI.ptr<deriv_type>() + p.y()*dstep + p.x()*cn2;

      // int ival = CV_DESCALE(I.at<uchar>(y, x)*iw00 + I.at<uchar>(y, x + 1)*iw01 +
      //                       I.at<uchar>(y + 1, x)*iw10 + I.at<uchar>(y + 1, x + 1)*iw11, W_BITS1-5);

      x = 0;
      int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                            // src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1-5);
                            src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1);
      int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                             dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
      int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                             dsrc[dstep+cn2+1]*iw11, W_BITS1);

      IWinBuf[i] = (short) ival; // 赋值图像灰度值
      dIxWinBuf[i] = (short) ixval;
      dIyWinBuf[i] = (short) iyval;

      iA11 += (itemtype)(ixval*ixval);
      iA12 += (itemtype)(ixval*iyval);
      iA22 += (itemtype)(iyval*iyval);

    }

    A11 = iA11*FLT_SCALE;
    A12 = iA12*FLT_SCALE;
    A22 = iA22*FLT_SCALE;

    float D = A11*A22 - A12*A12;
    float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                    4.f*A12*A12))/(2*PATTERN_SIZE);

    if( minEig < 1e-4 || D < FLT_EPSILON )
    {
      if( level == 0 && status )
          status[ptidx] = false;
      continue;
    }

    w = J.cols; h = J.rows;
    D = 1.f/D;

    Vector2f prevDelta;

    for( j = 0; j < 30; j++ )
    {
      inextPt.x() = static_cast<int>(nextPt.x());
      inextPt.y() = static_cast<int>(nextPt.y());

      if( inextPt.x() < -winSize.width || inextPt.x() >= J.cols ||
          inextPt.y() < -winSize.height || inextPt.y() >= J.rows )
      {
          if( level == 0 && status )
              status[ptidx] = false;
          break;
      }

      a = nextPt.x() - inextPt.x();
      b = nextPt.y() - inextPt.y();
      iw00 = wxRound((1.f - a)*(1.f - b)*(1 << W_BITS));
      iw01 = wxRound(a*(1.f - b)*(1 << W_BITS));
      iw10 = wxRound((1.f - a)*b*(1 << W_BITS));
      iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
      acctype ib1 = 0, ib2 = 0;
      float b1, b2;

      for (int i = 0; i < PATTERN_SIZE; i++) 
      {
        Vector2i p = inextPt + pattern2.col(i).template cast<int>();

        if(!(2 <= p.x() && p.x() < (w - 2 - 1) && 2 <= p.y() && p.y() < (h - 2 - 1))  || IWinBuf[i] < 0)
        {
          continue ;
        }

        const uchar* Jptr = J.ptr() + p.y()*stepJ + p.x()*cn;

        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                              Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                              // W_BITS1-5) - IWinBuf[i];
                              W_BITS1) - IWinBuf[i];
        ib1 += (itemtype)(diff*dIxWinBuf[i]);
        ib2 += (itemtype)(diff*dIyWinBuf[i]);

      }

      b1 = ib1*FLT_SCALE;
      b2 = ib2*FLT_SCALE;

      Vector2f delta( (float)((A12*b2 - A22*b1) * D),
                      (float)((A12*b1 - A11*b2) * D));

      nextPt += delta;
      nextPts[ptidx] = nextPt;

      if( delta.squaredNorm() <= 0.01 ) // 位移增量的2范数的平方小于阈值认为收敛
        break;

      // 如果两次迭代的位移差异非常小，认为已经收敛
      if( j > 0 && std::abs(delta.x() + prevDelta.x()) < 0.01 &&
        std::abs(delta.y() + prevDelta.y()) < 0.01 )
      {
        // 如果迭代过程收敛，微调特征点的位置（减去一半的位移量）并跳出循环
        // nextPts[ptidx] -= delta*0.5f;
        break;
      }
      prevDelta = delta; // 更新 prevDelta 为当前的 delta，为下一次迭代做准备。  
    } 

  }

}
#endif

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
    tbb::parallel_for(tbb::blocked_range<size_t>(0, rows), WXScharrDerivInvoker(src, dst));
#endif    
}

// void ScharrDerivInvoker::operator()(const Range& range) const
void WXScharrDerivInvoker::operator()(const tbb::blocked_range<size_t>& range) const
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

}

namespace basalt {

  enum TrackFailedReason
  {
    FORW_FLOW_FAILED = 1, // Forward Optical Flow
    BACK_FLOW_FAILED, // Backward Optical Flow
    GT_MAX_RECOVERED_DIS, // great than optical_flow_max_recovered_dist2
  };

/// Unlike PatchOpticalFlow, FrameToFrameOpticalFlow always tracks patches
/// against the previous frame, not the initial frame where a track was created.
/// While it might cause more drift of the patch location, it leads to longer
/// tracks in practice.
template <typename Scalar, template <typename> typename Pattern>
class FrameToFrameOpticalFlow : public OpticalFlowBase {
 public:
  typedef OpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;
  typedef OpticalFlowPatch<Scalar, Pattern441<Scalar>> PatchT2;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Sophus::SE2<Scalar> SE2;

  FrameToFrameOpticalFlow(const VioConfig& config,
                          const basalt::Calibration<double>& calib)
      : t_ns(-1), frame_counter(0), last_keypoint_id(0), config(config) {
  #if 0  
    {
      // optical flow test.
      cv::Mat prevImg = cv::imread("/root/lwx_dataset/compile_humble/root/lwx/rvv_tutorial/cv_optical_flow/assets/frame1.png", cv::IMREAD_GRAYSCALE);
      cv::Mat nextImg = cv::imread("/root/lwx_dataset/compile_humble/root/lwx/rvv_tutorial/cv_optical_flow/assets/frame2.png", cv::IMREAD_GRAYSCALE);

      if (prevImg.empty() || nextImg.empty()) {
        std::cerr << "无法加载图像" << std::endl;
        return ;
      }

      // 特征点检测（例如 ShiTomasi 角点）
      std::vector<cv::Point2f> prevPts, nextPts;
      cv::goodFeaturesToTrack(prevImg, prevPts, 100, 0.3, 7);

      // 金字塔图像
      std::vector<cv::Mat> prevPyr, nextPyr;
      int maxLevel = 3; // 最大金字塔层数
      bool withDerivatives = false;

      // 生成金字塔
      cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(21, 21), maxLevel, withDerivatives);
      cv::buildOpticalFlowPyramid(nextImg, nextPyr, cv::Size(21, 21), maxLevel, withDerivatives);

      // 存储跟踪结果
    #if 0
      std::vector<uchar> status;
      std::vector<float> err;

      // 计算金字塔光流
      cv::calcOpticalFlowPyrLK(prevPyr, nextPyr, prevPts, nextPts, status, err, cv::Size(21, 21), maxLevel);
    #elif 0
      std::vector<uchar> status;
      std::vector<float> err;
      wx::liu::calcOpticalFlowPyrLK(prevPyr, nextPyr, prevPts, nextPts, status, err, cv::Size(21, 21), maxLevel);
    #else
      //
      int num_points = prevPts.size();
      std::cout << " prevPts.size=" << num_points << std::endl;

      Eigen::Vector2f prevPts1[num_points];
      Eigen::Vector2f nextPts1[num_points];
      uchar status[num_points];
      for(int i = 0; i < num_points; i++)
      {
        prevPts1[i] = Eigen::Vector2f(prevPts[i].x, prevPts[i].y);
        status[i] = true;
      }

      const cv::Size winSize(21, 21);
      constexpr int derivDepth = cv::DataType<short>::depth;

      cv::Mat derivIBuf;
      derivIBuf.create(prevPyr[0].rows + winSize.height*2, prevPyr[0].cols + winSize.width*2, CV_MAKETYPE(derivDepth, prevPyr[0].channels() * 2));
      //
      
      // int maxLevel = config.optical_flow_levels;
      for (int level = maxLevel; level >= 0; level--)
      {
        // 计算图像梯度
        cv::Mat derivI;
        cv::Size imgSize = prevPyr[level].size();
        cv::Mat _derivI( imgSize.height + winSize.height*2,
            imgSize.width + winSize.width*2, derivIBuf.type(), derivIBuf.ptr() );
        derivI = _derivI(cv::Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        wx::liu::calcScharrDeriv(prevPyr[level], derivI); // 计算图像的Scharr导数
        cv::copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED); // 扩展边界

        tbb::blocked_range<size_t> range(0, num_points, num_points);
        // tbb::blocked_range<size_t> range(0, num_points);
        tbb::parallel_for(range, wx::liu::WXTrackerInvoker<Scalar, Pattern<Scalar>>(prevPyr[level], derivI,
                                                  nextPyr[level], prevPts1, nextPts1,
                                                  status, level, maxLevel));
      //                                     
      }

      for(int i = 0; i < num_points; i++)
      {
        nextPts.emplace_back(cv::Point2f(nextPts1[i](0), nextPts1[i][1]));
      }
    #endif

      // 可视化结果
      for (size_t i = 0; i < prevPts.size(); i++) {
          if (status[i]) {
              cv::line(nextImg, prevPts[i], nextPts[i], cv::Scalar(0, 255, 0), 2);
              cv::circle(nextImg, nextPts[i], 5, cv::Scalar(0, 0, 255), -1);
          }
      }

      // 显示结果
      cv::imshow("Optical Flow", nextImg);
      cv::waitKey(0);

      return ;
    }
  #endif

    grid_size_ = config.optical_flow_detection_grid_size;
    max_iterations_ = config.optical_flow_max_iterations;

    if(isSlowVelocity_)
    {
      // grid_size_ = config.optical_flow_detection_grid_size * 2;
      // grid_size_ = config.optical_flow_detection_grid_size + 20;
      grid_size_ = config.optical_flow_detection_grid_size + config.delta_grid_size;
      std::cout << "grid size: " << grid_size_ << std::endl;
    }

    // 设置输入队列大小: 输入队列设置容量  
    input_queue.set_capacity(10);

    // 相机内参
    this->calib = calib.cast<Scalar>();

    // 转换pattern数据类型
    patch_coord = PatchT::pattern2.template cast<float>();

    // 如果视觉前端为双目，构建基础矩阵 // 如果是双目相机 计算两者的E矩阵
    if (calib.intrinsics.size() > 1) {
      Eigen::Matrix4d Ed;
      Sophus::SE3d T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[1];
      computeEssential(T_i_j, Ed); // 计算基础矩阵
      E = Ed.cast<Scalar>();
    }

    // 开启处理线程
    processing_thread.reset(
        new std::thread(&FrameToFrameOpticalFlow::processingLoop, this));

#ifdef _GOOD_FEATURE_TO_TRACK_
    mask = cv::Mat(calib.resolution[0].x(), calib.resolution[0].y(), CV_8UC1, cv::Scalar(255));
    uint8_t* data_out = mask.ptr();
    size_t half_size = mask.cols * mask.rows;
    half_size = half_size / 2;
    for (size_t i = 0; i < half_size / 2; i++) {
      data_out[i] = 0;
    }
    std::cout << "mask cols=" << mask.cols << " rows=" << mask.rows << std::endl;
    if(1)
    {
      if(mask.empty())
          std::cout << "mask is empty " << std::endl;
      if (mask.type() != CV_8UC1)
          std::cout << "mask type wrong " << std::endl;
      // if (mask.size() != forw_img.size())
          // cout << "wrong size " << endl;
    }
#endif 

    // opencv optical flow on 2024-12-24
    // row & col: calib.resolution[0].x(), calib.resolution[0].y()
    COL = calib.resolution[0].x();
    ROW = calib.resolution[0].y();
    std::cout << "COL=" << COL << " ROW=" << ROW << std::endl;
    // cv::Mat::Mat	(	int 	rows,
    //                 int 	cols,
    //                 int 	type 
    //               )	
    forw_img[0] = cv::Mat::zeros(ROW, COL, CV_8UC1);  // CV_8UC3
    forw_img[1] = cv::Mat::zeros(ROW, COL, CV_8UC1);  // CV_8UC3

    FISHEYE = g_yaml_ptr->fisheye;
    if(FISHEYE)
    {
      fisheye_mask = cv::imread(g_yaml_ptr->fisheye_mask, 0);
      if(!fisheye_mask.data)
      {
        std::cout << ("load mask fail") << std::endl;
        // ROS_BREAK();
        FISHEYE = 0;
      }
      else
        std::cout << ("load mask success") << std::endl;
    }

    x_start = (COL % grid_size_) / 2;
    x_stop = x_start + grid_size_ * (COL / grid_size_ - 1);

    y_start = (ROW % grid_size_) / 2;
    y_stop = y_start + grid_size_ * (ROW / grid_size_ - 1);

    cells.setZero(ROW / grid_size_ + 1, COL / grid_size_ + 1);

    // , m_dscam[0](calib.intrinsics[0].getParam())
    // ds_cam[2]
    for (size_t i = 0; i < calib.intrinsics.size(); i++) {
      if(calib.intrinsics[i].getName() == "ds")
      {
        // using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        // Eigen::Matrix<Scalar, Eigen::Dynamic, 1> param = calib.intrinsics[i].getParam();
        Eigen::Matrix<double, Eigen::Dynamic, 1> param = calib.intrinsics[i].getParam().template cast<double>();
        // ds_cam[i] = std::make_shared<hm::DoubleSphereCamera>(param[0], param[1], param[2], param[3], param[4], param[5]);
        ds_cam[i].reset(new hm::DoubleSphereCamera(param[0], param[1], param[2], param[3], param[4], param[5]));

        FOCAL_LENGTH[i] = param[0];
      }
    }

  }

  ~FrameToFrameOpticalFlow() { processing_thread->join(); }

  // 2023-11-19
  void Reset()
  {
    isReset_ = true;

    // reset_mutex.lock();
    // t_ns = -1;
    // frame_counter = 0;
    // last_keypoint_id = 0;
    // OpticalFlowInput::Ptr curr_frame;
    // while (!input_queue.empty()) input_queue.pop(curr_frame); // drain input_queue
    
    // reset_mutex.unlock();
  }
  // the end.

  virtual void SetZeroVelocity(bool bl) {
    
    if(isZeroVelocity_ != bl)
    {
      isZeroVelocity_ = bl;
      if(isZeroVelocity_)
      {
        // grid_size_ = config.optical_flow_detection_grid_size * 2;
        max_iterations_ = config.optical_flow_max_iterations / 2;
      }
      else
      {
        // grid_size_ = config.optical_flow_detection_grid_size;
        max_iterations_ = config.optical_flow_max_iterations;
      }

      // std::cout << "grid size: " << grid_size_ << std::endl;
      std::cout << "max iterations: " << max_iterations_ << std::endl;
    }

  }

   virtual void SetSlowVelocity(bool bl) {
    if(isSlowVelocity_ != bl)
    {
      isSlowVelocity_ = bl;
      if(isSlowVelocity_)
      {
        // grid_size_ = config.optical_flow_detection_grid_size * 2;
        // grid_size_ = config.optical_flow_detection_grid_size + 20;
        grid_size_ = config.optical_flow_detection_grid_size + config.delta_grid_size;
      }
      else
      {
        grid_size_ = config.optical_flow_detection_grid_size;
      }

      std::cout << "grid size: " << grid_size_ << std::endl;
    }
   }

  void processingLoop() {
    OpticalFlowInput::Ptr input_ptr;

    // processingLoop循环处理部分：拿到一帧的数据指针、处理一帧processFrame
    while (true) {
      if(GetReset()) // 2023-11-20 11:12
      {
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // only imitate program to quit and run
        t_ns = -1;
        frame_counter = 0;
        last_keypoint_id = 0;
        
        OpticalFlowInput::Ptr curr_frame;
        while (!input_queue.empty()) input_queue.pop(curr_frame); // drain input_queue
        OpticalFlowResult::Ptr output_frame;
        while (!output_queue->empty()) output_queue->pop(output_frame); // drain output_queue

        std::cout << "reset front end.\n";
        // vio_cv.notify_one();
        reset_();

        SetReset(false);
        
      }
      // reset_mutex.lock(); // 2023-11-19
      input_queue.pop(input_ptr); // 从输入流获得图片

      // 如果获得的图片为空，在输出队列添加一个空的元素
      if (!input_ptr.get()) {
        if (output_queue) output_queue->push(nullptr);
        break;
      }

      // 追踪特征点，添加特征点，剔除外点，将追踪结果push到输出队列
      processFrame(input_ptr->t_ns, input_ptr);
      // reset_mutex.unlock(); // 2023-11-19
    }
  }

  //
  bool inBorder(const cv::Point2f &pt)
  {
      const int BORDER_SIZE = 1;
      int img_x = cvRound(pt.x);
      int img_y = cvRound(pt.y);
      return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
  }

  void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
  {
      int j = 0;
      int size = int(v.size());
      // for (int i = 0; i < int(v.size()); i++)
      for (int i = 0; i < size; i++)
          if (status[i])
              v[j++] = v[i];
      v.resize(j);
  }

  void reduceVector(vector<int> &v, vector<uchar> status)
  {
      int j = 0;
      int size = int(v.size());
      for (int i = 0; i < size; i++)
          if (status[i])
              v[j++] = v[i];
      v.resize(j);
  }

  void setMask2(vector<cv::Point2f>& forw_pts, vector<int>& track_cnt, vector<int>& ids)
  {
    cells.setZero(ROW / grid_size_ + 1, COL / grid_size_ + 1);

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // 按点被跟踪的次数倒序排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
        {
            return a.first > b.first;
        });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
      const cv::Point2f &p = it.second.first;
      // if (mask.at<uchar>(it.second.first) == 255) continue ;
      if (fisheye_mask.at<uchar>(it.second.first) == 0) continue ;

      if (p.x >= x_start && p.y >= y_start && p.x < x_stop + grid_size_ &&
        p.y < y_stop + grid_size_) {
        int x = (p.x - x_start) / grid_size_;
        int y = (p.y - y_start) / grid_size_;

        if(cells(y, x) == 0)
        {
          forw_pts.push_back(it.second.first);
          ids.push_back(it.second.second);
          track_cnt.push_back(it.first);

          cells(y, x) = 1;
        }

      }

    }
  }

  void setMask(vector<cv::Point2f>& forw_pts, vector<int>& track_cnt, vector<int>& ids)
  {
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); // 构建一个纯白色的图像
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // 按点被跟踪的次数倒序排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
        {
            return a.first > b.first;
        });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            // 如果点对应于mask图像所在的位置是白色的，那么就保留该点
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            
            /*
            void cv::circle	(InputOutputArray 	img,
                            Point 	            center,
                            int 	            radius,
                            const Scalar & 	color,
                            int 	            thickness = 1,
                            int 	            lineType = LINE_8,
                            int 	            shift = 0 
                            )
            */
            // 同时以该点为圆心，在半径为MIN_DIST的范围内将mask对应区域的点颜色置为黑色。那么下次其它点落在该区域的全部都被剔除掉了。
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
  }

  void rejectWithF(vector<cv::Point2f> &cur_pts, vector<cv::Point2f> &forw_pts, vector<int> &ids, vector<int> &track_cnt, std::shared_ptr<hm::DoubleSphereCamera> m_camera, int FOCAL_LENGTH)
  {
    if(calib.intrinsics[0].getName() == "ds" && m_camera.get())
    {
      if (forw_pts.size() >= 8)
      {
          // ROS_DEBUG("FM ransac begins");
          // TicToc t_f;
          vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
          for (unsigned int i = 0; i < cur_pts.size(); i++)
          {
              Eigen::Vector3d tmp_p;
              m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
              tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
              tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
              un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

              // std::cout << "cur_pts: uv=" << cur_pts[i].x << ", " << cur_pts[i].y << "   un_cur_pts: uv=" << tmp_p.x() << "," << tmp_p.y() << std::endl;

              m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
              tmp_p.x() = FOCAL_LENGTH* tmp_p.x() / tmp_p.z() + COL / 2.0;
              tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
              un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

              // std::cout<< "forw_pts: uv=" << forw_pts[i].x << ", " << forw_pts[i].y  << "   un_forw_pts: uv=" << tmp_p.x() << "," << tmp_p.y() << std::endl;
          }

          vector<uchar> status;
          // cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
          cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, 3., 0.99, status);
          // cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, 2., 0.99, status);
          // cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, 1., 0.80, status);
          // int size_a = cur_pts.size();
          // reduceVector(prev_pts, status);
          reduceVector(cur_pts, status);
          reduceVector(forw_pts, status);
          // reduceVector(cur_un_pts, status);
          reduceVector(ids, status);
          reduceVector(track_cnt, status);
          // ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
          // ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
      }
    } // if == "ds"

  }

  // return view of image
  const cv::Mat SubImage(const cv::Mat& img, size_t x, size_t y, size_t width, size_t height) const {
    // 确保子图像不超出原图范围
    assert((x + width) <= img.cols && (y + height) <= img.rows);

    // 使用 ROI（Region of Interest）特性来提取子矩阵
    cv::Rect roi(x, y, width, height); // 定义感兴趣区域
    return img(roi); // 返回ROI区域对应的子图像
  }

  inline bool InBounds(const cv::Mat& img, float x, float y, float border) {
    // 检查坐标 (x, y) 是否在图像有效区域内，并且不在边界区域内
    return border <= x && x < (img.cols - border) && border <= y && y < (img.rows - border);
  }

  void detectKeypoints3(
    const cv::Mat& img_raw, KeypointsData& kd,
    int PATCH_SIZE, int num_points_cell,
    const Eigen::aligned_vector<Eigen::Vector2d>& current_points) {
  
    // 清空输出结果容器
    kd.corners.clear();
    kd.corner_angles.clear();
    kd.corner_descriptors.clear();

    // 提取的特征点具有中心对称，即提取的特征点均匀分布在图像的中间, 意味着不能被PATCH_SIZE整除时，边边角角的地方就不会拿来选点
    // 比如对于PATCH_SIZE=50，640 * 480的图像来说x_start=20, x_stop=570, y_start=15, y_stop=415
    // 那么选的点的范围是：coloumn pixel ∈ [20, 620]，row pixel ∈ [15, 465].
    const size_t x_start = (img_raw.cols % PATCH_SIZE) / 2;
    const size_t x_stop = x_start + PATCH_SIZE * (img_raw.cols / PATCH_SIZE - 1);

    const size_t y_start = (img_raw.rows % PATCH_SIZE) / 2;
    const size_t y_stop = y_start + PATCH_SIZE * (img_raw.rows / PATCH_SIZE - 1);

    //  std::cerr << "x_start " << x_start << " x_stop " << x_stop << std::endl;
    //  std::cerr << "y_start " << y_start << " y_stop " << y_stop << std::endl;

    // 计算cell的总个数， cell按照矩阵的方式排列：e.g. 480 / 80 + 1, 640 / 80 + 1
    Eigen::MatrixXi cells;
    cells.setZero(img_raw.rows / PATCH_SIZE + 1, img_raw.cols / PATCH_SIZE + 1);

    // 统计已拥有的特征点在对应栅格cell中出现的个数
    for (const Eigen::Vector2d& p : current_points) {
      if (p[0] >= x_start && p[1] >= y_start && p[0] < x_stop + PATCH_SIZE &&
          p[1] < y_stop + PATCH_SIZE) {
        int x = (p[0] - x_start) / PATCH_SIZE;
        int y = (p[1] - y_start) / PATCH_SIZE;
        // cells记录每个cell中成功追踪的特征点数
        cells(y, x) += 1;
      }
    }

    // 2024-5-22.
    // const size_t skip_count = (y_stop - y_start) * 1 / (PATCH_SIZE * 2) + 1;
    const size_t cell_rows = img_raw.rows / PATCH_SIZE;
    const size_t cell_cols = img_raw.cols / PATCH_SIZE;
    const size_t skip_top_count = std::ceil(cell_rows * g_yaml_ptr->skip_top_ratio); // 0.5
    const size_t skip_bottom_count = std::ceil(cell_rows * g_yaml_ptr->skip_bottom_ratio);
    const size_t skip_left_count = std::ceil(cell_cols * g_yaml_ptr->skip_left_ratio);
    const size_t skip_right_count = std::ceil(cell_cols * g_yaml_ptr->skip_right_ratio);
    const size_t x_start2 = x_start + skip_left_count * PATCH_SIZE;
    const size_t x_stop2 = x_stop - skip_right_count * PATCH_SIZE;
    const size_t y_start2 = y_start + skip_top_count * PATCH_SIZE;
    const size_t y_stop2 = y_stop - skip_bottom_count * PATCH_SIZE;
    // std::cout << "y_start=" << y_start << " y_stop=" << y_stop << "y_start2=" << y_start2 << "PATCH_SIZE=" << PATCH_SIZE << std::endl;
    // the end.

    // for (size_t x = x_start; x <= x_stop; x += PATCH_SIZE) {
    for (size_t x = x_start2; x <= x_stop2; x += PATCH_SIZE) {
      // for (size_t y = y_start; y <= y_stop; y += PATCH_SIZE) {
      for (size_t y = y_start2; y <= y_stop2; y += PATCH_SIZE) {

        // 已经拥有特征点的栅格不再提取特征点
        if (cells((y - y_start) / PATCH_SIZE, (x - x_start) / PATCH_SIZE) > 0)
          continue;

        /*const basalt::Image<const uint16_t> sub_img_raw =
            img_raw.SubImage(x, y, PATCH_SIZE, PATCH_SIZE);

        cv::Mat subImg(PATCH_SIZE, PATCH_SIZE, CV_8U);

        // 16bit 移位操作只取后8位，因为原始图像存放在高8位
        for (int y = 0; y < PATCH_SIZE; y++) {
          uchar* sub_ptr = subImg.ptr(y);
          for (int x = 0; x < PATCH_SIZE; x++) {
            sub_ptr[x] = (sub_img_raw(x, y) >> 8); // 将图像转换为cv::Mat格式
          }
        }*/

        const cv::Mat subImg = SubImage(img_raw, x, y, PATCH_SIZE, PATCH_SIZE);

      if(g_yaml_ptr->FAST_algorithm)
      {
        int points_added = 0;
        int threshold = g_yaml_ptr->FAST_threshold;//40;
        // 每个cell 提取一定数量的特征点，阈值逐渐减低的
        // while (points_added < num_points_cell && threshold >= 5) {
        while (points_added < num_points_cell && threshold >= g_yaml_ptr->FAST_min_threshold) {
          std::vector<cv::KeyPoint> points;
          cv::FAST(subImg, points, threshold); // 对每一个grid提取一定数量的FAST角点
          // cv::FAST(subImg, points, g_yaml_ptr->FAST_threshold); // 对每一个grid提取一定数量的FAST角点
          // 以下是关于FAST角点的一些分析：
          // FAST算法将要检测的像素作为圆心，当具有固定半径的圆上的其他像素与圆心的像素之间的灰度差足够大时，该点被认为是角点。
          // 然而，FAST角点不具有方向和尺度信息，它们不具有旋转和尺度不变性。
          // 2012年，Rublee等人（2012）提出了基于FAST角点和BRIEF描述符的定向FAST和旋转BRIEF（ORB）算法。
          // 该算法首先在图像上构建图像金字塔，然后检测FAST关键点并计算关键点的特征向量。
          // ORB的描述符采用了二进制字符串特征BRIEF描述符的快速计算速度（Michael等人，2010），
          // 因此ORB计算速度比具有实时特征检测的fast算法更快。此外ORB受噪声影响较小，具有良好的旋转不变性和尺度不变性，可应用于实时SLAM系统。
          // 2016年，Chien等人（2016）比较并评估了用于VO应用的SIFT、SURF和ORB特征提取算法。
          // 通过对KITTI数据集的大量测试（Geiger等人，2013），可以得出结论，SIFT在提取特征方面最准确，而ORB的计算量较小。
          // 因此，作为计算能力有限的嵌入式计算机，ORB方法被认为更适合自动驾驶车辆的应用。
          // ORB描述子，或者说基于ORB的特征点，是目前最好的选择了。

          // 按照特征响应强度排序，避免角点集中（扎堆）的问题。
          std::sort(points.begin(), points.end(),
                    [](const cv::KeyPoint& a, const cv::KeyPoint& b) -> bool {
                      return a.response > b.response;
                    });

          //        std::cout << "Detected " << points.size() << " points.
          //        Threshold "
          //                  << threshold << std::endl;

          // 只取前n个特征点，且判断是否在指定范围内
          for (size_t i = 0; i < points.size() && points_added < num_points_cell;
              i++)
            if (InBounds(img_raw, x + points[i].pt.x, y + points[i].pt.y,
                                EDGE_THRESHOLD)) {
              if (fisheye_mask.at<uchar>(cv::Point2f(x + points[i].pt.x, y + points[i].pt.y)) == 0) continue ;
              kd.corners.emplace_back(x + points[i].pt.x, y + points[i].pt.y);
              points_added++;
            }

          if(g_yaml_ptr->extract_one_time) break;

          // 如果没有满足数量要求，降低阈值
          threshold /= 2;
        }
      }  
  // #else
        else //if(1) // 尝试shi-Tomas角点提取
        {
          std::vector<cv::Point2f> tmp_pts;
          // 第八个参数用于指定角点检测的方法, 默认为false, 如果是true则使用Harris角点检测，false则使用Shi Tomasi算法
          // cv::goodFeaturesToTrack(subImg, tmp_pts, num_points_cell, 0.01, 10); // 提取一个点
          // cv::goodFeaturesToTrack(subImg, tmp_pts, num_points_cell, 0.01, 8, mask); // 提取一个点
          // cv::goodFeaturesToTrack(subImg, tmp_pts, num_points_cell, 0.01, 8); // 提取一个点
          // 假设最好的点是100，最差的点就是100 *0.1，低于最差的点被reject
          // The corners with the quality measure less than the product are rejected. 
          // For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , 
          // then all the corners with the quality measure less than 15 are rejected.
          cv::goodFeaturesToTrack(subImg, tmp_pts, num_points_cell, 0.1, 8); // 提取一个点

          // 2024-5-23
          //指定亚像素计算迭代标注
          cv::TermCriteria criteria = cv::TermCriteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
            40,
            0.01);

          //亚像素检测
          if(tmp_pts.size() > 0)
          cv::cornerSubPix(subImg, tmp_pts, cv::Size(5, 5), cv::Size(-1, -1), criteria);
          // the end.

          if (tmp_pts.size() > 0 && InBounds(img_raw, x + tmp_pts[0].x, y + tmp_pts[0].y,
                                //  EDGE_THRESHOLD)) {
                                0)) {
              // std::cout << "add new point" << std::endl;
              // float val = img_raw.interp(x + tmp_pts[0].x, y + tmp_pts[0].y) >> 8;

              // 使用bilinear interpolation
              int ix = (int)(x + tmp_pts[0].x);
              int iy = (int)(y + tmp_pts[0].y);

              float dx = x + tmp_pts[0].x - ix; // 小数部分
              float dy = y + tmp_pts[0].y - iy;

              float ddx = float(1.0) - dx;
              float ddy = float(1.0) - dy;

              // 双线性插值
              // 使用 at<T>(y, x) 来访问像素值
              float val = ddx * ddy * (img_raw.at<uchar>(iy, ix) >> 8) + ddx * dy * (img_raw.at<uchar>(iy + 1, ix) >> 8) +
                    dx * ddy * (img_raw.at<uchar>(iy, ix + 1) >> 8) + dx * dy * (img_raw.at<uchar>(iy + 1, ix + 1) >> 8);

              // if(val <= 255)
              if(val <= g_yaml_ptr->max_intensity)
              {
                kd.corners.emplace_back(x + tmp_pts[0].x, y + tmp_pts[0].y);
              }

            }
        }
  // #endif
      } //for (size_t y
    } //for (size_t x

  }
  // detectKeypoints3 the end.

  void detectKeypoints2(
    const basalt::Image<const uint16_t>& img_raw, KeypointsData& kd,
    int PATCH_SIZE, int num_points_cell,
    const Eigen::aligned_vector<Eigen::Vector2d>& current_points/*, const cv::Mat &fisheye_mask*/) {
  
    // 清空输出结果容器
    kd.corners.clear();
    kd.corner_angles.clear();
    kd.corner_descriptors.clear();

    // 提取的特征点具有中心对称，即提取的特征点均匀分布在图像的中间, 意味着不能被PATCH_SIZE整除时，边边角角的地方就不会拿来选点
    // 比如对于PATCH_SIZE=50，640 * 480的图像来说x_start=20, x_stop=570, y_start=15, y_stop=415
    // 那么选的点的范围是：coloumn pixel ∈ [20, 620]，row pixel ∈ [15, 465].
    const size_t x_start = (img_raw.w % PATCH_SIZE) / 2;
    const size_t x_stop = x_start + PATCH_SIZE * (img_raw.w / PATCH_SIZE - 1);

    const size_t y_start = (img_raw.h % PATCH_SIZE) / 2;
    const size_t y_stop = y_start + PATCH_SIZE * (img_raw.h / PATCH_SIZE - 1);

    //  std::cerr << "x_start " << x_start << " x_stop " << x_stop << std::endl;
    //  std::cerr << "y_start " << y_start << " y_stop " << y_stop << std::endl;

    // 计算cell的总个数， cell按照矩阵的方式排列：e.g. 480 / 80 + 1, 640 / 80 + 1
    Eigen::MatrixXi cells;
    cells.setZero(img_raw.h / PATCH_SIZE + 1, img_raw.w / PATCH_SIZE + 1);

    // 统计已拥有的特征点在对应栅格cell中出现的个数
    for (const Eigen::Vector2d& p : current_points) {
      if (p[0] >= x_start && p[1] >= y_start && p[0] < x_stop + PATCH_SIZE &&
          p[1] < y_stop + PATCH_SIZE) {
        int x = (p[0] - x_start) / PATCH_SIZE;
        int y = (p[1] - y_start) / PATCH_SIZE;
        // cells记录每个cell中成功追踪的特征点数
        cells(y, x) += 1;
      }
    }

    // 2024-5-22.
    // const size_t skip_count = (y_stop - y_start) * 1 / (PATCH_SIZE * 2) + 1;
    const size_t cell_rows = img_raw.h / PATCH_SIZE;
    const size_t cell_cols = img_raw.w / PATCH_SIZE;
    const size_t skip_top_count = std::ceil(cell_rows * g_yaml_ptr->skip_top_ratio); // 0.5
    const size_t skip_bottom_count = std::ceil(cell_rows * g_yaml_ptr->skip_bottom_ratio);
    const size_t skip_left_count = std::ceil(cell_cols * g_yaml_ptr->skip_left_ratio);
    const size_t skip_right_count = std::ceil(cell_cols * g_yaml_ptr->skip_right_ratio);
    const size_t x_start2 = x_start + skip_left_count * PATCH_SIZE;
    const size_t x_stop2 = x_stop - skip_right_count * PATCH_SIZE;
    const size_t y_start2 = y_start + skip_top_count * PATCH_SIZE;
    const size_t y_stop2 = y_stop - skip_bottom_count * PATCH_SIZE;
    // std::cout << "y_start=" << y_start << " y_stop=" << y_stop << "y_start2=" << y_start2 << "PATCH_SIZE=" << PATCH_SIZE << std::endl;
    // the end.

    // for (size_t x = x_start; x <= x_stop; x += PATCH_SIZE) {
    for (size_t x = x_start2; x <= x_stop2; x += PATCH_SIZE) {
      // for (size_t y = y_start; y <= y_stop; y += PATCH_SIZE) {
      for (size_t y = y_start2; y <= y_stop2; y += PATCH_SIZE) {

        // 已经拥有特征点的栅格不再提取特征点
        if (cells((y - y_start) / PATCH_SIZE, (x - x_start) / PATCH_SIZE) > 0)
          continue;

        const basalt::Image<const uint16_t> sub_img_raw =
            img_raw.SubImage(x, y, PATCH_SIZE, PATCH_SIZE);

        cv::Mat subImg(PATCH_SIZE, PATCH_SIZE, CV_8U);

        // 16bit 移位操作只取后8位，因为原始图像存放在高8位
        for (int y = 0; y < PATCH_SIZE; y++) {
          uchar* sub_ptr = subImg.ptr(y);
          for (int x = 0; x < PATCH_SIZE; x++) {
            sub_ptr[x] = (sub_img_raw(x, y) >> 8); // 将图像转换为cv::Mat格式
          }
        }
  // #ifndef _GOOD_FEATURE_TO_TRACK_
  //#if 0//_GOOD_FEATURE_TO_TRACK_ == 0
      if(g_yaml_ptr->FAST_algorithm)
      {
        int points_added = 0;
        int threshold = g_yaml_ptr->FAST_threshold;//40;
        // 每个cell 提取一定数量的特征点，阈值逐渐减低的
        // while (points_added < num_points_cell && threshold >= 5) {
        while (points_added < num_points_cell && threshold >= g_yaml_ptr->FAST_min_threshold) {
          std::vector<cv::KeyPoint> points;
          cv::FAST(subImg, points, threshold); // 对每一个grid提取一定数量的FAST角点
          // cv::FAST(subImg, points, g_yaml_ptr->FAST_threshold); // 对每一个grid提取一定数量的FAST角点
          // 以下是关于FAST角点的一些分析：
          // FAST算法将要检测的像素作为圆心，当具有固定半径的圆上的其他像素与圆心的像素之间的灰度差足够大时，该点被认为是角点。
          // 然而，FAST角点不具有方向和尺度信息，它们不具有旋转和尺度不变性。
          // 2012年，Rublee等人（2012）提出了基于FAST角点和BRIEF描述符的定向FAST和旋转BRIEF（ORB）算法。
          // 该算法首先在图像上构建图像金字塔，然后检测FAST关键点并计算关键点的特征向量。
          // ORB的描述符采用了二进制字符串特征BRIEF描述符的快速计算速度（Michael等人，2010），
          // 因此ORB计算速度比具有实时特征检测的fast算法更快。此外ORB受噪声影响较小，具有良好的旋转不变性和尺度不变性，可应用于实时SLAM系统。
          // 2016年，Chien等人（2016）比较并评估了用于VO应用的SIFT、SURF和ORB特征提取算法。
          // 通过对KITTI数据集的大量测试（Geiger等人，2013），可以得出结论，SIFT在提取特征方面最准确，而ORB的计算量较小。
          // 因此，作为计算能力有限的嵌入式计算机，ORB方法被认为更适合自动驾驶车辆的应用。
          // ORB描述子，或者说基于ORB的特征点，是目前最好的选择了。

          // 按照特征响应强度排序，避免角点集中（扎堆）的问题。
          std::sort(points.begin(), points.end(),
                    [](const cv::KeyPoint& a, const cv::KeyPoint& b) -> bool {
                      return a.response > b.response;
                    });

          //        std::cout << "Detected " << points.size() << " points.
          //        Threshold "
          //                  << threshold << std::endl;

          // 只取前n个特征点，且判断是否在指定范围内
          for (size_t i = 0; i < points.size() && points_added < num_points_cell;
              i++)
            if (img_raw.InBounds(x + points[i].pt.x, y + points[i].pt.y,
                                EDGE_THRESHOLD)) {
              if (fisheye_mask.at<uchar>(cv::Point2f(x + points[i].pt.x, y + points[i].pt.y)) == 0) continue ;
              kd.corners.emplace_back(x + points[i].pt.x, y + points[i].pt.y);
              points_added++;
            }

          if(g_yaml_ptr->extract_one_time) break;

          // 如果没有满足数量要求，降低阈值
          threshold /= 2;
        }
      }  
  // #else
        else //if(1) // 尝试shi-Tomas角点提取
        {
          std::vector<cv::Point2f> tmp_pts;
          // 第八个参数用于指定角点检测的方法, 默认为false, 如果是true则使用Harris角点检测，false则使用Shi Tomasi算法
          // cv::goodFeaturesToTrack(subImg, tmp_pts, num_points_cell, 0.01, 10); // 提取一个点
          // cv::goodFeaturesToTrack(subImg, tmp_pts, num_points_cell, 0.01, 8, mask); // 提取一个点
          // cv::goodFeaturesToTrack(subImg, tmp_pts, num_points_cell, 0.01, 8); // 提取一个点
          // 假设最好的点是100，最差的点就是100 *0.1，低于最差的点被reject
          // The corners with the quality measure less than the product are rejected. 
          // For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , 
          // then all the corners with the quality measure less than 15 are rejected.
          cv::goodFeaturesToTrack(subImg, tmp_pts, num_points_cell, 0.1, 8); // 提取一个点

          // 2024-5-23
          //指定亚像素计算迭代标注
          cv::TermCriteria criteria = cv::TermCriteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
            40,
            0.01);

          //亚像素检测
          if(tmp_pts.size() > 0)
          cv::cornerSubPix(subImg, tmp_pts, cv::Size(5, 5), cv::Size(-1, -1), criteria);
          // the end.

          if (tmp_pts.size() > 0 && img_raw.InBounds(x + tmp_pts[0].x, y + tmp_pts[0].y,
                                //  EDGE_THRESHOLD)) {
                                0)) {
              // std::cout << "add new point" << std::endl;
              // float val = img_raw.interp(x + tmp_pts[0].x, y + tmp_pts[0].y) >> 8;

              // 使用bilinear interpolation
              int ix = (int)(x + tmp_pts[0].x);
              int iy = (int)(y + tmp_pts[0].y);

              float dx = x + tmp_pts[0].x - ix; // 小数部分
              float dy = y + tmp_pts[0].y - iy;

              float ddx = float(1.0) - dx;
              float ddy = float(1.0) - dy;

              // 双线性插值
              float val = ddx * ddy * (img_raw(ix, iy) >> 8) + ddx * dy * (img_raw(ix, iy + 1) >> 8) +
                    dx * ddy * (img_raw(ix + 1, iy) >> 8) + dx * dy * (img_raw(ix + 1, iy + 1) >> 8);

              // if(val <= 255)
              if(val <= g_yaml_ptr->max_intensity)
              {
                kd.corners.emplace_back(x + tmp_pts[0].x, y + tmp_pts[0].y);
              }
              // else
              // {
              //   std::cout << "val=" << val << std::endl;
              // }

            }
        }
  // #endif
      } //for (size_t y
    } //for (size_t x

    // std::cout << "Total points: " << kd.corners.size() << std::endl;

    //  cv::TermCriteria criteria =
    //      cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
    //  cv::Size winSize = cv::Size(5, 5);
    //  cv::Size zeroZone = cv::Size(-1, -1);
    //  cv::cornerSubPix(image, points, winSize, zeroZone, criteria);

    //  for (size_t i = 0; i < points.size(); i++) {
    //    if (img_raw.InBounds(points[i].pt.x, points[i].pt.y, EDGE_THRESHOLD)) {
    //      kd.corners.emplace_back(points[i].pt.x, points[i].pt.y);
    //    }
    //  }
  }
  // the end.

  // 2025-1-2
  void addPaddingToPyramid(const std::vector<cv::Mat>& pyramid, std::vector<cv::Mat>& paddedPyramid, 
                         const cv::Size& winSize = cv::Size(21,21), int pyrBorder = cv::BORDER_REFLECT_101) {
      int pyrSize = pyramid.size();
      paddedPyramid.resize(pyrSize);;
      
      for (size_t level = 0; level < pyramid.size(); ++level) {

          cv::Mat& temp = paddedPyramid.at(level);
          const cv::Mat &img = pyramid.at(level);

          if(!temp.empty())
              temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
          if(temp.type() != img.type() || temp.cols != winSize.width*2 + img.cols || temp.rows != winSize.height * 2 + img.rows)
              temp.create(img.rows + winSize.height*2, img.cols + winSize.width*2, img.type());

          /*
          if(pyrBorder == BORDER_TRANSPARENT)
              img.copyTo(temp(Rect(winSize.width, winSize.height, img.cols, img.rows)));
          else
              copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder);

          */
          int border = pyrBorder;
          if(level != 0) border = pyrBorder|cv::BORDER_ISOLATED;
          if(pyrBorder != cv::BORDER_TRANSPARENT)
              // copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder|BORDER_ISOLATED);
              cv::copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, border);

          temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);

      }
  }

  void addPadding2Img(const cv::Mat& img, cv::Mat& paddedImg, 
                    const cv::Size& winSize = cv::Size(21,21), int pyrBorder = cv::BORDER_REFLECT_101) {  
      //
      cv::Mat& temp = paddedImg;

      if(!temp.empty())
          temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
      if(temp.type() != img.type() || temp.cols != winSize.width*2 + img.cols || temp.rows != winSize.height * 2 + img.rows)
          temp.create(img.rows + winSize.height*2, img.cols + winSize.width*2, img.type());

      int border = pyrBorder;
      // if(level != 0) border = pyrBorder|BORDER_ISOLATED;
      // cv::copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder);
      cv::copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, border);
      temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
  }
  // the end.

  // 2024-12-31.
  cv::Mat stereo3Image2cvImage(const Image<const uint16_t>& img)
  {
    const uint16_t* data_in = nullptr;
    uint8_t* data_out = nullptr;

    // cv::Mat cv_image;
    // cv_image = cv::Mat::zeros(img.h, img.w, CV_8UC1);  // CV_8UC3
    cv::Mat cv_image(img.h, img.w, CV_8UC1);
    data_out = cv_image.ptr();
#if 0    
    data_in = img.ptr;

    size_t full_size = img.size();  // forw_img.cols * forw_img.rows;
    for (size_t i = 0; i < full_size; i++) {
      int val = data_in[i];
      val = val >> 8;
      data_out[i] = val;
    }
#else
    size_t i = 0;
    for (size_t r = 0; r < img.h; ++r) {
      const uint16_t* row = img.RowPtr(r);

      for(size_t c = 0; c < img.w; ++c) {
        int val = row[c];
        val = val >> 8;
        data_out[i++] = val;
      }

    }
#endif

    return cv_image;
  }

  std::vector<cv::Mat> getPyramidImage(const basalt::ManagedImagePyr<uint16_t> & pyr)
  {
    std::vector<cv::Mat> pyramid_images;
    for(int level = 0; level <= config.optical_flow_levels; level++)
    {
      const Image<const uint16_t> img = pyr.lvl(level);
      pyramid_images.emplace_back(stereo3Image2cvImage(img));
    }

    std::vector<cv::Mat> paddedPyramid;
    addPaddingToPyramid(pyramid_images, paddedPyramid);

    // return std::move(pyramid_images);
    return std::move(paddedPyramid);
  }
  // the end.

  // processFrame这里进行图像金字塔创建+跟踪+剔除特征点
  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    
    // 如果图像的数据为空（指针）直接返回
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }

    // opencv optical flow on 2024-12-24
    // convert image
    uint16_t* data_in = nullptr;
    uint8_t* data_out = nullptr;
    for(size_t i = 0; i < calib.intrinsics.size(); i++)
    // for(size_t i = 0; i < 1; i++) // TODO
    {
      basalt::ImageData imageData = new_img_vec->img_data[i];

      data_in = imageData.img->ptr;
      // forw_img = cv::Mat::zeros(imageData.img->h, imageData.img->w, CV_8UC1);  // CV_8UC3
      data_out = forw_img[i].ptr();

      size_t full_size = imageData.img->size();  // forw_img.cols * forw_img.rows;
      for (size_t i = 0; i < full_size; i++) {
        int val = data_in[i];
        val = val >> 8;
        data_out[i] = val;
      }
    }
    // the end.
    
    if (t_ns < 0) { // 第一次进入: 第一次处理帧时，t_ns == -1.
      // 开始初始化
      std::cout <<"front end init.\n";
      t_ns = curr_t_ns;

      transforms.reset(new OpticalFlowResult);
      // step l : feature 像素位姿的观测容器transforms初始化
      transforms->observations.resize(calib.intrinsics.size()); // 设置观测容器大小，对于双目来说size就是2
      transforms->t_ns = t_ns; // 时间戳复制
// #if !defined(_CV_OF_PYR_LK_)
      // step 2 : 设置图像金字塔,注意为金字塔开辟的是一个数组
      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>); // 初始化容器
      // step 2.1 金字塔的个数对应相机的个数
      pyramid->resize(calib.intrinsics.size());

      // step2.2 并行构建金字塔：多线程执行图像金子塔的构建
      // 参数1.指定参数范围 参数2匿名的函数体
      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()), // 迭代范围用数学区间表示是[0, 2)
      // tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size(), 2), // 迭代范围用数学区间表示是[0, 2) // single thread.
                        [&](const tbb::blocked_range<size_t>& r) { // [&]表示以引用方式捕获外部作用域的所有变量, [&]表示外部参数传引用，如果没有const修饰时可修改值。
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            //遍历每一个相机，构建图像金字塔
                            //参数1 : 原始图片, 参数2 : 建金字塔层数
                            // basalt的构建图像金字塔是自己实现的
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });

      // show pyramid image.
      #if 0
      {
        // basalt::ManagedImagePyr<uint16_t>
        // config.optical_flow_levels
        
        // 没有拷贝构造函数不能使用该赋值操作
        // basalt::ManagedImagePyr<uint16_t> pyr = pyramid->at(0);
        // 使用智能指针访问 pyramid 的元素
        // basalt::ManagedImagePyr<uint16_t>& pyr = (*pyramid)[0];
        // or use reference directly
        basalt::ManagedImagePyr<uint16_t> &pyr = pyramid->at(0);

        // const Image<const uint16_t> img = pyr.lvl(level);
        const Image<const uint16_t> img = pyr.mipmap();

        const uint16_t* data_in = nullptr;
        uint8_t* data_out = nullptr;

        // cv::Mat pyramid_image;
        // pyramid_image = cv::Mat::zeros(img.h, img.w, CV_8UC1);  // CV_8UC3
        cv::Mat pyramid_image(img.h, img.w, CV_8UC1);
        data_in = img.ptr;
        data_out = pyramid_image.ptr();

        size_t full_size = img.size();  // forw_img.cols * forw_img.rows;
        for (size_t i = 0; i < full_size; i++) {
          int val = data_in[i];
          val = val >> 8;
          data_out[i] = val;
        }

        cv::imshow("pyramid image", pyramid_image);
        cv::waitKey(0);
      }
      #endif

      #if 0
      {
        //
        basalt::ManagedImagePyr<uint16_t> &pyr = pyramid->at(0);
        const Image<const uint16_t> img = pyr.lvl(config.optical_flow_levels);
        // const Image<const uint16_t> img = pyr.lvl(0);

        const uint16_t* data_in = nullptr;
        uint8_t* data_out = nullptr;

        // cv::Mat cv_image;
        // cv_image = cv::Mat::zeros(img.h, img.w, CV_8UC1);  // CV_8UC3
        cv::Mat cv_image(img.h, img.w, CV_8UC1);
        data_out = cv_image.ptr();
#if 0        
        data_in = img.ptr;
        size_t full_size = img.size();  // forw_img.cols * forw_img.rows;
        for (size_t i = 0; i < full_size; i++) {
          int val = data_in[i];
          val = val >> 8;
          data_out[i] = val;
        }
#else
        size_t i = 0;
        for (size_t r = 0; r < img.h; ++r) {
          const uint16_t* row = img.RowPtr(r);

          for(size_t c = 0; c < img.w; ++c) {
            int val = row[c];
            val = val >> 8;
            data_out[i++] = val;
          }
        }
#endif

        cv::imshow("pyramid image", cv_image);
        cv::waitKey(0);
      }
      #endif
      // the end.

#if !defined(_CV_OF_PYR_LK_)
      // step3: 将图像的指针放入到transforms中，用于可视化
      transforms->input_images = new_img_vec;

      // step4: 添加特征点（因为是第一帧，故而直接提取新的特征点）
      addPoints();
#else
      transforms->input_images = new_img_vec; 
      #if !defined(_USE_S3_PYRAMID_IMG_)
      addPoints2(forw_img[0], forw_img[1]);
      #else
      addPoints_pyr(getPyramidImage(pyramid->at(0)), getPyramidImage(pyramid->at(1)));
      #endif
#endif      
      // step5: 使用对极几何剔除外点
      filterPoints();
      // 初始化结束
      // 2023-11-20
      // std::cout << "optical flow: cam0 observation count: " << transforms->observations.at(0).size() << std::endl;
      std::cout << "optical flow: cam0 observation count: " << transforms->observations.at(0).size() 
        << " / cam1 obs count: " << transforms->observations.at(1).size() << std::endl;
      
      
      /*
       * comment on 2024-12-22.  
      if(transforms->observations.at(0).size() <= 0)
      {
        t_ns = -1;
        if (output_queue && frame_counter % config.optical_flow_skip_frames == 0)
        output_queue->push(transforms);

        return ;
      }*/
      // the end.
    } else { // 非第一次进入

      // 开始追踪
      // 追踪简要流程：
      // 对每个特征点在金字塔高层级到低层级进行追踪(由粗到精)
      // 追踪当前帧的所有特征点
      // 反向追踪上的才算真正成功的

      // step 1: 更新时间
      t_ns = curr_t_ns; // 拷贝时间戳
// #if !defined(_CV_OF_PYR_LK_)
      // step 2.1: 更新last image的金子塔
      old_pyramid = pyramid; // 保存上一图像的金字塔

      // step2.2: 构造current image 的金宇塔
      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>); // 重新设置新指针
      pyramid->resize(calib.intrinsics.size());
      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });
#if !defined(_CV_OF_PYR_LK_)
      // step3: 追踪特征点
      OpticalFlowResult::Ptr new_transforms; // 新的返回参数
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->observations.resize(calib.intrinsics.size());
      new_transforms->t_ns = t_ns;

      // lwx: last left to current left , last right to current right // 对当前帧的和上一帧进行跟踪（左目与左目 右目与右目）
      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        if(i == 0)
        {
          forw_track_failed_cnt = 0;
          back_track_failed_cnt = 0;
          gt_max_recovered_dis_cnt = 0;
        }
        #if 0
        trackPoints(old_pyramid->at(i), pyramid->at(i),
                    transforms->observations[i],
                    new_transforms->observations[i]);
        #elif 1
        trackPointsFA(old_pyramid->at(i), pyramid->at(i),
                    transforms->observations[i],
                    new_transforms->observations[i]);           
        #else
        trackPoints3(old_pyramid->at(i), pyramid->at(i),
                    transforms->observations[i],
                    new_transforms->observations[i]);
        #endif

        if(0)//(i == 0)
        {
          std::ostringstream oss;
          oss << "forw_track_failed_cnt=" << forw_track_failed_cnt << " back_track_failed_cnt="
            << back_track_failed_cnt << " gt_max_recovered_dis_cnt=" << gt_max_recovered_dis_cnt << std::endl;
          
          std::cout << oss.str();
        }            
      }
#else
      // TODO: use opencv Lk optical flow
      
      /*
      void cv::calcOpticalFlowPyrLK	(	InputArray 	prevImg,
                                      InputArray 	nextImg,
                                      InputArray 	prevPts,
                                      InputOutputArray 	nextPts,
                                      OutputArray 	status,
                                      OutputArray 	err,
                                      Size 	winSize = Size(21, 21),
                                      int 	maxLevel = 3,
                                      TermCriteria 	criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                      int 	flags = 0,
                                      double 	minEigThreshold = 1e-4 
                                    )
      */
      //

      OpticalFlowResult::Ptr new_transforms; // 新的返回参数
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->observations.resize(calib.intrinsics.size());
      new_transforms->t_ns = t_ns;
      
      vector<uchar> status;
      vector<float> err;

      for (size_t i = 0; i < calib.intrinsics.size(); i++) 
      // for (size_t i = 0; i < 1; i++) // TODO
      {
        forw_pts[i].clear();

        // prev cam0 to current cam0  &  prev cam1 to current cam1
        status.clear();

        #if !defined(_USE_S3_PYRAMID_IMG_)
        cv::calcOpticalFlowPyrLK(cur_img[i], forw_img[i], cur_pts[i], forw_pts[i], status, err, cv::Size(21, 21), 3);
        #else
        //getPyramidImage(pyramid->at(0)), getPyramidImage(pyramid->at(1))
        cv::calcOpticalFlowPyrLK(getPyramidImage(old_pyramid->at(i)), getPyramidImage(pyramid->at(i)), cur_pts[i], forw_pts[i], status, err, cv::Size(21, 21), 3);
        #endif

        // if(i == 0) std::cout << "cur_pts[0]=" << cur_pts[i].size() << " forw_pts[0]=" << forw_pts[i].size() << std::endl;

        int size = int(forw_pts[i].size());
        for (int j = 0; j < size; j++)
            // Step 2 通过图像边界剔除outlier
            if (status[j] && !inBorder(forw_pts[i][j]))    // 追踪状态好检查在不在图像范围
                status[j] = 0;

        reduceVector(cur_pts[i], status);
        reduceVector(forw_pts[i], status);
        reduceVector(kpt_ids[i], status);  // 特征点的id
        reduceVector(track_cnt[i], status);    // 追踪次数

        rejectWithF(cur_pts[i], forw_pts[i], kpt_ids[i], track_cnt[i], ds_cam[i], FOCAL_LENGTH[i]);

        // setMask(forw_pts[i], track_cnt[i], kpt_ids[i]);
        setMask2(forw_pts[i], track_cnt[i], kpt_ids[i]);

        size = forw_pts[i].size();
        // if(i == 0) std::cout << "after filter: forw_pts[i]=" << forw_pts[i].size() << std::endl;    
        for(int j = 0; j < size; j++)
        {
          // 
          Eigen::AffineCompact2f transform;
          transform.setIdentity(); //旋转 设置为单位阵
          transform.translation() = Eigen::Vector2d(forw_pts[i][j].x, forw_pts[i][j].y).cast<Scalar>(); // kd.corners[i].cast<Scalar>(); // 角点坐标，保存到transform的平移部分
          new_transforms->observations[i].emplace(kpt_ids[i][j], std::make_pair(transform, track_cnt[i][j] + 1));
        }

      }
      // the end.
#endif

      // step 4: save track result
      transforms = new_transforms; // 这里transforms重新赋值追踪之后的特征点
      transforms->input_images = new_img_vec;

      // step 5: add feature 增加点
#if !defined(_CV_OF_PYR_LK_)
      addPoints(); // 追踪之后，继续提取新的点（对于非第一帧而言，先是追踪特征点，然后再提取新的点）
#else
      #if !defined(_USE_S3_PYRAMID_IMG_)
      addPoints2(forw_img[0], forw_img[1]);
      #else
      addPoints_pyr(getPyramidImage(pyramid->at(0)), getPyramidImage(pyramid->at(1)));
      #endif
#endif
      // step 6: 如果是双目相机，使用对极几何剔除外点
      filterPoints(); // 使用双目E剔除点
      // 追踪结束
    }

    // opencv optical flow
    for(size_t i = 0; i < calib.intrinsics.size(); i++)
    // for (size_t i = 0; i < 1; i++) // TODO
    {
      track_cnt[i].clear();
      kpt_ids[i].clear();
      // forw_pts[i].clear();
      cur_pts[i].clear();

      for (const auto& kv_obs : transforms->observations[i]) {
        // int kpt_id = kv_obs.first; // key point id
        Eigen::Matrix<Scalar, 2, 1> pos = kv_obs.second.first.translation().cast<Scalar>();
        // forw_pts[i].emplace_back(cv::Point2f(pos.x(), pos[1]));
        cur_pts[i].emplace_back(cv::Point2f(pos.x(), pos[1]));

        kpt_ids[i].emplace_back(kv_obs.first);
        track_cnt[i].emplace_back(kv_obs.second.second);
      }

      // cur_img[i] = forw_img[i];
      cur_img[i] = forw_img[i].clone();
      // cur_pts[i] = forw_pts[i];
    }
    // the end.


    // show track
    {
      const int ROW = new_img_vec->img_data[0].img->h;
      const int WINDOW_SIZE = config.vio_max_states + config.vio_max_kfs;

      // step1 convert image
      uint16_t* data_in = nullptr;
      uint8_t* data_out = nullptr;
      cv::Mat disp_frame;

      // for(int cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++)
      for (int cam_id = 0; cam_id < 1; cam_id++) {
        // img_data is a vector<ImageData>
        basalt::ImageData imageData = new_img_vec->img_data[cam_id];
        // std::cout << "w=" << imageData.img->w << "  h=" << imageData.img->h <<
        // std::endl;
        data_in = imageData.img->ptr;
        disp_frame = cv::Mat::zeros(imageData.img->h, imageData.img->w, CV_8UC1);  // CV_8UC3
        data_out = disp_frame.ptr();

        size_t full_size = imageData.img->size();  // disp_frame.cols * disp_frame.rows;
        for (size_t i = 0; i < full_size; i++) {
          int val = data_in[i];
          val = val >> 8;
          data_out[i] = val;
          // disp_frame.at(<>)
        }

        // cv::cvtColor(disp_frame, disp_frame, CV_GRAY2BGR);  // CV_GRAY2RGB

        // SHOW_TRACK
        cv::Mat tmp_img = disp_frame.rowRange(cam_id * ROW, (cam_id + 1) * ROW);
        cv::cvtColor(disp_frame, tmp_img, cv::COLOR_GRAY2BGR); // CV_GRAY2RGB

        // for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
        for (const auto& kv : transforms->observations[cam_id])
        {
            double len = std::min(1.0, 1.0 * kv.second.second / WINDOW_SIZE);
            auto p = kv.second.first.translation();
            cv::circle(tmp_img, cv::Point2f(p.x(), p[1]), 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);

            //draw speed line
            // Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
            // Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
            // Vector3d tmp_prev_un_pts;
            // tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
            // tmp_prev_un_pts.z() = 1;
            // Vector2d tmp_prev_uv;
            // trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
            // cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);

            //char name[10];
            //sprintf(name, "%d", trackerData[i].ids[j]);
            //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        // cv::imshow("vis", disp_frame);
        cv::imshow("tracked image", tmp_img);
        cv::waitKey(5);
        // cv::waitKey(0);


      }

      // 2025-1-2
      if(1) // show cam0 cam1 tracked image.
      {
        cv::Mat result_img;
        cv::hconcat(forw_img[0], forw_img[1], result_img);
        cv::cvtColor(result_img, result_img, cv::COLOR_GRAY2BGR);
        for (const auto& kv_obs : transforms->observations[0]) {
          Eigen::Matrix<Scalar, 2, 1> pos = kv_obs.second.first.translation().cast<Scalar>();
          cv::Point2f p0(pos.x(), pos[1]);
          // cv::circle(result_img, cv::Point2f(p.x(), p[1]), 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
          cv::circle(result_img, p0, 2, cv::Scalar(255, 0, 0), 2);

          auto it = transforms->observations[1].find(kv_obs.first);
          // if(transforms->observations[1].count(kv_obs.first))
          if(it != transforms->observations[1].end())
          {
            Eigen::Matrix<Scalar, 2, 1> pos = it->second.first.translation().cast<Scalar>();
            cv::Point2f p1(pos.x() + COL, pos[1]);
            cv::circle(result_img, p1, 2, cv::Scalar(0, 0, 255), 2);
            cv::line(result_img, p0, p1, cv::Scalar(0, 255, 0), 1);
          }
        }

        std::stringstream ss;
        ss << "cam0: " << transforms->observations[0].size() << "  cam1: " << transforms->observations[1].size() << "";
        std::string strText = ss.str();
        // show text
        int font_face = cv::FONT_HERSHEY_SIMPLEX;  // cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 0.5;                   // 1;//2;//1;
        int thickness = 1;  // 2; // 字体笔画的粗细程度，有默认值1
        int baseline;
        // 获取文本框的长宽
        // cv::Size text_size = cv::getTextSize(text, font_face, font_scale,
        // thickness, &baseline);
        cv::Size text_size =
            cv::getTextSize(strText, font_face, font_scale, thickness, &baseline);

        // 将文本框右上角绘制
        cv::Point origin;  // 计算文字左下角的位置
        origin.x = result_img.cols - text_size.width;
        origin.y = text_size.height;

        cv::putText(result_img, strText, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, false);  
        cv::imshow("cam0 - cam1 tracked image ", result_img);
        cv::waitKey(3);
      }
      // the end.
      

      return ;
    }

    // 判断是否定义了输出队列,如果输出队列不为空，将结果push到输出队列
    // 类似vins指定频率发布图像，防止imu相比视觉频率低导致相邻帧没有imu数据，使图像跳帧播放
    if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
      output_queue->push(transforms); // 光流结果推送到输出队列里 //- 其实是将光流法追踪的结果推送到了后端状态估计器
      // 这里补充说一点，basalt的前端相当于生产者，生产者线程向队列中添加特征；basalt的后端相当于消费者，消费者线程从队列中获取特征进行处理。
      // std::cout << "\033[1;33m" << "output_queue push data." << "\033[0m"<< std::endl;
    }
// std::cout << "\033[1;32m" << "frame_counter=" << frame_counter << "\033[0m"<< std::endl;
    // 跟踪数量增加
    frame_counter++; // 图像的数目累加
  }

  // forward additive & Scharr operator 2025-1-16
   
  void trackPointsFA(const basalt::ManagedImagePyr<uint16_t>& pyr_1,
                    const basalt::ManagedImagePyr<uint16_t>& pyr_2,
                    const Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>>& transform_map_1,
                    Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>>& transform_map_2)
  {
    //
    // num_points为1中点的个数
    size_t num_points = transform_map_1.size();

    std::vector<std::pair<KeypointId, TrackCnt>> ids;
    ids.reserve(num_points);

    // std::vector<Eigen::Vector2f> prevPts;
    // std::vector<Eigen::Vector2f> nextPts;
    // nextPts.resize(num_points);
    // prevPts.reserve(num_points);
    Eigen::Vector2f prevPts[num_points];
    Eigen::Vector2f nextPts[num_points];
    uchar status[num_points];
    int ptidx = 0;

    // 1.特征点类型转换map->vector
    for (const auto& kv : transform_map_1) {
      ids.push_back(std::make_pair(kv.first, kv.second.second)); // 1中点的id
      // prevPts.push_back(kv.second.first.translation()); // 1中点的信息（在2d图像上的旋转和平移信息）
      prevPts[ptidx++] = kv.second.first.translation();
      status[ptidx] = true;
    }

    tbb::concurrent_unordered_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>,
                                  std::hash<KeypointId>>
        result;

    //
    vector<cv::Mat> prevPyr = getPyramidImage(pyr_1);
    vector<cv::Mat> nextPyr = getPyramidImage(pyr_2);
    const cv::Size winSize(21, 21);
    constexpr int derivDepth = cv::DataType<short>::depth;

    cv::Mat derivIBuf;
    derivIBuf.create(prevPyr[0].rows + winSize.height*2, prevPyr[0].cols + winSize.width*2, CV_MAKETYPE(derivDepth, prevPyr[0].channels() * 2));
    //
    
    int maxLevel = config.optical_flow_levels;
    for (int level = maxLevel; level >= 0; level--)
    {
      // 计算图像梯度
      cv::Mat derivI;
      cv::Size imgSize = prevPyr[level].size();
      cv::Mat _derivI( imgSize.height + winSize.height*2,
          imgSize.width + winSize.width*2, derivIBuf.type(), derivIBuf.ptr() );
      derivI = _derivI(cv::Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
      wx::liu::calcScharrDeriv(prevPyr[level], derivI); // 计算图像的Scharr导数
      cv::copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED); // 扩展边界

      tbb::blocked_range<size_t> range(0, num_points, num_points);
      // tbb::blocked_range<size_t> range(0, num_points);
      tbb::parallel_for(range, wx::liu::WXTrackerInvoker<Scalar, Pattern<Scalar>>(prevPyr[level], derivI,
                                                nextPyr[level], prevPts, nextPts,
                                                status, level, maxLevel));/**/
      //                                     
    }

    // TODO : store tracked points
    for(int i = 0; i < num_points; i++)
    {
      if(status[i])
      {
        const std::pair<KeypointId, TrackCnt> id = ids[i]; // 得到点的id
        Eigen::AffineCompact2f transform;
        transform.setIdentity(); //旋转 设置为单位阵
        transform.translation() = nextPts[i];
        result[id.first] = std::make_pair(transform, id.second + 1);
      }
    }

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());

  }
  // the end.

  // 2025-1-7
  void trackPoints2(const basalt::ManagedImagePyr<uint16_t>& pyr_1,
                    const basalt::ManagedImagePyr<uint16_t>& pyr_2,
                    const Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>>& transform_map_1,
                    Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>>& transform_map_2)
  {
  #if 1
    #if 0
    vector<cv::Mat> prevPyr = getPyramidImage(pyr_1);
    vector<cv::Mat> nextPyr = getPyramidImage(pyr_2);
    const cv::Size winSize(21, 21);

    cv::Mat derivIBuf;
    derivIBuf.create(prevPyr[0].rows + winSize.height*2, prevPyr[0].cols + winSize.width*2, CV_MAKETYPE(derivDepth, prevPyr[0].channels() * 2));
    //
    for (int level = config.optical_flow_levels; level >= 0; level--)
    {
      // 计算图像梯度
      cv::Mat derivI;
      cv::Size imgSize = prevPyr[level * lvlStep1].size();
      cv::Mat _derivI( imgSize.height + winSize.height*2,
          imgSize.width + winSize.width*2, derivIBuf.type(), derivIBuf.ptr() );
      derivI = _derivI(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
      cv::calcScharrDeriv(prevPyr[level * lvlStep1], derivI); // 计算图像的Scharr导数
      cv::copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, BORDER_CONSTANT|BORDER_ISOLATED); // 扩展边界
      
      //
      // 使用并行计算加速光流追踪
      /*
      typedef cv::detail::LKTrackerInvoker LKTrackerInvoker;
      parallel_for_(Range(0, npoints), LKTrackerInvoker(prevPyr[level * lvlStep1], derivI,
                                                        nextPyr[level * lvlStep2], prevPts, nextPts,
                                                        status, err,
                                                        winSize, criteria, level, maxLevel,
                                                        flags, (float)minEigThreshold));*/
    }
    #endif
    //
  #else  
    size_t num_points = transform_map_1.size();
    std::vector<std::pair<KeypointId, TrackCnt>> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;
    ids.reserve(num_points);
    init_vec.reserve(num_points);
    for (const auto& kv : transform_map_1) {
      ids.push_back(std::make_pair(kv.first, kv.second.second));
      init_vec.push_back(kv.second.first);
    }
    tbb::concurrent_unordered_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>,
                                  std::hash<KeypointId>> result;
    //
    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      //
      for (size_t r = range.begin(); r != range.end(); ++r) { // r表示点在vector容器中的序号
        const std::pair<KeypointId, TrackCnt> id = ids[r]; // 得到点的id
        const Eigen::AffineCompact2f& transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = transform_1;
        // trackpoint2()
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);
    tbb::parallel_for(range, compute_func);

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
  #endif  
  }

  #if 0
  void trackPoints3(const basalt::ManagedImagePyr<uint16_t>& pyr_1,
                    const basalt::ManagedImagePyr<uint16_t>& pyr_2,
                    const Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>>& transform_map_1,
                    Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>>& transform_map_2)
  {
    size_t num_points = transform_map_1.size();
    std::vector<std::pair<KeypointId, TrackCnt>> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;
    ids.reserve(num_points);
    init_vec.reserve(num_points);
    for (const auto& kv : transform_map_1) {
      ids.push_back(std::make_pair(kv.first, kv.second.second));
      init_vec.push_back(kv.second.first);
    }
    tbb::concurrent_unordered_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>,
                                  std::hash<KeypointId>> result;
    //
    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      //
      for (size_t r = range.begin(); r != range.end(); ++r) { // r表示点在vector容器中的序号
        const std::pair<KeypointId, TrackCnt> id = ids[r]; // 得到点的id

        const Eigen::AffineCompact2f& transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = transform_1;

        bool valid = trackPoint3(pyr_1, pyr_2, transform_1, transform_2);
        if(valid)
        {
          result[id.first] = std::make_pair(transform_2, id.second + 1);
        }
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);
    tbb::parallel_for(range, compute_func);

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
  }

  inline bool trackPoint3(const basalt::ManagedImagePyr<uint16_t>& old_pyr,
                         const basalt::ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform,
                         Eigen::AffineCompact2f& transform) const {
    //
    bool patch_valid = true;
    transform.linear().setIdentity();
    //
    for (int level = config.optical_flow_levels; level >= 0 && patch_valid; level--)
    {
      //
      const Scalar scale = 1 << level;
      transform.translation() /= scale;

      PatchT2 p(old_pyr.lvl(level), old_transform.translation() / scale);

      patch_valid &= p.valid;
      if (patch_valid) {
        // Perform tracking on current level
        patch_valid &= trackPointAtLevel3(pyr.lvl(level), p, transform);
      }

      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  inline bool trackPointAtLevel3(const Image<const uint16_t>& img_2,
                                const PatchT2& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    // 指定循环次数，且patch合法
    // int iteration = 0; // added this line for test 2023-12-15.
    for (int iteration = 0;
        //  patch_valid && iteration < config.optical_flow_max_iterations;
         patch_valid && iteration < max_iterations_;
         iteration++) {
      typename PatchT2::VectorP res;

      // patch旋转平移到当前帧
      typename PatchT2::Matrix2P transformed_pat =
          transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      // 计算参考帧和当前帧patern对应的像素值差
      patch_valid &= dp.residual(img_2, transformed_pat, res);

      if (patch_valid) {
        // 计算增量，扰动更新
        const Vector3 inc = -dp.H_se2_inv_J_se2_T * res; // 求增量Δx = - H^-1 * J^T * r

        // avoid NaN in increment (leads to SE2::exp crashing)
        patch_valid &= inc.array().isFinite().all();

        // avoid very large increment
        patch_valid &= inc.template lpNorm<Eigen::Infinity>() < 1e6;

        if (patch_valid) {
          transform *= SE2::exp(inc).matrix(); // 更新状态量

          const int filter_margin = 2;

          // 判断更新后的像素坐标是否在图像img_2范围内
          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
        }
      }
    }

    // std::cout << "num_it = " << iteration << std::endl;

    return patch_valid;
  }
  // the end.
  #endif

  // trackPoints函数是用来追踪两幅图像的特征点的，输入是 金字塔1, 金字塔2, 1中的点, 输出的是2追踪1的点（即1中的点，被经过追踪之后，得到在图像2中的像素坐标）
  // 这里的1指的是上一帧，那么2就是当前帧；或者1指的是当前帧的左目， 那么2指的是当前帧的右目
  void trackPoints(const basalt::ManagedImagePyr<uint16_t>& pyr_1,
                   const basalt::ManagedImagePyr<uint16_t>& pyr_2,
                   const Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>>&
                       transform_map_1, // 1中的点
                   Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>>&
                       transform_map_2) /*const*/ { // 用于返回追踪到的点
    // num_points为1中点的个数
    size_t num_points = transform_map_1.size();

    std::vector<std::pair<KeypointId, TrackCnt>> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);

    // 1.特征点类型转换map->vector
    for (const auto& kv : transform_map_1) {
      ids.push_back(std::make_pair(kv.first, kv.second.second)); // 1中点的id
      init_vec.push_back(kv.second.first); // 1中点的信息（在2d图像上的旋转和平移信息）
    }

    // 定义输出结果的容器
    tbb::concurrent_unordered_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>,
                                  std::hash<KeypointId>>
        result;

    #ifdef _CALC_TIME_
    clock_t clock_sum = 0;
    #endif

    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      
      // 遍历每一个特征点
      for (size_t r = range.begin(); r != range.end(); ++r) { // r表示点在vector容器中的序号
        const std::pair<KeypointId, TrackCnt> id = ids[r]; // 得到点的id

        // 取出特征点在参考帧或者左目中的像素位置transform_1
        const Eigen::AffineCompact2f& transform_1 = init_vec[r];
        //用transform_1 初始化特征点在当前帧或者右目的位置
        Eigen::AffineCompact2f transform_2 = transform_1;

        clock_t clock_one_point = 0; // added on 2024-11-28.

        // 使用特征点 进行光流正向追踪
        // bool valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2);
        bool valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2, clock_one_point);
        #ifdef _CALC_TIME_
        clock_sum += clock_one_point;
        #endif

        if (valid) {
          // 如果正向追踪合法，使用反向追踪，由当前帧追踪参考帧
          Eigen::AffineCompact2f transform_1_recovered = transform_2;

          // valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);
          valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered, clock_one_point);
          #ifdef _CALC_TIME_
          clock_sum += clock_one_point;
          #endif

          if (valid) {
            // 计算反向光流恢复的前一帧的坐标，跟前一帧已知的坐标之间的欧氏距离的平方（点坐标求差之后再求二范数的平方）
            Scalar dist2 = (transform_1.translation() -
                            transform_1_recovered.translation())
                               .squaredNorm();

            // 判断正向光流和反向光流的误差是否合法，合法则保存结果
            if (dist2 < config.optical_flow_max_recovered_dist2) {
              // result[id] = transform_2;
              result[id.first] = std::make_pair(transform_2, id.second + 1);
            }
            // 2024-12-27.
            else
            {
              // map_track_failed.emplace(id.first, GT_MAX_RECOVERED_DIS);
              gt_max_recovered_dis_cnt++;
            }
            // the end.
          }
          // 2024-12-27.
          else
          {
            // map_track_failed.emplace(id.first, BACK_FLOW_FAILED);
            back_track_failed_cnt++;
          }
          // the end.
        }
        // 2024-12-27.
        else
        {
          // map_track_failed.emplace(id.first, FORW_FLOW_FAILED);
          forw_track_failed_cnt++;
        }
        // the end.
      }
    };

    #ifndef _CALC_TIME_
    tbb::blocked_range<size_t> range(0, num_points); // 定义遍历范围，用数学半开半闭区间表示为[0, num_points).
    #else
    tbb::blocked_range<size_t> range(0, num_points, num_points); // for test single thread on 2024-11-13.
    #endif

    // auto start = std::chrono::high_resolution_clock::now(); // 2024-11-15
    // 并行（计算）追踪特征点，SPMD（Single Program/Multiple Data 单一的过程中，多个数据或单个程序，多数据）
    tbb::parallel_for(range, compute_func);
    // compute_func(range);
    // auto end = std::chrono::high_resolution_clock::now(); // 2024-11-15
    // auto diff = end - start;
    // std::cout << std::setprecision(6) << std::fixed << "trackPoints " << "[" << num_points << "]: "
      // << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;

#ifdef _CALC_TIME_
    std::cout << "trackPoints " << "[" << num_points << "]: "
      << clock_sum /*<< " clock"*/ << std::endl;
#endif

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
  }

  // trackPoint函数是追踪一个点
  inline bool trackPoint(const basalt::ManagedImagePyr<uint16_t>& old_pyr,
                         const basalt::ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform,
                         Eigen::AffineCompact2f& transform, clock_t &clock_one_point) const {
    bool patch_valid = true;

    // AffineCompact2f有旋转和平移成员
    // 投影或仿射变换矩阵，参见Transform类。泛型仿射变换用Transform类表示，其实质是（Dim+1）^2的矩阵
    transform.linear().setIdentity(); //? 这里是将transform(本质是一个矩阵，包含旋转和平移)设为单位矩阵吗，应该仅仅是旋转部分设为单位阵

    /*
     * 类`Transform`表示使用齐次运算的仿射变换或投影变换。 
     *
     * Transform::linear()直接返回变换矩阵中的线性部分it returns the linear part of the transformation.
     * Transform::rotation()返回线性部分中的旋转分量。
     * 但由于线性部分包含 not only rotation but also reflection, shear and scaling
     *
     * Eigen::Transform::linear()用法的补充说明：
     * 例如，一个仿射变换'A'是由一个线性部分'L'和一个平移't'组成的，这样由'A'变换一个点'p'等价于：p' = L * p + t
     * 
     *                 | L   t |
     * 所以变换矩阵 T = |       |    的Linear部分即为左上角的Eigen::Matrix3d旋转矩阵。
     *                 | 0   1 |
     * 
     * 代码中另外一个表示矩阵的方式：
     * [L, t]
     * [0, 1]
     * 
     * 对于本程序来说，Eigen::AffineCompact2f其实就是 Eigen::Transform 的别名，
     * 其本质是3*3的矩阵，左上角2*2 用来表示旋转， 右边2*1表示平移，
     * 平移最根本的目的，在这里是为了表示点在二维像素平面上的坐标。 
     * 
     * 用齐次向量表示：
     * [p'] = [L t] * [p] = A * [p]
     * [1 ]   [0 1]   [1]       [1]
     * 
     * with:
     * 
     * A = [L t]
     *     [0 1]
     * 
     * 所以线性部分对应于齐次矩阵表示的左上角。它对应于旋转、缩放和剪切的混合物。
     * 
     */


    #ifdef _CALC_TIME_
    clock_one_point = 0;
    #endif

    // 从金字塔最顶层到最底层迭代优化
    for (int level = config.optical_flow_levels; level >= 0 && patch_valid;
         level--) {
      
      // 计算尺度
      const Scalar scale = 1 << level; // 相当于 scale = 2 ^ level

      transform.translation() /= scale; // 像素坐标，缩放到对应层

      #ifdef _CALC_TIME_
      PatchT::clock_sum = 0;
      #endif
      // 获得patch在对应层初始位置
      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale); // 参考帧的金字塔对应层，以及对应的坐标
      #ifdef _CALC_TIME_
      clock_one_point += PatchT::clock_sum;
      #endif
#if !defined(FORWARD_ADDITIVE_APPROACH)
      patch_valid &= p.valid;
#else
      // if(!p.valid && level == 0) patch_valid = false;
      patch_valid &= (p.valid? true: (level != 0));
#endif      
      if (patch_valid) {
        // Perform tracking on current level
#if !defined(FORWARD_ADDITIVE_APPROACH)        
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);
#else        
        patch_valid &= trackPointAtLevel2(pyr.lvl(level), p, transform);
#endif        
      }

      // 得到的结果变换到第0层对应的尺度
      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const uint16_t>& img_2,
                                const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    // 指定循环次数，且patch合法
    // int iteration = 0; // added this line for test 2023-12-15.
    for (int iteration = 0;
        //  patch_valid && iteration < config.optical_flow_max_iterations;
         patch_valid && iteration < max_iterations_;
         iteration++) {
      typename PatchT::VectorP res;

      // patch旋转平移到当前帧
      typename PatchT::Matrix2P transformed_pat =
          transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      // 计算参考帧和当前帧patern对应的像素值差
      patch_valid &= dp.residual(img_2, transformed_pat, res);

      if (patch_valid) {
        // 计算增量，扰动更新
        const Vector3 inc = -dp.H_se2_inv_J_se2_T * res; // 求增量Δx = - H^-1 * J^T * r

        // avoid NaN in increment (leads to SE2::exp crashing)
        patch_valid &= inc.array().isFinite().all();

        // avoid very large increment
        patch_valid &= inc.template lpNorm<Eigen::Infinity>() < 1e6;

        if (patch_valid) {
          transform *= SE2::exp(inc).matrix(); // 更新状态量

          const int filter_margin = 2;

          // 判断更新后的像素坐标是否在图像img_2范围内
          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
        }
      }
    }

    // std::cout << "num_it = " << iteration << std::endl;

    return patch_valid;
  }

  // 2025-1-14
  inline bool trackPointAtLevel2(const Image<const uint16_t>& img_2,
                                const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    typename PatchT::Vector2 prevDelta;
    prevDelta.setZero();

    // 指定循环次数，且patch合法
    // int iteration = 0; // added this line for test 2023-12-15.
    for (int iteration = 0;
        //  patch_valid && iteration < config.optical_flow_max_iterations;
         patch_valid && iteration < max_iterations_;
         iteration++) {
      typename PatchT::Vector2 delta;

      // 计算参考帧和当前帧patern对应的像素值差
      patch_valid &= dp.calcResidual(img_2, transform.translation(), delta);

      if (patch_valid) {
        // 计算增量，扰动更新
        // const Vector3 inc = -dp.H_se2_inv_J_se2_T * res; // 求增量Δx = - H^-1 * J^T * r
        

        // avoid NaN in increment (leads to SE2::exp crashing)
        patch_valid &= delta.array().isFinite().all();
        if(!patch_valid) std::cout << "1 delta= " << delta.transpose() << std::endl;

        // avoid very large increment
        patch_valid &= delta.template lpNorm<Eigen::Infinity>() < 1e6;
        if(!patch_valid) std::cout << "2 delta= " << delta.transpose() << std::endl;

        if (patch_valid) {
          // 更新状态量
          transform.translation() += delta;

          const int filter_margin = 2;

          // 判断更新后的像素坐标是否在图像img_2范围内
          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
          if(!patch_valid) std::cout << "3 transform= " << transform.translation().transpose() << std::endl;
        }
      }

      //
      if( delta.squaredNorm() <= 0.01 ) // 位移增量的2范数的平方小于阈值认为收敛
        break;

      // 如果两次迭代的位移差异非常小，认为已经收敛
      if( iteration > 0 && std::abs(delta.x() + prevDelta.x()) < 0.01 &&
        std::abs(delta.y() + prevDelta.y()) < 0.01 )
      {
        // 如果迭代过程收敛，微调特征点的位置（减去一半的位移量）并跳出循环
        // nextPts[ptidx] -= delta*0.5f;
        break;
      }
      prevDelta = delta; // 更新 prevDelta 为当前的 delta，为下一次迭代做准备。    
      //
    }

    // std::cout << "num_it = " << iteration << std::endl;

    return patch_valid;
  }
  // the end.

  void addPoints_pyr(const std::vector<cv::Mat> &pyr0, const std::vector<cv::Mat> &pyr1) {
    // test
    #if 0
    {
      for(int i = 0; i < pyr0.size(); i++)
      {
        cv::imshow("pyramid image", pyr0.at(i));
        cv::waitKey(0);
      }

      for(int i = 0; i < pyr1.size(); i++)
      {
        std::cout << "Image channels: " << pyr1[i].channels() << std::endl;
        std::cout << "Image size: " << pyr1[i].size() << std::endl;
        std::cout << "Image isContinuous: " << pyr1[i].isContinuous() << std::endl;
        std::cout << "Image step: " << pyr1[i].step[0] << std::endl;
        // cv::imshow("pyramid image", pyr1.at(i));
        // cv::waitKey(0);
      }
      
    }
    #endif

    // step 1 在当前帧第0层检测特征(划分网格，在网格中只保存响应比较好的和以前追踪的)
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    // 将以前追踪到的点放入到pts0,进行临时保存
    // kv为Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>类型的容器中的一个元素（键值对）
    for (const auto& kv : transforms->observations.at(0)) { // at(0)表示取左目的观测数据
      pts0.emplace_back(kv.second.first.translation().cast<double>()); // 取2d图像的平移部分，即二维像素坐标
    }

    KeypointsData kd; // 用来存储新检测到的特征点
    // detectKeypoints2(pyramid->at(0).lvl(0), kd, grid_size_, 1, pts0);
    detectKeypoints3(pyr0.at(0), kd, grid_size_, 1, pts0);

    // Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>> new_poses0,
        // new_poses1;

    vector<cv::Point2f> cam0_pts, cam1_pts;
    vector<int> ids;    

    // step 2 遍历特征点, 每个特征点利用特征点 初始化光流金字塔的初值
    // 添加新的特征点的观测值
    for (size_t i = 0; i < kd.corners.size(); i++) {
      Eigen::AffineCompact2f transform;
      transform.setIdentity(); //旋转 设置为单位阵
      transform.translation() = kd.corners[i].cast<Scalar>(); // 角点坐标，保存到transform的平移部分

      cam0_pts.emplace_back(cv::Point2f(kd.corners[i].x(), kd.corners[i].y()));

      // 特征点转换成输出结果的数据类型map
      // transforms->observations.at(0)[last_keypoint_id] = transform; // 键值对来存储特征点，类型为Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>
      transforms->observations.at(0)[last_keypoint_id] = std::make_pair(transform, 0);

      // new_poses0[last_keypoint_id] = std::make_pair(transform, 0);

      ids.emplace_back(last_keypoint_id);

      last_keypoint_id++; // last keypoint id 是一个类成员变量
    }

    // step 3 如果是有双目的话，使用左目新提的点，进行左右目追踪。右目保留和左目追踪上的特征点
    //如果是双目相机,我们使用光流追踪算法，即计算Left image 提取的特征点在right image图像中的位置
    if (calib.intrinsics.size() > 1 && cam0_pts.size()) {//相机内参是否大于1
      /*
      // 使用左目提取的特征点使用光流得到右目上特征点的位置
      trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);
      // 保存结果，因为是右目的点，因此保存到下标1中
      for (const auto& kv : new_poses1) {
        transforms->observations.at(1).emplace(kv);
      }*/

      vector<uchar> status;
      vector<float> err;
      // std::cout << "cam0_pts.size=" << cam0_pts.size() << std::endl;
      // cv::calcOpticalFlowPyrLK(img0, img1, cam0_pts, cam1_pts, status, err, cv::Size(21, 21), 3);
      cv::calcOpticalFlowPyrLK(pyr0, pyr1, cam0_pts, cam1_pts, status, err, cv::Size(21, 21), 3);

      int size = int(cam1_pts.size());
        for (int j = 0; j < size; j++)
            // 通过图像边界剔除outlier
            if (status[j] && !inBorder(cam1_pts[j]))    // 追踪状态好检查在不在图像范围
                status[j] = 0;

        // reduceVector(cam0_pts, status);
        reduceVector(cam1_pts, status);
        reduceVector(ids, status);
        // reduceVector(track_cnt[i], status);

        // rejectWithF(cam0_pts, cam1_pts, kpt_ids[i], track_cnt[i], ds_cam[i], FOCAL_LENGTH[i]);

        for (size_t i = 0; i < cam1_pts.size(); i++) {

          Eigen::AffineCompact2f transform;
          transform.setIdentity(); //旋转 设置为单位阵
          transform.translation() = Eigen::Vector2d(cam1_pts[i].x, cam1_pts[i].y).cast<Scalar>();

          transforms->observations.at(1).emplace(ids[i], std::make_pair(transform, 1));
        }

    }
  }

  void addPoints2(const cv::Mat &img0, const cv::Mat &img1) {
     // step 1 在当前帧第0层检测特征(划分网格，在网格中只保存响应比较好的和以前追踪的)
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    // 将以前追踪到的点放入到pts0,进行临时保存
    // kv为Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>类型的容器中的一个元素（键值对）
    for (const auto& kv : transforms->observations.at(0)) { // at(0)表示取左目的观测数据
      pts0.emplace_back(kv.second.first.translation().cast<double>()); // 取2d图像的平移部分，即二维像素坐标
    }

    KeypointsData kd; // 用来存储新检测到的特征点
    // detectKeypoints2(pyramid->at(0).lvl(0), kd, grid_size_, 1, pts0);
    detectKeypoints3(img0, kd, grid_size_, 1, pts0);

    // Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>> new_poses0,
        // new_poses1;

    vector<cv::Point2f> cam0_pts, cam1_pts;
    vector<int> ids;    

    // step 2 遍历特征点, 每个特征点利用特征点 初始化光流金字塔的初值
    // 添加新的特征点的观测值
    for (size_t i = 0; i < kd.corners.size(); i++) {
      Eigen::AffineCompact2f transform;
      transform.setIdentity(); //旋转 设置为单位阵
      transform.translation() = kd.corners[i].cast<Scalar>(); // 角点坐标，保存到transform的平移部分

      cam0_pts.emplace_back(cv::Point2f(kd.corners[i].x(), kd.corners[i].y()));

      // 特征点转换成输出结果的数据类型map
      // transforms->observations.at(0)[last_keypoint_id] = transform; // 键值对来存储特征点，类型为Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>
      transforms->observations.at(0)[last_keypoint_id] = std::make_pair(transform, 0);

      // new_poses0[last_keypoint_id] = std::make_pair(transform, 0);

      ids.emplace_back(last_keypoint_id);

      last_keypoint_id++; // last keypoint id 是一个类成员变量
    }

    // step 3 如果是有双目的话，使用左目新提的点，进行左右目追踪。右目保留和左目追踪上的特征点
    //如果是双目相机,我们使用光流追踪算法，即计算Left image 提取的特征点在right image图像中的位置
    if (calib.intrinsics.size() > 1 && cam0_pts.size()) {//相机内参是否大于1
      /*
      // 使用左目提取的特征点使用光流得到右目上特征点的位置
      trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);
      // 保存结果，因为是右目的点，因此保存到下标1中
      for (const auto& kv : new_poses1) {
        transforms->observations.at(1).emplace(kv);
      }*/

      vector<uchar> status;
      vector<float> err;
      // std::cout << "cam0_pts.size=" << cam0_pts.size() << std::endl;
      cv::calcOpticalFlowPyrLK(img0, img1, cam0_pts, cam1_pts, status, err, cv::Size(21, 21), 3);

      int size = int(cam1_pts.size());
        for (int j = 0; j < size; j++)
            // 通过图像边界剔除outlier
            if (status[j] && !inBorder(cam1_pts[j]))    // 追踪状态好检查在不在图像范围
                status[j] = 0;

        // reduceVector(cam0_pts, status);
        reduceVector(cam1_pts, status);
        reduceVector(ids, status);
        // reduceVector(track_cnt[i], status);

        // rejectWithF(cam0_pts, cam1_pts, kpt_ids[i], track_cnt[i], ds_cam[i], FOCAL_LENGTH[i]);

        for (size_t i = 0; i < cam1_pts.size(); i++) {

          Eigen::AffineCompact2f transform;
          transform.setIdentity(); //旋转 设置为单位阵
          transform.translation() = Eigen::Vector2d(cam1_pts[i].x, cam1_pts[i].y).cast<Scalar>();

          transforms->observations.at(1).emplace(ids[i], std::make_pair(transform, 1));
        }

    }
  }

  void addPoints() {

    // step 1 在当前帧第0层检测特征(划分网格，在网格中只保存响应比较好的和以前追踪的)
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    // 将以前追踪到的点放入到pts0,进行临时保存
    // kv为Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>类型的容器中的一个元素（键值对）
    for (const auto& kv : transforms->observations.at(0)) { // at(0)表示取左目的观测数据
      pts0.emplace_back(kv.second.first.translation().cast<double>()); // 取2d图像的平移部分，即二维像素坐标
    }

    KeypointsData kd; // 用来存储新检测到的特征点

    // 每个cell的大小默认是50 ， 每个cell提取1个特征点
    // 检测特征点
    // 参数1.图像 参数2.输出特征点容器 参数3,制定cell大小 参数4,每个cell多少特征点 参数5.成功追踪的特征点传递进去
/*
 * comment 2023-12-15.
    detectKeypoints(pyramid->at(0).lvl(0), kd,
                    config.optical_flow_detection_grid_size, 1, pts0);                
*/
    
    // detectKeypoints(pyramid->at(0).lvl(0), kd, grid_size_, 1, pts0);
    detectKeypoints2(pyramid->at(0).lvl(0), kd, grid_size_, 1, pts0);
    // detectKeypoints(pyramid->at(0).lvl(0), kd, grid_size_, 1, pts0, mask);

    Eigen::aligned_map<KeypointId, std::pair<Eigen::AffineCompact2f, TrackCnt>> new_poses0,
        new_poses1;

    // step 2 遍历特征点, 每个特征点利用特征点 初始化光流金字塔的初值
    // 添加新的特征点的观测值
    for (size_t i = 0; i < kd.corners.size(); i++) {
      Eigen::AffineCompact2f transform;
      transform.setIdentity(); //旋转 设置为单位阵
      transform.translation() = kd.corners[i].cast<Scalar>(); // 角点坐标，保存到transform的平移部分

      // 特征点转换成输出结果的数据类型map
      // transforms->observations.at(0)[last_keypoint_id] = transform; // 键值对来存储特征点，类型为Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>
      transforms->observations.at(0)[last_keypoint_id] = std::make_pair(transform, 0);
      // std::pair<KeypointId, TrackCnt> key(last_keypoint_id, 1);
      // transforms->observations.at(0)[key] = transform;
      // transforms->observations.at(0)[std::make_pair(last_keypoint_id, 1)] = transform; // another method 2024-12-22
      // new_poses0[last_keypoint_id] = transform;
      new_poses0[last_keypoint_id] = std::make_pair(transform, 0);

      last_keypoint_id++; // last keypoint id 是一个类成员变量
    }

    // step 3 如果是有双目的话，使用左目新提的点，进行左右目追踪。右目保留和左目追踪上的特征点
    //如果是双目相机,我们使用光流追踪算法，即计算Left image 提取的特征点在right image图像中的位置
    if (calib.intrinsics.size() > 1) {//相机内参是否大于1
      // 使用左目提取的特征点使用光流得到右目上特征点的位置
      #if 0
      trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);
      #elif 1
      trackPointsFA(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);
      #else
      trackPoints3(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);
      #endif
      // 保存结果，因为是右目的点，因此保存到下标1中
      for (const auto& kv : new_poses1) {
        transforms->observations.at(1).emplace(kv);
      }
    }
  }

  void filterPoints() {
    // 如果相机内参小于2，无法使用双目基础矩阵剔除外点
    if (calib.intrinsics.size() < 2) return;

    // set记录哪些特征点需要被剔除
    std::set<KeypointId> lm_to_remove;

    std::vector<KeypointId> kpid;
    Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

    // step l: 获得left image 和 right image 都可以看到的feature
    // 遍历右目特征点，查询是否在左目中被观测
    for (const auto& kv : transforms->observations.at(1)) {
      auto it = transforms->observations.at(0).find(kv.first);

      // 如果在左目中查询到，则把特征点对应ID保存
      if (it != transforms->observations.at(0).end()) {
        proj0.emplace_back(it->second.first.translation());
        proj1.emplace_back(kv.second.first.translation());
        kpid.emplace_back(kv.first);
      }
    }

    // step 2: 将feature 反投影为归一化坐标的3d点
    Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
    std::vector<bool> p3d0_success, p3d1_success;

    calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success);
    calib.intrinsics[1].unproject(proj1, p3d1, p3d1_success);

    // step 3: 使用对极几何剔除外点
    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) {
        const double epipolar_error =
            std::abs(p3d0[i].transpose() * E * p3d1[i]);

        //如果距离大于判定阈值 则不合法
        if (epipolar_error > config.optical_flow_epipolar_error) {
          lm_to_remove.emplace(kpid[i]);
        }
      } else {
        lm_to_remove.emplace(kpid[i]);
      }
    }

    // step 4: 只剔除外点在right image中的观测
    for (int id : lm_to_remove) {
      transforms->observations.at(1).erase(id);
      // 只剔除右目，保留左目，就不会造成左目光流中断
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  int64_t t_ns;

  size_t frame_counter;

  KeypointId last_keypoint_id;

  VioConfig config;
  basalt::Calibration<Scalar> calib;

  OpticalFlowResult::Ptr transforms; // 用于存放第一次提取的全新特征点，或者光流追踪之后的特征点加上提取的新点
  std::shared_ptr<std::vector<basalt::ManagedImagePyr<uint16_t>>> old_pyramid,
      pyramid; // 智能指针指向vector，下标0，表示左目的金字塔，下标1表示右目的金字塔
               // old_pyramid表示上一图像的金字塔，pyramid表示当前图像的金字塔

  Matrix4 E;

  std::shared_ptr<std::thread> processing_thread;

  // 2023-11-19.
  std::mutex reset_mutex; 
  bool isReset_ { false };
  void SetReset(bool bl) { isReset_ =bl; }
  inline bool GetReset() { return isReset_;}
  // the end.

  bool isZeroVelocity_ { false };
  bool isSlowVelocity_ { true };
  std::atomic<int> grid_size_ { 0 };
  std::atomic<int> max_iterations_ { 0 };

#ifdef _GOOD_FEATURE_TO_TRACK_
  cv::Mat mask;
#endif

// #ifdef _CALC_TIME_
  // clock_t clock_one_point = 0;
// #endif

/*
  vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
  cv::Mat prev_img, cur_img, forw_img;
*/

  vector<cv::Point2f> cur_pts[2], forw_pts[2];
  cv::Mat cur_img[2], forw_img[2];

  int ROW;
  int COL;
  int FISHEYE { 0 };
  int MIN_DIST = 30;
  cv::Mat mask;
  cv::Mat fisheye_mask;

  vector<int> kpt_ids[2];
  vector<int> track_cnt[2];

  size_t x_start;
  size_t x_stop;

  size_t y_start;
  size_t y_stop;

  // 计算cell的总个数， cell按照矩阵的方式排列：e.g. 480 / 80 + 1, 640 / 80 + 1
  Eigen::MatrixXi cells;

  constexpr static int EDGE_THRESHOLD = 19;

  // hm::DoubleSphereCamera *m_pDsCam[2];
  std::shared_ptr<hm::DoubleSphereCamera> ds_cam[2];
  int FOCAL_LENGTH[2];
  double F_THRESHOLD = 1.0;

  // std::unordered_map<KeypointId, TrackFailedReason> map_track_failed;
  int forw_track_failed_cnt = 0;
  int back_track_failed_cnt = 0;
  int gt_max_recovered_dis_cnt = 0;

};

}  // namespace basalt

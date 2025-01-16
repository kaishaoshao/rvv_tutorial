
#pragma once

#include <tbb/blocked_range.h>

#include <memory>
#include <opencv2/opencv.hpp>

#ifndef CV_OVERRIDE
#  define CV_OVERRIDE override
#endif

#ifndef WX_OVERRIDE
#  define WX_OVERRIDE //override
#endif

#ifndef WX_VIRTUAL
#  define WX_VIRTUAL //virtual
#endif

namespace wx::liu
{
using InputArray = cv::InputArray;
using InputOutputArray = cv::InputOutputArray;
using OutputArray = cv::OutputArray;
using Size = cv::Size;
using TermCriteria = cv::TermCriteria;
using Mat = cv::Mat;
// using template<> DataType<short> = cv::DataType<short>;
using Point2f = cv::Point2f;
using _InputArray = cv::_InputArray;
using Point2i = cv::Point2i;
// using Point = cv::Point;
typedef cv::Point2i Point;
typedef short deriv_type;
using Range = cv::Range;

typedef std::string String;

// class CV_EXPORTS ParallelLoopBody
// {
// public:
//     virtual ~ParallelLoopBody();
//     virtual void operator() (const Range& range) const = 0;
// };

struct ScharrDerivInvoker// : cv::ParallelLoopBody
{
    ScharrDerivInvoker(const Mat& _src, const Mat& _dst)
        : src(_src), dst(_dst)
    { }

    // void operator()(const Range& range) const CV_OVERRIDE;
    void operator()(const tbb::blocked_range<size_t>& range) const WX_OVERRIDE;

    const Mat& src;
    const Mat& dst;
};

struct LKTrackerInvoker// : cv::ParallelLoopBody
{
    LKTrackerInvoker( const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                        const Point2f* _prevPts, Point2f* _nextPts,
                        uchar* _status, float* _err,
                        Size _winSize, TermCriteria _criteria,
                        int _level, int _maxLevel, int _flags, float _minEigThreshold );

    // void operator()(const Range& range) const CV_OVERRIDE;
    void operator()(const tbb::blocked_range<size_t>& range) const WX_OVERRIDE;

    const Mat* prevImg;
    const Mat* nextImg;
    const Mat* prevDeriv;
    const Point2f* prevPts;
    Point2f* nextPts;
    uchar* status;
    float* err;
    Size winSize;
    TermCriteria criteria;
    int level;
    int maxLevel;
    int flags;
    float minEigThreshold;
};

struct LKOpticalFlow
{
public:
    using Ptr = std::shared_ptr<LKOpticalFlow>; // 定义指针

    LKOpticalFlow(Size winSize_ = Size(21,21),
                  int maxLevel_ = 3,
                  TermCriteria criteria_ = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                  int flags_ = 0,
                  double minEigThreshold_ = 1e-4) :
        winSize(winSize_), maxLevel(maxLevel_), criteria(criteria_), flags(flags_), minEigThreshold(minEigThreshold_)
    {
        //
    }

    WX_VIRTUAL Size getWinSize() const WX_OVERRIDE { return winSize;}
    WX_VIRTUAL void setWinSize(Size winSize_) WX_OVERRIDE { winSize = winSize_;}

    WX_VIRTUAL int getMaxLevel() const WX_OVERRIDE { return maxLevel;}
    WX_VIRTUAL void setMaxLevel(int maxLevel_) WX_OVERRIDE { maxLevel = maxLevel_;}

    WX_VIRTUAL TermCriteria getTermCriteria() const WX_OVERRIDE { return criteria;}
    WX_VIRTUAL void setTermCriteria(TermCriteria& crit_) WX_OVERRIDE { criteria=crit_;}

    WX_VIRTUAL int getFlags() const WX_OVERRIDE { return flags; }
    WX_VIRTUAL void setFlags(int flags_) WX_OVERRIDE { flags=flags_;}

    WX_VIRTUAL double getMinEigThreshold() const WX_OVERRIDE { return minEigThreshold;}
    WX_VIRTUAL void setMinEigThreshold(double minEigThreshold_) WX_OVERRIDE { minEigThreshold=minEigThreshold_;}

    WX_VIRTUAL void calc(InputArray prevImg, InputArray nextImg,
                        InputArray prevPts, InputOutputArray nextPts,
                        OutputArray status,
                        OutputArray err = cv::noArray()) WX_OVERRIDE;

    WX_VIRTUAL String getDefaultName() const WX_OVERRIDE { return "SparseOpticalFlow.SparsePyrLKOpticalFlow"; }

private:    
    Size winSize;
    int maxLevel;
    TermCriteria criteria;
    int flags;
    double minEigThreshold;
};

void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                           InputArray prevPts, InputOutputArray nextPts,
                           OutputArray status, OutputArray err,
                           Size winSize = Size(21,21), int maxLevel = 3,
                           TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                           int flags = 0, double minEigThreshold = 1e-4 );

} // namespace wx::liu
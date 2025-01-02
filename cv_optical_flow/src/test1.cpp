//
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

/*int cv::buildOpticalFlowPyramid	(	InputArray 	img,
                                            OutputArrayOfArrays 	pyramid,
                                            Size 	winSize,
                                            int 	maxLevel,
                                            bool 	withDerivatives = true,
                                            int 	pyrBorder = BORDER_REFLECT_101,
                                            int 	derivBorder = BORDER_CONSTANT,
                                            bool 	tryReuseInputImage = true 
                                            )

Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.

Parameters
img	8-bit input image.
pyramid	output pyramid.
winSize	window size of optical flow algorithm. Must be not less than winSize argument of calcOpticalFlowPyrLK. It is needed to calculate required padding for pyramid levels.
maxLevel	0-based maximal pyramid level number.
withDerivatives	set to precompute gradients for the every pyramid level. If pyramid is constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.
pyrBorder	the border mode for pyramid layers.
derivBorder	the border mode for gradients.
tryReuseInputImage	put ROI of input image into the pyramid if possible. You can pass false to force data copying.
Returns
number of levels in constructed pyramid. Can be less than maxLevel.	

*/

/*
void addPaddingToPyramid(const std::vector<Mat>& pyramid, const Size& winSize, std::vector<Mat>& paddedPyramid) {
    for (size_t i = 0; i < pyramid.size(); ++i) {
        const Mat& img = pyramid[i];
        
        // 计算填充大小
        int top = winSize.height / 2;
        int bottom = top;
        int left = winSize.width / 2;
        int right = left;
        
        // 添加边界填充
        Mat paddedImg;
        copyMakeBorder(img, paddedImg, top, bottom, left, right, BORDER_REFLECT_101);

        // 保存填充后的图像
        paddedPyramid.push_back(paddedImg);
    }
}*/

// void addPaddingToPyramid(const std::vector<Mat>& pyramid, const Size& winSize, int pyrBorder, std::vector<Mat>& paddedPyramid) {
void addPaddingToPyramid(const std::vector<Mat>& pyramid, std::vector<Mat>& paddedPyramid, 
                         const Size& winSize = Size(21,21), int pyrBorder = BORDER_REFLECT_101) {
    int pyrSize = pyramid.size();
    paddedPyramid.resize(pyrSize);;
    
    for (size_t i = 0; i < pyramid.size(); ++i) {

        Mat& temp = paddedPyramid.at(i);
        const Mat &img = pyramid.at(i);

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
        if(i != 0) border = pyrBorder|BORDER_ISOLATED;
        if(pyrBorder != BORDER_TRANSPARENT)
            // copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder|BORDER_ISOLATED);
            copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, border);

        temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);

    }
}

void addPadding2Img(const cv::Mat& img, cv::Mat& paddedImg, 
                    const Size& winSize = Size(21,21), int pyrBorder = BORDER_REFLECT_101) {     
    //
    Mat& temp = paddedImg;

    if(!temp.empty())
        temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
    if(temp.type() != img.type() || temp.cols != winSize.width*2 + img.cols || temp.rows != winSize.height * 2 + img.rows)
        temp.create(img.rows + winSize.height*2, img.cols + winSize.width*2, img.type());

    copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder);
    temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
}

void addPaddingToImage(const cv::Mat& img, const Size& winSize, int pyrBorder, cv::Mat& paddedImg) {     
    //
    Mat& temp = paddedImg;

    if(!temp.empty())
        temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
    if(temp.type() != img.type() || temp.cols != winSize.width*2 + img.cols || temp.rows != winSize.height * 2 + img.rows)
        temp.create(img.rows + winSize.height*2, img.cols + winSize.width*2, img.type());

    if(pyrBorder == BORDER_TRANSPARENT)
        img.copyTo(temp(Rect(winSize.width, winSize.height, img.cols, img.rows)));
    else
        copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder);
    temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);

}

int main(int argc, char* argv[]) 
{
    // 读取两帧图像
#if 1    
    cv::Mat prevImg = cv::imread("../assets/1726299898318057216.png", cv::IMREAD_GRAYSCALE);
    cv::Mat nextImg = cv::imread("../assets/1726299898366026496.png", cv::IMREAD_GRAYSCALE);
#else
    cv::Mat prevImg = cv::imread("../assets/frame1.png", cv::IMREAD_GRAYSCALE );
    cv::Mat nextImg = cv::imread("../assets/frame2.png", cv::IMREAD_GRAYSCALE );

    // cv::Mat prevImg = cv::imread("../assets/frame1.png", CV_LOAD_IMAGE_COLOR );
    // cv::Mat nextImg = cv::imread("../assets/frame2.png", CV_LOAD_IMAGE_COLOR );

    // cv::Mat prevImg = cv::imread("../assets/frame1.png", cv::IMREAD_COLOR );
    // cv::Mat nextImg = cv::imread("../assets/frame2.png", cv::IMREAD_COLOR );
#endif

    if (prevImg.empty() || nextImg.empty()) {
        std::cerr << "无法加载图像" << std::endl;
        return -1;
    }

    if(1)
    {
        // 假设你已经构建了图像金字塔
        std::vector<Mat> pyramid;

        // 示例: 创建一个底层图像 (灰度图)
        cv::Mat prevImg = cv::imread("pyramid_level_0.png", cv::IMREAD_GRAYSCALE);
        pyramid.push_back(prevImg);

        std::vector<cv::Point2f> prevPts, nextPts;
        cv::goodFeaturesToTrack(prevImg, prevPts, 100, 0.3, 7);

        Mat img = cv::imread("pyramid_level_1.png", cv::IMREAD_GRAYSCALE);
        pyramid.emplace_back(img);

        img = cv::imread("pyramid_level_2.png", cv::IMREAD_GRAYSCALE);
        pyramid.emplace_back(img);

        img = cv::imread("pyramid_level_3.png", cv::IMREAD_GRAYSCALE);
        pyramid.emplace_back(img);


        std::vector<Mat> paddedPyramid;
        Size winSize(21, 21); // 光流计算窗口大小 21x21
        // addPaddingToPyramid(pyramid, winSize, BORDER_REFLECT_101, paddedPyramid);
        addPaddingToPyramid(pyramid, paddedPyramid);

        int i = 0;
        std::cout << "Image channels: " << paddedPyramid[i].channels() << std::endl;
        std::cout << "Image size: " << paddedPyramid[i].size() << std::endl;
        std::cout << "Image isContinuous: " << paddedPyramid[i].isContinuous() << std::endl;
        std::cout << "Image step: " << paddedPyramid[i].step[0] << std::endl;

        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(paddedPyramid, paddedPyramid, prevPts, nextPts, status, err, cv::Size(21, 21), 3);

        // 可视化结果
        for (size_t i = 0; i < prevPts.size(); i++) {
            if (status[i]) {
                cv::line(nextImg, prevPts[i], nextPts[i], cv::Scalar(0, 255, 0), 2);
                cv::circle(nextImg, nextPts[i], 5, cv::Scalar(0, 0, 255), -1);
            }
        }

        // 显示结果
        cv::imshow("2 Optical Flow", nextImg);
        cv::waitKey(0);

        return 0;
    }

    if(0)
    {
        cv::Mat pyr_image = cv::imread("pyramid_level_0.png", cv::IMREAD_GRAYSCALE);
        if (pyr_image.empty()) {
            std::cerr << "Error: Image 'pyramid_level_0' not found." << std::endl;
            // return -1;
        }
        else
        {
            std::cout << "pyramid_level_0" << std::endl;
            std::cout << "Image channels: " << pyr_image.channels() << std::endl;
            std::cout << "Image size: " << pyr_image.size() << std::endl;
            std::cout << "Image isContinuous: " << pyr_image.isContinuous() << std::endl;
            std::cout << "Image step: " << pyr_image.step[0] << std::endl;
            std::cout << "pyramid_level_0 the end." << std::endl;

            cv::Mat paddedImg;
            cv::Size winSize(21, 21); // 例如：窗口大小是 21x21
            addPaddingToImage(pyr_image, winSize, BORDER_REFLECT_101, paddedImg);

            std::cout << "after paddedImg:\n";
            std::cout << "Image channels: " << paddedImg.channels() << std::endl;
            std::cout << "Image size: " << paddedImg.size() << std::endl;
            std::cout << "Image isContinuous: " << paddedImg.isContinuous() << std::endl;
            std::cout << "Image step: " << paddedImg.step[0] << std::endl;
            cv::imshow("paddedImg", paddedImg);
            cv::waitKey(0);

            return 0;
        }
    }

    if(0)
    {
        // 加载图像
        cv::Mat image = cv::imread("../assets/1726299898318057216.png", cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Error: Image not found." << std::endl;
            return -1;
        }

        // std::cout << "Image channels: " << image.channels() << std::endl;
        // std::cout << "Image size: " << image.size() << std::endl;
        // cv::imshow("raw image", image);
        // cv::waitKey(0);

        
        bool 	withDerivatives = true;

        // 构建金字塔
        std::vector<cv::Mat> pyramid;
        // int maxLevel = cv::buildOpticalFlowPyramid(image, pyramid, cv::Size(21, 21), 5);

        // withDerivatives = true;
        withDerivatives = false;
        int maxLevel = cv::buildOpticalFlowPyramid(image, pyramid, cv::Size(21, 21), 5, withDerivatives);

        std::cout << "maxLevel=" << maxLevel << " pyramid.size=" << pyramid.size() << std::endl;

        // cv::imshow("Pyramid Level " + std::to_string(0), pyramid[0]);
        // cv::waitKey(0);

        // cv::imshow("Pyramid Level " + std::to_string(4), pyramid[8]);
        // cv::waitKey(0);

        if(!withDerivatives)
        for (size_t i = 0; i < pyramid.size(); ++i) {
            std::cout << i << ":" << std::endl;
            std::cout << "Image channels: " << pyramid[i].channels() << std::endl;
            std::cout << "Image size: " << pyramid[i].size() << std::endl;
            std::cout << "Image isContinuous: " << pyramid[i].isContinuous() << std::endl;
            std::cout << "Image step: " << pyramid[i].step[0] << std::endl;

            cv::imshow("Pyramid Level " + std::to_string(i), pyramid[i]);
            cv::imwrite("pyramid_level_" + std::to_string(i) + ".png", pyramid[i]);
        }

        // 显示金字塔的每一层
        if(withDerivatives)
        for (size_t i = 0; i <= maxLevel; ++i) {
            cv::imshow("Pyramid Level " + std::to_string(i), pyramid[i * 2]);
            // pyramid[i * 2 + 1]的channels为2，存放的是第i层图像的每个像素点x和y方向的梯度值
        }

        // 等待按键，然后退出
        cv::waitKey(0);

        return 0;
    }

    // 特征点检测（例如 ShiTomasi 角点）
    std::vector<cv::Point2f> prevPts, nextPts;
    cv::goodFeaturesToTrack(prevImg, prevPts, 100, 0.3, 7);

    // 金字塔图像
    std::vector<cv::Mat> prevPyramid, nextPyramid;
    int maxLevel = 3; // 最大金字塔层数
    bool withDerivatives = false;

    // 生成金字塔
    cv::buildOpticalFlowPyramid(prevImg, prevPyramid, cv::Size(21, 21), maxLevel, withDerivatives);
    cv::buildOpticalFlowPyramid(nextImg, nextPyramid, cv::Size(21, 21), maxLevel, withDerivatives);

    std::cout << "prevPyramid.size=" << prevPyramid.size() << std::endl;
    /*for(auto img : prevPyramid)
    {
        // cv::imshow("pyramid image", pyr1.at(i));
        cv::imshow("pyramid image", img);
        cv::waitKey(0);
    }*/

    // 显示每一层的金字塔图像
    for (size_t i = 0; i < prevPyramid.size(); ++i) {
        std::string windowName = "Pyramid Level " + std::to_string(i);
        cv::imshow(windowName, prevPyramid[i]);

        // 保存金字塔图像（可选）
        // cv::imwrite("pyramid_level_" + std::to_string(i) + ".png", prevPyramid[i]);
    }

    // 等待用户按键
    cv::waitKey(0);

    // 存储跟踪结果
    std::vector<uchar> status;
    std::vector<float> err;

    // 计算金字塔光流
    cv::calcOpticalFlowPyrLK(prevPyramid, nextPyramid, prevPts, nextPts, status, err, cv::Size(21, 21), maxLevel);

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

    return 0;
}

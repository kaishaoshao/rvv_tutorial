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

using std::vector;
using std::pair;
using std::make_pair;

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
        #if 1
        trackPoints(old_pyramid->at(i), pyramid->at(i),
                    transforms->observations[i],
                    new_transforms->observations[i]);
        #else
        trackPoints3(old_pyramid->at(i), pyramid->at(i),
                    transforms->observations[i],
                    new_transforms->observations[i]);
        #endif

        if(i == 0)
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

      patch_valid &= p.valid;
      if (patch_valid) {
        // Perform tracking on current level
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);
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
      #if 1
      trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);
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

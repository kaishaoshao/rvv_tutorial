
// @date 2024-12-20
// @author wxliu

#include <iostream>
#include <string>

#include <basalt/optical_flow/frame_to_frame_optical_flow.h>
#include <basalt/utils/optical_flow_config.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

#include "wx_ros1_io.h"
using namespace wx;
using namespace basalt;

// VIO variables
basalt::Calibration<double> calib;
basalt::VioConfig vio_config;
basalt::OpticalFlowBase::Ptr opt_flow_ptr;

// Configuration variable
TYamlIO* g_yaml_ptr = nullptr;

// GUI functions

// Load functions
void load_data(const std::string& calib_path);

// Feed functions
void feedImage(basalt::OpticalFlowInput::Ptr data);

#include <opencv2/opencv.hpp>
typedef Eigen::Matrix<float, 2, 441> Mat2x441; // wxliu
Mat2x441 pattern_win;

int main(int argc, char* argv[])
{
    std::cout << "\033[0;31m main() \033[0m" << std::endl;

    // generate patch 2025-1-20
    if(1)
    {
      int size = 9;
      cv::Size winSize(size, size);
      cv::Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
      // cv::Point2f prevPt(pos.x(), pos[1]);
      // prevPt -= halfWin;
      int i = 0;
      float *pOffset = pattern_win.data();
      int x, y;
      for( y = 0; y < winSize.height; y++ )
      {
        x = 0;
        for( ; x < winSize.width; x++)
        {
          //
          // Eigen::Matrix<float, 2, 1> &col = pattern_win.col(i++);
          // col[0] = x - halfWin.x;
          // col[1] = y - halfWin.y;
          pOffset[i++] = x - halfWin.x;
          pOffset[i++] = y - halfWin.y;
        }
      }

      // std::cout << "pattern_win=\n" << pattern_win << std::endl;
      int count = size * size;
      for(int i = 0; i < count; i++)
      {
        // if( i > 0 && i % 10 == 0) std::cout << std::endl;
        if(i % size == 0) std::cout << std::endl;
        // std::cout << pattern_win.col(i).transpose();// << std::endl;
        auto offset = pattern_win.col(i);
        std::cout << "{" << offset[0] << ", " << offset(1) << "}, ";
        
      }

      return 0;
    }
    // the end.

    if(0)
    {
      int i = 0;
      for(i = 0; i < 441; i++)
      {
        if(i % 21 == 0) std::cout << std::endl;
        std::cout << std::setfill('0') << std::setw(3) << i << "  ";// << std::endl;
      }

      return 0;
    }

    if(0)
    {
      cv::Size winSize(21, 21);
      cv::Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
      // cv::Point2f prevPt(pos.x(), pos[1]);
      // prevPt -= halfWin;
      int i = 0;
      float *pOffset = pattern_win.data();
      int x, y;
      for( y = 0; y < winSize.height; y++ )
      {
        x = 0;
        for( ; x < winSize.width; x++)
        {
          //
          // Eigen::Matrix<float, 2, 1> &col = pattern_win.col(i++);
          // col[0] = x - halfWin.x;
          // col[1] = y - halfWin.y;
          pOffset[i++] = x - halfWin.x;
          pOffset[i++] = y - halfWin.y;
        }
      }

      // std::cout << "pattern_win=\n" << pattern_win << std::endl;
      for(int i = 0; i < 441; i++)
      {
        // if( i > 0 && i % 10 == 0) std::cout << std::endl;
        if(i % 21 == 0) std::cout << std::endl;
        // std::cout << pattern_win.col(i).transpose();// << std::endl;
        auto offset = pattern_win.col(i);
        std::cout << "{" << offset[0] << ", " << offset(1) << "}, ";
        
      }

      return 0;
    }
    
    // ros
    ros::init(argc, argv, "optical_flow_node");  // node name
    ros::MultiThreadedSpinner spinner(6);  // use 6 threads
    ros::NodeHandle n("~");

    std::string sys_yaml_file;
    n.param<std::string>("sys_yaml_file", sys_yaml_file, "");
    std::cout << "sys_yaml_file=" << sys_yaml_file << std::endl;

    if(sys_yaml_file.empty())
    {
        // get binary file's absolute path
        // char szModulePath[260] = { 0 };
        // if(0 == GetModuleFileName(szModulePath, 260)) {
        //     //szModulePath 就是.so文件的绝对路径。
        //     // module path:./zc_server
        //     std::cout << "module path:" << szModulePath << std::endl;
        // }

        char absolutePath[1024] = { 0 };
        // wx::TFileSystemHelper::getAbsolutePath(absolutePath, argv[0]);
        printf("absolutePath = %s\n", absolutePath);

        // szModulePath: /root/dev/stereo3_ros1_ws/install/lib/stereo3/stereo3
        char *ptr = strrchr(absolutePath, '/');
        *ptr = '\0';

        if(sys_yaml_file.empty())
        {
            sys_yaml_file = absolutePath;
            sys_yaml_file += "/sys.yaml";
            std::cout << "sys_yaml_file=" << sys_yaml_file << std::endl;
        }


    }

    struct TYamlIO yaml;
    g_yaml_ptr = &yaml;
    yaml.ReadConfiguration(sys_yaml_file);
    std::cout << "calib_path=" << yaml.cam_calib_path << std::endl
    << "config_path=" << yaml.config_path << std::endl;


    if (!yaml.config_path.empty()) {
      vio_config.load(yaml.config_path);
    }

    load_data(yaml.cam_calib_path);

    #if 0
    opt_flow_ptr.reset(new FrameToFrameOpticalFlow<float, Pattern51>(vio_config, calib));
    #else
    opt_flow_ptr.reset(new FrameToFrameOpticalFlow<float, Pattern441>(vio_config, calib));
    #endif
    
    // wx::CRos1IO node {ros::NodeHandle{"~"}, yaml };
    wx::CRos1IO node{n, yaml};

    node.feedImage_ = std::bind(&feedImage, std::placeholders::_1);

    // ros::spin(); // comment 
    spinner.spin();
    node.stop_();

    return 0;

}

void feedImage(basalt::OpticalFlowInput::Ptr data) 
{
  opt_flow_ptr->input_queue.push(data);
}

void load_data(const std::string& calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib);
    std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras"
              << std::endl;

  } else {
    std::cerr << "could not load camera calibration " << calib_path
              << std::endl;
    std::abort();
  }
}
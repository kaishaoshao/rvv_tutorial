
// @author: wxliu
// @date: 2023-11-9

// #include "wx_system.h"

#include "wx_ros1_io.h"
#include <basalt/image/image.h>
// #include <basalt/utils/vis_utils.h>
#include <opencv2/highgui.hpp>

#include <basalt/io/dataset_io.h>

#include <queue>
#include <vector>

extern basalt::OpticalFlowBase::Ptr opt_flow_ptr;  // 2023-11-28.

// #include <boost/circular_buffer.hpp>

// #define _TEST_ROS2_IO

// #define _FILTER_IN_SECONDS_
// #define _VELOCITY_FILTER_
#define _MULTI_VELOCITY_FILTER_ // comment on 2024-6-6. use this one before now

// #define _REMOVE_OUTLIER_FILTER_

// #define _Linear_Fitted5_ // good way to smooth velocity curve.

#define _VERIFY_CONFIG_FILE

// #define _KF_


#ifdef _KF_
#include "kalman_filter.h"
Kalman_Filter_Struct kalman_filter_pt1;
/*  =
  {
    .Q_ProcessNoise = 0.001,
    .R_MeasureNoise = 0.1,
    .estimate_value_init = 25,
    .estimate_variance_init = 1,
    .estimate_value_curr = 25,
    .estimate_variance_curr = 1,
  };*/
#endif


using namespace wx;

CRos1IO::CRos1IO(const ros::NodeHandle& pnh, const TYamlIO& yaml) noexcept
    // CRos1IO::CRos1IO(const ros::NodeHandle& pnh, bool use_imu, int fps, long
    // dt_ns) noexcept
    : pnh_(pnh),
      yaml_(yaml),
      fps_(yaml_.fps),
      dt_ns_(yaml_.dt_ns),
      sub_image0_(pnh_, yaml.image0_topic, 100),//5),
      sub_image1_(pnh_, yaml.image1_topic, 100)//5)
      // sub_image0_(pnh_, "/image_left", 5),
      // sub_image1_(pnh_, "/image_right", 5)
// , sub_image0_info_(this, "/image_left_info")
// , sub_image1_info_(this, "/image_right_info")
// , sync_stereo_(sub_image0_, sub_image1_, sub_image0_info_, sub_image1_info_,
// 10) // method 1
{

  yaml_.ReadAdaptiveThreshold();
  // intensity_diff_threshold = yaml_.intensity_diff_threshold;
  std::cout << "yaml_.intensity_diff_threshold=" << yaml_.intensity_diff_threshold << std::endl;

  if(yaml_.photometric_calibration)
  {
    img_undistorter = new undist_lite(yaml_.image_width, yaml_.image_height, 
      yaml_.gamma1, yaml_.vignette1, yaml_.gamma2, yaml_.vignette2);
  }

#ifdef _KF_
  // Kalman_Filter_Init(&kalman_filter_pt1, 0.001, 2, 25, 1);
  // Kalman_Filter_Init(&kalman_filter_pt1, 0.001, 2, 0, 1);
  Kalman_Filter_Init(&kalman_filter_pt1, yaml_.kf_q, yaml_.kf_r, 0, 1);
#endif

#ifdef _unprint_// _VERIFY_CONFIG_FILE
  std::cout << std::boolalpha << "CRos1IO---\n"
            << "calib_path=" << yaml_.cam_calib_path << std::endl
            << "config_path=" << yaml_.config_path << std::endl
            << "dt_ns = " << yaml_.dt_ns << std::endl
            << "slow_velocity = " << yaml_.slow_velocity
            << "zero_velocity = " << yaml_.zero_velocity << std::endl
            << "mean_value = " << yaml_.mean_value << std::endl
            << "number_of_255 = " << yaml_.number_of_255 << std::endl
            << "record_bag = " << yaml_.record_bag << std::endl
            << "record_duration = " << yaml_.record_duration << std::endl
            << "photometric_calibration = " << yaml_.photometric_calibration << std::endl
            << "image_width = " << yaml_.image_width << std::endl
            << "image_height = " << yaml_.image_height << std::endl
            << "gamma1 = " << yaml_.gamma1 << std::endl
            << "vignette1 = " << yaml_.vignette1 << std::endl
            << "gamma2 = " << yaml_.gamma2 << std::endl
            << "vignette2 = " << yaml_.vignette2 << std::endl
            << "computing_mode = " << yaml_.computing_mode << std::endl
            << "acc_zero_velocity = " << yaml_.acc_zero_velocity << std::endl
            << "ang_zero_velocity = " << yaml_.ang_zero_velocity << std::endl
            << "d_p = " << yaml_.d_p << std::endl
            << "equalize = " << yaml_.equalize << std::endl
            << "kf_q = " << yaml_.kf_q << std::endl
            << "kf_r = " << yaml_.kf_r << std::endl
            << "manual_set_intensity_diff = " << yaml_.manual_set_intensity_diff << std::endl
            << "manual_intensity_diff_threshold = " << yaml_.manual_intensity_diff_threshold << std::endl
            << "image0_topic = " << yaml_.image0_topic << std::endl
            << "image1_topic = " << yaml_.image1_topic << std::endl
            << "imu_topic = " << yaml_.imu_topic << std::endl;

  int cnt = yaml_.vec_tracked_points.size();
  std::string strConfidenceInterval = "tracked_points:[";
  for (int i = 0; i < cnt; i++) {
    strConfidenceInterval += std::to_string(yaml_.vec_tracked_points[i]) + ",";
    if (i == cnt - 1) {
      strConfidenceInterval[strConfidenceInterval.size() - 1] = ']';
    }
  }

  std::cout << strConfidenceInterval << std::endl;

  cnt = yaml_.vec_confidence_levels.size();
  strConfidenceInterval = "confidence_levels:[";
  for (int i = 0; i < cnt; i++) {
    strConfidenceInterval += std::to_string(yaml_.vec_confidence_levels[i]);
    if (i == cnt - 1) {
      strConfidenceInterval[strConfidenceInterval.size() - 1] = ']';
    } else {
      strConfidenceInterval += ",";
    }
  }

  std::cout << strConfidenceInterval << std::endl;
  if(yaml_.gnss_enable)
  {
    std::cout << std::boolalpha << "gnss_enable=" << yaml_.gnss_enable << std::endl
      << "rtk_topic=" << yaml_.rtk_topic << std::endl
      << "gnss_meas_topic=" << yaml_.gnss_meas_topic << std::endl
      << "gnss_ephem_topic=" << yaml_.gnss_ephem_topic << std::endl
      << "gnss_glo_ephem_topic=" << yaml_.gnss_glo_ephem_topic << std::endl
      << "gnss_iono_params_topic=" << yaml_.gnss_iono_params_topic << std::endl
      << "gnss_tp_info_topic=" << yaml_.gnss_tp_info_topic << std::endl
      << "gnss_elevation_thres=" << yaml_.gnss_elevation_thres << std::endl
      << "gnss_psr_std_thres=" << yaml_.gnss_psr_std_thres << std::endl
      << "gnss_dopp_std_thres=" << yaml_.gnss_dopp_std_thres << std::endl
      << "gnss_track_num_thres=" << yaml_.gnss_track_num_thres << std::endl
      << "gnss_ddt_sigma=" << yaml_.gnss_ddt_sigma << std::endl
      << "GNSS_DDT_WEIGHT=" << yaml_.GNSS_DDT_WEIGHT << std::endl
      << "gnss_local_online_sync=" << yaml_.gnss_local_online_sync << std::endl
      << "local_trigger_info_topic=" << yaml_.local_trigger_info_topic << std::endl
      << "gnss_local_time_diff=" << yaml_.gnss_local_time_diff << std::endl;
      
    int cnt = yaml_.GNSS_IONO_DEFAULT_PARAMS.size();
    std::string str = "gnss_iono_default_parameters:[";
    for (int i = 0; i < cnt; i++) {
      str += std::to_string(yaml_.GNSS_IONO_DEFAULT_PARAMS[i]);
      if (i == cnt - 1) {
        str[str.size() - 1] = ']';
      } else {
        str += ",";
      }
    }
    std::cout << str << std::endl;
    std::cout << "num_yaw_optimized=" << yaml_.num_yaw_optimized << std::endl;
  }

  std::cout << "CRos1IO---THE END---\n";
#endif

  if(!yaml_.tks_pro_integration)
  {
    SetForward(true);
  }

  // create work thread
#ifdef _BACKEND_WX_ 
  t_publish_myodom = std::thread(&CRos1IO::PublishMyOdomThread, this);
#endif

#ifdef _RECORD_BAG_
  if (yaml_.record_bag)
    t_record_bag = std::thread(&CRos1IO::RecordBagThread, this);
#endif

#ifdef _RECORD_BAG_
  t_write_bag = std::thread(&CRos1IO::WriteBagThread, this);
#endif

#ifdef _IS_IMU_STATIONARY
  t_check_imu_still = std::thread(&CRos1IO::CheckImuStillThread, this);
#endif

  use_imu_ = yaml_.use_imu;
  sync_stereo_.emplace(sub_image0_, sub_image1_, 100);//5);  // method 2
  sync_stereo_->registerCallback(&CRos1IO::StereoCb, this);

#ifdef _BACKEND_WX_ 
  pub_point_ = pnh_.advertise<ros_pointcloud>("/point_cloud2", 10);
  pub_odom_ = pnh_.advertise<nav_msgs::Odometry>("/pose_odom", 10);

  path_msg_.poses.reserve(
      1024);  // reserve的作用是更改vector的容量（capacity），使vector至少可以容纳n个元素。
              // 如果n大于vector当前的容量，reserve会对vector进行扩容。其他情况下都不会重新分配vector的存储空间
  
  spp_path_msg_.poses.reserve(1024);
  slam_path_msg_.poses.reserve(1024);
  rtk_path_msg_.poses.reserve(1024);

  // pub_path_ = this->create_publisher<nav_msgs::msg::Path>("/path_odom", 2);
  pub_path_ = pnh_.advertise<nav_msgs::Path>("/path_odom", 1);
  pub_spp_path_ = pnh_.advertise<nav_msgs::Path>("/spp_path_odom", 1);
  pub_slam_path_ = pnh_.advertise<nav_msgs::Path>("/slam_path_odom", 1);
  pub_rtk_path_ = pnh_.advertise<nav_msgs::Path>("/rtk_path_odom", 1);

  if (use_imu_ || yaml_.loose_coupling_imu) {
    // sub_imu_ = pnh.subscribe("/imu", 2000, imu_callback,
    // ros::TransportHints().tcpNoDelay());
    sub_imu_ = pnh_.subscribe(yaml_.imu_topic, 2000, &CRos1IO::imu_callback, this,
                              ros::TransportHints().tcpNoDelay());
  }

  // pub_warped_img =
  // this->create_publisher<sensor_msgs::msg::Image>("/feature_img", 1); //
  // feature_img // warped_img
  pub_warped_img = pnh_.advertise<sensor_msgs::Image>("feature_img", 5);

  pub_my_odom_ = pnh_.advertise<myodom::MyOdom>("/my_odom", 10);
#endif  

#ifdef _TEST_POINT_DEPTH_
  pub_my_img_ = pnh_.advertise<sensor_msgs::Image>("/my_img", 5);
  // pub_my_point_ = pnh_.advertise<mypoint::MyPoint>("/my_point", 10);
  pub_my_point_ = pnh_.advertise<ros_pointcloud>("/my_point", 10);
#endif

  cmap = GetColorMap("jet");

#ifdef _GNSS_WX_
  // gnss
  if(yaml_.gnss_enable)
  {
    // 订阅两个不同的星历话题，是因为两个导航系统下的星历格式不一样
    // GNSS_EPHEM_TOPIC包含3种系统的星历信息：GPS, Galileo, BeiDou ephemeris
    // 订阅星历信息：卫星的位置、速度、时间偏差等信息
    sub_ephem = pnh_.subscribe(yaml_.gnss_ephem_topic, 100, &CRos1IO::gnss_ephem_callback, this); // GNSS星历信息
    sub_glo_ephem = pnh_.subscribe(yaml_.gnss_glo_ephem_topic, 100, &CRos1IO::gnss_glo_ephem_callback, this); // GLO：GLONASS。格洛纳斯星历信息
    // 卫星的观测信息：GNSS原始测量
    sub_gnss_meas = pnh_.subscribe(yaml_.gnss_meas_topic, 100, &CRos1IO::gnss_meas_callback, this);
    // 卫星的观测信息：RTK定位结果
    sub_rtk = pnh_.subscribe(yaml_.rtk_topic, 100, &CRos1IO::rtk_callback, this);
    // 电离层参数订阅：GNSS广播电离层参数
    sub_gnss_iono_params = pnh_.subscribe(yaml_.gnss_iono_params_topic, 100, &CRos1IO::gnss_iono_params_callback, this);

    /*
      * GNSS和Local坐标系时间差的补偿:
      * 因为GVINS处理的过程中用到GNSS和VIO的结果，但两者其实是不同空间的产物，也有可能是不同时间的，
      * 因此，两者之间有时间差“time_diff_gnss_local”很正常，需要进行补偿
      * 
      */
    // GNSS和VIO的时间是否同步判断
    if (yaml_.gnss_local_online_sync) // 在线同步
    {
        sub_gnss_time_pluse_info = pnh_.subscribe(yaml_.gnss_tp_info_topic, 100, 
            &CRos1IO::gnss_tp_info_callback, this); // 订阅GNSS脉冲信息
        sub_local_trigger_info = pnh_.subscribe(yaml_.local_trigger_info_topic, 100, 
            &CRos1IO::local_trigger_info_callback, this); // 订阅相机触发时间
    }
    else
    {
        time_diff_gnss_local = yaml_.gnss_local_time_diff;
        // inputGNSSTimeDiff_(time_diff_gnss_local); // comment because 'inputGNSSTimeDiff_' is not initialized.
        time_diff_valid = true;
    }
  }
#endif  
}

#ifdef _RECORD_BAG_
void CRos1IO::OpenRosbag() {
  try {
    CloseRosBag();
    std::cout << "1 open rosbag" << std::endl;
    bag_.open(bag_name.c_str(), rosbag::bagmode::Write);
    SetRecordBag(true);
  } catch (rosbag::BagException& e) {
    ROS_ERROR("open rosbag failed: %s", e.what());
    // return ;
  }
}

void CRos1IO::CloseRosBag() {
  if (GetRecordBag()) {
    SetRecordBag(false);
    usleep(1000 * 100);  // 100ms
    std::cout << "1 close rosbag" << std::endl;
    bag_.close();
  }
}

void CRos1IO::WriteBagThread() {
  TStereoImage stereo_img_msg;
  sensor_msgs::ImuConstPtr imu_msg;
  atp_info::atp atp_msg;
  while (!bQuit) {
    if (!record_bag) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    // if (!imu_queue.empty()) {
    while (!imu_queue.empty()) {
      imu_queue.pop(imu_msg);
      // handle data now.
      bag_.write("/imu", imu_msg->header.stamp, *imu_msg);
    }

    if (!stereo_img_queue.empty()) {
      stereo_img_queue.pop(stereo_img_msg);
      // handle data now.
      bag_.write("/image_left", stereo_img_msg.image0_ptr->header.stamp,
                 *stereo_img_msg.image0_ptr);
      bag_.write("/image_right", stereo_img_msg.image1_ptr->header.stamp,
                 *stereo_img_msg.image1_ptr);
    }

    // write atp info. into bag file.
    if (!atp_queue.empty()) {
      atp_queue.pop(atp_msg);
      // handle data now.
      bag_.write("/atp_info", atp_msg.header.stamp, atp_msg);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }  // while(1)
}

#endif

CRos1IO::~CRos1IO() {
  bQuit = true;
#if 1

#ifdef _BACKEND_WX_ 
  t_publish_myodom.join();
#endif

#ifdef _RECORD_BAG_  
  if (yaml_.record_bag)
  t_record_bag.join();
  if (yaml_.record_bag) CloseRosBag();
  t_write_bag.join();
#endif

#ifdef _IS_IMU_STATIONARY
  t_check_imu_still.join();
#endif

#else

  t_publish_myodom.detach();

#ifdef _RECORD_BAG_
// #if 0  
  if (yaml_.record_bag)
  t_record_bag.detach();
// #endif
  if (yaml_.record_bag) CloseRosBag();
  t_write_bag.detach();
#endif

#ifdef _IS_IMU_STATIONARY
  t_check_imu_still.detach();
#endif

#endif
}

#ifdef _RECORD_BAG_
void CRos1IO::RecordBagThread() {
  while (!bQuit) {
    OpenRosbag();
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::seconds(yaml_.record_duration));
  }
}
#endif

#ifdef _BACKEND_WX_ 
void CRos1IO::PublishMyOdomThread() {
  basalt::PoseVelBiasState<double>::Ptr data;
  while (!bQuit) {
    // std::chrono::milliseconds dura(200);
    // std::this_thread::sleep_for(dura);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    while (!pvb_queue.empty()) {
      pvb_queue.pop(data);
      // if (!data.get()) { // 如果当前帧为空指针，则退出循环
      //   break;
      // }

      PublishMyOdom(data, pvb_queue.empty());

      // handle data now.
    }

  }  // while(1)
}
#endif

#ifdef _GNSS_WX_
// gnss
// GNSS星历-回调函数：除GLONASS之外的其它三个卫星系统的星历回调函数
void CRos1IO::gnss_ephem_callback(const GnssEphemMsgConstPtr &ephem_msg)
{
    EphemPtr ephem = msg2ephem(ephem_msg);
    inputEphem_(ephem);
}

// GLONASS星历-回调函数
void CRos1IO::gnss_glo_ephem_callback(const GnssGloEphemMsgConstPtr &glo_ephem_msg)
{
    GloEphemPtr glo_ephem = msg2glo_ephem(glo_ephem_msg);
    inputEphem_(glo_ephem);
}

// 电离层参数订阅 ionospheric
// troposphere [ˈtrɒpəsfɪə(r)] 对流层  ionosphere [aɪˈɒnəsfɪə(r)] 电离层
/*
卫星信号在传播的过程中会受到电离层和对流层的影响，且如果建模不正确或不考虑两者的影响，会导致定位结果变差，
因此，通常都会对两者进行建模处理；后面我们在选择卫星信号时，会考虑卫星的仰角，也是因为对于仰角小的卫星，
其信号在电离层和对流层中经过的时间较长，对定位影响大，这样的卫星我们就会排除。
*/
void CRos1IO::gnss_iono_params_callback(const StampedFloat64ArrayConstPtr &iono_msg)
{
    double ts = iono_msg->header.stamp.toSec();
    std::vector<double> iono_params;
    std::copy(iono_msg->data.begin(), iono_msg->data.end(), std::back_inserter(iono_params));
    assert(iono_params.size() == 8);
    inputIonoParams_(ts, iono_params);
}

// 订阅GNSS measurements （伪距、多普勒频率等）
void CRos1IO::gnss_meas_callback(const GnssMeasMsgConstPtr &meas_msg)
{
    // 从ros信息解析GNSS测量值
    std::vector<ObsPtr> gnss_meas = msg2meas(meas_msg);

    latest_gnss_time = time2sec(gnss_meas[0]->time); // TODO: latest_gnss_time似乎在stereo3中用不到?
    inputLatestGNSSTime_(latest_gnss_time);

    // cerr << "gnss ts is " << std::setprecision(20) << time2sec(gnss_meas[0]->time) << endl;
    if (!time_diff_valid)   return;

    /*m_buf.lock();
    gnss_meas_buf.push(std::move(gnss_meas)); // 得到GNSS观测值的秒时间，并把观测信息放在全局变量gnss_meas_buf里面，供后面使用
    m_buf.unlock();
    con.notify_one();*/
    feedGnss_(std::move(gnss_meas));
}

// 订阅RTK 信息
void CRos1IO::rtk_callback(const sensor_msgs::NavSatFix::ConstPtr &nav_sat_msg)
{
    uint64_t t = nav_sat_msg->header.stamp.toNSec();
    Eigen::Vector3d gt_geo, gt_ecef;
    gt_geo.x() = nav_sat_msg->latitude;
    gt_geo.y() = nav_sat_msg->longitude;
    gt_geo.z() = nav_sat_msg->altitude;
    if (gt_geo.hasNaN()) return;
    feedRTK_(t, nav_sat_msg->status.status, gt_geo);
}

// 订阅相机触发时间
/*获得local 和 GNSS的时间差；
   trigger_msg记录的是相机被GNSS脉冲触发的时间，也可以理解成图像的命名（以时间命名），和真正的GNSS时间是有差别的
   因为存在硬件延迟等，这也是后面为什么校正local 和 world时间的原因
*/
void CRos1IO::local_trigger_info_callback(const LocalSensorExternalTriggerConstPtr &trigger_msg)
{
    std::lock_guard<std::mutex> lg(m_time);

    if (next_pulse_time_valid)
    {
        time_diff_gnss_local = next_pulse_time - trigger_msg->header.stamp.toSec();
        inputGNSSTimeDiff_(time_diff_gnss_local);
        if (!time_diff_valid)       // just get calibrated
            std::cout << "time difference between GNSS and VI-Sensor got calibrated: "
                << std::setprecision(15) << time_diff_gnss_local << " s\n";
        time_diff_valid = true;
    }/**/
}

void CRos1IO::gnss_tp_info_callback(const GnssTimePulseInfoMsgConstPtr &tp_msg)
{
    gtime_t tp_time = gpst2time(tp_msg->time.week, tp_msg->time.tow);
    if (tp_msg->utc_based || tp_msg->time_sys == SYS_GLO)
        tp_time = utc2gpst(tp_time);
    else if (tp_msg->time_sys == SYS_GAL)
        tp_time = gst2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_BDS)
        tp_time = bdt2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_NONE)
    {
        std::cerr << "Unknown time system in GNSSTimePulseInfoMsg.\n";
        return;
    }
    double gnss_ts = time2sec(tp_time);

    std::lock_guard<std::mutex> lg(m_time);
    next_pulse_time = gnss_ts;
    next_pulse_time_valid = true;/**/
}
#endif

void CRos1IO::StereoCb(const sm::ImageConstPtr& image0_ptr,
                       const sm::ImageConstPtr& image1_ptr) {
  // std::cout << "received images.\n";
#ifdef _RECORD_BAG_
  // if(yaml_.record_bag)
  if (record_bag) {
    // m_rec_.lock();
    // bag_.write("/image_left", image0_ptr->header.stamp, *image0_ptr);
    // bag_.write("/image_right", image1_ptr->header.stamp, *image1_ptr);
    // m_rec_.unlock();

    TStereoImage stereoImage;
    stereoImage.image0_ptr = image0_ptr;
    stereoImage.image1_ptr = image1_ptr;
    stereo_img_queue.push(stereoImage);
  }
#endif

#ifdef _IS_FORWARD_
  // if (isForward_ && !isForward_()) return;
  if (!GetForward()) return;
#endif

#if 0
  char szLog[255] = "";//{ '\0' };
  snprintf(szLog, 255, "raw image ts=%lf", image0_ptr->header.stamp.toSec() + yaml_.gnss_local_time_diff);
  wx::TFileSystemHelper::WriteLog(szLog);
#endif

  static u_int64_t prev_t_ns = 0;

  u_int64_t t_ns =
      image0_ptr->header.stamp.nsec + image0_ptr->header.stamp.sec * 1e9;

  if (prev_t_ns >= t_ns)  // 2023-12-18.
  {
    return;
  } else {
    prev_t_ns = t_ns;
  }

  basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);
  data->img_data.resize(CRos1IO::NUM_CAMS);
  data->t_ns = t_ns;
  // std::cout << t_ns << std::endl;

  const uint8_t* data_in0 = nullptr;
  const uint8_t* data_in1 = nullptr;

#ifdef _NOISE_SUPPRESSION_
  auto image0 = cb::toCvShare(image0_ptr)->image;  // cv::Mat类型
  auto image1 = cb::toCvShare(image1_ptr)->image;
  // auto image0 = cb::toCvCopy(image0_ptr)->image;  // cv::Mat类型
  // auto image1 = cb::toCvCopy(image1_ptr)->image;

  // cv::imshow("original image", image0);
  // cv::waitKey(0);
#ifdef _GAUSSIAN_BlUR_
  cv::GaussianBlur(image0, image0, cv::Size(3, 3), 0);  // 高斯滤波
  cv::GaussianBlur(image1, image1, cv::Size(3, 3), 0);  // 高斯滤波
#else
  cv::medianBlur(image0, image0, 3);  // 中值滤波
  cv::medianBlur(image1, image1, 3);  // 中值滤波
#endif

  data_in0 = image0.ptr();
  data_in1 = image1.ptr();

  // cv::imshow("noise suppression image", image0); // image_ns
  // cv::waitKey(0);
  // return ;
#else
  cv::Mat image0, image1;
  if (yaml_.photometric_calibration) {
    // auto image0 = cb::toCvCopy(image0_ptr)->image;
    // auto image1 = cb::toCvCopy(image1_ptr)->image;

    image0 = cb::toCvCopy(image0_ptr)->image;
    image1 = cb::toCvCopy(image1_ptr)->image;

    if (image0.type() == CV_8UC3) {
      cv::cvtColor(image0, image0, cv::COLOR_BGR2GRAY);
    }
    if (image0.type() != CV_8U) {
      printf("image0 did something strange! this may segfault. %i \n",
            image0.type());
      return;
    }

    if (image1.type() == CV_8UC3) {
      cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
    }
    if (image1.type() != CV_8U) {
      printf("image1 did something strange! this may segfault. %i \n",
            image1.type());
      return;
    }

    img_undistorter->undist(image0, image1);

    // cv::imshow("undistorted image", image0);
    // cv::waitKey(1);
    // return ;

    data_in0 = image0.ptr();
    data_in1 = image1.ptr();
  }
  else
  {
    // data_in0 = (const uint8_t*)image0_ptr->data.data();
    // data_in1 = (const uint8_t*)image1_ptr->data.data(); // comment on 2024-5-9.

    // constexpr bool EQUALIZE = true;
    // if (EQUALIZE)
    if (yaml_.equalize)
    {
      // cv::Mat image0, image1;
      auto src_image0 = cb::toCvShare(image0_ptr)->image;  // cv::Mat类型
      auto src_image1 = cb::toCvShare(image1_ptr)->image;
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
      // TicToc t_c;
      clahe->apply(src_image0, image0);
      clahe->apply(src_image1, image1);
      // ROS_DEBUG("CLAHE costs: %fms", t_c.toc());

      cv::GaussianBlur(image0, image0, cv::Size(3, 3), 0);  // 高斯滤波
      cv::GaussianBlur(image1, image1, cv::Size(3, 3), 0);  // 高斯滤波

      data_in0 = image0.ptr();
      data_in1 = image1.ptr();
    }
    else
    {
      data_in0 = (const uint8_t*)image0_ptr->data.data();
      data_in1 = (const uint8_t*)image1_ptr->data.data();
    }
  }
#endif

  
  {
    auto image0 = cb::toCvShare(image0_ptr)->image;  // cv::Mat类型
#if 0
    auto start = std::chrono::steady_clock::now();
    cv::Mat downsampled_image;
    cv::pyrDown(image0, downsampled_image, cv::Size(image0.cols/2, image0.rows/2));
    // cv::resize(image, downsampled_image, cv::Size(), 0.5, 0.5);

    if (!prev_image_.empty()) {
        // Compute absolute difference with previous image
        cv::Mat abs_diff_image;
        cv::absdiff(downsampled_image, prev_image_, abs_diff_image);
        cv::Scalar meanIntensity = cv::mean(abs_diff_image);
        // std::cout << "1 Mean intensity: " << 255*meanIntensity[0] << std::endl;
        // std::cout << "2 Mean intensity: " << meanIntensity[0] << std::endl;
        if(GetComputeIntensityDiff())
        {
          if(meanIntensity[0] > intensity_diff_threshold) {
            intensity_diff_threshold = meanIntensity[0];
            yaml_.intensity_diff_threshold = intensity_diff_threshold;
            yaml_.WriteAdaptiveThreshold();
            char szLog[255] = { 0 };
            sprintf(szLog, "thres:%.6f", yaml_.intensity_diff_threshold);
            wx::TFileSystemHelper::WriteLog(szLog);
          }
        }
        else
        {
          isStationary_ = (meanIntensity[0] > yaml_.intensity_diff_threshold) ? false : true;
        }

    }

    prev_image_ = downsampled_image.clone();
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end - start;

    std::cout << std::fixed << " 1 time cost : " << diff.count() << " s\n";
#else
    // method 2
    {
      // auto start2 = std::chrono::steady_clock::now();
      if (!prev_image_.empty()) {
        // Compute absolute difference with previous image
        cv::Mat abs_diff_image;
        cv::absdiff(image0, prev_image_, abs_diff_image);
        cv::Scalar meanIntensity = cv::mean(abs_diff_image);
        // std::cout << "1 Mean intensity: " << 255*meanIntensity[0] << std::endl;
        // std::cout << "2 Mean intensity: " << meanIntensity[0] << std::endl;
        if(GetComputeIntensityDiff() && !yaml_.manual_set_intensity_diff)
        {
          #if 0
          if(meanIntensity[0] > intensity_diff_threshold) {
            intensity_diff_threshold = meanIntensity[0];
            yaml_.intensity_diff_threshold = intensity_diff_threshold;
            yaml_.WriteAdaptiveThreshold();
            char szLog[255] = { 0 };
            sprintf(szLog, "thres:%.6f", yaml_.intensity_diff_threshold);
            wx::TFileSystemHelper::WriteLog(szLog);
          }
          #endif

          intensity_diff_counter_++;
          intensity_diff_threshold += meanIntensity[0];
          yaml_.intensity_diff_threshold = static_cast<double>(intensity_diff_threshold / intensity_diff_counter_);
          yaml_.WriteAdaptiveThreshold();

          char szLog[255] = { 0 };
          sprintf(szLog, "curr intensity diff:%.6f  mean intensity diff:%.6f", meanIntensity[0], yaml_.intensity_diff_threshold);
          // wx::TFileSystemHelper::WriteLog(szLog);

        }
        else
        {
          if(yaml_.manual_set_intensity_diff)
          {
            isStationary_ = (meanIntensity[0] > yaml_.manual_intensity_diff_threshold) ? false : true;
          }
          else
          {
            isStationary_ = (meanIntensity[0] > yaml_.intensity_diff_threshold) ? false : true;
          }
          
        }

      }

      prev_image_ = image0.clone();
      // auto end2 = std::chrono::steady_clock::now();
      // std::chrono::duration<double> diff2 = end2 - start2;
      // std::cout << std::fixed << " 2 time cost : " << diff2.count() << " s\n";
    }
    // the end.
#endif
  }

#if 0  // for check bright points
  auto image0 = cb::toCvShare(image0_ptr)->image;
  // cv::Mat mask = image0 > 255;
  cv::Mat mask = image0 == 255;
  int count = cv::countNonZero(mask);
  // ROS_INFO("Number of pixels with value > 250: %d", count);
  ROS_INFO("Number of pixels equal to 255: %d", count);


  cv::Scalar meanValue = cv::mean(image0);
  float MyMeanValue = meanValue.val[0];//.val[0]表示第一个通道的均值
  std::cout<<"Average of all pixels in image0 with 1st channel is "<< MyMeanValue << std::endl;
#endif

  // 拷贝左目图像数据
  data->img_data[0].img.reset(new basalt::ManagedImage<uint16_t>(
      image0_ptr->width, image0_ptr->height));

  // 图像数据转化为uint16_t
  // const uint8_t* data_in = (const uint8_t*)image0_ptr->data.data();
  uint16_t* data_out = data->img_data[0].img->ptr;
  size_t full_size = image0_ptr->width * image0_ptr->height;
  for (size_t i = 0; i < full_size; i++) {
    int val = data_in0[i];
    val = val << 8;
    data_out[i] = val;
  }

  // 拷贝右目图像数据
  data->img_data[1].img.reset(new basalt::ManagedImage<uint16_t>(
      image1_ptr->width, image1_ptr->height));
  // data_in = (const uint8_t*)image1_ptr->data.data();
  data_out = data->img_data[1].img->ptr;
  full_size = image1_ptr->width * image1_ptr->height;
  for (size_t i = 0; i < full_size; i++) {
    int val = data_in1[i];
    val = val << 8;
    data_out[i] = val;
  }

  feedImage_(data);
}

#ifdef _BACKEND_WX_ 
void CRos1IO::imu_callback(const sensor_msgs::ImuConstPtr& imu_msg)  // const
{

#ifdef _IS_IMU_STATIONARY
  tmp_imu_queue_.push(imu_msg);
#endif

#ifdef _RECORD_BAG_
  // if(yaml_.record_bag)
  if (record_bag) {
    // if imu_callback is a const callback function, then i cant write anymore.
    // so...

    // m_rec_.lock();
    // bag_.write("/imu", imu_msg->header.stamp, *imu_msg);
    // m_rec_.unlock();

    imu_queue.push(imu_msg);
  }
#endif

#ifdef _IS_FORWARD_
  // if (isForward_ && !isForward_()) return;
  if (!GetForward()) return;
#endif

  // double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nsec * (1e-9);
  double dx = imu_msg->linear_acceleration.x;
  double dy = imu_msg->linear_acceleration.y;
  double dz = imu_msg->linear_acceleration.z;
  double rx = imu_msg->angular_velocity.x;
  double ry = imu_msg->angular_velocity.y;
  double rz = imu_msg->angular_velocity.z;

// #if USE_TIGHT_COUPLING
// #else  // LOOSE_COUPLING
// #endif

  if (yaml_.use_imu) {
    int64_t t_ns = imu_msg->header.stamp.nsec + imu_msg->header.stamp.sec * 1e9;
    basalt::ImuData<double>::Ptr data(new basalt::ImuData<double>);
    data->t_ns = t_ns;
    data->accel = Vector3d(dx, dy, dz);
    data->gyro = Vector3d(rx, ry, rz);

    feedImu_(data);
  }
  else if (yaml_.loose_coupling_imu) {
    double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nsec * (1e-9);
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    inputIMU_(t, acc, gyr);

    // pg_.inputIMU(t, acc, gyr);
  }


#if 0
  double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nsec * (1e-9);
  char szTime[60] = { 0 };
  sprintf(szTime, "%f", t);
  // std::cout << "[lwx] imu timestamp t=" << szTime << "acc = " << acc.transpose() << "  gyr = " << gyr.transpose() << std::endl;
  std::cout << "[lwx] imu timestamp t=" << szTime << "acc = " << data->accel.transpose() << "  gyr = " << data->gyro.transpose() << std::endl;
#endif
}

void CRos1IO::atp_cb(atp_info::atp& atp_msg)
{
#ifdef _RECORD_BAG_
  // if(yaml_.record_bag)
  if (record_bag) {
    atp_queue.push(atp_msg);
  }
#endif
}

void CRos1IO::PublishPoints(basalt::VioVisualizationData::Ptr data) {
  if (pub_point_.getNumSubscribers() == 0) return;

  ros_pointcloud cloud_msg;
  // Time (int64_t nanoseconds=0, rcl_clock_type_t clock=RCL_SYSTEM_TIME)
  cloud_msg.header.stamp = ros::Time(data->t_ns * 1.0 * 1e-9);  // this->now();
  // cloud_msg.header.stamp.sec = data->t_ns / 1e9;
  // cloud_msg.header.stamp.nsec = data->t_ns % (1e9);
  cloud_msg.header.frame_id = "odom";
  cloud_msg.point_step =
      12;  // 16 // 一个点占16个字节 Length of a point in bytes
           // cloud.fields = MakePointFields("xyzi"); //
  // 描述了二进制数据块中的通道及其布局。
  cloud_msg.fields = MakePointFields("xyz");  // tmp comment 2023-12-6
  /*
    cloud_msg.height = 1;
    cloud_msg.width = 0;
    cloud_msg.fields.resize(3);
    cloud_msg.fields[0].name = "x";
    cloud_msg.fields[0].offset = 0;
    cloud_msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    cloud_msg.fields[0].count = 1;
    cloud_msg.fields[1].name = "y";
    cloud_msg.fields[1].offset = 4;
    cloud_msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    cloud_msg.fields[1].count = 1;
    cloud_msg.fields[2].name = "z";
    cloud_msg.fields[2].offset = 8;
    cloud_msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    cloud_msg.fields[2].count = 1;
    cloud_msg.is_bigendian = false;
    cloud_msg.point_step = 12;
    cloud_msg.row_step = 0;
    cloud_msg.is_dense = true;

    // 将点云数据转换为二进制数据，并存储在 PointCloud2 消息对象中
    cloud_msg.width = data->points.size();
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
    cloud_msg.data.resize(cloud_msg.row_step);
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    for (const auto& point : data->points) {
      *iter_x = point.x();
      *iter_y = point.y();
      *iter_z = point.z();
      ++iter_x;
      ++iter_y;
      ++iter_z;
    }
  */

  // int total_size = data->points.size();
  cloud_msg.height = 1;
  cloud_msg.width = data->points.size();
  cloud_msg.data.resize(
      cloud_msg.width *
      cloud_msg.point_step);  // point_step指示每个点的字节数为16字节

  int i = 0;
  for (const auto& point : data->points) {
    auto* ptr = reinterpret_cast<float*>(cloud_msg.data.data() +
                                         i * cloud_msg.point_step);

    ptr[0] = point.x();
    ptr[1] = point.y();
    ptr[2] = point.z();
    // ptr[3] = static_cast<float>(patch.vals[0] / 255.0); // 图像强度转换为颜色

    i++;
  }

  // 发布点云
  pub_point_.publish(cloud_msg);
}

void CRos1IO::PublishOdometry(basalt::PoseVelBiasState<double>::Ptr data) {
  if (pub_odom_.getNumSubscribers() == 0) return;

  // 创建nav_msgs::msg::Odometry消息对象
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = ros::Time(data->t_ns * 1.0 * 1e-9);  // this->now();
  odom_msg.header.frame_id = "odom";
  odom_msg.pose.pose.position.x = data->T_w_i.translation().x();
  odom_msg.pose.pose.position.y = data->T_w_i.translation().y();
  odom_msg.pose.pose.position.z = data->T_w_i.translation().z();
  odom_msg.pose.pose.orientation.x = data->T_w_i.unit_quaternion().x();
  odom_msg.pose.pose.orientation.y = data->T_w_i.unit_quaternion().y();
  odom_msg.pose.pose.orientation.z = data->T_w_i.unit_quaternion().z();
  odom_msg.pose.pose.orientation.w = data->T_w_i.unit_quaternion().w();

  // 发布位姿
  pub_odom_.publish(odom_msg);
}

void CRos1IO::Reset() {
  std::cout << "clear path_msg_.poses" << std::endl;
  path_msg_.poses.clear();
  path_msg_.poses.reserve(1024);

  spp_path_msg_.poses.clear();
  spp_path_msg_.poses.reserve(1024);

  slam_path_msg_.poses.clear();
  slam_path_msg_.poses.reserve(1024);

  rtk_path_msg_.poses.clear();
  rtk_path_msg_.poses.reserve(1024);

  isResetOdom = true; // 2024-1-11
}

void CRos1IO::PublishAllPositions(TAllPositions::Ptr data)
{
/*  
  std::cout << "spp_pos=" << data->spp_pos.transpose() 
    << " slam_pos=" << data->slam_pos.transpose()
    << " rtk_pos=" << data->rtk_pos.transpose() << std::endl;
*/
  if (pub_spp_path_.getNumSubscribers() > 0)
  {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = ros::Time(data->t_ns / 1.0e9);
    pose_msg.header.frame_id = "odom";

    pose_msg.pose.position.x = data->spp_pos.x();
    pose_msg.pose.position.y = data->spp_pos.y();
    pose_msg.pose.position.z = data->spp_pos.z();
    pose_msg.pose.orientation.x = 0;
    pose_msg.pose.orientation.y = 0;
    pose_msg.pose.orientation.z = 0;
    pose_msg.pose.orientation.w = 1;

    spp_path_msg_.header.stamp = ros::Time(data->t_ns / 1.0e9);
    spp_path_msg_.header.frame_id = "odom";
    spp_path_msg_.poses.push_back(pose_msg);
    pub_spp_path_.publish(spp_path_msg_);
  }

  if (pub_slam_path_.getNumSubscribers() > 0)
  {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = ros::Time(data->t_ns / 1.0e9);
    pose_msg.header.frame_id = "odom";

    pose_msg.pose.position.x = data->slam_pos.x();
    pose_msg.pose.position.y = data->slam_pos.y();
    pose_msg.pose.position.z = data->slam_pos.z();
    pose_msg.pose.orientation.x = 0;
    pose_msg.pose.orientation.y = 0;
    pose_msg.pose.orientation.z = 0;
    pose_msg.pose.orientation.w = 1;

    slam_path_msg_.header.stamp = ros::Time(data->t_ns / 1.0e9);
    slam_path_msg_.header.frame_id = "odom";
    slam_path_msg_.poses.push_back(pose_msg);
    pub_slam_path_.publish(slam_path_msg_);
  }

  if (pub_rtk_path_.getNumSubscribers() > 0)
  {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = ros::Time(data->t_ns / 1.0e9);
    pose_msg.header.frame_id = "odom";

    pose_msg.pose.position.x = data->rtk_pos.x();
    pose_msg.pose.position.y = data->rtk_pos.y();
    pose_msg.pose.position.z = data->rtk_pos.z();
    pose_msg.pose.orientation.x = 0;
    pose_msg.pose.orientation.y = 0;
    pose_msg.pose.orientation.z = 0;
    pose_msg.pose.orientation.w = 1;

    rtk_path_msg_.header.stamp = ros::Time(data->t_ns / 1.0e9);
    rtk_path_msg_.header.frame_id = "odom";
    rtk_path_msg_.poses.push_back(pose_msg);
    pub_rtk_path_.publish(rtk_path_msg_);
  }


}

void CRos1IO::PublishPoseAndPath(basalt::PoseVelBiasState<double>::Ptr data) {
  geometry_msgs::PoseStamped pose_msg;
  // pose_msg.header.stamp = this->now();
  pose_msg.header.stamp =
      ros::Time(data->t_ns * 1.0 / 1e9);  ///< timestamp of the state in seconds
                                          ///< [nanoseconds];  // 帧采集时间
  // pose_msg.header.stamp.sec = data->t_ns / 1e9;
  // pose_msg.header.stamp.nsec = data->t_ns % 1e9;

  pose_msg.header.frame_id = "odom";
  // Sophus2Ros(tf, pose_msg.pose);
  pose_msg.pose.position.x = data->T_w_i.translation().x();
  pose_msg.pose.position.y = data->T_w_i.translation().y();
  pose_msg.pose.position.z = data->T_w_i.translation().z();
  pose_msg.pose.orientation.x = data->T_w_i.unit_quaternion().x();
  pose_msg.pose.orientation.y = data->T_w_i.unit_quaternion().y();
  pose_msg.pose.orientation.z = data->T_w_i.unit_quaternion().z();
  pose_msg.pose.orientation.w = data->T_w_i.unit_quaternion().w();

#ifdef _PUBLISH_VELOCITY_
#if 0
  PublishMyOdom(data, true);
#else
  pvb_queue.push(data);
#endif
#endif

  if (pub_odom_.getNumSubscribers() > 0) {
    nav_msgs::Odometry odom_msg;
    // odom_msg.header.stamp = time;
    // odom_msg.header.frame_id = "odom";
    odom_msg.header = pose_msg.header;
    odom_msg.pose.pose = pose_msg.pose;

    // 发布位姿
    pub_odom_.publish(odom_msg);
  }

  if (pub_path_.getNumSubscribers() == 0) return;

  // 发布轨迹   path_odom话题
  // path_msg_.header.stamp = this->now();
  path_msg_.header.stamp = ros::Time(
      data->t_ns * 1.0 /
      1e9);  ///< timestamp of the state in nanoseconds;  // 帧采集时间
  path_msg_.header.frame_id = "odom";
  path_msg_.poses.push_back(pose_msg);
  pub_path_.publish(path_msg_);

  // std::cout << " postion: " << pose_msg.pose.position.x << ", "
  //   << pose_msg.pose.position.y << ", " << pose_msg.pose.position.z <<
  //   std::endl;
}

void bubble_sort(double* arr, int length) {
  int i = 0;
  for (i = 0; i < length - 1; i++) {
    int flag = 0;  // flag标识
    int j = 0;
    for (j = 0; j < length - 1 - i; j++) {
      if (arr[j] > arr[j + 1]) {
        int temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
        flag = 1;  // 发生数据交换，置flag为1
      }
    }
    if (flag == 0)  // 若flag为0，则没有发生交换，跳出循环
    {
      break;
    }
  }
}

bool sortByY(const double& p1, const double& p2) {
  return p1 < p2;  // 升序排列
}

double ClacFitted(std::vector<double>& vec, double vel) {
  double new_velocity;
  if (vec.size() < 5)
  // if(q_velocities.size() < 5)
  {
    vec.emplace_back(vel);
    // q_velocities.push(period_distance / delta_s);
    new_velocity = vel;
  } else {
    // q_velocities.pop();
    // q_velocities.push(period_distance / delta_s);

    for (int i = 0; i < 4; i++) {
      vec[i] = vec[i + 1];
    }
    vec[4] = vel;

    // for(auto vel : vec) std::cout << vel << " ";
    // std::cout << std::endl;
    std::vector<double> vec_tmp(vec);

    std::sort(vec_tmp.begin(), vec_tmp.end(), sortByY);

    new_velocity = (vec_tmp[1] + vec_tmp[2] + vec_tmp[3]) / 3;
  }

  return new_velocity;
}

void CRos1IO::PublishMyOdom(basalt::PoseVelBiasState<double>::Ptr data,
                            bool bl_publish /* = false*/) {
  // #if 1//def _PUBLISH_VELOCITY_
  //  if(1) // compute velocity.

  // use member variable 'dt_ns_'
  double fps_inv = 1.0 / fps_;
  // int DATA_CNT = 0.2 * fps_;

  // static int odometry_cnt = 0;
  static int64_t t_ns = 0;
  static int64_t last_t_ns = 0;
  static double total_distance = 0.0;
  static double period_distance = 0.0;
  static Vector2d last_pose(0, 0);
  static double prev_velocity = 0.0;
  static double total_odom = 0.0;
  static bool reset_velociy = false;
#ifdef _FILTER_IN_SECONDS_
  static constexpr int ELEM_CNT =
      4;  // 10;//5; // if elem_cnt equal to 10. it's mean that we use about 2
          // seconds's data to average.
  static double array_period_odom[ELEM_CNT] = {0};
  static double array_delta_s[ELEM_CNT] = {0};
  static int keep_count = 0;
#endif

#ifdef _MULTI_VELOCITY_FILTER_
  static constexpr int ELEM_CNT = 4;
  static double array_velocity[ELEM_CNT] = {0};
  static constexpr double array_weight[ELEM_CNT] = {0.1, 0.2, 0.3, 0.4};
  static int keep_count = 0;
  static unsigned char reset_cnt = 0;
#endif

#ifdef _REMOVE_OUTLIER_FILTER_
  static constexpr int ELEM_CNT = 5;  // 6
  static double array_velocity[ELEM_CNT] = {0};
  static constexpr double array_weight[ELEM_CNT] = {0.0, 0.0, 0.1, 0.2,
                                                    0.3};  //, 0.4 };
  static int keep_count = 0;
#endif

#ifdef _Linear_Fitted5_
  static std::vector<double> vec_velocities[20];
  // static std::queue<double> q_velocities;

  static constexpr int ELEM_CNT = 2;  // 4;
  static double array_velocity[ELEM_CNT] = {0};
  // static constexpr double array_weight[ELEM_CNT] = { 0.1, 0.2, 0.3, 0.4 };
  static constexpr double array_weight[ELEM_CNT] = {0.45, 0.55};
  static int keep_count = 0;
  static unsigned char reset_cnt = 0;
  static std::vector<double> vec_new_vel;

#endif


  if (isResetOdom) {
    if(fabs(data->T_w_i.translation().x()) < 1e-6 && fabs(data->T_w_i.translation().y()) < 1e-6)
    {
      std::cout << std::boolalpha << "isResetOdom=" << isResetOdom << std::endl;
      wx::TFileSystemHelper::WriteLog("reset odom.");
      prev_velocity = 0.0;
      period_distance = 0.0;
      total_distance = 0.0;
      last_t_ns = 0;
      t_ns = 0;
      total_odom = 0.0;
      last_pose.x() = 0;
      last_pose.y() = 0;
  #ifdef _MULTI_VELOCITY_FILTER_
      keep_count = 0;
      reset_cnt = 0;
  #endif

      isResetOdom = false;
    }
  }

  if (t_ns == 0) {
    t_ns = data->t_ns - fps_inv * 1e9;
    last_t_ns = data->t_ns - fps_inv * 1e9;
  }

  if (last_pose.x() == 0 && last_pose.y() == 0) {
    last_pose.x() = data->T_w_i.translation().x();
    last_pose.y() = data->T_w_i.translation().y();

    wx::TFileSystemHelper::WriteLog("begin to reset last pose.");
  }

  Vector2d curr_pose;
  curr_pose.x() = data->T_w_i.translation().x();
  curr_pose.y() = data->T_w_i.translation().y();
  double delta_distance = (curr_pose - last_pose).norm();

  double frame_delta_s = (data->t_ns - last_t_ns) * 1.0 * (1e-9);

  period_distance += delta_distance;
  total_distance += delta_distance;
  // std::cout << " delta_distance=" << delta_distance << "  period_distance="
  // << period_distance << std::endl;

  double delta_s = (data->t_ns - t_ns) * 1.0 * (1e-9);

  // if (period_distance / delta_s > 25.0) {
  //   std::cout << "frame_delta_distance=" << delta_distance
  //             << "  frame_delta_s=" << frame_delta_s
  //             << "  period_distance=" << period_distance
  //             << "  delta_s=" << delta_s << std::endl;

  //   std::cout << "curr_pose" << curr_pose.transpose() << std::endl;
  //   std::cout << "last_pose" << last_pose.transpose() << std::endl;
  // }

#if 0 // change 1 to 0.
  char szLog[512] = { 0 };
  // sprintf(szLog, "last_pose=(%.2f, %.2f), curr_pose=(%.2f, %.2f), delta_distance=%.2f, time interval=%.2f", 
  //   last_pose.x(), last_pose.y(), curr_pose.x(), curr_pose.y(), delta_distance, delta_s);

  // sprintf(szLog, "prev_p=(%.2f, %.2f), curr_p=(%.2f, %.2f), delta_s=%.2f", 
  //   last_pose.x(), last_pose.y(), curr_pose.x(), curr_pose.y(), delta_s);

  sprintf(szLog, "p=(%.2f, %.2f), delta_d=%.2f, delta_s=%.2f", curr_pose.x(), curr_pose.y(), delta_distance, delta_s);

  wx::TFileSystemHelper::WriteLog(szLog);
#endif

  last_pose = curr_pose;

  last_t_ns = data->t_ns;

#ifdef _Linear_Fitted5_
  // double new_vel = delta_distance / delta_s;
  double new_vel = delta_distance / frame_delta_s;
  // new_vel = ClacFitted(vec_velocities, new_vel);
  // new_vel = ClacFitted(vec_velocities2, new_vel);
  // new_vel = ClacFitted(vec_velocities3, new_vel);
  // new_vel = ClacFitted(vec_velocities4, new_vel);
  // new_vel = ClacFitted(vec_velocities5, new_vel);

  for (int i = 0; i < 20; i++) {
    new_vel = ClacFitted(vec_velocities[i], new_vel);
  }

  vec_new_vel.emplace_back(new_vel);

#endif

  // if(odometry_cnt == 0)
  // if(delta_s > 0.18) // 20ms
  if (bl_publish) {
    double curr_velocity = 0.0;
    // std::cout << "delta_s:" << delta_s << std::endl;
#ifdef _FILTER_IN_SECONDS_
    if (keep_count < ELEM_CNT) {
      array_period_odom[keep_count++] = period_distance;
      array_delta_s[keep_count++] = delta_s;
    } else {
      int i = 0;
      for (i = 0; i < ELEM_CNT - 1; i++) {
        array_period_odom[i] = array_period_odom[i + 1];
        array_delta_s[i] = array_delta_s[i + 1];
      }

      array_period_odom[i] = period_distance;
      array_delta_s[i] = delta_s;
    }

    if (1) {
      int i = 0;
      double tmp_odom = 0.0;
      double tmp_delta_s = 0.0;
      for (i = 0; i < keep_count; i++) {
        tmp_odom += array_period_odom[i];
        tmp_delta_s += array_delta_s[i];
      }

      curr_velocity = tmp_odom / tmp_delta_s;
    }

#elif defined _MULTI_VELOCITY_FILTER_

    if (keep_count < ELEM_CNT) {
      array_velocity[keep_count++] = period_distance / delta_s;

      #if 1
      if((period_distance / delta_s) > yaml_.abnormal_velocity)
      {
        char szLog[256] = { 0 };
        sprintf(szLog, " period_distance=%.2f, delta_s=%.2f", period_distance, delta_s);
        wx::TFileSystemHelper::WriteLog(szLog);
      }
      #endif


    } else {
      int i = 0;
      for (i = 0; i < ELEM_CNT - 1; i++) {
        array_velocity[i] = array_velocity[i + 1];
      }

      array_velocity[i] = period_distance / delta_s;
    }

    if (keep_count < ELEM_CNT) {
      for (int i = 0; i < keep_count; i++) {
        curr_velocity += array_velocity[i];
      }

      curr_velocity = curr_velocity / keep_count;
    } else {
      for (int i = 0; i < keep_count; i++) {
        curr_velocity += array_velocity[i] * array_weight[i];
      }
    }

#elif defined _REMOVE_OUTLIER_FILTER_
/*
  if(keep_count < ELEM_CNT)
  {
    array_velocity[keep_count++] = period_distance / delta_s;
  }
  else
  {
    int i = 0;
    for(i = 0; i < ELEM_CNT - 1; i++)
    {
      array_velocity[i] = array_velocity[i + 1];
    }

    array_velocity[i] = period_distance  / delta_s;
  }

  //-
  if(keep_count < ELEM_CNT)
  {
    for(int i = 0; i < keep_count; i++)
    {
      curr_velocity += array_velocity[i];
    }

    // curr_velocity = curr_velocity / keep_count;
    curr_velocity = period_distance / delta_s;
  }
  else
  {
    // for(int i = 0; i < keep_count; i++)
    // {
    //   curr_velocity += array_velocity[i] * array_weight[i];
    // }

    // int N = ELEM_CNT;
    // curr_velocity = (-array_velocity[N - 5] + 4.0 * array_velocity[N - 4]
  - 6.0 * array_velocity[N - 3] +
    //   4.0 * array_velocity[N - 2] + 69.0 * array_velocity[N - 1]) / 70.0;

    bubble_sort(array_velocity, 5);
    curr_velocity = (array_velocity[1] + array_velocity[2] + array_velocity[3])
  / 3;
  }
*/
/*
  if(vec_velocities.size() < 5)
  // if(q_velocities.size() < 5)
  {
    vec_velocities.emplace_back(period_distance / delta_s);
    // q_velocities.push(period_distance / delta_s);
    curr_velocity = period_distance / delta_s;
  }
  else
  {
    // q_velocities.pop();
    // q_velocities.push(period_distance / delta_s);

    for(int i = 0; i < 4; i++)
    {
      vec_velocities[i] = vec_velocities[i + 1];
    }
    vec_velocities[4] = period_distance / delta_s;

    // for(auto vel : vec_velocities) std::cout << vel << " ";
    // std::cout << std::endl;
    std::vector<double> vec_tmp(vec_velocities);

    std::sort(vec_tmp.begin(), vec_tmp.end(), sortByY);

    curr_velocity = (vec_tmp[1] + vec_tmp[2] + vec_tmp[3]) / 3;
  }
*/
#elif defined _Linear_Fitted5_

    // curr_velocity = new_velocity2;
    // curr_velocity = new_velocity4;

    // curr_velocity = new_vel;
    for (auto vel : vec_new_vel) curr_velocity += vel;
    curr_velocity /= vec_new_vel.size();

    vec_new_vel.clear();

    if (keep_count < ELEM_CNT) {
      array_velocity[keep_count++] = curr_velocity;
    } else {
      int i = 0;
      for (i = 0; i < ELEM_CNT - 1; i++) {
        array_velocity[i] = array_velocity[i + 1];
      }

      array_velocity[i] = curr_velocity;
    }

    if (keep_count < ELEM_CNT) {
      for (int i = 0; i < keep_count; i++) {
        curr_velocity += array_velocity[i];
      }

      curr_velocity = curr_velocity / keep_count;
    } else {
      curr_velocity = 0.0;
      for (int i = 0; i < keep_count; i++) {
        curr_velocity += array_velocity[i] * array_weight[i];
      }
    }

#elif defined _KF_
  curr_velocity = period_distance / delta_s;
  curr_velocity = Kalman_Filter_Iterate(&kalman_filter_pt1, curr_velocity);

#else
    curr_velocity = period_distance / delta_s;
#endif

    if (prev_velocity < 1e-6) {
      prev_velocity = curr_velocity;
    }

#ifdef _FILTER_IN_SECONDS_
    // double publish_velocity = curr_velocity;
    double publish_velocity = pow(curr_velocity, yaml_.coefficient);
    // std::cout << " curr_veloctiy=" << curr_velocity << " publish_velocity="
    // << publish_velocity << std::endl;

#elif defined _MULTI_VELOCITY_FILTER_
    double publish_velocity = curr_velocity;
    // double publish_velocity = pow(curr_velocity, yaml_.coefficient);
    switch(yaml_.computing_mode)
    {
    case 1:
      publish_velocity = pow(curr_velocity, yaml_.coefficient);
      break ;
    case 2:
      publish_velocity = curr_velocity * yaml_.coefficient;
      break ;
    default:
      // publish_velocity = curr_velocity;
      break ;
    }
#elif defined _KF_
    double publish_velocity = curr_velocity;
    // double publish_velocity = pow(curr_velocity, yaml_.coefficient);
    switch(yaml_.computing_mode)
    {
    case 1:
      publish_velocity = pow(curr_velocity, yaml_.coefficient);
      break ;
    case 2:
      publish_velocity = curr_velocity * yaml_.coefficient;
      break ;
    default:
      // publish_velocity = curr_velocity;
      break ;
    }

#else

#ifdef _VELOCITY_FILTER_
    double publish_velocity = (prev_velocity + curr_velocity) / 2;
#else
    double publish_velocity = curr_velocity;
#endif

#endif


    if (publish_velocity > yaml_.abnormal_velocity) {
      std::cout << "The speed " << publish_velocity
                << " is ABNORMAL. reset algorithm." << std::endl;

      // std::cout << std::boolalpha << "isResetOdom=" << isResetOdom << std::endl;

      char szLog[256] = { 0 };
      sprintf(szLog, "The velocity '%.2f' is ABNORMAL. reset algorithm.", publish_velocity);
      wx::TFileSystemHelper::WriteLog(szLog);

      prev_velocity = 0.0;
      period_distance = 0.0;
      total_distance = 0.0;
      last_t_ns = 0;
      t_ns = 0;
      total_odom = 0.0;
      last_pose.x() = 0;
      last_pose.y() = 0;
    #ifdef _MULTI_VELOCITY_FILTER_
      keep_count = 0;
      reset_cnt = 0;
    #endif

      reset_();
      return;
    }

#ifdef _MULTI_VELOCITY_FILTER_

#if 1
    // if(fabs(publish_velocity - prev_velocity) > 2) // 3 or 2 or other number
    // ?
    if (fabs(publish_velocity - prev_velocity) >
        0.5)  // 3 or 2 or other number ?
    {
      // std::cout << " --- reset count:" << (int)reset_cnt << " prev_v:" <<
      // prev_velocity << " curr_v:" << publish_velocity << std::endl;
      if (reset_cnt < 3) {
        // std::cout << " --- reset velocity.---\n";
        publish_velocity = prev_velocity * 0.5 + publish_velocity * 0.5;
        reset_cnt++;
        // if(keep_count >= 1)
        array_velocity[keep_count - 1] = publish_velocity;
      } else {
        reset_cnt = 0;
      }

    } else {
      reset_cnt = 0;
    }
#else
    if (fabs(publish_velocity - prev_velocity) > 0.4) {
      double acc = (array_velocity[2] - array_velocity[0]) / 0.6;
      std::cout << " prev_v:" << prev_velocity
                << " curr_v:" << publish_velocity;
      publish_velocity = array_velocity[2] + acc * 0.2;
      std::cout << "  acc:" << acc << "  new curr_v:" << publish_velocity
                << std::endl;
      array_velocity[keep_count - 1] = publish_velocity;
    }
#endif

    if (publish_velocity > 200.0 / 9)  // 80 km/h limit
    {
      #if 1
      char szLog[256] = { 0 };
      sprintf(szLog, "80 km/h limit. v > 200.0 / 9: publish_velocity=%.2f, prev_velocity=%.2f", publish_velocity, prev_velocity);
      wx::TFileSystemHelper::WriteLog(szLog);
      #endif
      publish_velocity = prev_velocity;
      // if(keep_count >= 1)
      array_velocity[keep_count - 1] = publish_velocity;
    }

#elif defined _REMOVE_OUTLIER_FILTER_
    /*
      if(fabs(publish_velocity - prev_velocity) > 0.5)
      {
        // if(keep_count < ELEM_CNT)
        // {
        //   curr_velocity = array_velocity[keep_count - 1];
        // }
        // else
        {
          double acc_vel[5] = { 0 };

          for(int i = 0; i < keep_count - 1; i++)
          {
            acc_vel[i] = array_velocity[i + 1] - array_velocity[i];
          }

          bubble_sort(acc_vel, 5);

          double acc_average = (acc_vel[1] + acc_vel[2] + acc_vel[3]) / 3;

          if(acc_average > 0.5)  acc_average = 0;
          else if (acc_average < -0.5)  acc_average = -0;

          publish_velocity = prev_velocity + acc_average;
        }
      }
    */

#else

#ifdef _VELOCITY_FILTER_
    if (reset_velociy == false) {
      // if(fabs(publish_velocity - prev_velocity) > 2) // 3 or 2 or other
      // number ?
      if (fabs(publish_velocity - prev_velocity) >
          0.5)  // 3 or 2 or other number ?
      {
        std::cout << " --- reset velocity.---\n";
        publish_velocity = prev_velocity;
        reset_velociy = true;
      }
    } else {
      reset_velociy = false;
    }
#endif

    if (publish_velocity > 200.0 / 9)  // 80 km/h limit
    {
      publish_velocity = prev_velocity;
    }

#endif

    bool is_reliable = true;
    if (fabs(publish_velocity - prev_velocity) >= 0.3) {
      is_reliable = false;
    }

#ifdef _IS_IMU_STATIONARY
    // if(isStill_)// && prev_velocity < 1e-6 && publish_velocity < 0.5)
    // if(isStill_ && prev_velocity < 1e-6 && publish_velocity < 0.5)
    // if(0 && isStill_ && publish_velocity < 0.5)
    if(isStill_ && publish_velocity < 0.5)
    {
      publish_velocity = 0.0;
    }

    // if(0 && (isPreintegrationstationary_() || isStill_)) // 2024-5-16.
    if(0 && isPreintegrationstationary_() && isStill_) // 2024-5-16.
    {
      // std::cout << "isPreintegrationstationary_() && isStill_ == true" << std::endl;
      publish_velocity = 0.0;
    }
#endif

    if(isStationary_)
    {
      publish_velocity = 0.0;
    }

    if(isLightToggled) // for test.
    {
      publish_velocity = 0.0;
    }

    if (publish_velocity <
        yaml_.zero_velocity)  // 0.05) // put here better than other place.
    {
      zeroVelocity_(true);
      publish_velocity = 0.00;
    } else {
      zeroVelocity_(false);
    }

    if (publish_velocity < yaml_.slow_velocity)  // 3.0)
    {
      slowVelocity_(true);
    } else {
      slowVelocity_(false);
    }


    


    prev_velocity = publish_velocity;

    double period_odom = publish_velocity * delta_s;
    total_odom += period_odom;


    // char szLog[256] = { 0 };
    // sprintf(szLog, "publish velocity: %.2f", publish_velocity);
    // wx::TFileSystemHelper::WriteLog(szLog);

    // if (pub_my_odom_.getNumSubscribers() > 0)
    {
      myodom::MyOdom odom_msg;
      // odom_msg.header.stamp = rclcpp::Time(data->t_ns);
      odom_msg.header.stamp = ros::Time((data->t_ns - dt_ns_) * 1.0 /
                                        1e9);  // complementary timestamp
      odom_msg.header.frame_id = "odom";

      // double delta_s = (data->t_ns - t_ns) * 1.0 * (1e-9);

      odom_msg.velocity = publish_velocity;  // period_distance / delta_s;
      odom_msg.delta_time = delta_s;
      // odom_msg.period_odom = period_distance;
      odom_msg.period_odom = period_odom;
      // odom_msg.total_odom = total_distance; // TODO:
      odom_msg.total_odom = total_odom;

      int nTrackedPoints = data->bias_accel.x();
      int nOptFlowPatches = data->bias_accel.y();
      int nUseImu = data->bias_accel.z();
      /*
            if(nUseImu == 1 || nTrackedPoints <= 10 || nOptFlowPatches <= 10)
            {
              odom_msg.confidence_coefficient = 0;
            }
            else if(nTrackedPoints < 20)
            {
              odom_msg.confidence_coefficient = 0.5;
            }
            else
            {
              odom_msg.confidence_coefficient = 1.0;
            }
      */
      double confidence_coefficient = 0.0;
      int nSize = yaml_.vec_tracked_points.size();

      if (nTrackedPoints > yaml_.vec_tracked_points[nSize - 1]) {
        confidence_coefficient = 1.0;
      } else {
        for (int i = 0; i < nSize; i++) {
          // if(nTrackedPoints <= yaml_.vec_tracked_points[i])
          if (nTrackedPoints <= yaml_.vec_tracked_points[i] &&
              publish_velocity >= 0.05)  // if velociy is 0, tracked points is
                                         // fewer than moving 2023-12-15
          {
            confidence_coefficient = yaml_.vec_confidence_levels[i];
#if 0
            nTracked_confidence_count++;
#endif
            break;
          }
        }
      }

#if 0
      if(confidence_coefficient < 1e-6 && isLampOn_)
      {
        zero_confidence ++;
      }

      if(isLampOn_)
      {
        total_confidence ++;
      }
      std::cout << " total_confidence: " << total_confidence 
        << "zero_confidence :" << zero_confidence 
        << " Percentage:" << (zero_confidence * 1.0 / total_confidence * 100) << "%" << std::endl;
#endif

#if 0
      if(confidence_coefficient == 1 && is_reliable == false)
      {
        nVelocity_confidence_count++;
      }

      if(confidence_coefficient == 1 && isLampOn_ == false)
      {
        nLamp_confidence_count++;
      }

      std::cout << "---statistics---\n" << "track - confidence == 0 counter: " << nTracked_confidence_count << std::endl
          << "velocity - confidence == 0 counter: " << nVelocity_confidence_count << std::endl
          << "lamp - confidence == 0 counter: " << nLamp_confidence_count << std::endl
          << "---the end---" << std::endl;
#endif

      // if(nUseImu == 1)
      if ((nUseImu == 1) || (is_reliable == false) || isLampOn_ == false) {
        confidence_coefficient = 0.0;
      }

      odom_msg.confidence_coefficient = confidence_coefficient;

      // publish velocity, period odom and total odom.
      if (pub_my_odom_.getNumSubscribers() > 0) pub_my_odom_.publish(odom_msg);

      if (add_odom_frame_) 
      {
        add_odom_frame_(publish_velocity, period_odom, total_odom,
                        confidence_coefficient);
      }

      // std::cout << "confidence : " << data->bias_accel.transpose() << "
      // confidence coefficient:" << confidence_coefficient << std::endl; // for
      // test.
    }

    t_ns = data->t_ns;
    period_distance = 0.0;
  }

  // #endif
}
#endif

inline void CRos1IO::getcolor(float p, float np, float& r, float& g, float& b) {
  float inc = 4.0 / np;
  float x = p * inc;
  r = 0.0f;
  g = 0.0f;
  b = 0.0f;

  if ((0 <= x && x <= 1) || (5 <= x && x <= 6))
    r = 1.0f;
  else if (4 <= x && x <= 5)
    r = x - 4;
  else if (1 <= x && x <= 2)
    r = 1.0f - (x - 1);

  if (1 <= x && x <= 3)
    g = 1.0f;
  else if (0 <= x && x <= 1)
    g = x - 0;
  else if (3 <= x && x <= 4)
    g = 1.0f - (x - 3);

  if (3 <= x && x <= 5)
    b = 1.0f;
  else if (2 <= x && x <= 3)
    b = x - 2;
  else if (5 <= x && x <= 6)
    b = 1.0f - (x - 5);
}

#ifdef _BACKEND_WX_
void CRos1IO::PublishFeatureImage(basalt::VioVisualizationData::Ptr data) {
  // if (pub_warped_img.getNumSubscribers() == 0)
  //   return;

  static cv::Mat disp_frame;

  static int prev_intensity_of_max_count = -1;

#ifdef _RESET_ALOGORITHM_BY_LAMP_ON_
  static bool is_reset_alogorithm_by_lamp_on = true;
#endif

  // step1 convert image
  uint16_t* data_in = nullptr;
  uint8_t* data_out = nullptr;

  // for(int cam_id = 0; cam_id < NUM_CAMS; cam_id++)
  for (int cam_id = 0; cam_id < 1; cam_id++) {
    // img_data is a vector<ImageData>
    basalt::ImageData imageData =
        data->opt_flow_res->input_images->img_data[cam_id];
    // std::cout << "w=" << imageData.img->w << "  h=" << imageData.img->h <<
    // std::endl;
    data_in = imageData.img->ptr;
    disp_frame =
        cv::Mat::zeros(imageData.img->h, imageData.img->w, CV_8UC1);  // CV_8UC3
    data_out = disp_frame.ptr();

    size_t full_size =
        imageData.img->size();  // disp_frame.cols * disp_frame.rows;
    for (size_t i = 0; i < full_size; i++) {
      int val = data_in[i];
      val = val >> 8;
      data_out[i] = val;
      // disp_frame.at(<>)
    }

    // check if lamp is on or off 2023-12-20.
    // if(1)
    {
#if 1
      // static int rev_cnt = 0;
      cv::Mat mask = disp_frame == 255;
      int count = cv::countNonZero(mask);
#ifdef _INTENSITY_OF_MAX_CNT_
      // find maximum count of a intensity.
      std::vector<int> vec_counter;
      for(int i = 0; i < 256; i++)
      {
        cv::Mat mask = disp_frame == i;
        int count = cv::countNonZero(mask);
        vec_counter.emplace_back(count);
      }
      // std::sort(vec_counter.begin(), vec_counter.end());
      int intensity = 0;
      int max_count = vec_counter[0];
      for(int i = 1; i < 256; i++)
      {
        if(vec_counter[i] > max_count)
        {
          intensity = i;
          max_count = vec_counter[i];
        }
      }
      // std::cout << "maximum count of a intensity is " << intensity << " it's count is" << max_count << std::endl;

      if(prev_intensity_of_max_count == -1)
      {
        prev_intensity_of_max_count = intensity;
      }

      if(abs(prev_intensity_of_max_count - intensity) >= 5)
      {
        // std::cout << "toggle light: " << " maximum count of a intensity is " << intensity << " it's count is" << max_count << std::endl;
        std::cout << "toggle light: " << " curr intensity is " << intensity << " prev intensity is " << prev_intensity_of_max_count << std::endl;
        isLightToggled = true;
      }
      else
      {
        isLightToggled = false;
      }
      prev_intensity_of_max_count = intensity;
#endif
      // if (count > 15)
      // std::cout << "Number of intensity equal to 255 is " << count
      // << std::endl;
      // more than 30 pixels with 255 denote lamp on.

      // if(rev_cnt == 0)
      // {
      //   rev_cnt = count;
      // }
      // int nAverage = (rev_cnt + count) / 2;
      // rev_cnt = count;
#else
      cv::Scalar meanValue = cv::mean(disp_frame);
      float MyMeanValue = meanValue.val[0];  //.val[0]表示第一个通道的均值
      // std::cout<<"Average of all pixels in image0 with 1st channel is "<<
      // MyMeanValue << std::endl;
#endif
      // if(MyMeanValue >= yaml_.mean_value)
      if (count >= yaml_.number_of_255) {
        if (!isLampOn_) {
          isLampOn_ = true;
          // std::cout << std::boolalpha << "lamp on :" << isLampOn_ << std::endl;
#ifdef _RESET_ALOGORITHM_BY_LAMP_ON_
          if (is_reset_alogorithm_by_lamp_on) {
            reset_();
            is_reset_alogorithm_by_lamp_on = false;
          }
#endif
        }

      } else {
        if (isLampOn_) {
          isLampOn_ = false;
          // std::cout << std::boolalpha << "lamp on :" << isLampOn_ << std::endl;
        }
      }
    }

    if (pub_warped_img.getNumSubscribers() == 0) return;
    // the end.

    /*
        cv::Mat img(cv::Size(imageData.img->w, imageData.img->h), CV_16UC1,
       imageData.img->ptr); #ifdef _DEBUG_ cv::imshow("feature_img", img);
        // cv::waitKey(0);
        cv::waitKey(1);
        #endif

        cv::Mat disp_frame;
        img.convertTo(disp_frame, CV_8UC1);

    */
    // disp_frame.convertTo(disp_frame, CV_8UC3);
    // just gray to bgr can show colored text and pixel.
    cv::cvtColor(disp_frame, disp_frame, CV_GRAY2BGR);  // CV_GRAY2RGB

#ifdef _DEBUG_
    cv::imshow("feature_img", disp_frame);
    // cv::waitKey(0);
    cv::waitKey(1);
#endif

// bool show_obs = true;
// if (show_obs) { // 显示追踪的特征点
#ifdef SHOW_TRACKED_POINTS

    const auto& points = data->projections[cam_id];

    if (points.size() > 0) {
      double min_id = points[0][2], max_id = points[0][2];

      for (const auto& points2 : data->projections)
        for (const auto& p : points2) {
          min_id = std::min(min_id, p[2]);
          max_id = std::max(max_id, p[2]);
        }

      // const IntervalD idepth_range(0.0, max_id);
      const IntervalD idepth_range(min_id, max_id);

      for (const auto& c : points) {
        const float radius = 6.5;

      #if 1  
        float r, g, b;
        getcolor(c[2] - min_id, max_id - min_id, b, g, r);
        // glColor3f(r, g, b);
        // pangolin::glDrawCirclePerimeter(c[0], c[1], radius);
        // pangolin里面的1.0，对应opencv里面的255
        b *= 255;
        g *= 255;
        r *= 255;
        cv::circle(disp_frame, cv::Point(c[0], c[1]), radius,
                   cv::Scalar(b, g, r));
        // cv::circle(disp_frame, cv::Point(c[0], c[1]), radius, cv::Scalar(255,
        // 255, 255));

      #else
        cv::Scalar bgr;
        Eigen::Map<Eigen::Vector3d> bgr_map(&bgr[0]);
        const double x = idepth_range.Normalize(c[2]);
        bgr_map = cmap.GetBgr(x) * 255;

        cv::circle(disp_frame, cv::Point(c[0], c[1]), radius, bgr);
      #endif
      }
    }

    // glColor3f(1.0, 0.0, 0.0); // to r, g, b
    // pangolin::GlFont::I()
    //     .Text("Tracked %d points", points.size())
    //     .Draw(5, 20);

    if (1) {
      /*
      void cv::putText  ( InputOutputArray  img,  // 要添加备注的图片
      const String &  text,  // 要添加的文字内容
      Point  org,  //
     要添加的文字基准点或原点坐标，左上角还是左下角取决于最后一个参数bottomLeftOrigin的取值
                   // Bottom-left corner of the text string in the image.
      int  fontFace,
      double  fontScale, //
     字体相较于最初尺寸的缩放系数。若为1.0f，则字符宽度是最初字符宽度，若为0.5f则为默认字体宽度的一半
      Scalar  color,  // 字体颜色
      int  thickness = 1,  // 字体笔画的粗细程度
      int  lineType = LINE_8,  // 字体笔画线条类型

      bool  bottomLeftOrigin = false  // 如果取值为TRUE，则Point
     org指定的点为插入文字的左上角位置，如果取值为默认值false则指定点为插入文字的左下角位置.
                                      // When true, the image data origin is at
     the bottom-left corner. Otherwise, it is at the top-left corner.
     )

      fontFace文字的字体类型（Hershey字体集），可供选择的有
      FONT_HERSHEY_SIMPLEX：正常大小无衬线字体
      FONT_HERSHEY_PLAIN：小号无衬线字体
      FONT_HERSHEY_DUPLEX：正常大小无衬线字体，比FONT_HERSHEY_SIMPLEX更复杂
      FONT_HERSHEY_COMPLEX：正常大小有衬线字体
      FONT_HERSHEY_TRIPLEX：正常大小有衬线字体，比FONT_HERSHEY_COMPLEX更复杂
      FONT_HERSHEY_COMPLEX_SMALL：FONT_HERSHEY_COMPLEX的小译本
      FONT_HERSHEY_SCRIPT_SIMPLEX：手写风格字体
      FONT_HERSHEY_SCRIPT_COMPLEX：手写风格字体，比FONT_HERSHEY_SCRIPT_SIMPLEX更复杂
      这些参数和FONT_ITALIC同时使用就会得到相应的斜体字

    */

      /*
        //创建空白图用于绘制文字
              cv::Mat image = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
              //设置蓝色背景
              image.setTo(cv::Scalar(100, 0, 0));
       */
      // 设置绘制文本的相关参数

      std::string strText = "";
      char szText[100] = {0};
      sprintf(szText, "Tracked %d points", points.size());
      strText = szText;
      // show text
      int font_face = cv::FONT_HERSHEY_SIMPLEX;  // cv::FONT_HERSHEY_COMPLEX;
      double font_scale = 0.5;                   // 2;//1;
      int thickness = 1;  // 2; // 字体笔画的粗细程度，有默认值1
      int baseline;
      // 获取文本框的长宽
      // cv::Size text_size = cv::getTextSize(text, font_face, font_scale,
      // thickness, &baseline);
      cv::Size text_size =
          cv::getTextSize(strText, font_face, font_scale, thickness, &baseline);

      // 将文本框居中绘制
      cv::Point origin;  // 计算文字左下角的位置
      // origin.x = image.cols / 2 - text_size.width / 2;
      // origin.y = image.rows / 2 + text_size.height / 2;
      origin.x = 0;  // 2023-11-28
      // origin.x = disp_frame.cols - text_size.width;
      origin.y = text_size.height;
      // cv::putText(disp_frame, text, origin, font_face, font_scale,
      // cv::Scalar(0, 255, 255), thickness, 8, 0);
      //  cv::putText(disp_frame, text, origin, font_face, font_scale,
      //  cv::Scalar(0, 0, 0), thickness, 8, false);
      cv::putText(disp_frame, strText, origin, font_face, font_scale,
                  cv::Scalar(255, 0, 255), thickness, 8, false);
    }
// }
#endif

// bool show_flow = true;
// if (show_flow) { // 显示光流patch
#ifdef SHOW_FLOW_PATCHES

    const Eigen::aligned_map<basalt::KeypointId, Eigen::AffineCompact2f>&
        kp_map = data->opt_flow_res->observations[cam_id];

    for (const auto& kv : kp_map) {
      Eigen::MatrixXf transformed_patch =
          kv.second.linear() * opt_flow_ptr->patch_coord;
      transformed_patch.colwise() += kv.second.translation();

      for (int i = 0; i < transformed_patch.cols(); i++) {
        const Eigen::Vector2f c = transformed_patch.col(i);
        // pangolin::glDrawCirclePerimeter(c[0], c[1], 0.5f);
        cv::circle(disp_frame, cv::Point(c[0], c[1]), 0.5f,
                   cv::Scalar(0, 0, 255));
      }

      const Eigen::Vector2f c = kv.second.translation();
    }

    // pangolin::GlFont::I()
    //     .Text("%d opt_flow patches", kp_map.size())
    //     .Draw(5, 20);

    std::string strText = "";
    char szText[100] = {0};
    sprintf(szText, "%d opt_flow patches", kp_map.size());
    strText = szText;
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

    // 将文本框居中绘制
    cv::Point origin;  // 计算文字左下角的位置
    // origin.x = image.cols / 2 - text_size.width / 2;
    // origin.y = image.rows / 2 + text_size.height / 2;
    origin.x = disp_frame.cols - text_size.width;
    origin.y = text_size.height;
    // cv::putText(disp_frame, text, origin, font_face, font_scale,
    // cv::Scalar(0, 255, 255), thickness, 8, 0);
    //  cv::putText(disp_frame, text, origin, font_face, font_scale,
    //  cv::Scalar(0, 0, 0), thickness, 8, false);
    cv::putText(disp_frame, strText, origin, font_face, font_scale,
                cv::Scalar(0, 255, 255), thickness, 8, false);

// }
#endif

#ifdef _OPEN_CV_SHOW_
    cv::imshow("feature_img", disp_frame);
    // cv::waitKey(0);
    cv::waitKey(1);
#endif

    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", disp_frame).toImageMsg();
    msg->header.stamp = ros::Time(data->t_ns * 1.0 / 1.0e9);
    msg->header.frame_id = "odom";
    // warped_img.header = image0_ptr->header;
    /*
     * 以下设置似乎显得多余?  yeah.
        msg->height = disp_frame.rows;
        msg->width = disp_frame.cols;
        // msg->is_bigendian = false;
        // msg->step = 640; // 640 * 1 * 3
        // warped_img.data = image0_ptr->data;
        // msg->encoding = "mono8";
    */
    pub_warped_img.publish(*msg);
  }
}
#endif

#ifdef _TEST_POINT_DEPTH_
void CRos1IO::PublishImageAndPoints(basalt::VioVisualizationData::Ptr data)
{
  if(data->projections[0].size() == 0) return ;

  if (pub_my_img_.getNumSubscribers() == 0)
    return;

  constexpr int cam_id = 0;

  // convert image
  static cv::Mat disp_frame;
  uint16_t* data_in = nullptr;
  uint8_t* data_out = nullptr;

  basalt::ImageData imageData =
  data->opt_flow_res->input_images->img_data[cam_id];
  // std::cout << "w=" << imageData.img->w << "  h=" << imageData.img->h << std::endl;
  data_in = imageData.img->ptr;
  disp_frame = cv::Mat::zeros(imageData.img->h, imageData.img->w, CV_8UC1);  // CV_8UC3
  data_out = disp_frame.ptr();

  size_t full_size = imageData.img->size();  // disp_frame.cols * disp_frame.rows;
  for (size_t i = 0; i < full_size; i++) {
    int val = data_in[i];
    val = val >> 8;
    data_out[i] = val;
  }

  cv::cvtColor(disp_frame, disp_frame, CV_GRAY2BGR);  // CV_GRAY2RGB

  sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", disp_frame).toImageMsg();
  // msg->header.stamp = ros::Time(data->t_ns * 1.0 / 1e9);
  msg->header.stamp = ros::Time(data->t_ns * 1.0 * 1e-9);
  msg->header.frame_id = "odom";
  pub_my_img_.publish(*msg);

  if (pub_my_point_.getNumSubscribers() == 0) return;

  ros_pointcloud cloud_msg;
  cloud_msg.header.stamp = ros::Time(data->t_ns * 1.0 * 1e-9);
  cloud_msg.header.frame_id = "odom";
  cloud_msg.point_step =
      12;  // 16 // 一个点占16个字节 Length of a point in bytes
           // cloud.fields = MakePointFields("xyzi"); //
  // 描述了二进制数据块中的通道及其布局。
  cloud_msg.fields = MakePointFields("xyz");

  const auto& points = data->projections[cam_id];
  cloud_msg.height = 1;
  cloud_msg.width = points.size();
  cloud_msg.data.resize(cloud_msg.width * cloud_msg.point_step);
  int i = 0;
  for (const auto& c : points) {
    // cv::Point(c[0], c[1])
    // KeypointId kpt_id = c[3];
    // const Keypoint<Scalar>& kpt_pos = lmdb.getLandmark(kpt_id);
    // float depth = static_cast<float>(1.0 / kpt_pos.inv_dist); // 逆深度
    float u = static_cast<float>(c[0]);
    float v = static_cast<float>(c[1]);
    float d = static_cast<float>(1.0 / c[2]);
    // std::cout << "c[2]=" << c[2] << "  d=" << d << std::endl;
    // std::cout << "u=" << u << "  v=" << v << std::endl;

    auto* ptr = reinterpret_cast<float*>(cloud_msg.data.data() +
                                         i * cloud_msg.point_step);

    ptr[0] = u;
    ptr[1] = v;
    ptr[2] = d;

    i++;
  }

  pub_my_point_.publish(cloud_msg);

/*
  const Vec2 p0 = opt_flow_meas->observations.at(0).at(lm_id).translation().cast<Scalar>();

  for (const auto& kv_obs : opt_flow_meas->observations[cam_id]) {
    int kpt_id = kv_obs.first; // 特征点id
      
    // 特征点在路标数据库中是否存在， 即判断路标点是否跟踪成功
    if (lmdb.landmarkExists(kpt_id)) { 
      Vec2 pos = kv_obs.second.translation().cast<Scalar>();
    }
  }
  */
}
#endif

#ifdef _IS_IMU_STATIONARY
void CRos1IO::CheckImuStillThread() {
  std::vector<sensor_msgs::ImuConstPtr> vecTmpImuData;
  constexpr int MAX_IMU_CNT = 20;  // 8;
  constexpr int Continuous_number = 3;//6;

  static int nBig_acc_cnt = 0;
  static int nSmall_acc_cnt = 0;
  sensor_msgs::ImuConstPtr imu_msg;

  std::cout << "CHECK if imu is stationary" << std::endl;

  // time_t vo_start_time;
  while (!bQuit) {

    vecTmpImuData.clear();
    // m_imu_data.lock();
    int nSize = tmp_imu_queue_.size();
    if (nSize >= MAX_IMU_CNT) {
      // vecTmpImuData.assign(vecImuData.begin(),
      //                      vecImuData.begin() + MAX_IMU_CNT);
      // vecImuData.clear();

      
      int i = 0;
      while (!tmp_imu_queue_.empty()) {
        tmp_imu_queue_.pop(imu_msg);
        vecTmpImuData.emplace_back(imu_msg);
        i++ ;
        if(i == MAX_IMU_CNT)
        {
          break ;
        }
      }
    }
    // m_imu_data.unlock();

    if (vecTmpImuData.size() == MAX_IMU_CNT) {
        int i = 0;
        double dx0 = 0, dy0 = 0, dz0 = 0, rx0 = 0, ry0 = 0, rz0 = 0;
        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
        double acc_sum = 0.0, ang_sum = 0.0;
        double acc_aver = 0.0, ang_aver = 0.0;

        for (auto& imu_msg : vecTmpImuData) {
          // double t = imu_msg->header.stamp.toSec();

          if (i == 0) {
            dx0 = imu_msg->linear_acceleration.x;
            dy0 = imu_msg->linear_acceleration.y;
            dz0 = imu_msg->linear_acceleration.z;
            rx0 = imu_msg->angular_velocity.x;
            ry0 = imu_msg->angular_velocity.y;
            rz0 = imu_msg->angular_velocity.z;
          }

          dx = imu_msg->linear_acceleration.x;
          dy = imu_msg->linear_acceleration.y;
          dz = imu_msg->linear_acceleration.z;
          rx = imu_msg->angular_velocity.x;
          ry = imu_msg->angular_velocity.y;
          rz = imu_msg->angular_velocity.z;

          Vector3d acc0(dx0, dy0, dz0), angular0(rx0, ry0, rz0);
          Vector3d acc(dx, dy, dz), angular(rx, ry, rz);
#if 1          
          acc_sum += (acc - acc0).norm();
          ang_sum += (angular - angular0).norm();

#if 0
          dx0 = dx;
          dy0 = dy;
          dz0 = dz;

          rx0 = rx;
          ry0 = ry;
          rz0 = rz;
#endif          

#else
          acc_sum += acc.norm();
          ang_sum += angular.norm();
#endif
          i++;
        }
#if 1
        acc_aver = acc_sum / (MAX_IMU_CNT - 1);
        ang_aver = ang_sum / (MAX_IMU_CNT - 1);
#else
        acc_aver = acc_sum / MAX_IMU_CNT;
        ang_aver = ang_sum / MAX_IMU_CNT;

        acc_sum = 0.0;
        ang_sum = 0.0;
        for (auto& imu_msg : vecTmpImuData) {
          dx = imu_msg->linear_acceleration.x;
          dy = imu_msg->linear_acceleration.y;
          dz = imu_msg->linear_acceleration.z;
          rx = imu_msg->angular_velocity.x;
          ry = imu_msg->angular_velocity.y;
          rz = imu_msg->angular_velocity.z;

          Vector3d acc(dx, dy, dz), angular(rx, ry, rz);
          acc_sum += (acc.norm() - acc_aver) * (acc.norm() - acc_aver);
          ang_sum += (angular.norm() - ang_aver) * (angular.norm() - ang_aver);

        }

        acc_aver = acc_sum / MAX_IMU_CNT;
        ang_aver = ang_sum / MAX_IMU_CNT;
#endif
        if (yaml_.acc_zero_velocity > 0 && yaml_.ang_zero_velocity > 0) {
          // if(acc_aver < 0.1 && ang_aver < 0.08) // for realsense.
          if (acc_aver < yaml_.acc_zero_velocity &&
              ang_aver < yaml_.ang_zero_velocity/**/)
          {
            // std::cout << "nSmall_acc_cnt=" << nSmall_acc_cnt << std::endl;
            nSmall_acc_cnt++;
            if(nSmall_acc_cnt > Continuous_number)
            {
              if (!isStill_) {
                // cout << "active 2 inactive: average of accelerate is " <<
                // acc_aver << " average of angular speed is " << ang_aver <<
                // std::endl;

                #if 0
                std::cout << "active 2 inactive: acc_aver= " << acc_aver
                    << " ang_aver= " << ang_aver
                    << std::endl;
                #endif
              }

              isStill_ = true;
              // std::cout << "imu is stationary" << std::endl;
              nSmall_acc_cnt = 0;
            }

            nBig_acc_cnt = 0;

          } else {
            nBig_acc_cnt++;
            // if(nBig_acc_cnt > 2)
            if (nBig_acc_cnt > Continuous_number)  // && g_new_feature_cnt > 20)
            {
              if (isStill_) {
                // cout << "inactive 2 active: average of accelerate is " <<
                // acc_aver << " average of angular speed is " << ang_aver <<
                // std::endl;

                #if 0
                std::cout << "inactive 2 active: acc_aver= " << acc_aver
                     << " ang_aver= " << ang_aver
                     << std::endl;
                #endif     
              }

              isStill_ = false;

              // std::cout << " imu is NOT stationary." << std::endl;

              nBig_acc_cnt = 0;
            }

            nSmall_acc_cnt = 0;
          }
        }

        // std::cout << " average of accelerate is " << acc_aver 
        //   << " average of angular speed is " << ang_aver << std::endl;


        // std::cout << " variance of accelerate is " << acc_aver 
        //   << " variance of angular speed is " << ang_aver << std::endl;  

      }

    usleep(5000);  // sleep 5 ms
  }
}
#endif
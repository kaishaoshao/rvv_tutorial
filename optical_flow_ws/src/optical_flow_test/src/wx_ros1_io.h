
// created by wxliu on 2023-11-9
#ifndef _WX_ROS1_IO_H_
#define _WX_ROS1_IO_H_

// #define _RECORD_BAG_

// #define _TEST_POINT_DEPTH_

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include <tbb/concurrent_queue.h>
#ifdef _RECORD_BAG_
#include <rosbag/bag.h>
#endif
#include <cv_bridge/cv_bridge.h>

// #include "std_msgs/string.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
//#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

// #include <sensor_pub/ImageInfo.h>
// #include "sensor_pub/ImageInfo.h" // tmp comment on 2024-12-20.
#include <std_msgs/Empty.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
// #include <sensor_msgs/PointCloud2Iterator.h>
#include <sensor_msgs/NavSatFix.h>

#include <basalt/optical_flow/optical_flow.h>

#ifdef _BACKEND_WX_
#include <basalt/imu/imu_types.h>
#include <basalt/vi_estimator/vio_estimator.h>
#endif

// #include <mypoint/MyPoint.h>

// #include <myodom/MyOdom.h> // tmp comment on 2024-12-20.

// #include <atp_info/atp.h> // tmp comment on 2024-12-20.

#include "wx_yaml_io.h"

#include "undistorted_photometric.h"

#include "util/cmap.h"

// #include "kalman_filter.h"

//#include <Eigen/Dense>

#ifdef _GNSS_WX_
// gnss
// #include <stereo3/LocalSensorExternalTrigger.h>
#include <gnss_comm/LocalSensorExternalTrigger.h>
#include <gnss_comm/gnss_ros.hpp>
#include <gnss_comm/gnss_utility.hpp>
#include <gnss_comm/gnss_spp.hpp>
using namespace gnss_comm;
#include "gnss_msg.h"
#endif

using namespace std::chrono_literals;

using std::placeholders::_1;


// #define USE_TIGHT_COUPLING 1

// #define _NOISE_SUPPRESSION_

#define _IS_FORWARD_
// #define _IS_IMU_STATIONARY // stationary

namespace wx {

using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
// using Quaterniond = Eigen::Quaterniond; 
using PointFields = std::vector<sensor_msgs::PointField>;
using sensor_msgs::PointField;
using ros_pointcloud = sensor_msgs::PointCloud2;

namespace cb = cv_bridge;
namespace sm = sensor_msgs;
// namespace gm = geometry_msgs;
namespace mf = message_filters;

typedef struct _TStereoImage
{
    sm::ImageConstPtr image0_ptr;
    sm::ImageConstPtr image1_ptr;
}TStereoImage;

class CRos1IO
{
public: 
    CRos1IO(const ros::NodeHandle& pnh, const TYamlIO &yaml) noexcept;
    // CRos1IO(const ros::NodeHandle& pnh, bool use_imu, int fps, long dt_ns = 0) noexcept;
    // CRos1IO(const ros::NodeHandle& pnh, bool use_imu, int fps, 
    //     std::vector<int> vec_tracked_points, std::vector<double> vec_confidence_levels, long dt_ns = 0) noexcept;
    ~CRos1IO();
#ifdef _BACKEND_WX_     
    void PublishPoints(basalt::VioVisualizationData::Ptr data);
    void PublishOdometry(basalt::PoseVelBiasState<double>::Ptr data);
#endif

    void Reset();
#ifdef _BACKEND_WX_    
    void PublishPoseAndPath(basalt::PoseVelBiasState<double>::Ptr data); // TODO: transfer timestamp of sampling.
    void PublishFeatureImage(basalt::VioVisualizationData::Ptr data);
    void PublishMyOdom(basalt::PoseVelBiasState<double>::Ptr data, bool bl_publish = false);
    void PublishAllPositions(TAllPositions::Ptr data);
#endif    
    void ResetPublishedOdom() { isResetOdom = true; }
    #ifdef _RECORD_BAG_
    void OpenRosbag();
    void CloseRosBag();// { bag_.close(); }
    void SetRecordBag(bool bl) { record_bag = bl; }
    inline bool GetRecordBag() { return record_bag; }
    void WriteBagThread();
    #endif

    void SetForward(bool bl) { is_Forward_ = bl; }
    inline bool GetForward() { return is_Forward_; }

    void SetComputeIntensityDiff(bool bl) { 
        prev_image_ = cv::Mat();
        intensity_diff_threshold = 0.0;
        intensity_diff_counter_ = 0;
        
        isComputeIntensityDiff_ = bl; 
    }
    inline bool GetComputeIntensityDiff() { return isComputeIntensityDiff_; }

#ifdef _ATP_WX_
    void atp_cb(atp_info::atp& atp_msg);
#endif

#ifdef _TEST_POINT_DEPTH_
    void PublishImageAndPoints(basalt::VioVisualizationData::Ptr data); // 2024-4-28.
#endif

private: 
    void PublishMyOdomThread();
 #ifdef _RECORD_BAG_
    void RecordBagThread();
#endif
#ifdef _IS_IMU_STATIONARY
    void CheckImuStillThread();
#endif
    void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg);// const; 

    void StereoCb(const sm::ImageConstPtr& image0_ptr,
        const sm::ImageConstPtr& image1_ptr);

    inline void getcolor(float p, float np, float& r, float& g, float& b);

    /// @brief Get size of datatype from PointFiled
    int GetPointFieldDataTypeBytes(uint8_t dtype) noexcept {
        // INT8 = 1u,
        // UINT8 = 2u,
        // INT16 = 3u,
        // UINT16 = 4u,
        // INT32 = 5u,
        // UINT32 = 6u,
        // FLOAT32 = 7u,
        // FLOAT64 = 8u,
        switch (dtype) {
            case 0U:
            return 0;
            case 1U:  // int8
            case 2U:  // uint8
            return 1;
            case 3U:  // int16
            case 4U:  // uint16
            return 2;
            case 5U:  // int32
            case 6U:  // uint32
            case 7U:  // float32
            return 4;
            case 8U:  // float64
            return 8;
            default:
            return 0;
        }
    }

    inline PointFields MakePointFields(const std::string& fstr) {
        std::vector<PointField> fields;
        fields.reserve(fstr.size());

        int offset{0};
        PointField field;

        for (auto s : fstr) {
            s = std::tolower(s);
            if (s == 'x' || s == 'y' || s == 'z') {
            field.name = s;
            field.offset = offset;
            field.datatype = PointField::FLOAT32;
            field.count = 1;
            } else if (s == 'i') {
            field.name = "intensity";
            field.offset = offset;
            field.datatype = PointField::FLOAT32;
            field.count = 1;
            } else {
            continue;
            }

            // update offset
            offset += GetPointFieldDataTypeBytes(field.datatype) * field.count;
            fields.push_back(field);
        }

        return fields;
    }

#ifdef _GNSS_WX_ 
// gnss
private:
    void gnss_ephem_callback(const GnssEphemMsgConstPtr &ephem_msg);
    void gnss_glo_ephem_callback(const GnssGloEphemMsgConstPtr &glo_ephem_msg);
    void gnss_meas_callback(const GnssMeasMsgConstPtr &meas_msg);
    void rtk_callback(const sensor_msgs::NavSatFix::ConstPtr &meas_msg);
    void gnss_iono_params_callback(const StampedFloat64ArrayConstPtr &iono_msg);
    void gnss_tp_info_callback(const GnssTimePulseInfoMsgConstPtr &tp_msg);
    void local_trigger_info_callback(const LocalSensorExternalTriggerConstPtr &trigger_msg);
#endif

public:
    std::function<void(basalt::OpticalFlowInput::Ptr)> feedImage_;
#ifdef _BACKEND_WX_ 
    std::function<void(basalt::ImuData<double>::Ptr)> feedImu_;
#endif    
    std::function<void(void)> stop_;
    std::function<void(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)>inputIMU_;
    std::function<void(bool bl)> zeroVelocity_;
    std::function<void(bool bl)> slowVelocity_;
    std::function<void(void)> reset_; // reset alogorithm.
    // std::function<void(double speed_, double calc_odom_result_, float confidence_coefficient_)>add_odom_frame_;
    std::function<void(double, double, double, float)>add_odom_frame_;
    std::function<bool(void)> isForward_;
    std::function<bool(void)> isPreintegrationstationary_;
#ifdef _GNSS_WX_    
    // gnss
    std::function<void(EphemBasePtr)> inputEphem_;
    std::function<void(double, const std::vector<double> &)> inputIonoParams_;
    std::function<void(const double)> inputGNSSTimeDiff_;
    std::function<void(const double)> inputLatestGNSSTime_;
    std::function<void(std::vector<ObsPtr>)> feedGnss_;//void feedGnss(std::vector<ObsPtr> data)
    std::function<void(uint64_t t, int status, const Eigen::Vector3d& lla)> feedRTK_;//void feedGnss(std::vector<ObsPtr> data)
#endif

private:
    ros::NodeHandle pnh_;
#ifdef _BACKEND_WX_     
    std::thread t_publish_myodom;
    tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> pvb_queue;
#endif

    TYamlIO yaml_;
    // double dt_s_ { 0.0 }; // if dt_s_ equal to '0.0', it's mean our sensor is already timestamp synchronization with atp.
    long dt_ns_ { 0 };
    int fps_ { 50 };
    bool use_imu_ = false;
    ros::Subscriber sub_imu_;
    mf::Subscriber<sm::Image> sub_image0_;
    mf::Subscriber<sm::Image> sub_image1_;
    // mf::Subscriber<sensor_pub::msg::ImageInfo> sub_image0_info_;
    // mf::Subscriber<sensor_pub::msg::ImageInfo> sub_image1_info_;
    static constexpr int NUM_CAMS = 2;
/*
 * 如果直接定义sync_stereo_，需要构造CRos2IO时，提前构造之，因为其没有缺省的构造函数
 * sync_stereo_(sub_image0_, sub_image1_, sub_image0_info_, sub_image1_info_, 10)
 * 另一个方法是借助std::optional
    mf::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image,
        sensor_pub::msg::ImageInfo, sensor_pub::msg::ImageInfo> sync_stereo_; // method 1
*/
    using SyncStereo = mf::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image>;
    std::optional<SyncStereo> sync_stereo_; // method 2

#if 0
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr sub_stop_;
#endif    
    ros::Publisher pub_point_;
    ros::Publisher pub_odom_;
    ros::Publisher pub_path_;
    ros::Publisher pub_spp_path_;
    ros::Publisher pub_slam_path_;
    ros::Publisher pub_rtk_path_;
    nav_msgs::Path path_msg_;
    nav_msgs::Path spp_path_msg_;
    nav_msgs::Path slam_path_msg_;
    nav_msgs::Path rtk_path_msg_;

    ros::Publisher pub_warped_img;

    ros::Publisher pub_my_odom_;
#ifdef _TEST_POINT_DEPTH_
    ros::Publisher pub_my_img_;
    ros::Publisher pub_my_point_;
#endif
    std::atomic<bool> isLampOn_ { true };

#if 0 //def _DEBUG_MODE_
    int nTracked_confidence_count { 0 };
    int nVelocity_confidence_count { 0 };
    int nLamp_confidence_count { 0 };
#endif
#if 0   
    int zero_confidence { 0 };
    int total_confidence { 0 };
#endif
    // Kalman_Filter_Struct kalman_filter_pt1;
    /* =
    {
        .Q_ProcessNoise = 0.001,
        .R_MeasureNoise = 0.1,
        .estimate_value_init = 25,
        .estimate_variance_init = 1,
        .estimate_value_curr = 25,
        .estimate_variance_curr = 1,
    };*/

    bool bQuit { false };
    bool isResetOdom { false };
    #ifdef _RECORD_BAG_
    rosbag::Bag bag_;
    std::string bag_name { "test.bag" };
    bool record_bag { false };
    std::mutex m_rec_;
#ifdef _BACKEND_WX_     
    tbb::concurrent_bounded_queue<TStereoImage> stereo_img_queue;
    tbb::concurrent_bounded_queue<sensor_msgs::ImuConstPtr> imu_queue;
    // tbb::concurrent_bounded_queue<atp_info::atpPtr> atp_queue; // atp_info::atp
    tbb::concurrent_bounded_queue<atp_info::atp> atp_queue; // atp_info::atp
#endif    
    std::thread t_write_bag;
    std::thread t_record_bag;
    #endif
#ifdef _IS_IMU_STATIONARY
    bool isStill_ { false };
    tbb::concurrent_bounded_queue<sensor_msgs::ImuConstPtr> tmp_imu_queue_;
    std::thread t_check_imu_still;
#endif

    bool isLightToggled { false };
    bool is_Forward_ { true };
    bool isComputeIntensityDiff_ { false };
    double intensity_diff_threshold { 0.0 };
    uint16_t intensity_diff_counter_ { 0 };
    cv::Mat prev_image_;
    // cv::Mat prev_image2_;
    bool isStationary_ { false };

    undist_lite* img_undistorter { nullptr };
    ColorMap cmap;

#ifdef _GNSS_WX_
    // gnss
    ros::Subscriber sub_ephem, sub_glo_ephem, sub_gnss_meas, sub_rtk, sub_gnss_iono_params;
    ros::Subscriber sub_gnss_time_pluse_info, sub_local_trigger_info;
    bool time_diff_valid { false };
    double time_diff_gnss_local { 0.0 };
    double latest_gnss_time { -1 };
    double next_pulse_time;
    bool next_pulse_time_valid {false};
    std::mutex m_time;
#endif
/*
 * minimal publisher and subscriber
 *
 */

};

}

#endif
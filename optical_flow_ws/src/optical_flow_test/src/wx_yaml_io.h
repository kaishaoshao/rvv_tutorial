

// created by wxliu on 2023-11-10

#ifndef _WX_YAML_IO_H_
#define _WX_YAML_IO_H_

#include <string>
using std::string;

#include <vector>
#include <Eigen/Dense>

namespace wx {

struct TYamlIO
{
    TYamlIO() = default;
    // virtual ~TYamlIO() noexcept = default;

    bool loose_coupling_imu { true };
    bool use_gvio { true }; // if true use 'SqrtKeypointGvioEstimator' else use 'SqrtKeypointVioEstimator'.
    bool show_gui = false;
    bool print_queue = false;
    std::string image0_topic {"/image_left"};
    std::string image1_topic {"/image_right"};
    std::string imu_topic {"/imu"};
    // gnss parameters.
    bool gnss_enable { true };
    std::string rtk_topic {"/ublox_driver/receive_lla"};           // rtk ground truth topic
    std::string gnss_meas_topic {"/ublox_driver/range_meas"};           // GNSS raw measurement topic
    std::string gnss_ephem_topic {"/ublox_driver/ephem"};              // GPS, Galileo, BeiDou ephemeris
    std::string gnss_glo_ephem_topic {"/ublox_driver/glo_ephem"};       // GLONASS ephemeris
    std::string gnss_iono_params_topic {"/ublox_driver/iono_params"};   // GNSS broadcast ionospheric parameters
    std::string gnss_tp_info_topic {"/ublox_driver/time_pulse_info"};   // PPS time info
    double gnss_elevation_thres {30 };            // satellite elevation threshold (degree)
    double gnss_psr_std_thres {2.0 };            // pseudo-range std threshold
    double gnss_dopp_std_thres {2.0 };           // doppler std threshold
    uint32_t gnss_track_num_thres {20 };           // number of satellite tracking epochs before entering estimator
    double gnss_ddt_sigma { 0.1 };
    double GNSS_DDT_WEIGHT { 10.0 };
    bool gnss_local_online_sync {true};                       // if perform online synchronization betwen GNSS and local time
    std::string local_trigger_info_topic {"/external_trigger"};   // external trigger info of the local sensor, if `gnss_local_online_sync` is 1
    double gnss_local_time_diff {18.0 };                      // difference between GNSS and local time (s), if `gnss_local_online_sync` is 0
    std::vector<double> GNSS_IONO_DEFAULT_PARAMS { 0.1118E-07, 0.2235E-07, -0.4172E-06, 0.6557E-06, 0.1249E+06, -0.4424E+06, 0.1507E+07, -0.2621E+06 };
    std::string diff_pose_path;
    int num_yaw_optimized { 200 };
    // the end.

    std::string cam_calib_path = "/root/lwx_dataset/config/viobot_b_calib_vo.json";
    std::string dataset_path;
    std::string dataset_type;
    std::string config_path = "/root/lwx_dataset/config/viobot_b_vio_config.json";
    //std::string config_path = "../data/kitti_config.json";
    std::string result_path;
    std::string trajectory_fmt;
    bool trajectory_groundtruth { false };
    int num_threads = 0;
    bool use_imu = false;
    bool use_double = false;
    std::string marg_data_path;
    std::vector<Eigen::Matrix3d> RIC;
    std::vector<Eigen::Vector3d> TIC;
    Eigen::Vector3d TGI; // t_gnss <- imu
    long dt_ns { 0 }; // if dt_s_ equal to 0, it's mean our sensor is already timestamp synchronization with atp.
    int fps { 50 };
    std::vector<int> vec_tracked_points;
    std::vector<double> vec_confidence_levels;
    double coefficient { 1.0 };
    double slow_velocity { 3.0 };
    double zero_velocity { 0.05 };
    double mean_value { 0.0 };
    
    bool tks_pro_integration { true };
    std::string src_ip_address = "192.168.55.28";
    std::string dest_ip_adderss = "192.168.55.21";
    bool debug_mode { false };
    std::string output_data_file = "/home/ita560/docker/data_output/";
    bool data_output { true };
    bool data_display { true };
    bool bag_flag { true };
    int atp_id { 0 };
    int number_of_255 { 36 };
    int log_freq{ 0 };

    double abnormal_velocity { 25.0 };
    bool record_bag { false };
    int record_duration { 180 }; // in second.
    // double acc_zero_velocity { 0.002 }; //{ 0.1 };
    // double ang_zero_velocity { 2e-6 }; // { 0.003 };

    double acc_zero_velocity { 0.1 }; // 0.08
    double ang_zero_velocity { 0.003 };
    double acc_high_velocity { 15.0 };
    double ang_high_velocity { 10.0 };
    double dt_threshold { 5.0 };
    double change_end_wait_time { 1.0 }; // in seconds.
    bool output_log { true };
    bool photometric_calibration { false };

    std::string gamma1 {};
    std::string vignette1 {};

    std::string gamma2 {};
    std::string vignette2 {};

    int image_width { 640 };
    int image_height { 528 };

    int computing_mode { 1 };
    double d_p { 0.003 };
    bool FAST_algorithm { true };
    float skip_top_ratio { 0.0 };
    float skip_bottom_ratio { 0.0 };
    float skip_left_ratio { 0.0 };
    float skip_right_ratio { 0.0 };
    int min_tracked_points { 2 };
    float max_init_depth { 0 };
    bool equalize { false }; // EQUALIZE
    int max_intensity { 255 };
    double intensity_diff_threshold { 0.0 };
    float kf_q { 0.001 };
    float kf_r { 2.0 };

    bool manual_set_intensity_diff { false };
    float manual_intensity_diff_threshold { 0.34 };
    int FAST_threshold { 40 }; // threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.
    int FAST_min_threshold { 5 };
    bool extract_one_time { false };

    // loop closure
    int add_keyframe_mode { 0 };

    int motion_flag = 1; // 0 -- staionary; 1 -- normal; 2 -- aggressive

    int fisheye { 0 };
    std::string fisheye_mask {};

    void ReadConfiguration(std::string sys_yaml_file = "");
#ifdef _NEED_WRITE_YAML    
    void WriteConfiguration();
#endif

    void ReadAdaptiveThreshold();
    void WriteAdaptiveThreshold();//(double val);
    inline void SetDiffPosePath(std::string str) { diff_pose_path = str; }

#if 0 // TODO
    TYamlIO& operator=(const TYamlIO& right);
    TYamlIO(const TYamlIO& right);
#endif
};

}

#endif

// @author: wxliu
// @date: 2023-11-10

#include "wx_yaml_io.h"

#include <fstream>
#include <yaml-cpp/yaml.h>

#include <unistd.h>
#include <basalt/utils/assert.h>

using namespace wx;

#if 0
TYamlIO::TYamlIO(const TYamlIO& right)
{
  //TODO
  // show_gui = right.show_gui;
    
}

TYamlIO& TYamlIO::operator=(const TYamlIO& right)
{
  //TODO
}
#endif

void TYamlIO::ReadConfiguration(std::string sys_yaml_file)
{

  // read parameters from file.
  // std::string config_file = "/home/lwx/./sys.yaml";
  // std::string config_file = "/root/./sys.yaml";

  // std::string pkg_path = ament_index_cpp::get_package_share_directory("stereo3");
  // std::string config_file = pkg_path + "/config/stereo3.yaml";

  // std::string config_file = "/root/stereo3_ws/install/share/stereo3/config/sys.yaml";
  // std::string config_file = "/home/ll/C/ws/basalt-lwx-ws/src/basalt-lwx/config/sys.yaml";
  std::string config_file = "/root/dev/stereo3_ros1_ws/install/share/stereo3/config/sys.yaml";
  if(!sys_yaml_file.empty()) config_file = sys_yaml_file;
  std::cout << "sys.yaml: " << config_file << std::endl;
  if(access(config_file.c_str(), 0) != 0)
  {
    // '!= 0' indicate that file do not exist.
    std::ofstream fout(config_file);

    YAML::Node config = YAML::LoadFile(config_file);
    
    // example of writting items to file
    //config["score"] = 99;
    //config["time"] = time(NULL);

    config["loose_coupling_imu"] = loose_coupling_imu;
    config["use_gvio"] = use_gvio;
    config["show_gui"] = show_gui;
    config["print_queue"] = print_queue;
    config["image0_topic"] = image0_topic;
    config["image1_topic"] = image1_topic;
    config["imu_topic"] = imu_topic;
    config["cam_calib_path"] = cam_calib_path;
    config["dataset_path"] = dataset_path;
    config["dataset_type"] = dataset_type;
    config["config_path"] = config_path;
    config["result_path"] = result_path;
    config["trajectory_fmt"] = trajectory_fmt;
    config["trajectory_groundtruth"] = trajectory_groundtruth;
    config["num_threads"] = num_threads;
    config["use_imu"] = use_imu;
    config["use_double"] = use_double;
    config["dt_ns"] = dt_ns;
    config["fps"] = fps;
    config["coefficient"] = coefficient;
    config["slow_velocity"] = slow_velocity;
    config["zero_velocity"] = zero_velocity;
    config["mean_value"] = mean_value;

    config["tks_pro_integration"] = tks_pro_integration;
    config["src_ip_address"] = src_ip_address;
    config["dest_ip_adderss"] = dest_ip_adderss;
    config["debug_mode"] = debug_mode;
    config["output_data_file"] = output_data_file;
    config["data_output"] = data_output;
    config["data_display"] = data_display;
    config["bag_flag"] = bag_flag;
    config["atp_id"] = atp_id;

    config["number_of_255"] = number_of_255;
    config["log_freq"] = log_freq;
    config["abnormal_velocity"] = abnormal_velocity;
    config["record_bag"] = record_bag;
    config["record_duration"] = record_duration;

    config["acc_zero_velocity"] = acc_zero_velocity;
    config["ang_high_velocity"] = ang_high_velocity;
    config["acc_high_velocity"] = acc_high_velocity;
    config["ang_zero_velocity"] = ang_zero_velocity;
    config["dt_threshold"] = dt_threshold;
    config["change_end_wait_time"] = change_end_wait_time;
    config["output_log"] = output_log;
    config["photometric_calibration"] = photometric_calibration;

    config["image_width"] = image_width;
    config["image_height"] = image_height;

    config["camera1"]["gamma"] = gamma1;
    config["camera1"]["vignette"] = vignette1;

    config["camera2"]["gamma"] = gamma2;
    config["camera2"]["vignette"] = vignette2;

    config["computing_mode"] = computing_mode;

    config["d_p"] = d_p;

    config["FAST_algorithm"] = FAST_algorithm;
    
    config["skip_top_ratio"] = skip_top_ratio;
    config["skip_bottom_ratio"] = skip_bottom_ratio;
    config["skip_left_ratio"] = skip_left_ratio;
    config["skip_right_ratio"] = skip_right_ratio;
    config["min_tracked_points"] = min_tracked_points;
    config["max_init_depth"] = max_init_depth;
    config["equalize"] = equalize;
    config["max_intensity"] = max_intensity;
    config["kf_q"] = kf_q;
    config["kf_r"] = kf_r;
    config["manual_set_intensity_diff"] = manual_set_intensity_diff;
    config["manual_intensity_diff_threshold"] = manual_intensity_diff_threshold;

    config["FAST_threshold"] = FAST_threshold;
    config["FAST_min_threshold"] = FAST_min_threshold;
    config["extract_one_time"] = extract_one_time;
    config["add_keyframe_mode"] = add_keyframe_mode;

/*
    std::vector<double> vector_T{1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0};
    
    config["body_T_cam0"]["data"] = vector_T;
*/

    // set gnss parameter.
    config["gnss_enable"] = gnss_enable;
    config["rtk_topic"] = rtk_topic;
    config["gnss_meas_topic"] = gnss_meas_topic;
    config["gnss_ephem_topic"] = gnss_ephem_topic;
    config["gnss_glo_ephem_topic"] = gnss_glo_ephem_topic;
    config["gnss_iono_params_topic"] = gnss_iono_params_topic;
    config["gnss_tp_info_topic"] = gnss_tp_info_topic;
    config["gnss_elevation_thres"] = gnss_elevation_thres;
    config["gnss_psr_std_thres"] = gnss_psr_std_thres;
    config["gnss_dopp_std_thres"] = gnss_dopp_std_thres;
    config["gnss_track_num_thres"] = gnss_track_num_thres;
    config["gnss_ddt_sigma"] = gnss_ddt_sigma;
    // GNSS_DDT_WEIGHT
    config["gnss_local_online_sync"] = gnss_local_online_sync;
    config["local_trigger_info_topic"] = local_trigger_info_topic;
    config["gnss_local_time_diff"] = gnss_local_time_diff;
    // config["gnss_iono_default_parameters"]["data"] = GNSS_IONO_DEFAULT_PARAMS;
    config["num_yaw_optimized"] = num_yaw_optimized;
    // the end.

    fout << config;

    fout.close();
   
  }
  else
  {
    // read items:
    YAML::Node config = YAML::LoadFile(config_file);
    /*
        * here is a demo for reading:
    int score = config["score"].as<int>();
    time_t time = config["time"].as<time_t>();
    if(config["image0_topic"].Type() == YAML::NodeType::Scalar)
    std::string image0_topic = config["image0_topic"].as<std::string>();
    */

    if(config["loose_coupling_imu"].Type() == YAML::NodeType::Scalar)
    loose_coupling_imu = config["loose_coupling_imu"].as<bool>();

    if(config["show_gui"].Type() == YAML::NodeType::Scalar)
    show_gui = config["show_gui"].as<bool>();

    if(config["print_queue"].Type() == YAML::NodeType::Scalar)
    print_queue = config["print_queue"].as<bool>();

    if(config["image0_topic"].Type() == YAML::NodeType::Scalar)
    image0_topic = config["image0_topic"].as<std::string>();

    if(config["image1_topic"].Type() == YAML::NodeType::Scalar)
    image1_topic = config["image1_topic"].as<std::string>();

    if(config["imu_topic"].Type() == YAML::NodeType::Scalar)
    imu_topic = config["imu_topic"].as<std::string>();

    if(config["cam_calib_path"].Type() == YAML::NodeType::Scalar)
    cam_calib_path = config["cam_calib_path"].as<std::string>();

    if(config["dataset_path"].Type() == YAML::NodeType::Scalar)
    dataset_path = config["dataset_path"].as<std::string>();

    if(config["dataset_type"].Type() == YAML::NodeType::Scalar)
    dataset_type = config["dataset_type"].as<std::string>();

    if(config["config_path"].Type() == YAML::NodeType::Scalar)
    config_path = config["config_path"].as<std::string>();

    if(config["result_path"].Type() == YAML::NodeType::Scalar)
    result_path = config["result_path"].as<std::string>();

    if(config["trajectory_fmt"].Type() == YAML::NodeType::Scalar)
    trajectory_fmt = config["trajectory_fmt"].as<std::string>();

    if(config["trajectory_groundtruth"].Type() == YAML::NodeType::Scalar)
    trajectory_groundtruth = config["trajectory_groundtruth"].as<bool>();

    if(config["num_threads"].Type() == YAML::NodeType::Scalar)
    num_threads = config["num_threads"].as<int>();

    if(config["use_imu"].Type() == YAML::NodeType::Scalar)
    use_imu = config["use_imu"].as<bool>();

    if(config["use_double"].Type() == YAML::NodeType::Scalar)
    use_double = config["use_double"].as<bool>();

    if(config["dt_ns"].Type() == YAML::NodeType::Scalar)
    dt_ns = config["dt_ns"].as<long>();

    if(config["fps"].Type() == YAML::NodeType::Scalar)
    fps = config["fps"].as<int>();

    if(config["coefficient"].Type() == YAML::NodeType::Scalar)
    coefficient = config["coefficient"].as<double>();

    if(config["slow_velocity"].Type() == YAML::NodeType::Scalar)
    slow_velocity = config["slow_velocity"].as<double>();

    if(config["zero_velocity"].Type() == YAML::NodeType::Scalar)
    zero_velocity = config["zero_velocity"].as<double>();

    if(config["mean_value"].Type() == YAML::NodeType::Scalar)
    mean_value = config["mean_value"].as<double>();

    // for tks.
    if(config["tks_pro_integration"].Type() == YAML::NodeType::Scalar)
    tks_pro_integration = config["tks_pro_integration"].as<bool>();

    if(config["src_ip_address"].Type() == YAML::NodeType::Scalar)
    src_ip_address = config["src_ip_address"].as<std::string>();

    if(config["dest_ip_adderss"].Type() == YAML::NodeType::Scalar)
    dest_ip_adderss = config["dest_ip_adderss"].as<std::string>();

    if(config["debug_mode"].Type() == YAML::NodeType::Scalar)
    debug_mode = config["debug_mode"].as<bool>();

    if(config["output_data_file"].Type() == YAML::NodeType::Scalar)
    output_data_file = config["output_data_file"].as<std::string>();

    if(config["data_output"].Type() == YAML::NodeType::Scalar)
    data_output = config["data_output"].as<bool>();

    if(config["data_display"].Type() == YAML::NodeType::Scalar)
    data_display = config["data_display"].as<bool>();

    if(config["bag_flag"].Type() == YAML::NodeType::Scalar)
    bag_flag = config["bag_flag"].as<bool>();

    if(config["atp_id"].Type() == YAML::NodeType::Scalar)
    atp_id = config["atp_id"].as<int>();

    if(config["number_of_255"].Type() == YAML::NodeType::Scalar)
    number_of_255 = config["number_of_255"].as<int>();

    if(config["log_freq"].Type() == YAML::NodeType::Scalar)
      log_freq = config["log_freq"].as<int>();

    if(config["abnormal_velocity"].Type() == YAML::NodeType::Scalar)  
      abnormal_velocity = config["abnormal_velocity"].as<double>();

    if(config["record_bag"].Type() == YAML::NodeType::Scalar)
    record_bag = config["record_bag"].as<bool>(); 

    if(config["record_duration"].Type() == YAML::NodeType::Scalar)
      record_duration = config["record_duration"].as<int>(); 

    if(config["acc_zero_velocity"].Type() == YAML::NodeType::Scalar)
    acc_zero_velocity = config["acc_zero_velocity"].as<double>();

    if(config["ang_zero_velocity"].Type() == YAML::NodeType::Scalar)
    ang_zero_velocity = config["ang_zero_velocity"].as<double>();

    if(config["acc_high_velocity"].Type() == YAML::NodeType::Scalar)
    acc_high_velocity = config["acc_high_velocity"].as<double>();

    if(config["ang_high_velocity"].Type() == YAML::NodeType::Scalar)
    ang_high_velocity = config["ang_high_velocity"].as<double>();

    if(config["dt_threshold"].Type() == YAML::NodeType::Scalar)
    dt_threshold = config["dt_threshold"].as<double>();

    if(config["change_end_wait_time"].Type() == YAML::NodeType::Scalar)
    change_end_wait_time = config["change_end_wait_time"].as<double>();

    if(config["output_log"].Type() == YAML::NodeType::Scalar)
    output_log = config["output_log"].as<bool>();

    if(config["photometric_calibration"].Type() == YAML::NodeType::Scalar)
    photometric_calibration = config["photometric_calibration"].as<bool>();

    if(config["image_width"].Type() == YAML::NodeType::Scalar)
    image_width = config["image_width"].as<int>();

    if(config["image_height"].Type() == YAML::NodeType::Scalar)
    image_height = config["image_height"].as<int>();

    // read photometric calibration file path.
    if(config["camera1"]["gamma"].Type() == YAML::NodeType::Scalar)
    gamma1 = config["camera1"]["gamma"].as<std::string>();

    if(config["camera1"]["vignette"].Type() == YAML::NodeType::Scalar)
    vignette1 = config["camera1"]["vignette"].as<std::string>();

    if(config["camera2"]["gamma"].Type() == YAML::NodeType::Scalar)
    gamma2 = config["camera2"]["gamma"].as<std::string>();

    if(config["camera2"]["vignette"].Type() == YAML::NodeType::Scalar)
    vignette2 = config["camera2"]["vignette"].as<std::string>();

    if(config["computing_mode"].Type() == YAML::NodeType::Scalar)
    computing_mode = config["computing_mode"].as<int>();

    if(config["d_p"].Type() == YAML::NodeType::Scalar)
    d_p = config["d_p"].as<double>();

    if(config["FAST_algorithm"].Type() == YAML::NodeType::Scalar)
    FAST_algorithm = config["FAST_algorithm"].as<bool>();

    if(config["skip_top_ratio"].Type() == YAML::NodeType::Scalar)
    skip_top_ratio = config["skip_top_ratio"].as<float>();

    if(config["skip_bottom_ratio"].Type() == YAML::NodeType::Scalar)
    skip_bottom_ratio = config["skip_bottom_ratio"].as<float>();

    if(config["skip_left_ratio"].Type() == YAML::NodeType::Scalar)
    skip_left_ratio = config["skip_left_ratio"].as<float>();

    if(config["skip_right_ratio"].Type() == YAML::NodeType::Scalar)
    skip_right_ratio = config["skip_right_ratio"].as<float>();

    if(config["min_tracked_points"].Type() == YAML::NodeType::Scalar)
    min_tracked_points = config["min_tracked_points"].as<int>();

    if(config["max_init_depth"].Type() == YAML::NodeType::Scalar)
    max_init_depth = config["max_init_depth"].as<float>();

    if(config["equalize"].Type() == YAML::NodeType::Scalar)
    equalize = config["equalize"].as<bool>();

    if(config["max_intensity"].Type() == YAML::NodeType::Scalar)
    max_intensity = config["max_intensity"].as<int>();

    if(config["kf_q"].Type() == YAML::NodeType::Scalar)
    kf_q = config["kf_q"].as<float>();

    if(config["kf_r"].Type() == YAML::NodeType::Scalar)
    kf_r = config["kf_r"].as<float>();

    if(config["manual_set_intensity_diff"].Type() == YAML::NodeType::Scalar)
    manual_set_intensity_diff = config["manual_set_intensity_diff"].as<bool>();

    if(config["manual_intensity_diff_threshold"].Type() == YAML::NodeType::Scalar)
    manual_intensity_diff_threshold = config["manual_intensity_diff_threshold"].as<float>();

    // for FAST algorithm
    if(config["FAST_threshold"].Type() == YAML::NodeType::Scalar)
    FAST_threshold = config["FAST_threshold"].as<int>();

    if(config["extract_one_time"].Type() == YAML::NodeType::Scalar)
    extract_one_time = config["extract_one_time"].as<bool>();

    if(config["FAST_min_threshold"].Type() == YAML::NodeType::Scalar)
    FAST_min_threshold = config["FAST_min_threshold"].as<int>();

    if(config["add_keyframe_mode"].Type() == YAML::NodeType::Scalar)
    add_keyframe_mode = config["add_keyframe_mode"].as<int>();

    // read imu_cam extrinsic
    std::vector<double> vector_T{1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0};
    vector_T = config["body_T_cam0"]["data"].as<std::vector<double>>();

    Eigen::Map<Eigen::Matrix4d> T(vector_T.data());
    Eigen::Matrix4d T2 = T.transpose();

    
    RIC.push_back(T2.block<3, 3>(0, 0));
    TIC.push_back(T2.block<3, 1>(0, 3));

    //imu_.setExtrinsic(RIC[0], TIC[0]);

    // read gnss_cam extrinsic
    std::vector<double> vector_TGI{0.0, 0.0, 0.0};
    vector_TGI = config["imu_T_gnss"]["data"].as<std::vector<double>>();
    TGI = Eigen::Map<Eigen::Vector3d>(vector_TGI.data());

    vec_tracked_points = config["confidence_interval"]["tracked_points"].as<std::vector<int>>();
    vec_confidence_levels = config["confidence_interval"]["confidence_levels"].as<std::vector<double>>();
/*
 * // no use in relase mode.    
    assert(vec_tracked_points.size() > 0);
    assert(vec_tracked_points.size() == vec_confidence_levels.size());
 */

    BASALT_ASSERT(vec_tracked_points.size() > 0);
    BASALT_ASSERT(vec_tracked_points.size() == vec_confidence_levels.size());

 
    // set gnss parameter. 2024-8-1
    // config["gnss_iono_default_parameters"]["data"] = GNSS_IONO_DEFAULT_PARAMS;

    if(config["use_gvio"].Type() == YAML::NodeType::Scalar)
    use_gvio = config["use_gvio"].as<bool>();

    if(config["gnss_enable"].Type() == YAML::NodeType::Scalar)
    gnss_enable = config["gnss_enable"].as<bool>();

    if(config["rtk_topic"].Type() == YAML::NodeType::Scalar)
    rtk_topic = config["rtk_topic"].as<std::string>();

    if(config["gnss_meas_topic"].Type() == YAML::NodeType::Scalar)
    gnss_meas_topic = config["gnss_meas_topic"].as<std::string>();

    if(config["gnss_ephem_topic"].Type() == YAML::NodeType::Scalar)
    gnss_ephem_topic = config["gnss_ephem_topic"].as<std::string>();

    if(config["gnss_glo_ephem_topic"].Type() == YAML::NodeType::Scalar)
    gnss_glo_ephem_topic = config["gnss_glo_ephem_topic"].as<std::string>();

    if(config["gnss_iono_params_topic"].Type() == YAML::NodeType::Scalar)
    gnss_iono_params_topic = config["gnss_iono_params_topic"].as<std::string>();

    if(config["gnss_tp_info_topic"].Type() == YAML::NodeType::Scalar)
    gnss_tp_info_topic = config["gnss_tp_info_topic"].as<std::string>();

    if(config["gnss_elevation_thres"].Type() == YAML::NodeType::Scalar)
    gnss_elevation_thres = config["gnss_elevation_thres"].as<double>();

    if(config["gnss_psr_std_thres"].Type() == YAML::NodeType::Scalar)
    gnss_psr_std_thres = config["gnss_psr_std_thres"].as<double>();

    if(config["gnss_dopp_std_thres"].Type() == YAML::NodeType::Scalar)
    gnss_dopp_std_thres = config["gnss_dopp_std_thres"].as<double>();

    if(config["gnss_track_num_thres"].Type() == YAML::NodeType::Scalar)
    gnss_track_num_thres = config["gnss_track_num_thres"].as<unsigned int>();

    if(config["gnss_ddt_sigma"].Type() == YAML::NodeType::Scalar)
    gnss_ddt_sigma = config["gnss_ddt_sigma"].as<double>();
    GNSS_DDT_WEIGHT = 1.0 / gnss_ddt_sigma;

    if(config["gnss_local_online_sync"].Type() == YAML::NodeType::Scalar)
    gnss_local_online_sync = config["gnss_local_online_sync"].as<bool>();

    if(config["local_trigger_info_topic"].Type() == YAML::NodeType::Scalar)
    local_trigger_info_topic = config["local_trigger_info_topic"].as<std::string>();

    if(config["gnss_local_time_diff"].Type() == YAML::NodeType::Scalar)
    gnss_local_time_diff = config["gnss_local_time_diff"].as<double>();

    if(config["gnss_iono_default_parameters"]["data"].Type() == YAML::NodeType::Scalar)
    GNSS_IONO_DEFAULT_PARAMS = config["gnss_iono_default_parameters"]["data"].as<std::vector<double>>();

    if(config["num_yaw_optimized"].Type() == YAML::NodeType::Scalar)
    num_yaw_optimized = config["num_yaw_optimized"].as<int>();
    
    // the end.

  }

}

#ifdef _NEED_WRITE_YAML
void TYamlIO::WriteConfiguration()
{
    std::string config_file = "./sys.yaml";
    if(access(config_file.c_str(), 0) != 0)
    {
        // '!= 0' indicate that file do not exist.
        std::ofstream fout(config_file);

        YAML::Node config = YAML::LoadFile(config_file);

        // example of writting item to file
        config["score"] = 99;
        config["time"] = time(NULL);

        fout << config;

        fout.close();
    }
    else
    {}

    YAML::Node config = YAML::LoadFile(config_file);
    int score = config["score"].as<int>();
    time_t time = config["time"].as<time_t>();

    std::ofstream fout(config_file);
    config["score"] = 100;
    config["time"] = time(NULL);

    fout << config;

    fout.close();
    
}
#endif

void TYamlIO::ReadAdaptiveThreshold()
{
 
  // read time from file.
  std::string config_file = "./adaptive.yaml";
  if(access(config_file.c_str(), 0) != 0)
  {
    // '!= 0' indicate that file do not exist.
    std::ofstream fout(config_file);

    YAML::Node config = YAML::LoadFile(config_file);

    config["intensity_diff_threshold"] = 0;

    fout << config;

    fout.close();
   
  }
  else
  {
    YAML::Node config = YAML::LoadFile(config_file);
    if(config["intensity_diff_threshold"].Type() == YAML::NodeType::Scalar)
    intensity_diff_threshold = config["intensity_diff_threshold"].as<double>();
  }

}

void TYamlIO::WriteAdaptiveThreshold()
{
  //
  std::string config_file = "./adaptive.yaml";
  if(access(config_file.c_str(), 0) != 0)
  {
      // '!= 0' indicate that file do not exist.
      std::ofstream fout(config_file);

      YAML::Node config = YAML::LoadFile(config_file);

      // example of writting item to file
      // config["intensity_diff_threshold"] = 0;
      config["intensity_diff_threshold"] = intensity_diff_threshold;

      fout << config;

      fout.close();

      return ;
  }
  else
  {}

  YAML::Node config = YAML::LoadFile(config_file);

  std::ofstream fout(config_file);
  config["intensity_diff_threshold"] = intensity_diff_threshold;
 
  fout << config;

  fout.close();
}
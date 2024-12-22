
#pragma once

#include <string>

namespace wx {

enum class LinearizationType { ABS_QR, ABS_SC, REL_SC };

struct OpticalFlowConfig {
  OpticalFlowConfig()
  {
    optical_flow_type = "frame_to_frame";
    optical_flow_detection_grid_size = 50;
    delta_grid_size = 0; //20;
    optical_flow_max_recovered_dist2 = 0.004; //0.09f;
    optical_flow_pattern = 51;
    optical_flow_max_iterations = 5;
    optical_flow_levels = 4; //3;
    optical_flow_epipolar_error = 0.005;
    optical_flow_skip_frames = 1;
  }
//   void load(const std::string& filename);
//   void save(const std::string& filename);

  std::string optical_flow_type;
  int optical_flow_detection_grid_size;
  int delta_grid_size; // 2023-12-19.
  float optical_flow_max_recovered_dist2;
  int optical_flow_pattern;
  int optical_flow_max_iterations;
  int optical_flow_levels;
  float optical_flow_epipolar_error;
  int optical_flow_skip_frames;
};

} // namespace wx
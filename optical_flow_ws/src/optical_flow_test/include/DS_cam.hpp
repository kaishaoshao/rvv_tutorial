#pragma once

#include <iostream>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace hm {

class DoubleSphereCamera
{
public:
    DoubleSphereCamera(const double fx,
                       const double fy,
                       const double cx,
                       const double cy,
                       const double xi,
                       const double alpha)
    {
        param_[0] = fx;
        param_[1] = fy;
        param_[2] = cx;
        param_[3] = cy;
        param_[4] = xi;
        param_[5] = alpha;

        int width = 640;
        int height = 480;
        cv::Mat map(height, width, CV_32FC1);
        undistor_image_map_x = map.clone();
        undistor_image_map_y = map.clone();
        undistor_point_map_x = map.clone();
        undistor_point_map_y = map.clone();
        liftProject_map_x = map.clone();
        liftProject_map_y = map.clone();
        for (int v = 0; v < height; v++)
        {
            for (int u = 0; u < width; u++)
            {
                // undistor
                const double x = (double(u) - cx) / fx;
                const double y = (double(v) - cy) / fy;

                const double r2 = x * x + y * y;
                const double d1 = std::sqrt(r2 + 1.0);
                const double d2 = std::sqrt(r2 + (xi * d1 + 1.0) * (xi * d1 + 1.0));
                const double scaling = 1.0f / (alpha * d2 + (1.0 - alpha) * (xi * d1 + 1.0));
                const double xd = x * scaling;
                const double yd = y * scaling;
                const double xDistorted = xd * fx + cx;
                const double yDistorted = yd * fy + cy;

                undistor_image_map_x.at<float>(v, u) = xDistorted;
                undistor_image_map_y.at<float>(v, u) = yDistorted;

                // liftProject
                const double xi2_2 = alpha * alpha;
                const double xi1_2 = xi * xi;
                const double sqrt2 = sqrt(double(1) - (double(2) * alpha - double(1)) * r2);
                const double norm2 = alpha * sqrt2 + double(1) - alpha;
                const double z = (double(1) - xi2_2 * r2) / norm2;
                const double z2 = z * z;

                const double norm1 = z2 + r2;
                const double sqrt1 = sqrt(z2 + (double(1) - xi1_2) * r2);
                const double k = (z * xi + sqrt1) / norm1;

                Eigen::Vector3d P;
                P << k * x, k * y, k * z - xi;

                P[0] /= P[2];
                P[1] /= P[2];
                P[2] = 1.0;
                liftProject_map_x.at<float>(v, u) = P[0];
                liftProject_map_y.at<float>(v, u) = P[1];

                undistor_point_map_x.at<float>(v, u) = P[0] * fx + cx;
                undistor_point_map_y.at<float>(v, u) = P[1] * fy + cy;
            }
        }
    }

    ~DoubleSphereCamera() {}

    inline void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P)
    {
        P[0] = liftProject_map_x.at<float>(p[1], p[0]);
        P[1] = liftProject_map_y.at<float>(p[1], p[0]);
        P[2] = 1.0;
    }

    inline void undistorPoint(const cv::Point2f &in_p, cv::Point2f &out_p)
    {
        out_p.x = undistor_point_map_x.at<float>(in_p.y, in_p.x);
        out_p.y = undistor_point_map_y.at<float>(in_p.y, in_p.x);
    }

    inline void undistorImage(cv::Mat &image, cv::Mat &undistor_image)
    {
        cv::remap(image, undistor_image, undistor_image_map_x, undistor_image_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

private:
    cv::Mat undistor_image_map_x, undistor_image_map_y;
    cv::Mat undistor_point_map_x, undistor_point_map_y;
    cv::Mat liftProject_map_x, liftProject_map_y;
    double param_[6];
};

} // namespace hm
#pragma once

#include <Eigen/Core>
#include <string>

namespace wx {

/// @brief A simple color map implementation inspired by
/// github.com/yuki-koyama/tinycolormap/blob/master/include/tinycolormap.hpp
class ColorMap {
 public:
  using Rgb = Eigen::Vector3d;

  ColorMap() = default;
  ColorMap(std::string name, const std::vector<Eigen::Vector3d>& colors);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const ColorMap& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Map input x to color rgb/bgr, assumes x is in [0, 1]
  Eigen::Vector3d GetRgb(double x) const noexcept;
  Eigen::Vector3d GetBgr(double x) const noexcept {
    return GetRgb(x).reverse();
  }

  bool Ok() const noexcept { return !data_.empty(); }
  const std::string& name() const noexcept { return name_; }
  int size() const noexcept { return static_cast<int>(data_.size()); }

 private:
  std::string name_;
  double step_{};
  std::vector<Eigen::Vector3d> data_;
};

/// @brief Factory
ColorMap MakeCmapJet();
ColorMap MakeCmapHeat();
ColorMap MakeCmapTurbo();
ColorMap MakeCmapPlasma();
ColorMap GetColorMap(std::string_view name);


template <typename T>
struct Interval {
  Interval() = default;
  Interval(const T& left, const T& right) noexcept
      : left_(left), right_(right) {
    // CHECK_LE(left, right);
  }

  T left_, right_;

  const T& a() const noexcept { return left_; }
  const T& b() const noexcept { return right_; }
  T width() const noexcept { return b() - a(); }
  bool empty() const noexcept { return b() <= a(); }
  bool ContainsClosed(const T& v) const noexcept {
    return (a() <= v) && (v <= b());
  }
  bool ContainsOpen(const T& v) const noexcept {
    return (a() < v) && (v < b());
  }

  /// Whether this interval contains other
  bool ContainsClosed(const Interval<T>& other) const noexcept {
    return a() <= other.a() && other.b() <= b();
  }

  /// Normalize v to [0, 1], assumes v in [left, right]
  /// Only enable if we have floating type
  T Normalize(const T& v) const noexcept {
    static_assert(std::is_floating_point_v<T>, "Must be floating point");
    return (v - a()) / width();
  }

  /// InvNormalize v to [left, right], assumes v in [0, 1]
  T InvNormalize(const T& v) const noexcept {
    static_assert(std::is_floating_point_v<T>, "Must be floating point");
    return v * width() + a();
  }
};

using IntervalF = Interval<float>;
using IntervalD = Interval<double>;

}  // namespace zc

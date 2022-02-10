#pragma once

#include "opencv2/core/mat.hpp"

/// \brief Represents a multivariate normal model.
///
/// Given some training samples, the class will train a multivariate normal model of the data.
class MultivariateNormalModel
{
public:
  /// Train a new model.
  explicit MultivariateNormalModel(const cv::Mat& samples);

  /// Compute the Mahalanobis distance between the model and a image of feature pixel vectors.
  cv::Mat computeMahalanobisDistances(const cv::Mat& image) const;

  /// Returns the trained mean.
  cv::Mat mean() const;

  /// Returns the trained covariance.
  cv::Mat covariance() const;

  /// Returns the inverse of the trained covariance.
  cv::Mat inverseCovariance() const;

private:
  cv::Mat mean_;
  cv::Mat covariance_;
  cv::Mat inverse_covariance_;

  /// Train a model given some training samples
  void performTraining(const cv::Mat& samples);
};

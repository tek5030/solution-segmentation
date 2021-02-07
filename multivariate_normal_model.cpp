#include "multivariate_normal_model.h"
#include "opencv2/imgproc.hpp"


MultivariateNormalModel::MultivariateNormalModel(const cv::Mat& samples)
{
  performTraining(samples);
}


void MultivariateNormalModel::performTraining(const cv::Mat& samples)
{
  cv::calcCovarMatrix(samples, covariance_, mean_, cv::COVAR_NORMAL | cv::COVAR_COLS);
  cv::invert(covariance_, inverse_covariance_, cv::DECOMP_SVD);
}


cv::Mat MultivariateNormalModel::computeMahalanobisDistances(const cv::Mat& image) const
{
  // We need to represent the distances in uint8 because of Otsu's method.
  // We get a pretty good representation by multiplying the distances with 1000.
  constexpr double dist_to_uint8_scale = 1000.0;

  // Convert to double precision and reshape to feature vector columns.
  cv::Mat samples_in_double_precision;
  image.convertTo(samples_in_double_precision, CV_64F);
  samples_in_double_precision = samples_in_double_precision.reshape(1, samples_in_double_precision.total()).t();

  cv::Mat mahalanobis_img(image.size(), CV_64FC1);

  // For the fastest possible access of image data
  // see https://docs.opencv.org/4.0.1/db/da5/tutorial_how_to_scan_images.html
  const auto mahalanobis_img_ptr = mahalanobis_img.ptr<double>();

  for (int i=0; i < samples_in_double_precision.cols; ++i)
  {
    const cv::Mat sample = samples_in_double_precision.col(i);
    mahalanobis_img_ptr[i] = cv::Mahalanobis(sample , mean_, inverse_covariance_);
  }

  // Scale and convert to uint8.
  mahalanobis_img.convertTo(mahalanobis_img, CV_8U, dist_to_uint8_scale);
  return mahalanobis_img;
}


cv::Mat MultivariateNormalModel::mean() const
{
  return mean_.clone();
}


cv::Mat MultivariateNormalModel::covariance() const
{
  return covariance_.clone();
}


cv::Mat MultivariateNormalModel::inverseCovariance() const
{
  return inverse_covariance_.clone();
}

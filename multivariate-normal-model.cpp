#include "multivariate-normal-model.h"
#include "opencv2/imgproc.hpp"

MultivariateNormalModel::MultivariateNormalModel(const cv::Mat& samples)
{
  performTraining(samples);
}


void MultivariateNormalModel::performTraining(const cv::Mat& samples)
{
  cv::calcCovarMatrix(samples, covariance_, mean_, cv::COVAR_NORMAL | cv::COVAR_ROWS);
  covariance_ /= (samples.rows - 1);
  cv::invert(covariance_, inverse_covariance_, cv::DECOMP_SVD);
}


cv::Mat MultivariateNormalModel::computeMahalanobisDistances(const cv::Mat& image) const
{
  // Convert to double precision and reshape to feature vector rows.
  cv::Mat samples_in_double_precision;
  image.convertTo(samples_in_double_precision, CV_64F);
  const auto num_samples = static_cast<int>(samples_in_double_precision.total());
  samples_in_double_precision = samples_in_double_precision.reshape(1, num_samples);

  cv::Mat mahalanobis_img(image.size(), CV_64FC1);

  // For the fastest possible access of image data
  // see https://docs.opencv.org/4.0.1/db/da5/tutorial_how_to_scan_images.html
  const auto mahalanobis_img_ptr = mahalanobis_img.ptr<double>();

  for (int i=0; i < samples_in_double_precision.rows; ++i)
  {
    const cv::Mat sample = samples_in_double_precision.row(i);
    mahalanobis_img_ptr[i] = cv::Mahalanobis(sample , mean_, inverse_covariance_);
  }

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

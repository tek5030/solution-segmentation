#include "multivariate-normal-model.h"
#include "opencv2/imgproc.hpp"

MultivariateNormalModel::MultivariateNormalModel(const cv::Mat& samples)
{
  performTraining(samples);
}


void MultivariateNormalModel::performTraining(const cv::Mat& samples)
{
  cv::calcCovarMatrix(samples, covariance_, mean_, cv::COVAR_NORMAL | cv::COVAR_ROWS, CV_32F);
  covariance_ /= (samples.rows - 1);
  cv::invert(covariance_, inverse_covariance_, cv::DECOMP_SVD);
}


cv::Mat MultivariateNormalModel::computeMahalanobisDistances(const cv::Mat& image) const
{
  // Convert to double precision and reshape to feature vector rows.
  cv::Mat float_image;
  image.convertTo(float_image, CV_32F);
  const auto num_samples = float_image.total();
  float_image = float_image.reshape(1, static_cast<int>(num_samples));

  cv::Mat mahalanobis_img(image.size(), CV_32F);

  // For the fastest possible access of image data
  // see https://docs.opencv.org/4.0.1/db/da5/tutorial_how_to_scan_images.html
  const auto mahalanobis_img_ptr = mahalanobis_img.ptr<float>();

  for (int i=0; i < float_image.rows; ++i)
  {
    const cv::Mat sample = float_image.row(i);
    mahalanobis_img_ptr[i] = static_cast<float>(cv::Mahalanobis(sample , mean_, inverse_covariance_));
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

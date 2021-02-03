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
  constexpr double dist_to_uint8_scale = 1000.0;

  cv::Mat mahalanobis_img(image.size(), CV_8UC1);

  using Pixel = cv::Vec<uint8_t, 3>;
  image.forEach<Pixel>(
      [&](const Pixel& pixel, const int* pos)
      {
        const cv::Vec3d double_pixel(pixel(0), pixel(1), pixel(2));
        const double mahalanobis = cv::Mahalanobis(double_pixel, mean_, inverse_covariance_);
        mahalanobis_img.at<uint8_t>(pos[0], pos[1]) = static_cast<uint8_t>(dist_to_uint8_scale * mahalanobis);
      });

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

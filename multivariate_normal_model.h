#pragma once

#include "opencv2/core/mat.hpp"

/// \brief A class that stores a trained multivariate normal model.
///
/// Given some training samples, the class will create a multivariate normal model of the data.
/// When successfully trained, the model can compute a transformed Mahalanobis distance that
/// indicates on a scale [0,1] how well a test pixel fits with the model.
class MultivariateNormalModel
{
public:
  /// Create a new, empty model.
  MultivariateNormalModel();

  /// Tell if the model has been trained yet or not.
  bool isTrained() const;

  /// Create a model given some training samples
  void performTraining(const cv::Mat& samples);

  /// If the model is already trained, compute transformed Mahalanobis distance image for all pixels in image.
  cv::Mat computeTransformedMahalanobisImage(const cv::Mat& image) const;

  /// Replace a ratio of samples_ with new_samples and retrain the model
  void update(const cv::Mat& new_samples, float update_ratio);

private:
  bool is_trained_;
  cv::Mat inv_covar_;
  cv::Mat mu_;
  cv::Mat samples_;

  cv::Mat computeMahalanobisDistances(const cv::Mat& image) const;
};

/// \brief Returns a binary image (values 0 or 255) which is the result of
/// thresholding the input_image with the threshold threshold_value.
/// \param[in] input_image Assumed to be of type CV_8U.
/// \param[in] threshold_value Assumed to be a number between 0 and 255
/// \param[in] use_otsu Option whether to use Otsu's method or not
/// \return The segmented image
cv::Mat segmentImage(const cv::Mat& input_image, int threshold_value, bool use_otsu);

/// \brief Creates a cv::Rect with size and position scaled relative to some input img_size
/// \param[in] img_size The size of the image you want to sample from
/// \return The computed cv::Rect
cv::Rect getSamplingRectangle(const cv::Size& img_size);

/// \brief Extract training samples from a rectangle within an image.
/// \param[in] source_image
/// \param[in] sampling_rectangle
/// \return A [m x n]-sized row-vector of samples, where [m x n] is the number of pixels within the sampling rectangle
/// and each sample is a column vector with values from each channel (which means that a [(m x n) x ch] matrix is returned).
cv::Mat extractTrainingSamples(const cv::Mat& source_image, const cv::Rect& sampling_rectangle);

/// \brief Draw a rectangle on an image
/// \param[in, out] image The image will be modified, a rectangle will be drawn on top of it.
/// \param[in] sampling_rectangle The rectangle which will be drawn.
void drawSamplingRectangle(cv::Mat& image, const cv::Rect& sampling_rectangle);

# pragma once

#include "opencv2/highgui.hpp"
#include <string>

class SegmentationLabGUI
{
public:
  SegmentationLabGUI(int initial_thresh_val, double max_thresh_val);

  ~SegmentationLabGUI();

  void setThreshold(int thresh);

  int getThreshold() const;

  void showFrame(const cv::Mat& frame_img) const;

  void showMahalanobis(const cv::Mat& mahalanobis_img) const;

  static char waitKey(int delay);

private:
  std::string win_name_input_;
  std::string win_name_result_;
  std::string threshold_trackbar_name_;
  int thresh_val_;
};

/// \brief Creates a cv::Rect with size and position scaled relative to some input img_size
/// \param[in] img_size  The size of the image you want to sample from
/// \param[in] rect_size The size of the sampling rectangle
/// \return The computed cv::Rect
cv::Rect getSamplingRectangle(const cv::Size& img_size, const cv::Size& rect_size = {100, 80});

/// \brief Draw a rectangle on an image
/// \param[in, out] image The image will be modified, a rectangle will be drawn on top of it.
/// \param[in] sampling_rectangle The rectangle which will be drawn.
void drawSamplingRectangle(cv::Mat& image, const cv::Rect& sampling_rectangle);

/// \brief Extract training samples from a rectangle within an image.
/// \param[in] source_image
/// \param[in] sampling_rectangle
/// \return A cv::Mat with sample vectors as columns.
/// For a m x n sampling rectangle of a k-channel image, this returns a cv::Mat with k rows, m*n columns and 1 channel.
cv::Mat extractTrainingSamples(const cv::Mat& source_image, const cv::Rect& sampling_rectangle);


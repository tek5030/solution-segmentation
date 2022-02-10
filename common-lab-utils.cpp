#include "common-lab-utils.h"
#include "opencv2/imgproc.hpp"

SegmentationLabGUI::SegmentationLabGUI(const int initial_thresh_val, const double max_thresh_val)
    : win_name_input_{"Segmentation - Input frame"}
    , win_name_result_{"Segmentation - Mahalanobis distances"}
    , threshold_trackbar_name_{"Threshold"}
    , thresh_val_{initial_thresh_val}
{
  cv::namedWindow(win_name_input_, cv::WINDOW_NORMAL);
  cv::namedWindow(win_name_result_, cv::WINDOW_NORMAL);
  cv::createTrackbar(threshold_trackbar_name_,
                     win_name_input_,
                     nullptr,
                     static_cast<int>(std::round(max_thresh_val)),
                     [](auto val, auto ptr)
                     {
                       reinterpret_cast<SegmentationLabGUI*>(ptr)->setThreshold(val);
                     }, this);
  setThreshold(thresh_val_);
}

SegmentationLabGUI::~SegmentationLabGUI()
{
  cv::destroyWindow(win_name_input_);
  cv::destroyWindow(win_name_result_);
}

void SegmentationLabGUI::setThreshold(const int thresh)
{
  thresh_val_ = thresh;
  cv::setTrackbarPos(threshold_trackbar_name_, win_name_input_, thresh_val_);
}

int SegmentationLabGUI::getThreshold() const
{ return thresh_val_; }

void SegmentationLabGUI::showFrame(const cv::Mat& frame_img) const
{
  cv::imshow(win_name_input_, frame_img);
}

void SegmentationLabGUI::showMahalanobis(const cv::Mat& mahalanobis_img) const
{
  cv::imshow(win_name_result_, mahalanobis_img);
}

char SegmentationLabGUI::waitKey(const int delay)
{ return static_cast<char>(cv::waitKey(delay)); }

cv::Rect getSamplingRectangle(const cv::Size& img_size, const cv::Size& rect_size)
{
  int center_x = img_size.width / 2;
  int center_y = 4 * img_size.height / 5;
  int width = rect_size.width;
  int height = rect_size.height;
  int top_left_x = center_x - width / 2;
  int top_left_y = center_y - height / 2;

  const cv::Rect sampling_rectangle(top_left_x, top_left_y, width, height);
  const cv::Rect entire_image(0, 0, img_size.width, img_size.height);

  // This operation ensures that the boundaries of the returned sampling rectangle are within the image dimensions,
  // just in case the chosen width or height is too large.
  return (sampling_rectangle & entire_image);
}

void drawSamplingRectangle(cv::Mat& image, const cv::Rect& sampling_rectangle)
{
  const cv::Scalar colour{0, 0, 255};
  constexpr int thickness = 3;
  cv::rectangle(image, sampling_rectangle, colour, thickness);
}

cv::Mat extractTrainingSamples(const cv::Mat& source_image, const cv::Rect& sampling_rectangle)
{
  cv::Mat patch = source_image(sampling_rectangle).clone();
  cv::Mat samples = patch.reshape(1, static_cast<int>(patch.total()));
  return samples;
}

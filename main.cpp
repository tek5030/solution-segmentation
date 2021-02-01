#include "multivariate_normal_model.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

int main()
{
  cv::VideoCapture cap{0};
  if (!cap.isOpened())
  {
    std::cerr << "Could not open VideoCapture" << std::endl;
    return EXIT_FAILURE;
  }

  MultivariateNormalModel model;

  const std::string win_name_input{"Lab 11: Segmentation - Input"};
  cv::namedWindow(win_name_input, cv::WINDOW_NORMAL);

  const std::string win_name_result{"Lab 11: Segmentation - Transformed Mahalanobis distance"};
  cv::namedWindow(win_name_result, cv::WINDOW_NORMAL);

  const int max_threshold{255};
  int current_threshold{240}; //Start value of the threshold slider
  bool use_otsu{false};
  bool use_adaptive_model{false};
  float adaptive_update_ratio{0.1f};

  cv::createTrackbar("Threshold", win_name_input, &current_threshold, max_threshold);

  for (cv::Mat frame; cap.read(frame); )
  {
    cv::Mat feature_image = frame.clone();
    cv::Rect sampling_rectangle = getSamplingRectangle(feature_image.size());

    if (model.isTrained())
    {
      cv::Mat transformed_maha = model.computeTransformedMahalanobisImage(feature_image);
      transformed_maha.convertTo(transformed_maha, CV_8U, 255);
      cv::Mat segmented = segmentImage(transformed_maha, current_threshold, use_otsu);

      cv::morphologyEx(segmented, segmented, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {5,5}));
      cv::morphologyEx(segmented, segmented, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {7,7}));

      cv::imshow(win_name_result, transformed_maha);
      frame.setTo(cv::Scalar{0, 255, 0}, segmented);
    }

    drawSamplingRectangle(frame, sampling_rectangle);
    cv::imshow(win_name_input, frame);

    if (use_adaptive_model)
    {
      cv::Mat samples = extractTrainingSamples(feature_image, sampling_rectangle);
      if (model.isTrained())
        model.update(samples, adaptive_update_ratio);
      else
        model.performTraining(samples);
    }

    char key = static_cast<char>(cv::waitKey(100));
    if (key == ' ')
    {
      cv::Mat samples = extractTrainingSamples(feature_image, sampling_rectangle);
      model.performTraining(samples);
    }
    else if (key == 'o')
    { use_otsu = !use_otsu; }
    else if (key == 'a')
    { use_adaptive_model = !use_adaptive_model; }
    else if (key >= 0)
    { break; }
  }

}

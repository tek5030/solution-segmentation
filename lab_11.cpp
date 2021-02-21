#include "lab_11.h"
#include "multivariate_normal_model.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

// ------ Declarations for utility functions ------

/// \brief Replace a ratio of current_samples with new_samples
/// \param[in] old_samples Samples used in the current model.
/// \param[in] new_samples New samples.
/// \param[in] update_ratio The ratio of samples to replace on average.
void updateSamples(cv::Mat& old_samples, const cv::Mat& new_samples, float update_ratio);

/// \brief Returns a binary image (values 0 or 255) which is the result of segmenting the input.
/// \param[in] input_image Assumed to be of type CV_8U.
/// \param[in/out] threshold_value Assumed to be a number between 0 and 255. Is updated when Otsu's is used.
/// \param[in] use_otsu Option whether to use Otsu's method or not
/// \return The segmented image
cv::Mat performSegmentation(const cv::Mat& input_image, int& threshold_value, bool use_otsu);

/// \brief Extracts pixel features from an image.
/// \param[in] input_image Assumed to be of type CV_8U.
/// \return An image where each pixel is the feature vector for the corresponding pixel in the input.
cv::Mat extractFeatures(const cv::Mat& frame);

/// \brief Creates a cv::Rect with size and position scaled relative to some input img_size
/// \param[in] img_size The size of the image you want to sample from
/// \return The computed cv::Rect
cv::Rect getSamplingRectangle(const cv::Size& img_size);

/// \brief Extract training samples from a rectangle within an image.
/// \param[in] source_image
/// \param[in] sampling_rectangle
/// \return A cv::Mat with sample vectors as columns.
/// For a m x n sampling rectangle of a k-channel image, this returns a cv::Mat with k rows, m*n columns and 1 channel.
cv::Mat extractTrainingSamples(const cv::Mat& source_image, const cv::Rect& sampling_rectangle);

/// \brief Draw a rectangle on an image
/// \param[in, out] image The image will be modified, a rectangle will be drawn on top of it.
/// \param[in] sampling_rectangle The rectangle which will be drawn.
void drawSamplingRectangle(cv::Mat& image, const cv::Rect& sampling_rectangle);


void lab11()
{
  // Change to video file if you want to use that instead.
  constexpr int device_id = 0;
  cv::VideoCapture cap{device_id};

  if (!cap.isOpened())
  {
    throw std::runtime_error("Could not open camera " + std::to_string(device_id));
  }

  // Construct sampling region based on image dimensions.
  cv::Mat frame;
  cap >> frame;
  cv::Rect sampling_rectangle = getSamplingRectangle(frame.size());

  // Set up parameters.
  const int max_threshold{255};
  int current_threshold{30};
  bool use_otsu{false};
  bool use_adaptive_model{false};
  float adaptive_update_ratio{0.1f};

  // Create windows.
  const std::string win_name_input{"Lab 11: Segmentation - Input"};
  cv::namedWindow(win_name_input, cv::WINDOW_NORMAL);
  const std::string win_name_result{"Lab 11: Segmentation - Mahalanobis distance"};
  cv::namedWindow(win_name_result, cv::WINDOW_NORMAL);

  // Add a trackbar for setting the threshold.
  const std::string threshold_trackbar_name{"Threshold"};
  cv::createTrackbar(threshold_trackbar_name, win_name_input, &current_threshold, max_threshold);

  // The main loop in the program.
  std::shared_ptr<MultivariateNormalModel> model;
  cv::Mat current_samples;
  while (true)
  {
    // Read an image frame.
    cap >> frame;
    if (frame.empty())
    {
      throw std::runtime_error("The camera returned an empty frame. Is the camera ok?");
    }

    // Extract pixel features from the image.
    cv::Mat feature_image = extractFeatures(frame);

    if (model)
    {
      // Compute how well the pixel features fit with the model.
      cv::Mat mahalanobis_img = model->computeMahalanobisDistances(feature_image);
      cv::imshow(win_name_result, mahalanobis_img);

      // Segment out the areas of the image that fits well enough.
      cv::Mat segmented = performSegmentation(mahalanobis_img, current_threshold, use_otsu);
      cv::setTrackbarPos(threshold_trackbar_name, win_name_input, current_threshold);

      // Set segmented area to green.
      frame.setTo(cv::Scalar{0, 255, 0}, segmented);
    }

    // Update if using adaptive model.
    if (model && use_adaptive_model)
    {
      cv::Mat new_samples = extractTrainingSamples(feature_image, sampling_rectangle);
      updateSamples(current_samples, new_samples, adaptive_update_ratio);
      model = std::make_shared<MultivariateNormalModel>(current_samples);
    }

    // Draw current frame.
    drawSamplingRectangle(frame, sampling_rectangle);
    cv::imshow(win_name_input, frame);

    // Get input from keyboard
    char key = static_cast<char>(cv::waitKey(10));
    if (key == ' ')
    {
      // Press space to train a model based on the samples in the rectangle.
      current_samples = extractTrainingSamples(feature_image, sampling_rectangle);
      model = std::make_shared<MultivariateNormalModel>(current_samples);
    }
    else if (key == 'o')
    {
      // Press 'o' to toggle use of Otsu's method.
      use_otsu = !use_otsu;
    }
    else if (key == 'a')
    {
      // Press 'a' to toggle use of adaptive model.
      use_adaptive_model = !use_adaptive_model;
    }
    else if (key >= 0)
    {
      // Press any other key to exit the lab.
      break;
    }
  }
}


void updateSamples(cv::Mat& old_samples, const cv::Mat& new_samples, float update_ratio)
{
  // Draw uniformly distributed random numbers
  cv::Mat rand_num = cv::Mat::zeros(1,new_samples.cols,CV_32FC1);
  cv::randu(rand_num,0.,1.);

  // Update samples
  for (int i = 0; i<rand_num.cols; i++)
  {
    if (rand_num.at<float>(0,i) < update_ratio)
    {
      new_samples.col(i).copyTo(old_samples.col(i));
    }
  }
}


cv::Mat performSegmentation(const cv::Mat& input_image, int& threshold_value, bool use_otsu)
{
  int thresh_type = cv::THRESH_BINARY_INV;
  if (use_otsu)
  { thresh_type |= cv::THRESH_OTSU; }

  cv::Mat segmented_image;
  threshold_value = static_cast<int>(cv::threshold(input_image, segmented_image, threshold_value, 255, thresh_type));

  cv::morphologyEx(segmented_image, segmented_image, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {5,5}));
  cv::morphologyEx(segmented_image, segmented_image, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {5,5}));

  return segmented_image;
}


cv::Mat extractFeatures(const cv::Mat& frame)
{
  cv::Mat feature_image = frame.clone();
  return feature_image;
}


cv::Rect getSamplingRectangle(const cv::Size& img_size)
{
  int center_x = img_size.width/2;
  int center_y = 4*img_size.height/5;
  int width = 100;
  int height = 80;
  int top_left_x = center_x - width/2;
  int top_left_y = center_y - height/2;

  const cv::Rect sampling_rectangle(top_left_x, top_left_y, width, height);
  const cv::Rect entire_image(0,0,img_size.width,img_size.height);

  // This operation ensures that the boundaries of the returned sampling rectangle are within the image dimensions,
  // just in case the chosen width or height is to large.
  return (sampling_rectangle & entire_image );
}


cv::Mat extractTrainingSamples(const cv::Mat& source_image, const cv::Rect& sampling_rectangle)
{
  cv::Mat patch = source_image(sampling_rectangle).clone();
  cv::Mat samples = patch.reshape(1, patch.total()).t();
  return samples;
}


void drawSamplingRectangle(cv::Mat& image, const cv::Rect& sampling_rectangle)
{
  const cv::Scalar color{0, 0, 255};
  constexpr int thickness = 2;
  cv::rectangle(image, sampling_rectangle, color, thickness);
}

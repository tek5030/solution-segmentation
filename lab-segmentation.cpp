#include "lab-segmentation.h"

#include "common-lab-utils.h"
#include "multivariate-normal-model.h"
#include "opencv2/imgproc.hpp"
#include <iostream>

// ------ Declarations for utility functions ------

/// \brief Replace a ratio of old_samples with new_samples
/// \param[in] old_samples Samples used in the current model.
/// \param[in] new_samples New samples.
/// \param[in] update_ratio The ratio of samples to replace on average.
void updateSamples(cv::Mat& old_samples, const cv::Mat& new_samples, float update_ratio);

/// \brief Returns a binary image (values 0 or 255) which is the result of segmenting the input.
/// \param[in] input_image Assumed to be of type CV_8U.
/// \param[in/out] threshold_value Assumed to be a number between 0 and 255. Is updated when Otsu's is used.
/// \param[in] use_otsu Option whether to use Otsu's method or not
/// \param[in] max_dist_value The maximum distance value to represent after rescaling.
/// \return The segmented image
cv::Mat performSegmentation(const cv::Mat& input_image, int& threshold_value, bool use_otsu, double max_dist_value);

/// \brief Extracts pixel features from an image.
/// \param[in] input_image Assumed to be of type CV_8U.
/// \return An image where each pixel is the feature vector for the corresponding pixel in the input.
cv::Mat extractFeatures(const cv::Mat& frame);


void runSegmentationLab()
{
  // Set up parameters.
  bool use_otsu{false};
  bool use_adaptive_model{false};
  float adaptive_update_ratio{0.1f};
  const double max_distance = 20.;
  int initial_thresh_val{8};
  using ModelType = MultivariateNormalModel;

  // Set up a simple gui for the lab (based on OpenCV highgui) and run the main loop.
  SegmentationLabGUI gui(initial_thresh_val, max_distance);

  // Connect to the camera.
  // Change to video file if you want to use that instead.
  constexpr int video_source = 0;
  cv::VideoCapture cap{video_source};

  if (!cap.isOpened())
  {
    throw std::runtime_error("Could not open video source " + std::to_string(video_source));
  }

  // Read the first frame
  cv::Mat frame;
  cap >> frame;

  if (frame.empty())
  {
    throw std::runtime_error("Could not capture video frame from source " + std::to_string(video_source));
  }

  // Construct sampling region based on image dimensions.
  const cv::Rect sampling_rectangle = getSamplingRectangle(frame.size());

  // Train first model based on samples from the first image.
  cv::Mat feature_image = extractFeatures(frame);
  cv::Mat current_samples = extractTrainingSamples(feature_image, sampling_rectangle);

  ModelType model(current_samples);

  // The main loop in the program.
  while (true)
  {
    // Read an image frame.
    cap >> frame;
    if (frame.empty())
    {
      throw std::runtime_error("The camera returned an empty frame. Is the camera ok?");
    }

    // Extract pixel features from the image.
    feature_image = extractFeatures(frame);

    // Update if using adaptive model.
    if (use_adaptive_model)
    {
      cv::Mat new_samples = extractTrainingSamples(feature_image, sampling_rectangle);
      updateSamples(current_samples, new_samples, adaptive_update_ratio);
      model = ModelType(current_samples);
    }

    // Compute how well the pixel features fit with the model.
    cv::Mat mahalanobis_img = model.computeMahalanobisDistances(feature_image);

    // Segment out the areas of the image that fits well enough.
    int current_threshold = gui.getThreshold();
    cv::Mat segmented = performSegmentation(mahalanobis_img, current_threshold, use_otsu, max_distance);
    gui.setThreshold(current_threshold);

    // Set segmented area to green.
    //frame.setTo(frame * cv::Scalar{0, 1, 0}, segmented > 0);
    cv::bitwise_and(frame, cv::Scalar{0, 255, 0}, frame, segmented > 0);

    // Draw current frame.
    drawSamplingRectangle(frame, sampling_rectangle);

    gui.showFrame(frame);
    gui.showMahalanobis(mahalanobis_img / max_distance);

    // Update the GUI and wait a short time for input from the keyboard/
    const auto key = SegmentationLabGUI::waitKey(1);

    // React to commands from the keyboard
    if (key == 'q')
    {
      // Press 'q' to quit
      std::cout << "quitting" << std::endl;
      break;
    }
    else if (key == ' ')
    {
      // Press space to train a model based on the samples in the rectangle.
      std::cout << "Extracting samples manually" << std::endl;
      current_samples = extractTrainingSamples(feature_image, sampling_rectangle);
      model = ModelType(current_samples);
    }
    else if (key == 'o')
    {
      // Press 'o' to toggle use of Otsu's method
      use_otsu = !use_otsu;
      std::cout << "Use Otsu's: " << std::boolalpha << use_otsu << std::endl;
    }
    else if (key == 'a')
    {
      // Press 'a' to toggle use of adaptive model.
      use_adaptive_model = !use_adaptive_model;
      std::cout << "Use adaptive model: " << std::boolalpha << use_adaptive_model << std::endl;
    }
  }
  cap.release();
}


cv::Mat extractFeatures(const cv::Mat& frame)
{
  // Convert to float32
  cv::Mat feature_image;
  frame.convertTo(feature_image, CV_32F, 1. / 255.);

  // Choose a colour format:
  cv::cvtColor(feature_image, feature_image, cv::COLOR_BGR2YCrCb);

  return feature_image;
}


cv::Mat performSegmentation(
    const cv::Mat& input_image,
    int& threshold_value,
    const bool use_otsu,
    const double max_dist_value
)
{
  // We need to represent the distances in uint16 because of OpenCV's implementation of Otsu's method.
  const double scale = std::numeric_limits<uint16_t>::max() / max_dist_value;

  cv::Mat distances_scaled;
  input_image.convertTo(distances_scaled, CV_16UC1, scale);

  int thresh_type = cv::THRESH_BINARY_INV;
  if (use_otsu)
  { thresh_type |= cv::THRESH_OTSU; }

  cv::Mat segmented_image;
  const double scaled_threshold = cv::threshold(distances_scaled, segmented_image, threshold_value * scale, 255,
                                                thresh_type);
  threshold_value = static_cast<int>(std::round(scaled_threshold / scale));

  // Perform cleanup using morphological operations.
  cv::Mat structuring_element = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
  cv::morphologyEx(segmented_image, segmented_image, cv::MORPH_OPEN, structuring_element);
  cv::morphologyEx(segmented_image, segmented_image, cv::MORPH_CLOSE, structuring_element);

  return segmented_image;
}


void updateSamples(cv::Mat& old_samples, const cv::Mat& new_samples, const float update_ratio)
{
  // Draw uniformly distributed random numbers
  cv::Mat rand_num = cv::Mat::zeros(new_samples.size(), CV_32FC1);
  cv::randu(rand_num, 0., 1.);

  // Update samples
  new_samples.copyTo(old_samples, rand_num < update_ratio);
}

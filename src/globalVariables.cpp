#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "globalVariables.h"


//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
cv::RNG rng(12345);
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
//std::queue<cv::Point> avgLeftEye;
//std::queue<cv::Point> avgRightEye;

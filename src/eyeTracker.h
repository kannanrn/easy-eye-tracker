#ifndef EYE_TRACKER_H
#define EYE_TRACKER_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::vector<cv::Rect> detectFaces(cv::Mat gray_frame);
void findEyes(cv::Mat frame_gray, cv::Rect face, cv::Point &leftPupil, cv::Point &rightPupil);
cv::Mat findSkin (cv::Mat &frame);

#endif

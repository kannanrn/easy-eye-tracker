#ifndef HELPERS_H
#define HELPERS_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#if WIN32
    #include <windows.h>
#else
    #include <X11/Xlib.h>
#endif

void mouseMove(int x, int y, Display* display, Window &window);
void mouseMove(cv::Point2f position, Display* display, Window &window);
void detectAvgPupils(std::deque<cv::Point> &leftQueue,std::deque<cv::Point> &rightQueue, cv::Point &avgRightPupil, cv::Point &avgLeftPupil);
void getScreenResolution(int &width, int &height);
bool rectInImage(cv::Rect rect, cv::Mat image);
bool inMat(cv::Point p,int rows,int cols);
cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);


#endif

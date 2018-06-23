#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>
#include <cstdio>

#include "constants.h"

#if WIN32
  #include <windows.h>
#else
  #include <X11/Xlib.h>
#endif

void mouseMove(int x, int y, Display* display, Window &window){


  XSelectInput(display, window, KeyReleaseMask);

  XWarpPointer(display, None, window, 0, 0, 0, 0, x, y);

  XFlush(display);
}
void mouseMove(cv::Point2f position, Display* display, Window &window){


  XSelectInput(display, window, KeyReleaseMask);

  XWarpPointer(display, None, window, 0, 0, 0, 0, position.x, position.y);

  XFlush(display);
}

void detectAvgPupils(std::deque<cv::Point> &leftQueue,std::deque<cv::Point> &rightQueue, cv::Point &avgRightPupil, cv::Point &avgLeftPupil) {
  for(int i = 0; i < kQueueSize; i++) {
    avgRightPupil.x += rightQueue[i].x;
    avgRightPupil.y += rightQueue[i].y;
    avgLeftPupil.x += leftQueue[i].x;
    avgLeftPupil.y += leftQueue[i].y;
  }
  avgRightPupil.x /= kQueueSize;
  avgRightPupil.y /= kQueueSize;
  avgLeftPupil.x /= kQueueSize;
  avgLeftPupil.y /= kQueueSize;
}

void getScreenResolution(int &width, int &height) {
#if WIN32
  width = (int) GetSystemMetrics(SM_CXSCREEN);
  height = (int) GetSystemMetrics(SM_CYSCREEN);
#else
  Display* disp = XOpenDisplay(NULL);
  Screen* screen = DefaultScreenOfDisplay(disp);
  width = screen->width;
  height = screen->height;
  XCloseDisplay(disp);
#endif
}


bool rectInImage(cv::Rect rect, cv::Mat image) {
  return rect.x > 0 && rect.y > 0 && rect.x+rect.width < image.cols &&
  rect.y+rect.height < image.rows;
}


bool inMat(cv::Point p,int rows,int cols) {
  return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}


cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY) {
  cv::Mat mags(matX.rows,matX.cols,CV_64F);
  for (int y = 0; y < matX.rows; ++y) {
    const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
    double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < matX.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = sqrt((gX * gX) + (gY * gY));
      Mr[x] = magnitude;
    }
  }
  return mags;
}


double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor) {
  cv::Scalar stdMagnGrad, meanMagnGrad;
  cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
  double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
  return stdDevFactor * stdDev + meanMagnGrad[0];
}

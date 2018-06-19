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

int scale_x(int x, int ref_minx, int ref_width, int cam_width){
  return ((x-ref_minx)*ref_width)/cam_width;
}

int scale_y(int y, int ref_miny, int ref_height, int cam_height){
  return ((y-ref_miny)*ref_height)/cam_height;
}

void mouseMove(int x, int y){
  Display *dpy;
  Window root_window;

  dpy = XOpenDisplay(0);
  root_window = XRootWindow(dpy, 0);
  XSelectInput(dpy, root_window, KeyReleaseMask);

  XWarpPointer(dpy, None, root_window, 0, 0, 0, 0, x, y);

  XFlush(dpy);
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

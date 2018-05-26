#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "helpers.h"
#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "eyeTracker.h"
#include "globalVariables.h"


/**
 * @function main
 */
int main( int argc, const char** argv ) {
  cv::Mat frame;
  bool calibrate = true;

  int width;
  int height;
  getScreenResolution(width, height);
  printf("WIDTH:%d    HEIGHT:%d", width, height);
//yaourt -S libx11-dev
//da sistemare anche cmake
  std::deque<cv::Point> leftQueue;
  std::deque<cv::Point> rightQueue;
  cv::Point fillerPoint(0,0);
  for(int i = 0; i < kQueueSize; i++) {
    leftQueue.push_front(fillerPoint);
    rightQueue.push_front(fillerPoint);
  }

  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };

  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 400, 100);
  cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(face_window_name, 10, 100);
  cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
  cv::moveWindow("Right Eye", 10, 600);
  cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
  cv::moveWindow("Left Eye", 10, 800);


  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
          43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

  cv::VideoCapture capture(0);
  if( capture.isOpened() ) {

    cv::Point leftPupil;
    cv::Point rightPupil;
    cv::Point avgLeftPupil;
    cv::Point avgRightPupil;

    while(calibrate) {
      capture.read(frame);
      // mirror it
      cv::flip(frame, frame, 1);
      frame.copyTo(debugImage);
      // Apply the classifier to the frame
      if( !frame.empty() ) {
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);

        std::vector<cv::Rect> faces = detectFaces(gray_frame);
        //-- Show what you got
        if (faces.size() > 0) {
          findEyes(gray_frame, faces[0], leftPupil, rightPupil);
        }
        leftQueue.pop_back();
        leftQueue.push_front(leftPupil);
        rightQueue.pop_back();
        rightQueue.push_front(rightPupil);
        detectAvgPupils(leftQueue, rightQueue, avgRightPupil, avgLeftPupil);
        printf("R: %d   L:%d\n", avgRightPupil.x, avgRightPupil.y);
        //printf("R: %d   L:%d\n", leftPupil.x, leftPupil.y);
        //printf("R(%d, %d)   L(%d,%d) \n", rightQueue[4].x, rightQueue[4].y, leftQueue[4].x, leftQueue[4].y);
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      circle(debugImage, avgRightPupil, 3, 1234);
      circle(debugImage, avgLeftPupil, 3, 1234);
      imshow(main_window_name,debugImage);
      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }
    }

    while( !calibrate ) {
      capture.read(frame);
      // mirror it
      cv::flip(frame, frame, 1);
      frame.copyTo(debugImage);
      // Apply the classifier to the frame
      if( !frame.empty() ) {
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);

        std::vector<cv::Rect> faces = detectFaces(gray_frame);
        //-- Show what you got
        if (faces.size() > 0) {
          findEyes(gray_frame, faces[0], leftPupil, rightPupil);
        }
        leftQueue.pop_back();
        leftQueue.push_front(leftPupil);
        rightQueue.pop_back();
        rightQueue.push_front(rightPupil);
        detectAvgPupils(leftQueue, rightQueue, avgRightPupil, avgLeftPupil);
        printf("R: %d   L:%d\n", avgRightPupil.x, avgRightPupil.y);
        //printf("R: %d   L:%d\n", leftPupil.x, leftPupil.y);
        //printf("R(%d, %d)   L(%d,%d) \n", rightQueue[4].x, rightQueue[4].y, leftQueue[4].x, leftQueue[4].y);
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      circle(debugImage, avgRightPupil, 3, 1234);
      circle(debugImage, avgLeftPupil, 3, 1234);
      imshow(main_window_name,debugImage);
      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }

    //end while(true)
    }

  //end if cap.isOpened()
  }

  releaseCornerKernels();

  return 0;
}

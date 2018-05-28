#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "helpers.h"
#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "eyeTracker.h"
#include "globalVariables.h"

#define NUM_SECONDS 5


/**
 * @function main
 */
int main( int argc, const char** argv ) {

  // Matrix used to store the frames captured by the camera
  cv::Mat frame;

  // Variable used to detect if calibrate or not
  bool calibrate = true;

  // Heigh and width of the monitor
  int width;
  int height;
  getScreenResolution(width, height);

  // Those points represent the reference points measured during the calibration
  cv::Point refLeftPupil;
  cv::Point refRightPupil;

  // Dimension of the frames captured by the camera
  int cam_width;
  int cam_height;


  /*
  * The values of the points used to calculate the average point are stored inside those queues.
  * This is used to minimize the errors.
  */
  std::deque<cv::Point> leftQueue;
  std::deque<cv::Point> rightQueue;
  cv::Point fillerPoint(0,0);
  // Filling the queues to useless points
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

    // Current points of the pupils
    cv::Point leftPupil;
    cv::Point rightPupil;
    // Average points of the pupils, calculated using the queus
    cv::Point avgLeftPupil;
    cv::Point avgRightPupil;

    // Counter used to track every round of the calibration loop
    int cont = 0;
    // Inizializing the time
    clock_t this_time = clock();
    clock_t last_time = this_time;
    double time_counter = 0;
    // Setting the height and width of the frame captured by the camera
    capture.read(frame);
    cam_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    cam_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);


    //CALIBRATION LOOP
    while(calibrate) {
      capture.read(frame);
      // mirror it
      cv::flip(frame, frame, 1);
      frame.copyTo(debugImage);
      // Apply the classifier to the frame
      if( !frame.empty() ) {
        // Converting the image to grey scale
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);

        // Storing the detected faces
        std::vector<cv::Rect> faces = detectFaces(gray_frame);
        // Detect eyes in the stored face
        if (faces.size() > 0) {
          findEyes(gray_frame, faces[0], leftPupil, rightPupil);
        }

        // Updating the pupil queues
        leftQueue.pop_back();
        leftQueue.push_front(leftPupil);
        rightQueue.pop_back();
        rightQueue.push_front(rightPupil);

        // Calculating the average point of the pupils
        detectAvgPupils(leftQueue, rightQueue, avgRightPupil, avgLeftPupil);

        //Setting the reference position of the pupils
        refLeftPupil.x += avgLeftPupil.x;
        refLeftPupil.y += avgLeftPupil.y;
        refRightPupil.x += avgRightPupil.x;
        refRightPupil.y += avgRightPupil.y;

        // Calculating the time passed from the start of the calibration
        this_time = clock();
        time_counter += (double)(this_time-last_time);
        last_time = this_time;
        cont ++;
        // Check if the calibration time is over
        if(time_counter > (double)(NUM_SECONDS*CLOCKS_PER_SEC)) {
          calibrate = false;
        }
        // Drawing the point to watch during the calibration
        circle(debugImage, cv::Point(cam_width/2,cam_height/2), 5, cv::Scalar(0,0,255), CV_FILLED);
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      // Drawing the current position of the pupils
      circle(debugImage, avgRightPupil, 3, 1234);
      circle(debugImage, avgLeftPupil, 3, 1234);
      imshow(main_window_name,debugImage);

      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }
    }

    // End of calculating the reference points
    refLeftPupil.x /= cont;
    refLeftPupil.y /= cont;
    refRightPupil.x /= cont;
    refRightPupil.y /= cont;

    //-------END CALIBRATION---------

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

      circle(debugImage, refLeftPupil, 5, cv::Scalar(0,0,255), CV_FILLED);
      circle(debugImage, refRightPupil, 5, cv::Scalar(0,0,255), CV_FILLED);
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

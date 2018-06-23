#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <unistd.h>

#include "helpers.h"
#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "eyeTracker.h"
#include "globalVariables.h"

#define NUM_SECONDS 5

struct DetectFaceParams {
    bool draw = false;
    bool detectFace;
    cv::Point startPoint;
    cv::Point endPoint;
    cv::Rect face;
};


void mouse_callback(int event, int x, int y, int flag, void *param) {
    DetectFaceParams *detectFaceParams = (DetectFaceParams *) param;
    if (!detectFaceParams->detectFace)
        return;

    detectFaceParams->endPoint = cv::Point(x, y);
    if (event == cv::EVENT_LBUTTONDOWN) {
        detectFaceParams->draw = true;
        detectFaceParams->startPoint = cv::Point(x, y);
    }
    if (event == cv::EVENT_LBUTTONUP) {
        detectFaceParams->face = cv::Rect(detectFaceParams->startPoint, detectFaceParams->endPoint);
        detectFaceParams->detectFace = false;
    }
}

int scale_x(int x, int ref_minx, int ref_width, int cam_width){
  return ((x*ref_width)/cam_width)-ref_minx;
}

int scale_y(int y, int ref_miny, int ref_height, int cam_height){
  return ((y*ref_height)/cam_height)-ref_miny;
}

/**
 * @function main
 */
int main(int argc, const char **argv) {

    // Matrix used to store the frames captured by the camera
    cv::Mat frame;

    // Variable used to detect if calibrate or not
    bool calibrate = true;
    DetectFaceParams detectFaceParams;
    detectFaceParams.detectFace = true;

    Display *dpy;
    Window root_window;
    dpy = XOpenDisplay(0);
    root_window = XRootWindow(dpy, 0);

    // Heigh and width of the monitor
    int width;
    int height;
    getScreenResolution(width, height);

    // Those points represent the reference points measured during the calibration
    cv::Point2f refLeftPupil[6];
    cv::Point2f refRightPupil[6];
    std::cout << refRightPupil[0] << std::endl;


    // Dimension of the frames captured by the camera
    int cam_width;
    int cam_height;

    cv::Point2f dstQuad[] = {
            cv::Point2f(width/2, height),
            cv::Point2f(0, height),
            cv::Point2f(0,0),
            cv::Point2f(width/2, 0),
            cv::Point2f(width, 0),
            cv::Point2f(width, height)
    };

    /*
    * The values of the points used to calculate the average point are stored inside those queues.
    * This is used to minimize the errors.
    */
    std::deque<cv::Point> leftQueue;
    std::deque<cv::Point> rightQueue;
    cv::Point fillerPoint(0, 0);
    // Filling the queues of useless points
    for (int i = 0; i < kQueueSize; i++) {
        leftQueue.push_front(fillerPoint);
        rightQueue.push_front(fillerPoint);
    }

    // Load the cascades
    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
        return -1;
    };

    cv::namedWindow(main_window_name, CV_WINDOW_NORMAL);
    cv::namedWindow("Detecting Face", CV_WINDOW_NORMAL);
    cv::setMouseCallback("Detecting Face", mouse_callback, (void *) &detectFaceParams);
    cv::namedWindow(face_window_name, CV_WINDOW_NORMAL);
    cv::moveWindow(face_window_name, 10, 100);
    cv::namedWindow("Right Eye", CV_WINDOW_NORMAL);
    cv::moveWindow("Right Eye", 10, 600);
    cv::namedWindow("Left Eye", CV_WINDOW_NORMAL);
    cv::moveWindow("Left Eye", 10, 800);


    createCornerKernels();
    ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
            43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

    cv::VideoCapture capture(0);
    if (capture.isOpened()) {

        // Current points of the pupils
        cv::Point leftPupil;
        cv::Point rightPupil;
        // Average points of the pupils, calculated using the queus
        cv::Point avgLeftPupil;
        cv::Point avgRightPupil;

        // Setting the height and width of the frame captured by the camera
        capture.read(frame);
        cam_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        cam_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);


        cv::moveWindow(main_window_name, width / 2 - cam_width / 2, height / 2 - cam_height / 2);
        while (detectFaceParams.detectFace) {
            capture.read(frame);
            cv::flip(frame, frame, 1);
            cv::cvtColor(frame, frame, CV_BGR2GRAY);
            if (detectFaceParams.draw) {
                rectangle(frame, cv::Rect(detectFaceParams.startPoint, detectFaceParams.endPoint), 1234);
            }
            cv::imshow("Detecting Face", frame);
            int c = cv::waitKey(10);
            if ((char) c == 'c') { break; }
            if ((char) c == 'f') {
                imwrite("frame.png", frame);
            }
        }

        cv::destroyWindow("Detecting Face");


        cv::Mat referenceBackground(64, 64, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Point2f refPosition(width / 2 - 40, height - 105);
        cv::namedWindow("Reference Point");


        for (int i = 0; i < 6; i++) {

            // Counter used to track every round of the calibration loop
            int cont = 0;
            // Inizializing the time
            clock_t this_time = clock();
            clock_t last_time = this_time;
            double time_counter = 0;
            //bool redPoint = true;
            cv::moveWindow("Reference Point", refPosition.x, refPosition.y);
            cv::Scalar pointColor = cv::Scalar(0, 0, 255);

            //CALIBRATION LOOP
            while (calibrate) {
                capture.read(frame);
                // mirror it
                cv::flip(frame, frame, 1);
                frame.copyTo(debugImage);

                // Apply the classifier to the frame
                if (!frame.empty()) {
                    // Converting the image to grey scale
                    cv::Mat gray_frame;
                    cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);

                    findEyes(gray_frame, detectFaceParams.face, leftPupil, rightPupil);

                    // Updating the pupil queues
                    leftQueue.pop_back();
                    leftQueue.push_front(leftPupil);
                    rightQueue.pop_back();
                    rightQueue.push_front(rightPupil);

                    // Calculating the average point of the pupils
                    detectAvgPupils(leftQueue, rightQueue, avgRightPupil, avgLeftPupil);

                    // Setting the reference position of the pupils
                    refLeftPupil[i].x += avgLeftPupil.x;
                    refLeftPupil[i].y += avgLeftPupil.y;
                    refRightPupil[i].x += avgRightPupil.x;
                    refRightPupil[i].y += avgRightPupil.y;

                    // Calculating the time passed from the start of the calibration
                    this_time = clock();
                    time_counter += (double) (this_time - last_time);
                    last_time = this_time;
                    cont++;
                    if (time_counter > (double) ((NUM_SECONDS - 3) * CLOCKS_PER_SEC)) {
                        pointColor = cv::Scalar(0, 255, 0);
                    }
                    // Check if the calibration time is over
                    if (time_counter > (double) (NUM_SECONDS * CLOCKS_PER_SEC)) {
                        calibrate = false;
                    }

                    // Drawing the point to watch during the calibration
                    circle(referenceBackground, cv::Point2f(32, 32), 32, pointColor, CV_FILLED);
                    circle(referenceBackground, cv::Point2f(32,32), 28, cv::Scalar(255,255,255), CV_FILLED);
                    //line(referenceBackground, cv::Point2f(40, 0), cv::Point2f(40, 80), cv::Scalar(0, 0, 0), 2);
                    //line(referenceBackground, cv::Point2f(0, 40), cv::Point2f(80, 40), cv::Scalar(0, 0, 0), 2);
                    imshow("Reference Point", referenceBackground);
                } else {
                    printf(" --(!) No captured frame -- Break!");
                    break;
                }

                // Drawing the current position of the pupils
                circle(debugImage, avgRightPupil, 3, 1234);
                circle(debugImage, avgLeftPupil, 3, 1234);
                imshow(main_window_name, debugImage);

                int c = cv::waitKey(10);
                if ((char) c == 'c') { break; }
                if ((char) c == 'f') {
                    imwrite("frame.png", frame);
                }
            }

            // End of calculating the reference points
            refLeftPupil[i].x /= cont;
            refLeftPupil[i].y /= cont;
            refRightPupil[i].x /= cont;
            refRightPupil[i].y /= cont;
            switch (i) {
                case 0:
                    refPosition.x = 0;
                    calibrate = true;
                    break;
                case 1:
                    refPosition.y = 0;
                    calibrate = true;
                    break;
                case 2:
                    refPosition.x = width / 2 - 40;
                    calibrate = true;
                    break;
                case 3:
                    refPosition.x = width - 80;
                    calibrate = true;
                    break;
                case 4:
                    refPosition.y = height - 105;
                    calibrate = true;
                    break;
            }
        }
        //-------END CALIBRATION---------//


        cv::Point2f srcQuad[] = {
                cv::Point2f((refLeftPupil[0].x + refRightPupil[0].x)/2, (refLeftPupil[0].y + refRightPupil[0].y)/2),
                cv::Point2f((refLeftPupil[1].x + refRightPupil[1].x)/2, (refLeftPupil[1].y + refRightPupil[1].y)/2),
                cv::Point2f((refLeftPupil[2].x + refRightPupil[2].x)/2, (refLeftPupil[2].y + refRightPupil[2].y)/2),
                cv::Point2f((refLeftPupil[3].x + refRightPupil[3].x)/2, (refLeftPupil[3].y + refRightPupil[3].y)/2),
                cv::Point2f((refLeftPupil[4].x + refRightPupil[4].x)/2, (refLeftPupil[4].y + refRightPupil[4].y)/2),
                cv::Point2f((refLeftPupil[5].x + refRightPupil[5].x)/2, (refLeftPupil[5].y + refRightPupil[5].y)/2)
        };
        cv::Mat warp_mat = cv::getPerspectiveTransform(srcQuad, dstQuad);
        std::vector<cv::Point2f> pupilPositions;
        std::vector<cv::Point2f> cursorPositions;

        cv::destroyWindow("Reference Point");
        while (!calibrate) {
            capture.read(frame);
            // Mirror it
            cv::flip(frame, frame, 1);
            frame.copyTo(debugImage);
            // Apply the classifier to the frame
            if (!frame.empty()) {
                cv::Mat gray_frame;
                cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);
                findEyes(gray_frame, detectFaceParams.face, leftPupil, rightPupil);
                leftQueue.pop_back();
                leftQueue.push_front(leftPupil);
                rightQueue.pop_back();
                rightQueue.push_front(rightPupil);
                detectAvgPupils(leftQueue, rightQueue, avgRightPupil, avgLeftPupil);


                pupilPositions.push_back(cv::Point((avgRightPupil.x + avgLeftPupil.x)/2, (avgRightPupil.y+avgLeftPupil.y)/2));
                cv::perspectiveTransform(pupilPositions, cursorPositions, warp_mat);
                pupilPositions.pop_back();
                /*
                //Point calculation for leftPupil
                cv::Point2f scaledLeftPupil;
                scaledLeftPupil.x = scale_x(avgLeftPupil.x, refLeftPupil[2].x, refLeftPupil[4].x, cam_width);
                scaledLeftPupil.y = scale_y(avgLeftPupil.y, refLeftPupil[2].y, refLeftPupil[1].y, cam_height);
                printf("leftPupil x: %d y: %d\n", scaledLeftPupil.x, scaledLeftPupil.y);

                //Point calculation for rigthPupil
                cv::Point2f scaledRigthPupil;
                scaledRigthPupil.x = scale_x(avgRightPupil.x, refRightPupil[2].x, refRightPupil[4].x, cam_width);
                scaledRigthPupil.y = scale_y(avgRightPupil.y, refRightPupil[2].y, refRightPupil[1].y, cam_height);

                cv::Point2f scaledAvgPupil;
                scaledAvgPupil.x = (scaledLeftPupil.x+scaledRigthPupil.x)/2;
                scaledAvgPupil.y = (scaledLeftPupil.y+scaledRigthPupil.y)/2;

                //Set the mouse with the average of the two pupils
                printf("avgPupil x: %d y: %d\n", scaledAvgPupil.x, scaledAvgPupil.y);
                 */

                mouseMove(cursorPositions.front(), dpy, root_window);
                cursorPositions.pop_back();
            } else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            circle(debugImage, refLeftPupil[0], 5, cv::Scalar(0, 0, 255), CV_FILLED);
            circle(debugImage, refRightPupil[0], 5, cv::Scalar(0, 0, 255), CV_FILLED);
            circle(debugImage, avgRightPupil, 3, 1234);
            circle(debugImage, avgLeftPupil, 3, 1234);
            imshow(main_window_name, debugImage);
            int c = cv::waitKey(10);
            if ((char) c == 'c') { break; }
            if ((char) c == 'f') {
                imwrite("frame.png", frame);
            }

            //end while(true)
        }

        //end if cap.isOpened()
    }

    XCloseDisplay(dpy);
    releaseCornerKernels();

    return 0;
}

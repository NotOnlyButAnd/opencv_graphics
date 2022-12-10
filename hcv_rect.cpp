#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;
int main( int argc, char** argv ) {
    // Task1
    Mat image, img_hsv;
    image = imread("/home/pixel/VikaLabs/2.jpg", IMREAD_LOAD_GDAL);   // Read the file
    if(!image.data){                 
        return -1;
    }
    cvtColor(image,img_hsv,41); 
    imshow("hsv image", img_hsv);   
    imshow("image", image);   
    waitKey(1000000000);
    destroyAllWindows();

    // Task4
    VideoCapture cap(0);
    double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); 
    double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    Size frame_size(dWidth, dHeight);
    Mat frame;
    while (true) {
        cap.read(frame);
        imshow("Let`s take photo from camera", frame);
        if (waitKey(10) == 27){
            break;
        }
    }
    Mat frame_hsv;
    cvtColor(frame,frame_hsv,41);
    Vec3b color_frame_hsv = frame_hsv.at<Vec3b>(dHeight/2,dWidth/2);   // type CV_8UC3
     // b g r in rectangle
    int color_r = 0;
    int color_g = 0;
    int color_b = 0;
   if ((int)color_frame_hsv[0] > 0 && (int)color_frame_hsv[0] < 60) {
       color_r = 255;
       color_g = 0;
       color_b = 0;
   } else if ((int)color_frame_hsv[0] > 60 && (int)color_frame_hsv[0] < 120) {
       color_r = 0;
       color_g = 255;
       color_b = 0;
   } else {
       color_r = 0;
       color_g = 0;
       color_b = 255;
   }
    int height_rec = 300;
    int width_rec = 100;
    double x = dWidth/2 -width_rec/2;
    double y = dHeight/2 - height_rec/2;
    Point st(x, y);
    Point end(dWidth/2 + width_rec / 2, dHeight/2 + height_rec/2);
    Mat frame1 = frame;
    rectangle(frame, st, end,
              Scalar(0,0,255),
              1, LINE_8);
    int height_rec2 = width_rec;
    int width_rec2 = height_rec;
    Point st1(dWidth/2 - width_rec2 / 2, dHeight/2 - height_rec2/2);
    Point end1(dWidth/2 + width_rec2 / 2, dHeight/2 + height_rec2/2);
    rectangle(frame, st1, end1,
              Scalar(0, 0, 255),
              1, LINE_8);
    imshow("red cross", frame);  
    rectangle(frame, st, end,
              Scalar(color_b, color_g, color_r),
              -1, LINE_8);
    rectangle(frame, st1, end1,
              Scalar(color_b, color_g, color_r), -1, LINE_8);
    imshow("filled cross", frame);
    waitKey(1000000000);
    return 0;}

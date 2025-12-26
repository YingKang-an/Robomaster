#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main ()
{
    Mat src = imread("苹果.jpg", IMREAD_COLOR);
    
    if(src.empty())
    {
        cerr << "无法加载图片" << endl;
        return -1;
    }

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    Scalar lower_red1 = Scalar(0, 50, 50);
    Scalar upper_red1 = Scalar(10, 255, 255);
    Scalar lower_red2 = Scalar(160, 50, 50);
    Scalar upper_red2 = Scalar(180, 255, 255);

    Mat mask1, mask2, redMask;

    inRange(hsv, lower_red1, upper_red1, mask1);
    inRange(hsv, lower_red2, upper_red2, mask2);
    
    redMask = mask1 | mask2;           
    
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(12,12));
    
    morphologyEx(redMask, redMask, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(redMask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    Mat result = src.clone();

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if(area > 100)
        {
            drawContours(result, contours, i, Scalar(0, 255, 0), 5);
        }
    }

    imshow("apple", src);
    imshow("redMask", redMask);
    imshow("result", result);

    waitKey(0);
    
    destroyAllWindows();
    return 0;
}


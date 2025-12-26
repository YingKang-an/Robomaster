#include "opencv2/opencv.hpp"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
// 最小二乘法计算拟合圆
Vec3f fitCircle(const std::vector<cv::Point2f> &points)
{
    if (points.size() >= 3)
    {
        /*
    A 矩阵：存储点的坐标，用于构建线性方程。每一行表示一个点的信息。
    B 矩阵：存储与圆方程相关的值
        */
        cv::Mat A(points.size(), 3, CV_32FC1);
        cv::Mat B(points.size(), 1, CV_32FC1);

        for (size_t i = 0; i < points.size(); ++i)
        {
            float x = points[i].x;
            float y = points[i].y;
            /*
            对每个点 (x,y)，我们填充 A 和 B 矩阵：
            A第一列是 2x，第二列是 2y，第三列是常数 1。
            */
            A.at<float>(i, 0) = 2 * x;
            A.at<float>(i, 1) = 2 * y;
            A.at<float>(i, 2) = 1;

            B.at<float>(i, 0) = -(x * x + y * y);
        }
        cv::Mat solution;
        // cv::solve 函数用于求解方程 Ax=B。
        // 使用 SVD（奇异值分解）方法来求解，以提高数值稳定性。
        cv::solve(A, B, solution, cv::DECOMP_SVD);
        // 使用了负的平方和 −(x2+y2)−(x2+y2) 来表示圆方程。
        // 这导致在求解过程中得出的圆心坐标是相反的，所以需要取负值以得到正确的圆心位置。
        float center_x = -solution.at<float>(0, 0);
        float center_y = -solution.at<float>(1, 0);
        // 半径通过公式计算：其中 c 是 solution 的第三个元素。
        float radius = sqrt(center_x * center_x + center_y * center_y - solution.at<float>(2, 0));
        return Vec3f(center_x, center_y, radius);
    }
    else
        return Vec3f(7, 7, 7);
}
// 主函数实现
int main(int argc, char **argv)
{

    cv ::VideoCapture capture("../buff.mp4");

    if (!capture.isOpened())
    {
        std ::cout << "无法读取视频" << std ::endl;
        return 0;
    }
    cv ::Mat frame;
    // 存储二维点（Point2f 对象）的动态数组
    vector<cv::Point2f> points;
    // 定义一个旋转矩形
    RotatedRect rect;
    while (true)
    {
        // 从视频捕获对象 capture 中读取下一帧并将其存储在 frame 中。
        capture.read(frame);

        // 如果帧为空，表示视频播放完毕
        if (frame.empty())
        {
            // 重新打开视频文件
            capture.open("../buff.mp4");
            if (!capture.isOpened())
            {
                cout << "can't open video" << endl;
                break; // 退出循环
            }
            cout << "open video again" << endl;
            continue; // 跳过当前循环，开始新的帧读取
        }

        ///////////////////////
        // Image preprocessing//
        ///////////////////////
        
        // 定义了一个 vector 容器 imgChannels，用于存储拆分后的图像通道
        vector<Mat> imgChannels;
        // 将图像的颜色通道拆分开来
        /*
        imgChannels 的内容将依次为：
        imgChannels[0]: 蓝色通道
        imgChannels[1]: 绿色通道
        imgChannels[2]: 红色通道
        */
        split(frame, imgChannels);
        // 通道相减，突出图像中红色对蓝色的差异
        Mat midImage = imgChannels.at(2) - imgChannels.at(0);
        // 结构元素
        int structElementSize = 5;
        GaussianBlur(midImage, midImage, Size(2 * structElementSize + 1, 2 * structElementSize + 1), 0, 0, BORDER_DEFAULT);
        threshold(midImage, midImage, 100, 255, THRESH_BINARY);
        // 膨胀
        structElementSize = 3;
        Mat element = getStructuringElement(MORPH_RECT,
                                            Size(2 * structElementSize + 1, 2 * structElementSize + 1),
                                            Point(structElementSize, structElementSize));
        dilate(midImage, midImage, element);
        // 开操作
        structElementSize = 2;
        element = getStructuringElement(MORPH_RECT,
                                        Size(2 * structElementSize + 1, 2 * structElementSize + 1),
                                        Point(structElementSize, structElementSize));
        morphologyEx(midImage, midImage, MORPH_OPEN, element);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(midImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

        ///////////////////////
        // Features matching //
        ///////////////////////

        /*
        * hierarchy[i][0]：当前轮廓的【下一个同级轮廓】的索引（同一层级、平级的下一个轮廓，无则为-1）
        * hierarchy[i][1]：当前轮廓的【上一个同级轮廓】的索引（同一层级、平级的上一个轮廓，无则为-1）
        * hierarchy[i][2]：当前轮廓的【子轮廓】的索引（当前轮廓包含的内层小轮廓，无则为-1）
        * hierarchy[i][3]：当前轮廓的【父轮廓】的索引（包含当前轮廓的外层大轮廓，无则为-1）
        */
       
        if (!contours.empty())
        {
            // 遍历当前轮廓的所有同级轮廓，直到没有更多的同级轮廓（即 hierarchy[i][0] 为 -1）
            for (int i = 0; i >= 0; i = hierarchy[i][0])
            {
                // 检查当前轮廓是否有子轮廓（即 hierarchy[i][2] 不为 -1）
                if (hierarchy[i][2] != -1) // && hierarchy[hierarchy[i][2]][0] == -1&&hierarchy[hierarchy[i][2]][1]==-1)
                {
                    // 检查父轮廓的子轮廓上一个同级轮廓是否存在，以及该子轮廓的面积是否小于 500。这个检查通常用于过滤掉不感兴趣的小轮廓。
                    if (hierarchy[hierarchy[i][2]][0] == -1 || contourArea(contours[hierarchy[hierarchy[i][2]][0]]) < 500)
                    {
                        // 检查父轮廓的子轮廓的下一个同级轮廓是否存在以及面积。
                        if (hierarchy[hierarchy[i][2]][1] == -1 || contourArea(contours[hierarchy[hierarchy[i][2]][1]]) < 500)
                        {
                            // 计算了一个轮廓的最小面积包围矩形（即由轮廓定义的形状）
                            rect = minAreaRect(contours[hierarchy[i][2]]);
                            Point2f p[4];
                            // points 方法将矩形的四个角点填充到数组 p 中
                            rect.points(p);
                            // 计算矩形的中心点：
                            // 通过平均四个角点的坐标来计算矩形的中心点。然后，将这个中心点添加到 points 向量中
                            Point2f center(0, 0);
                            for (int i = 0; i < 4; i++)
                            {
                                center += p[i];
                            }
                            center /= 4;
                            points.push_back(center);
                            // 在 frame（图像或视频帧）上绘制拟合的圆。该圆用绿色绘制，厚度为 3 像素
                            Vec3f circl = fitCircle(points);
                            cv::circle(frame, Point(circl[0], circl[1]), circl[2], Scalar(0, 255, 0), 3);
                            // 在矩形的中心绘制一个小的品红色圆圈（半径为 5）。
                            cv::circle(frame, center, 5, Scalar(255, 0, 255), -1);
                            // 代码在矩形的中心添加了 "target" 标签，并略微偏移放置第二个文本，以避免重叠。
                            cv::putText(frame, "target", center, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, LINE_AA);
                            cv::putText(frame, "target", center + Point2f(2, 1), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, LINE_AA);
                            // 连接矩形的四个角，绘制出矩形的边界，使其在帧上可视化
                            for (int j = 0; j < 4; ++j)
                            {
                                line(frame, p[j], p[(j + 1) % 4], Scalar(255, 255, 255), 2);
                            }
                            break;
                        }
                    }
                }
            }
        }
        namedWindow("mid", cv::WINDOW_NORMAL);
        resizeWindow("mid", 640, 480);
        imshow("mid", midImage);
        namedWindow("frame", cv::WINDOW_NORMAL);
        resizeWindow("frame", 640, 480);
        imshow("frame", frame);
        waitKey(7);
    }

    return 0;
}

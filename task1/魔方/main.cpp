#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

struct ColorParams
{
    int h_min, h_max;       
    int s_min, s_max;     
    int v_min, v_max; 
    Scalar draw_color;
    string name;
    int total_count;
    Mat mask;
};

vector<ColorParams> colors =
{
    {11, 24, 210, 255, 100, 255, Scalar(20, 200, 255), "orange ", 0, Mat()},
    {38, 70, 17, 255, 28, 255, Scalar(10, 255, 10), "green ", 0, Mat()},
    {90, 120, 50, 255, 50, 255, Scalar(255, 10, 10), "blue ", 0, Mat()}
};

Mat frame, hsv, result;
bool is_paused = false;
vector<int> last_contours(3, 0);  //存储上一帧颜色的数量
bool is_twist = false;
double video_fps;
int frame_delay;

//更新掩码
void update_mask (int color_idx)                           
{
    if (hsv.empty())
    {
        return;
    }
    ColorParams& color = colors[color_idx];

    inRange(hsv, Scalar(color.h_min, color.s_min, color.v_min), Scalar(color.h_max, color.s_max, color.v_max), color.mask);

    //腐蚀+膨胀
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
    morphologyEx(color.mask, color.mask, MORPH_OPEN, kernel);
}

//判断正方形
bool is_square (vector<Point>& contour)
{
    //面积范围
    double area = contourArea(contour);
    if (area < 300 || area > 4000)
    {
        return false;
    }

    //多边形顶点
    double perimeter = arcLength (contour, true);
    vector<Point> approx;
    approxPolyDP(contour, approx, 0.03 * perimeter, true);
    if (approx.size() != 4)
    {
        return false;
    }

    //角度判断
    Point p0 = approx[0], p1 = approx[1], p2 = approx[2], p3 = approx[3];
    Point v1 = p1 - p0;
    Point v2 = p2 - p1;
    Point v3 = p3 - p2;
    Point v4 = p0 - p3;
    
    double dot1 = v1.x * v2.x + v1.y * v2.y;
    double len_v1 = sqrt(v1.x*v1.x + v1.y*v1.y);
    double len_v2 = sqrt(v2.x*v2.x + v2.y*v2.y);
    double cos1;
    if (len_v1 < 1e-2 || len_v2 < 1e-2)
    {
        cos1 = 1.0;
    }
    else
    {
        cos1 = fabs(dot1 / (len_v1 * len_v2));
    }

    double dot2 = v2.x * v3.x + v2.y * v3.y;
    double len_v3 = sqrt(v3.x*v3.x + v3.y*v3.y);
    double cos2;
    if (len_v2 < 1e-2 || len_v3 < 1e-2)
    {
        cos2 = 1.0;
    }
    else
    {
        cos2 = fabs(dot2 / (len_v2 * len_v3));
    }

    double dot3 = v3.x * v4.x + v3.y * v4.y;
    double len_v4 = sqrt(v4.x*v4.x + v4.y*v4.y);
    double cos3;
    if (len_v3 < 1e-2 || len_v4 < 1e-2)
    {
        cos3 = 1.0;
    }
    else
    {
        cos3 = fabs(dot3 / (len_v3 * len_v4));
    }

    double dot4 = v4.x * v1.x + v4.y * v1.y;
    double cos4;
    if (len_v4 < 1e-2 || len_v1 < 1e-2)
    {
        cos4 = 1.0;
    }
    else
    {
        cos4 = fabs(dot4 / (len_v4 * len_v1));
    }

    const double cos_threshold = 0.2;
    if (cos1 > cos_threshold || cos2 > cos_threshold || cos3 > cos_threshold || cos4 > cos_threshold)
    {
        return false;
    }

    //宽高比判断
    RotatedRect rrect = minAreaRect(contour);
    float ratio = min(rrect.size.width, rrect.size.height) / max(rrect.size.width, rrect.size.height);
    return (ratio > 0.75);
}

//检测并绘制轮廓
int detect_and_draw_contours(ColorParams& color, Mat& draw_frame) 
{
    vector<vector<Point>> contours;
    findContours(color.mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    int valid_count = 0;
    for (auto& cnt : contours) 
    {
        if (is_square(cnt)) 
        {
            valid_count++;
            drawContours(draw_frame, vector<vector<Point>>{cnt}, -1, color.draw_color, 3);
        }
    }
    return valid_count;
}

//扭动检测
void detect_twist(const vector<int>& current_contours) 
{
    int change = 0;
    for (int i = 0; i < 3; ++i) 
    {
        change += abs(current_contours[i] - last_contours[i]);
    }
    is_twist = (change > 0);//方块数变化 => 扭动
    last_contours = current_contours;
}

int main(int argc, char* argv[])
{
    VideoCapture cap("../魔方识别视频.mp4");
    if (!cap.isOpened()) 
    {
        cerr << "无法打开视频！" << endl;
        return -1;
    }

    video_fps = cap.get(CAP_PROP_FPS);
    frame_delay = (int)(1000 / video_fps);

    namedWindow("识别结果", WINDOW_AUTOSIZE);

    while (true) 
    {
        if (!is_paused) 
        {
            cap >> frame;
            if (frame.empty()) 
            {
                break;
            }
            cvtColor(frame, hsv, COLOR_BGR2HSV);
        }

        result = frame.clone();
        vector<int> current_contours(3, 0);

        for (int color_idx = 0; color_idx < 3; color_idx++) 
        {
       
            update_mask(color_idx);
            current_contours[color_idx] = detect_and_draw_contours(colors[color_idx], result);
        }

        // 检测扭动并更新计数
        detect_twist(current_contours);
        if (is_twist && !is_paused) 
        {
            for (int i = 0; i < 3; ++i) 
            {
                colors[i].total_count += current_contours[i];
            }
            is_twist = false;
        }

        // 显示统计信息
        int text_y = 50;
        for (auto& color : colors) 
        {
            putText(result, 
                    color.name + "current number: " + to_string(current_contours[&color - &colors[0]]),
                    Point(30, text_y), 
                    FONT_HERSHEY_SIMPLEX, 0.8, 
                    color.draw_color, 3);
            text_y += 40;
        }
        putText(result, "total numbers", Point(30, text_y), 
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 3);
        text_y += 40;
        for (auto& color : colors) 
        {
            putText(result, 
                    color.name + "total number: " + to_string(color.total_count),
                    Point(30, text_y), 
                    FONT_HERSHEY_SIMPLEX, 0.8, 
                    color.draw_color, 3);
            text_y += 40;
        }

        // 显示结果
        imshow("识别结果", result);

        // 按键控制
        char key;
        if (is_paused) 
        {
            key = waitKey(0);
        } 
        else 
        {
            key = waitKey(frame_delay);
        }
        if (key == 27) 
        {
            break;
        }
        if (key == 32) 
        {
            is_paused = !is_paused;
        }
    }

    // 输出最终统计
    cout << "\n处理结束！" << endl;
    for (auto& color : colors) 
    {
        cout << color.name << "累计出现次数：" << color.total_count << endl;
    }

    cap.release();
    destroyAllWindows();
    
    return 0;
}

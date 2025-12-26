#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

//========================== 全局参数 ==============================

//------------------------ 初始化暂停标志 ---------------------------
bool is_paused = false;                        // 初始化暂停标志

// ------------------------ 预处理阈值参数 --------------------------
int blue_sub_black_thres = 7;                   // 蓝差分黑色区域阈值
double gray_bright_thres = 140;                 // 灰度高亮阈值

//------------------------- 形态学结构元素 --------------------------
static const Mat ELEMENT = getStructuringElement(MORPH_RECT, Size(5, 5)); 

//------------------------ 灯条与装甲板筛选参数 ----------------------
const float LIGHT_MIN_AREA = 10.0f;             // 灯条最小面积
const float LIGHT_MAX_AREA = 3000.0f;           // 灯条最大面积
const float LIGHT_MIN_MAX_LEN = 7.0f;           // 灯条最小长边长度
const float LIGHT_MIN_RATIO = 1.0f;             // 灯条最小长宽比
const float LIGHT_MAX_RATIO = 25.0f;            // 灯条最大长宽比
const float LIGHT_TILT_JUDGE_THRESH = 45.0f;    // 灯条倾斜角度判断阈值
const float ARMOR_MAX_RATE = 1.5f;              // 装甲板灯条长度比例上限
const float ARMOR_MAX_ANGLE_SUB = 15.0f;        // 装甲板灯条角度差上限
const float ARMOR_MAX_Y_X_RATIO = 15.0f;        // 装甲板灯条y差/x差上限
const float ARMOR_MIN_DISTANCE_RATIO = 0.0f;    // 装甲板最小距离比例
const float ARMOR_MAX_DISTANCE_RATIO = 2.5f;    // 装甲板最大距离比例
const float ARMOR_MAX_HORIZONTAL_ANGLE = 35.0f; // 装甲板中心线与水平轴的最大夹角

//------------------------- HSV颜色识别参数 -------------------------
const Scalar RED2_HSV_LOW = Scalar(0, 0, 60);
const Scalar RED2_HSV_HIGH = Scalar(10, 255, 255);
const Scalar RED1_HSV_LOW = Scalar(140, 0, 60);
const Scalar RED1_HSV_HIGH = Scalar(180, 255, 255);
const Scalar BLUE_HSV_LOW = Scalar(100, 20, 80);
const Scalar BLUE_HSV_HIGH = Scalar(130, 255, 255);
const float COLOR_RATIO_THRESH = 0.1f;           // 颜色占比阈值 大于则判定为对应颜色
const float BIG_RECT_AREA_RATIO = 5.0f;          // 大矩形面积为灯条的n倍

//-------------------------- 中间图像存储 -----------------------------
Mat gray_img;                                    // 1.彩色转灰度
Mat gray_thres_img;                              // 2.灰度二值化
Mat color_sub_br_img;                            // 3.B-R差分图
Mat color_sub_br_show;                           // 4.B-R差分归一化显示
Mat blue_black_area;                             // 5.蓝差分黑色区域
Mat gray_bright_area;                            // 6.灰度高亮区域
Mat candidate;                                   // 7.黑+亮交集区域
Mat candidate_hori_erode;                        // 8.水平腐蚀后区域
Mat dilate_img;                                  // 9.垂直膨胀结果

//--------------------------- 灯条结构体 ------------------------------
struct LightBar
{
    RotatedRect rect;                             // 灯条的旋转矩形
    Point2f center;                               // 灯条中心点
    float max_len;                                // 灯条长边长度
    float min_len;                                // 灯条短边长度
    float area;                                   // 灯条轮廓面积
    float rect_area;                              // 灯条旋转矩形面积
    Rect big_rect;                                // 原始图像中面积n倍的大矩形
    string color;                                 // HSV识别的颜色 red|blue|unknown
    Mat big_rect_roi;                             // 大矩形的ROI区域
};

//============================ 图像预处理 =============================
void preprocessImage(const Mat& src)
{
    // 1.彩色转灰度
    cvtColor(src, gray_img, COLOR_BGR2GRAY);

    // 2.灰度二值化 提取白色区域
    threshold(gray_img, gray_thres_img, gray_bright_thres, 255, THRESH_BINARY);

    // 3.1 B-R差分 提取亮度高的白色区域
    vector<Mat> bgr_channels;
    split(src, bgr_channels);
    subtract(bgr_channels[0], bgr_channels[2], color_sub_br_img, noArray(), -1);
    // 3.2归一化
    normalize(color_sub_br_img, color_sub_br_show, 0, 255, NORM_MINMAX, -1, noArray());
    // 3.3整形化归一化操作可能出现浮点数
    color_sub_br_show.convertTo(color_sub_br_show, CV_8UC1);

    // 4.提取交集区域 + 水平腐蚀
    threshold(color_sub_br_img, blue_black_area, blue_sub_black_thres, 255, THRESH_BINARY_INV);
    threshold(gray_img, gray_bright_area, gray_bright_thres, 255, THRESH_BINARY);
    bitwise_and(blue_black_area, gray_bright_area, candidate);

    // 5.1水平方向腐蚀
    Mat hori_erode_element = getStructuringElement(MORPH_RECT, Size(3, 1));
    erode(candidate, candidate_hori_erode, hori_erode_element);
    // 5.2垂直方向膨胀
    Mat vert_dilate_element = getStructuringElement(MORPH_RECT, Size(1, 15));
    //Mat vert_dilate_element = Mat_<uchar>::ones(15,1);
    dilate(candidate_hori_erode, dilate_img, vert_dilate_element);
}

//========================= ROI的HSV颜色识别 ==========================
string judgeColorByHSV(const Mat& roi)
{
    if (roi.empty()) return "unknown";

    Mat hsv_roi;
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);

    // 红色掩码
    Mat red_mask1, red_mask2, red_mask;
    inRange(hsv_roi, RED1_HSV_LOW, RED1_HSV_HIGH, red_mask1);
    inRange(hsv_roi, RED2_HSV_LOW, RED2_HSV_HIGH, red_mask2);
    red_mask = red_mask1 | red_mask2; // 合并红色掩码
    float red_ratio = (float)countNonZero(red_mask) / (roi.rows * roi.cols);

    // 蓝色掩码
    Mat blue_mask;
    inRange(hsv_roi, BLUE_HSV_LOW, BLUE_HSV_HIGH, blue_mask);
    float blue_ratio = (float)countNonZero(blue_mask) / (roi.rows * roi.cols);

    // 判断颜色 先红，再蓝，否则未知   颜色占比阈值 大于则判定为对应颜色
    if (red_ratio >= COLOR_RATIO_THRESH)
    {
        return "red";
    }
    else if (blue_ratio >= COLOR_RATIO_THRESH)
    {
        return "blue";
    }
    else
    {
        return "unknown";
    }
}

//========================== 生成n倍矩形ROI  ==========================
Rect createBigRect(const Point2f& center, float bar_rect_area, const Mat& src)
{
    // 1.大矩形面积 = 灯条旋转矩形面积 * 系数
    float big_area = bar_rect_area * BIG_RECT_AREA_RATIO;
    // 2.计算正方形边长
    float big_side = sqrt(big_area);
    // 3.计算大矩形的左上角坐标
    int x = max(0, (int)(center.x - big_side / 2));
    int y = max(0, (int)(center.y - big_side / 2));
    // 4.计算宽度和高度
    int width = min((int)big_side, src.cols - x);
    int height = min((int)big_side, src.rows - y);

    return Rect(x, y, width, height);
}

//============================ 筛选灯条 ================================
vector<LightBar> filterLightBars(const Mat& src)
{
    vector<LightBar> light_bars;
    // 1.创建一个副本 接受膨胀填充小轮廓结果
    Mat optimized_light_region = dilate_img.clone();
    dilate(optimized_light_region, optimized_light_region, ELEMENT);

    // 2.1定义存储的点集  定义轮廓层级关系
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // 2.2筛选轮廓
    findContours(optimized_light_region, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 4遍历所有检测到的轮廓 逐个筛选是否为合格灯条
    for (size_t i = 0; i < contours.size(); i++)
    {
        // 4.1面积筛选
        float area = contourArea(contours[i]);
        if (area < LIGHT_MIN_AREA || area > LIGHT_MAX_AREA) continue;

        // 4.2.1计算轮廓的最小外接旋转矩形
        RotatedRect rect = minAreaRect(contours[i]);
        // 4.2.2获取旋转矩形的宽度和高度（w、h是矩形的两个边长，无固定长宽之分）
        float w = rect.size.width;
        float h = rect.size.height;
        if (w < 1 || h < 1) continue; // 避免长宽为0导致除0问题

        // 4.3统一灯条的长宽 筛选
        float max_len = max(w, h);
        float min_len = min(w, h);
        // 4.3.1长边长度筛选
        if (max_len < LIGHT_MIN_MAX_LEN) continue;
        float ratio = max_len / min_len;
        // 4.3.2长宽比筛选
        if (ratio < LIGHT_MIN_RATIO || ratio > LIGHT_MAX_RATIO) continue;

        // 4.4角度判断
        float angle = fabs(rect.angle);
        bool is_vertical = (angle < LIGHT_TILT_JUDGE_THRESH && h > w) || (angle > LIGHT_TILT_JUDGE_THRESH && w > h);
        if (!is_vertical) continue;

        // 5.计算灯条旋转矩形面积 用于创建大矩形
        float rect_area = w * h;
        // 5.1创建大矩形（面积n倍）
        Rect big_rect = createBigRect(rect.center, rect_area, src);
        // 5.2提取ROI
        Mat big_rect_roi = src(big_rect);
        // 5.3 ROI 的 HSV颜色识别
        string color = judgeColorByHSV(big_rect_roi);

        // 6.1定义临时灯条信息
        LightBar bar;
        // 6.2存储灯条信息
        bar.rect = rect;
        bar.center = rect.center;
        bar.max_len = max_len;
        bar.min_len = min_len;
        bar.area = area;
        bar.rect_area = rect_area;
        bar.big_rect = big_rect;
        bar.color = color;
        bar.big_rect_roi = big_rect_roi;
        // 6.3把灯条信息灯条集合
        light_bars.push_back(bar);
    }

    return light_bars;
}

//============================== 匹配装甲板 ============================
bool matchArmorByColor(const vector<LightBar>& light_bars, const string& target_color, Point2f& armor_center, pair<LightBar, LightBar>& best_pair)
{
    // 1.筛选指定颜色的灯条到color_bars
    vector<LightBar> color_bars;
    for (const auto& bar : light_bars)
    {
        if (bar.color == target_color)
        {
            color_bars.push_back(bar);
        }
    }

    // 2.初始化参数
    armor_center = Point2f(0, 0);
    best_pair = {LightBar(), LightBar()};
    bool found = false;
    // 得分机制选择最优装甲板
    float best_score = 0.0f;

    // 3.两层循环 两两配对 不重复 不遗漏
    for (size_t i = 0; i < color_bars.size(); i++)
    {
        for (size_t j = i + 1; j < color_bars.size(); j++)
        {
            const LightBar& bar1 = color_bars[i];
            const LightBar& bar2 = color_bars[j];

            // 4.1两个灯条中心点之间的直线距离
            float distance = norm(bar1.center - bar2.center);
            // 4.2两个灯条的最长边、最短边
            float max_len = max(bar1.max_len, bar2.max_len);
            float min_len = min(bar1.max_len, bar2.max_len);
            // 4.3灯条长度比
            float rate = max_len / min_len;
            // 4.4两个灯条的旋转角度差
            float angle_sub = fabs(bar1.rect.angle - bar2.rect.angle);
            // 4.5两个灯条中心点的垂直方向偏差
            float y_diff = fabs(bar1.center.y - bar2.center.y);
            // 4.6两个灯条中心点的水平方向偏差
            float x_diff = fabs(bar1.center.x - bar2.center.x);
            // 避免除0报错 跳过重叠灯条
            if (x_diff < 1e-6) continue;
            // 4.7装甲板与水平轴的夹角
            float dx = bar2.center.x - bar1.center.x;
            float dy = bar2.center.y - bar1.center.y;
            float horizontal_angle = atan2(fabs(dy), fabs(dx)) * 180.0 / CV_PI;

            // 5.1灯条距离合理
            bool condition1 = (distance > max_len * ARMOR_MIN_DISTANCE_RATIO) && (distance <= max_len * ARMOR_MAX_DISTANCE_RATIO);
            // 5.2两灯条长短相似
            bool condition2 = rate < ARMOR_MAX_RATE;
            // 5.3灯条要基本平行
            bool condition3 = angle_sub < ARMOR_MAX_ANGLE_SUB;
            // 5.4垂直偏差不大
            bool condition4 = y_diff < x_diff * ARMOR_MAX_Y_X_RATIO;
            // 5.5装甲板不太斜
            bool condition5 = horizontal_angle <= ARMOR_MAX_HORIZONTAL_ANGLE;
            
            // 6.满足上面五种情况进入下面评分筛选 去除多个灯条都满足的情况
            if (condition1 && condition2 && condition3 && condition4 && condition5)
            {
                // 计算得分 加权得分
                float score = (1.0f - angle_sub / ARMOR_MAX_ANGLE_SUB) * 12 + (1.0f - rate / ARMOR_MAX_RATE) * 5 + (1.0f - horizontal_angle / ARMOR_MAX_HORIZONTAL_ANGLE) * 3;
                if (score > best_score || !found)
                {
                    best_score = score;
                    armor_center = (bar1.center + bar2.center) / 2.0f;
                    best_pair = {bar1, bar2};
                    found = true;
                }
            }
        }
    }

    return found;
}

//========================== 绘制检测结果 ==============================
void drawDetectionResult(Mat& dst, const vector<LightBar>& light_bars)
{
    // 1.1绘制所有灯条 红=红框 蓝=蓝框 未知=黄框
    for (const auto& bar : light_bars)
    {
        Scalar light_color;
        if (bar.color == "red")
        {
            light_color = Scalar(0, 0, 255); // 红=红框
        }
        else if (bar.color == "blue")
        {
            light_color = Scalar(255, 0, 0); // 蓝=蓝框
        } else
        {
            light_color = Scalar(0, 255, 255); // 未知=黄框
        }
        // 1.2绘制灯条矩形
        Point2f pts[4];
        bar.rect.points(pts);
        for (int k = 0; k < 4; k++)
        {
            line(dst, pts[k], pts[(k + 1) % 4], light_color, 2);
        }
        // 1.3绘制ROI
        rectangle(dst, bar.big_rect, light_color, 1);
    }

    // 2.匹配并绘制红色装甲板
    Point2f red_armor_center;
    pair<LightBar, LightBar> red_best_pair;
    if (matchArmorByColor(light_bars, "red", red_armor_center, red_best_pair))
    {
        // 2.1绘制装甲板连线
        Point2f pts1[4], pts2[4];
        red_best_pair.first.rect.points(pts1);
        red_best_pair.second.rect.points(pts2);
        for (int k = 0; k < 4; k++)
        {
            line(dst, pts1[k], pts2[k], Scalar(0, 150, 255), 2); // 红装甲板：橙框
        }
        // 2.2绘制装甲板中心
        circle(dst, red_armor_center, 5, Scalar(0, 255, 0), -1);
        // 2.3标注文字
        string text = "red target";
        Point text_pos(red_armor_center.x + 20, red_armor_center.y);
        putText(dst, text, text_pos, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 3);
    }

    // 3.匹配并绘制蓝色装甲板
    Point2f blue_armor_center;
    pair<LightBar, LightBar> blue_best_pair;
    if (matchArmorByColor(light_bars, "blue", blue_armor_center, blue_best_pair))
    {
        Point2f pts1[4], pts2[4];
        blue_best_pair.first.rect.points(pts1);
        blue_best_pair.second.rect.points(pts2);
        for (int k = 0; k < 4; k++)
        {
            line(dst, pts1[k], pts2[k], Scalar(255, 255, 0), 2); // 蓝装甲板：青框
        }
        circle(dst, blue_armor_center, 5, Scalar(0, 255, 0), -1);
        string text = "blue target";
        Point text_pos(blue_armor_center.x + 20, blue_armor_center.y);
        putText(dst, text, text_pos, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 3);
    }
}

//============================= 装甲板检测 =============================
Mat detectArmor(const Mat& src)
{
    Mat result = src.clone();
    if (src.empty() || src.channels() != 3)
    {
        cerr << "Error: Input image is empty or not BGR format!" << endl;
        return result;
    }

    // 1.图像预处理
    preprocessImage(src);

    // 2. 筛选灯条（含大矩形创建+HSV颜色识别）
    vector<LightBar> light_bars = filterLightBars(src);

    // 3. 绘制检测结果
    drawDetectionResult(result, light_bars);

    return result;
}

//============================= 主函数 ================================
int main(int argc, char* argv[])
{
    // 1.打开视频 失败则用摄像头）
    VideoCapture capture("../deck_detection.mp4", CAP_FFMPEG);
    if (!capture.isOpened())
    {
        cerr << "Warning: 无法打开视频文件 打开摄像头！" << endl;
        capture.open(0);
        if (!capture.isOpened())
        {
            cerr << "Error: 无法打开摄像头！" << endl;
            return -1;
        }
    }

    Mat frame, processed_frame;
    double fps = capture.get(CAP_PROP_FPS);
    int delay = fps > 1e-6 ? static_cast<int>(round(1000.0 / fps)) : 30;

    // 2.创建显示窗口
    namedWindow("Frame", WINDOW_AUTOSIZE);
    namedWindow("Result", WINDOW_AUTOSIZE);
    namedWindow("Mask", WINDOW_AUTOSIZE);
    namedWindow("Middle process", WINDOW_AUTOSIZE);

    // 3.调整窗口位置
    moveWindow("Frame", 100, 100);
    moveWindow("Result", 800, 100);
    moveWindow("Middle process", 100, 700);
    moveWindow("Mask", 800, 700);

    // 4.主循环
    while (true)
    {
        // 4.1.判断暂停/播放完毕
        if (!is_paused)
        {
            capture >> frame;
            if (frame.empty())
            {
                cout << "视频播放完毕！" << endl;
                break;
            }
            processed_frame = detectArmor(frame);
        }

        // 4.2.显示处理步骤和结果
        imshow("Frame",frame);
        imshow("Result", processed_frame);
        imshow("Mask", dilate_img);
        imshow("Middle process", color_sub_br_show);

        // 4.3 ESC退出，空格暂停/继续
        int key = is_paused ? waitKey(0) : waitKey(delay);
        if (key == 27) // ESC键
        {
            cout << "用户手动退出！" << endl;
            break;
        }
        else if (key == 32) // 空格键
        {
            is_paused = !is_paused;
            cout << "视频已" << (is_paused ? "暂停" : "继续") << endl;
        }
    }

    // 5.释放资源
    capture.release();
    destroyAllWindows();

    return 0;
}

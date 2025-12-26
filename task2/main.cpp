#include "opencv2/opencv.hpp"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

// ================================ 全局变量 =======================================

// 定义连续角度
float last_continuous_theta = 0.0f;
// 首次检测到目标时进行初始化矩阵的标志
bool is_first_detect = true;
// 标记目标是否被成功检测到
bool is_target_detected = false;
// 视频帧率
double fps = 30.0;

// =============================== 角速度平滑 ======================================
// 存储最近多帧的角速度数据，用于计算平均值
vector<float> omega_history;
// 平滑窗口大小，取最近10帧的角速度
const int smooth_window = 8;
// 平滑后的角速度，用于一秒后预测，过滤波动
float smooth_omega = 0.0f;

// =============================== 一秒后预测缓动 ==================================
// 存储上一次的一秒后预测角度，用于缓动过渡
float last_predict_1s_theta = 0.0f;
// 标记是否是第一次进行一秒后预测
bool is_first_1s_predict = true;

// ================================ 误差判断阈值 ==================================
// 角度误差阈值为45度，角度偏差触发重置
float ANGLE_ERROR_THRESHOLD = CV_PI / 4;
// 角速度突变阈值为1.0rad/s，大的角速度变化触发重置
float OMEGA_ERROR_THRESHOLD = 1.0f;
// 存储上一次卡尔曼滤波预测的角度，计算角度误差
float last_kalman_predict_theta = 0.0f;
// 存储上一次的轮廓面积，用于判断视觉遮挡/突变
float last_contour_area = 0.0f;

// ================================= 连续超标统计 ==================================
// 初始化连续超标帧数
static int continuous_error_count = 0;
// 连续超标阈值为3,连续3帧超标重置
static const int ERROR_COUNT_THRESHOLD = 3;

// ============================== 最小二乘法计算拟合圆 ==============================
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
        // 使用了负的平方和 −(x2+y2) 来表示圆方程。
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

// ============================= 像素坐标转连续角度和角速度 ===========================
void xy2ContinuousAngle(const Point2f &center, const Point2f &point, float dt, float &continuous_theta, float &omega)
{
    // 计算目标相对于圆心的x、y偏移量
    float dx = point.x - center.x;
    float dy = point.y - center.y;
    // 计算原始极角（范围：-π ~ π）
    float raw_theta = atan2(dy, dx);

    if (is_first_detect)
    {
        // 首次检测到目标：初始化相关变量
        continuous_theta = raw_theta; // 初始输出的连续角度为当前极角
        omega = 0.0f;                 // 初始角速度为0
        is_first_detect = false;      // 标记首次检测完成
        is_target_detected = true;    // 标记目标已检测到
        omega_history.clear();        // 清空历史角速度数据
        smooth_omega = 0.0f;          // 初始化平滑角速度
    }
    else
    {
        // 处理角度跳变：将当前极角与上一帧的连续角度做对比，修正跳变（π -> -π / π -> π+1）
        // fmod(last_continuous_theta, 2*CV_PI)：将上一帧的连续角度转换回0~2π范围
        // 角度变化量 = 当前帧角度 - 上一帧角度
        float delta_theta = raw_theta - fmod(last_continuous_theta, 2 * CV_PI);
        // 如果角度差超过π（180度），说明发生了跳变，需要修正
        if (delta_theta > CV_PI)
            delta_theta -= 2 * CV_PI;
        else if (delta_theta < -CV_PI)
            delta_theta += 2 * CV_PI;

        // 更新连续角度：上一帧角度 + 角度变化量（突破0~2π限制）
        continuous_theta = last_continuous_theta + delta_theta;
        // 计算原始单帧角速度：角度变化量 / 时间间隔（rad/s）
        omega = delta_theta / dt;

// =========================== 角速度滑动平均：过滤单帧波动 ==========================
        // 将当前角速度加入历史数组
        omega_history.push_back(omega);
        // 只要最新数据
        if (omega_history.size() > smooth_window)
        {
            omega_history.erase(omega_history.begin());
        }
        // 计算历史角速度的平均值
        smooth_omega = 0.0f;
        for (size_t i = 0; i < omega_history.size(); i++)
        {
            float w = omega_history[i];
            smooth_omega += w;
        }
        smooth_omega /= omega_history.size();
    }

    // 保存当前连续角度，用于下一帧计算
    last_continuous_theta = continuous_theta;
}

// ========================= 连续角度转拟合圆上的像素坐标 =============================
Point2f continuousAngle2xy(const Point2f &center, float radius, float continuous_theta)
{
    // 将连续角度取模2π，转换回0~2π的范围（用于极坐标转笛卡尔坐标）
    float normalized_theta = fmod(continuous_theta, 2 * CV_PI);
    // 极坐标转笛卡尔坐标：计算拟合圆上的像素坐标
    float x = center.x + radius * cos(normalized_theta);
    float y = center.y + radius * sin(normalized_theta);
    return Point2f(x, y);
}

// ====================== 卡尔曼单步预测 下一帧高信任测量值 ======================
float kalmanPredictSingleStep(KalmanFilter &kalman, float current_theta, float current_omega, float dt)
{
    // 初始化卡尔曼滤波器（首次使用或重置后）
    if (kalman.statePost.empty())
    {
        // 初始化卡尔曼：状态数=2（角度、角速度），测量数=1（角度），控制数=0
        kalman.init(2, 1, 0);

        // ================ 卡尔曼转移矩阵A：匀速运动模型 =================
        // 状态转移方程：[θ(t+1); ω(t+1)] = [1, dt; 0, 1] * [θ(t); ω(t)]
        // 含义：下一帧角度 = 当前角度 + 角速度*时间间隔；下一帧角速度 = 当前角速度
        kalman.transitionMatrix = (Mat_<float>(2, 2) << 1, dt, 0, 1);

        // ================ 卡尔曼测量矩阵H：仅测量角度 ==================
        // 测量方程：z(t) = [1, 0] * [θ(t); ω(t)]
        // 含义：测量值只有角度
        kalman.measurementMatrix = (Mat_<float>(1, 2) << 1, 0);

        // ================= 卡尔曼参数配置：高信任测量值 =================
        // 过程噪声协方差Q：小值，轻微偏向模型，主要信任测量值
        setIdentity(kalman.processNoiseCov, Scalar::all(1e-2));
        // 测量噪声协方差R：很小值，高信任测量值
        setIdentity(kalman.measurementNoiseCov, Scalar::all(1e-4));
        // 后验误差协方差P：小值，让滤波启动更快，快速收敛到测量值
        setIdentity(kalman.errorCovPost, Scalar::all(1e-2));

        // ============== 卡尔曼初始状态：当前角度和角速度 ================
        kalman.statePost.at<float>(0) = current_theta;
        kalman.statePost.at<float>(1) = current_omega;
    }
    else
    {
        // 更新转移矩阵的dt
        kalman.transitionMatrix.at<float>(0, 1) = dt;
    }

// ========================= 卡尔曼预测步骤：预测下一帧的状态 =========================
    Mat prediction_single = kalman.predict();

// ========================= 卡尔曼校正步骤：用当前测量值更新 =========================
    Mat measurement = (Mat_<float>(1, 1) << current_theta);
    kalman.correct(measurement);

    // 保存本次预测的角度，用于后续误差判断
    last_kalman_predict_theta = prediction_single.at<float>(0);

    // 返回预测的下一帧角度
    return prediction_single.at<float>(0);
}

// ====================== 一秒后预测（平滑角速度+加权平均更新） ======================
//   current_theta：当前测量的连续角度
//   smooth_omega：平滑后的角速度
float predict1sBySmoothOmega(float current_theta, float smooth_omega)
{
    // 匀速模型计算一秒后的角度：当前角度 + 平滑角速度 * 1秒
    float current_predict_theta = current_theta + smooth_omega * 1.00f;

    // ====================== 缓动更新：让预测点缓慢过渡 ======================
    // 缓动系数：0~1  越小越平滑 但响应稍慢，越大越实时 但轻微闪烁
    float ease_factor = 0.7f;
    if (is_first_1s_predict)
    {
        // 第一次进行一秒后预测：直接使用计算值
        last_predict_1s_theta = current_predict_theta;
        is_first_1s_predict = false;
        return current_predict_theta;
    }

    // 缓动公式（加权平均）：上一次预测角度 * (1-系数) + 当前计算角度 * 系数
    // 作用：让预测点不会突然跳到新位置，而是缓慢过渡
    float ease_theta = last_predict_1s_theta * (1 - ease_factor) + current_predict_theta * ease_factor;

    // 保存本次缓动后的角度，用于下一帧
    last_predict_1s_theta = ease_theta;

    return ease_theta;
}

// ====================== 重置所有预测相关变量 ======================
void resetAllPredictionVars(KalmanFilter &kalman)
{
    // 1. 重置卡尔曼滤波器：清空所有状态，恢复初始状态
    kalman = KalmanFilter();

    // 2. 重置连续角度相关变量
    last_continuous_theta = 0.0f;
    is_first_detect = true;
    is_target_detected = false;

    // 3. 重置角速度平滑相关变量
    omega_history.clear();
    smooth_omega = 0.0f;

    // 4. 重置一秒后预测缓动相关变量
    last_predict_1s_theta = 0.0f;
    is_first_1s_predict = true;

    // 5. 重置误差判断相关变量
    last_kalman_predict_theta = 0.0f;
    last_contour_area = 0.0f;
    continuous_error_count = 0; // 重置连续超标帧数

    cout << "角度变化过大!目标已转移!重置卡尔曼滤波!" << endl;
}

//#################################################################################
// ==================================== 主函数 =====================================
//#################################################################################

int main(int argc, char **argv)
{
    // 打开视频文件
    VideoCapture capture("../buff.mp4");

    if (!capture.isOpened())
    {
        cout << "无法读取视频" << endl;
        return 0;
    }

    // 从视频中获取实际帧率
    fps = capture.get(CAP_PROP_FPS);
    if (fps <= 0)
        fps = 30.0; // 若获取失败，默认30帧/秒
    // 计算每帧的时间间隔 = 1秒/帧率
    const float dt_per_frame = 1.0f / fps;

    Mat frame;
    // 存储目标中心的像素点，用于最小二乘法拟合圆
    vector<cv::Point2f> points;
    // 定义旋转矩形，用于存储目标轮廓的最小外接矩形
    RotatedRect rect;

    // 声明卡尔曼滤波器对象
    KalmanFilter kalman;

    while (true)
    {
        // 读取视频的下一帧
        capture.read(frame);

        // 如果帧为空（视频播放完毕），重新打开视频循环播放
        if (frame.empty())
        {
            capture.open("../buff.mp4");
            if (!capture.isOpened())
            {
                cout << "can't open video" << endl;
                break;
            }
            cout << "open video again" << endl;
            // 重置检测标志，不重置卡尔曼，让之后的预测越来越准
            is_target_detected = false;
            is_first_detect = true;
            continue; // 跳过当前循环，开始新的帧读取
        }

// ================================= 图像预处理 =====================================

        /////////////////////////
        // Image preprocessing //
        /////////////////////////
        
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

// ==================================== 轮廓检测 ====================================
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
       
// ================================ 轮廓处理-绘图 ===================================
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
                            // 计算目标轮廓的最小面积包围矩形
                            rect = minAreaRect(contours[hierarchy[i][2]]);
                            Point2f p[4];
                            // 将矩形的四个角点填充到数组 p 中
                            rect.points(p);
                            // 计算矩形的中心点
                            Point2f center(0, 0);
                            for (int i = 0; i < 4; i++)
                            {
                                center += p[i];
                            }
                            center /= 4;
                            // 将中心点添加到points向量，用于拟合圆
                            points.push_back(center);
                            // 最小二乘法拟合圆
                            Vec3f circl = fitCircle(points);
                            // 在 frame（图像或视频帧）上绘制拟合的圆。该圆用绿色绘制，厚度为 3 像素
                            cv::circle(frame, Point(circl[0], circl[1]), circl[2], Scalar(0, 255, 0), 3);
                            // 在矩形的中心绘制一个小的品红色圆圈（半径为 5）。
                            cv::circle(frame, center, 5, Scalar(255, 0, 255), -1);
                            // 代码在矩形的中心添加了 "target" 标签，并略微偏移放置第二个文本，以避免重叠。
                            cv::putText(frame, "target", center, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, LINE_AA);
                            cv::putText(frame, "target", center + Point2f(2, 1), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, LINE_AA);
                            // 绘制白色矩形（目标轮廓）
                            for (int j = 0; j < 4; ++j)
                            {
                                line(frame, p[j], p[(j + 1) % 4], Scalar(255, 255, 255), 2);
                            }

                            // =================== kalman 核心逻辑 ===================
                            float current_theta, current_omega;
                            // 1. 像素坐标转连续角度和角速度
                            xy2ContinuousAngle(Point2f(circl[0], circl[1]), center, dt_per_frame, current_theta, current_omega);

                            // 2. 卡尔曼单步预测下一帧角度
                            float predict_theta_frame = kalmanPredictSingleStep(kalman, current_theta, current_omega, dt_per_frame);

                            // 3. 误差判断：大于阈值+连续超标 重置
                            // 3.1 计算角度误差
                            float angle_error = fabs(current_theta - last_kalman_predict_theta);
                            // 3.2 计算角速度突变误差
                            float omega_error = fabs(current_omega - smooth_omega);

                            // 判断当前帧内待打击目标是否变化：角度误差/角速度误差超过阈值
                            bool is_error_exceed = (angle_error > ANGLE_ERROR_THRESHOLD) || (omega_error > OMEGA_ERROR_THRESHOLD);

                            if (is_error_exceed)
                            {
                                // 当前帧超标：连续帧数+1
                                continuous_error_count++;
                                // 连续超标达到阈值，才执行重置
                                if (continuous_error_count >= ERROR_COUNT_THRESHOLD)
                                {
                                    resetAllPredictionVars(kalman); // 重置所有变量
                                    predict_theta_frame = current_theta; // 重置后，用当前测量值
                                    continuous_error_count = 0; // 重置后清零
                                }
                            }
                            else
                            {
                                // 当前帧未超标：清空连续帧数
                                continuous_error_count = 0;
                            }

                            // 4.一秒后预测：仅当目标未被重置时执行
                            float predict_theta_1s = current_theta; // 默认用当前测量值
                            if (is_target_detected)
                            {
                                predict_theta_1s = predict1sBySmoothOmega(current_theta, smooth_omega);
                            }

                            // 5.转换为像素坐标：将预测角度转换为拟合圆上的像素点
                            Point2f predict_point_frame = continuousAngle2xy(Point2f(circl[0], circl[1]), circl[2], predict_theta_frame);
                            Point2f predict_point_1s = continuousAngle2xy(Point2f(circl[0], circl[1]), circl[2], predict_theta_1s);

                            // 6.绘制预测点：下一帧红，一秒后蓝
                            cv::circle(frame, predict_point_frame, 5, Scalar(0, 0, 255), -1);
                            cv::putText(frame, "1frame forecast", predict_point_frame + Point2f(5, 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);

                            cv::circle(frame, predict_point_1s, 5, Scalar(255, 0, 0), -1);
                            cv::putText(frame, "1s forecast", predict_point_1s + Point2f(5, 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, LINE_AA);

                            // 找到目标后退出循环
                            break;
                        }
                    }
                }
            }
        }

        // ============================== 显示图像 ==================================
        namedWindow("mid", WINDOW_NORMAL);
        resizeWindow("mid", 640, 480);
        imshow("mid", midImage);

        namedWindow("frame", WINDOW_NORMAL);
        resizeWindow("frame", 640, 480);
        imshow("frame", frame);

        // 按ESC键退出视频循环
        if (waitKey(7) == 27)
        {
            break;
        }
    }

    return 0;
}
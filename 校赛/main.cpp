#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

using namespace cv;
using namespace std;

// ========================== 核心参数 ================================

// 1.颜色分割参数
const int H_LOW = 33, H_HIGH = 70;   // 绿色范围
const int S_LOW = 40, S_HIGH = 200;  // 饱和度
const int V_LOW = 0, V_HIGH = 255;   // 亮度

// 2.目录路径
const string IMG_INPUT_DIR = "../test/";        // 输入图片文件夹
const string INTERMEDIATE_DIR = "../intermediate/";  // 中间结果（掩膜 距离图 分水岭掩膜）
const string RESULT_FILES_DIR = "../result/";   // 最终结果文件夹

// 3.文件名称
const vector<string> TASK_FILES = {"task1.jpg", "task2.jpg", "task3.jpg", "task4.jpg", "task5.jpg"};
const vector<string> MASK_FILES = {"mask1.jpg", "mask2.jpg", "mask3.jpg", "mask4.jpg", "mask5.jpg"};
const vector<string> DIST_FILES = {"dist1.jpg", "dist2.jpg", "dist3.jpg", "dist4.jpg", "dist5.jpg"};
const vector<string> WATERSHED_MASK_FILES = {"watershed_mask1.jpg", "watershed_mask2.jpg", "watershed_mask3.jpg", "watershed_mask4.jpg", "watershed_mask5.jpg"};  // 分水岭后新掩膜
const vector<string> RESULT_FILES = {"result1.jpg", "result2.jpg", "result3.jpg", "result4.jpg", "result5.jpg"};

// 4.分水岭+极小值合并核心参数
const float DIST_THRESH_COEF = 0.80;    // 极小值区域筛选阈值 越小极小值越多
const float MARKER_MERGE_DIST = 0.04;   // 极小值合并距离 越大合并越激进
const int WATERSHED_KERNEL = 5;        // 距离变换核大小
const int MIN_MARKER_AREA = 3;        // 最小极小值面积 过滤噪点

// 5.弹丸筛选参数
const float RADIUS_TOLERANCE = 0.45;   // 基准模式：半径容差
const float MIN_CIRCULARITY = 0.35;    // 最小圆度 越低允许越不规则
const float MIN_AREA_RATIO = 0.15;      // 基准模式：最小面积占比 适应遮挡
const float DUPLICATE_DISTANCE = 0.5;  // 去重距离阈值
const int MASK_MIN_AREA = 40;          // 掩膜最小面积
const int MASK_MAX_AREA = 15000;       // 掩膜最大面积

// 6.层级过滤参数
const int MAX_HIERARCHY_DEPTH = 2;

// 基准参数结构体
struct BaseParam
{
    float radius;
    bool valid;
};

// ===================== 计算轮廓中心 ==========================================
Point2f getContourCenter(const vector<Point>& cnt)
{
    Moments mu = moments(cnt);
    return Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
}

// ===================== 计算轮廓的层级深度 ==================================
int calculateContourDepth(const vector<Vec4i>& hierarchy, int contour_idx)
{
    if (contour_idx == -1)
    { 
        return 0;
    }
    // 计算轮廓的深度：当前轮廓深度 = 1 + 子轮廓最大深度
    int child_idx = hierarchy[contour_idx][2];
    int child_depth = calculateContourDepth(hierarchy, child_idx);
    return 1 + child_depth;
}

// ===================== 颜色分割 形态学优化 层级过滤 ===========================
Mat colorSegmentation(const Mat& bgr)
{
    Mat hsv, mask;

    // 1.BGR转HSV
    cvtColor(bgr, hsv, COLOR_BGR2HSV);

    // 2.生成初始掩膜
    inRange(hsv, Scalar(H_LOW, S_LOW, V_LOW), Scalar(H_HIGH, S_HIGH, V_HIGH), mask);

    // 3.创建形态学操作核
    Mat kernel_2x2 = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
    Mat kernel_3x3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

    // 4.开运算：去除背景小噪点
    morphologyEx(mask, mask, MORPH_OPEN, kernel_2x2, Point(-1, -1), 1);

    // 5.腐蚀：制造粘连弹丸的缝隙
    erode(mask, mask, kernel_2x2, Point(-1, -1), 1);

    Mat kernel_1x1 = getStructuringElement(MORPH_ELLIPSE, Size(1, 1));
    erode(mask, mask, kernel_1x1, Point(-1, -1), 1);

    // 6.闭运算：填补弹丸内部孔洞
    morphologyEx(mask, mask, MORPH_CLOSE, kernel_3x3, Point(-1, -1), 1);

    // 7.轻微膨胀：恢复弹丸边缘
    dilate(mask, mask, kernel_2x2, Point(-1, -1), 1);

    // 8.第一步过滤：小面积噪点 面积<20
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (contourArea(contours[i]) < 20)
        {
            drawContours(mask, contours, (int)i, Scalar(0), -1);  // 填充为黑色过滤
        }
    }

    // 9.层级过滤：过滤复杂杂物（如纸抽、瓶子）
    contours.clear();
    hierarchy.clear();
    findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++)
    {
        int depth = calculateContourDepth(hierarchy, (int)i);
        double area = contourArea(contours[i]);
        if (depth > MAX_HIERARCHY_DEPTH || area > MASK_MAX_AREA)
        {
            drawContours(mask, contours, (int)i, Scalar(0), -1);  // 过滤杂物
            cout << "过滤层级杂物：轮廓索引" << i << "，层级深度" << depth << endl;
        }
    }

    // imshow("1. 颜色分割后的掩膜", mask);
    // waitKey(0);

    return mask;
}

// ===================== 距离变换+标记点生成 =================================
Mat generateDistanceTransform(const Mat& mask, Mat& dist_norm)
{
    Mat mask_8u = Mat::zeros(mask.size(), CV_8UC1);
    for (int i = 0; i < mask.rows; i++)
    {
        const uchar* src_row = mask.ptr<uchar>(i);
        uchar* dst_row = mask_8u.ptr<uchar>(i);
        for (int j = 0; j < mask.cols; j++)
        {
            dst_row[j] = (src_row[j] > 127) ? 255 : 0;
        }
    }

    // 1.距离变换：计算前景像素到最近背景的欧氏距离
    Mat dist;
    distanceTransform(mask_8u, dist, DIST_L2, WATERSHED_KERNEL);

    // 2.归一化：距离值缩放到0-1 方便统一阈值处理
    normalize(dist, dist_norm, 0, 1, NORM_MINMAX);

    // 3.阈值处理：筛选极小值区域
    Mat dist_thresh;
    threshold(dist_norm, dist_thresh, DIST_THRESH_COEF, 1, THRESH_BINARY);
    dist_thresh.convertTo(dist_thresh, CV_8UC1, 255);

    // 4.清除掩膜外的标记点
    for (int i = 0; i < dist_thresh.rows; i++)
    {
        uchar* dt_row = dist_thresh.ptr<uchar>(i);
        const uchar* mask_row = mask_8u.ptr<uchar>(i);
        for (int j = 0; j < dist_thresh.cols; j++)
        {
            if (mask_row[j] == 0) dt_row[j] = 0;   // 掩膜外强制设为黑色
        }
    }

    // 5.膨胀
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
    dilate(dist_thresh, dist_thresh, kernel, Point(-1, -1), 1);

    // 新增：再次确保膨胀后的标记点不超出掩膜
    for (int i = 0; i < dist_thresh.rows; i++)
    {
        uchar* dt_row = dist_thresh.ptr<uchar>(i);
        const uchar* mask_row = mask_8u.ptr<uchar>(i);
        for (int j = 0; j < dist_thresh.cols; j++)
        {
            if (mask_row[j] == 0) dt_row[j] = 0;
        }
    }

    // imshow("2. 距离变换图 亮处是弹丸中心", dist_norm);
    // imshow("3. 极小值区域（标记点候选）", dist_thresh);
    // waitKey(0);

    return dist_thresh;  // 返回极小值区域二值图
}

// ===================== 标记点去重极小值合并 掩膜约束===============================
vector<vector<Point>> mergeMarkers(const vector<vector<Point>>& marker_contours, const Mat& mask, float merge_dist)
{
    vector<vector<Point>> merged_markers;
    vector<Point2f> centers;  // 存储已合并标记点的中心

    for (const auto& cnt : marker_contours)
    {
        // 1.过滤噪点极小值
        if (contourArea(cnt) < MIN_MARKER_AREA) continue;

        Point2f center = getContourCenter(cnt);
        // 2.确保标记点中心在掩膜内
        if (mask.at<uchar>((int)center.y, (int)center.x) != 255) continue;

        bool is_duplicate = false;

        // 3.合并距离小于merge_dist的极小值区域 避免过度分割
        for (const auto& existing_center : centers)
        {
            if (norm(center - existing_center) < merge_dist)
            {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate)
        {
            merged_markers.push_back(cnt);
            centers.push_back(center);
        }
    }

    return merged_markers;
}

// ===================== 分水岭分割 =====================================
vector<vector<Point>> watershedSegmentation(const Mat& bgr, const Mat& mask, const Mat& dist_thresh, Mat& watershed_mask, float base_radius = 0)
{
    Mat mask_8u = Mat::zeros(mask.size(), CV_8UC1);
    for (int i = 0; i < mask.rows; i++)
    {
        const uchar* src_row = mask.ptr<uchar>(i);
        uchar* dst_row = mask_8u.ptr<uchar>(i);
        for (int j = 0; j < mask.cols; j++)
        {
            dst_row[j] = (src_row[j] > 127) ? 255 : 0;
        }
    }

    // 1.提取极小值区域轮廓
    vector<vector<Point>> marker_contours;
    vector<Vec4i> hierarchy;
    findContours(dist_thresh, marker_contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 2.极小值合并
    float merge_dist = base_radius > 0 ? (base_radius * MARKER_MERGE_DIST) : 20.0f;
    vector<vector<Point>> merged_markers = mergeMarkers(marker_contours, mask_8u, merge_dist);

    // 3.显示合并后的标记点
    Mat marker_img = bgr.clone();
    for (size_t i = 0; i < merged_markers.size(); i++)
    {
        drawContours(marker_img, merged_markers, i, Scalar(0, 255, 0), -1); // 绿色标记点
    }
    // imshow("4. 合并后的标记点（绿色为有效标记）", marker_img);
    // waitKey(0);

    // 4.分水岭标记矩阵
    Mat markers = Mat::zeros(mask.size(), CV_32SC1);
    int marker_id = 1;

    // 4.绘制前景标记点
    for (const auto& cnt : merged_markers)
    {
        Mat cnt_mask = Mat::zeros(mask.size(), CV_8UC1);
        drawContours(cnt_mask, vector<vector<Point>>{cnt}, -1, Scalar(255), -1);
        for (int i = 0; i < mask.rows; i++)
        {
            for (int j = 0; j < mask.cols; j++)
            {
                if (mask_8u.at<uchar>(i, j) == 255 && cnt_mask.at<uchar>(i, j) == 255)
                {
                    markers.at<int>(i, j) = marker_id;
                }
            }
        }
        marker_id++;
    }

    // 5.绘制背景标记
    markers.setTo(0, mask_8u == 0);  // 掩膜黑色区域=背景
    rectangle(markers, Rect(0, 0, mask.cols, 1), Scalar(0), -1);
    rectangle(markers, Rect(0, mask.rows-1, mask.cols, 1), Scalar(0), -1);
    rectangle(markers, Rect(0, 0, 1, mask.rows), Scalar(0), -1);
    rectangle(markers, Rect(mask.cols-1, 0, 1, mask.rows), Scalar(0), -1);

    // 6.分水岭分割
    Mat markers_copy = markers.clone();
    watershed(bgr, markers_copy);
    Mat edge = Mat::zeros(mask.size(), CV_8UC1);
    edge.setTo(255, markers_copy == -1);

    // 7.生成分水岭后新掩膜
    watershed_mask = Mat::zeros(mask.size(), CV_8UC1);
    for (int i = 0; i < markers_copy.rows; i++)
    {
        for (int j = 0; j < markers_copy.cols; j++)
        {
            if (mask_8u.at<uchar>(i, j) == 255 && markers_copy.at<int>(i, j) >= 1)
            {
                watershed_mask.at<uchar>(i, j) = 255;  // 掩膜内+前景ID≥1 → 白色
            }
            else
            {
                watershed_mask.at<uchar>(i, j) = 0;    // 其他情况→黑色
            }
        }
    }

    // 8.提取分割后的弹丸轮廓
    vector<vector<Point>> segmented_contours;
    for (int id = 1; id < marker_id; id++)
    {
        Mat marker_mask = Mat::zeros(markers_copy.size(), CV_8UC1);
        for (int i = 0; i < markers_copy.rows; i++)
        {
            for (int j = 0; j < markers_copy.cols; j++)
            {
                if (mask_8u.at<uchar>(i, j) == 255 && markers_copy.at<int>(i, j) == id)
                {
                    marker_mask.at<uchar>(i, j) = 255;
                }
            }
        }
        vector<vector<Point>> cnts;
        findContours(marker_mask, cnts, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (!cnts.empty())
        {
            // 取最大轮廓 过滤分割产生的小碎片
            auto max_cnt = *max_element(cnts.begin(), cnts.end(), [](const vector<Point>& a, const vector<Point>& b)
            {
                return contourArea(a) < contourArea(b);
            });
            if (contourArea(max_cnt) > 10)
            {
                segmented_contours.push_back(max_cnt);
            }
        }
    }

        // 9.新增形态学细化
    for (auto& cnt : segmented_contours) {
        // 10.对每个轮廓对应的区域进行腐蚀
        Mat cnt_mask = Mat::zeros(mask.size(), CV_8UC1);
        drawContours(cnt_mask, vector<vector<Point>>{cnt}, -1, Scalar(255), -1);
        erode(cnt_mask, cnt_mask, getStructuringElement(MORPH_ELLIPSE, Size(1,1)), Point(-1,-1), 1);
        // 11.重新提取细化后的轮廓
        vector<vector<Point>> temp_cnts;
        findContours(cnt_mask, temp_cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (!temp_cnts.empty()) {
            cnt = temp_cnts[0];  // 替换为细化后的轮廓
        }
    }


    // // 显示分割结果
    // Mat watershed_result = bgr.clone();
    // for (size_t i = 0; i < segmented_contours.size(); i++)
    // {
    //     drawContours(watershed_result, segmented_contours, i, Scalar(rand()%255, rand()%255, rand()%255), 2);
    // }
    // // 绘制掩膜边界
    // vector<vector<Point>> mask_contours;
    // findContours(mask_8u, mask_contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // drawContours(watershed_result, mask_contours, -1, Scalar(255, 0, 0), 1);
    // imshow("5. 分水岭分割后的轮廓", watershed_result);
    // waitKey(0);

    return segmented_contours;
}

// ===================== 基准弹丸识别 =======================================
BaseParam calibrateBaseRadiusPerImage(const Mat& bgr, const Mat& mask, const string& img_name)
{
    BaseParam base = {0.0f, false};

    // 1.生成距离变换图和极小值区域
    Mat dist_norm;
    Mat dist_thresh = generateDistanceTransform(mask, dist_norm);

    // 2.分水岭分割 无基准时用固定合并距离
    Mat temp_watershed_mask;
    vector<vector<Point>> segmented_contours = watershedSegmentation(bgr, mask, dist_thresh, temp_watershed_mask);

    // 3.筛选孤立的弹丸作为基准
    vector<float> valid_radii;
    for (const auto& cnt : segmented_contours)
    {
        double area = contourArea(cnt);
        if (area < 50 || area > 5000) continue;  // 过滤过小/过大轮廓

        // 计算圆度和外接圆
        Point2f center;
        float radius;
        minEnclosingCircle(cnt, center, radius);
        double perimeter = arcLength(cnt, true);
        double circularity = perimeter > 0 ? (4 * CV_PI * area) / (perimeter * perimeter) : 0;

        // 基准弹丸要求：高圆度+面积匹配+孤立
        bool is_circular = (circularity >= 0.8);
        bool area_match = (abs(area - CV_PI * radius * radius) / (CV_PI * radius * radius)) < 0.15;
        if (!is_circular || !area_match) continue;

        // 验证孤立性（与其他弹丸距离≥2.2倍半径）
        bool is_isolated = true;
        for (const auto& other_cnt : segmented_contours)
        {
            if (cnt == other_cnt) continue;
            Point2f other_center;
            float other_radius;
            minEnclosingCircle(other_cnt, other_center, other_radius);
            if (norm(center - other_center) < 2.2 * radius)
            {
                is_isolated = false;
                break;
            }
        }

        if (is_isolated)
        {
            valid_radii.push_back(radius);
            cout << "[" << img_name << "] 找到基准弹丸：半径=" << radius << " 圆度=" << circularity << endl;
        }
    }

    // 4.计算基准半径
    if (!valid_radii.empty())
    {
        base.radius = accumulate(valid_radii.begin(), valid_radii.end(), 0.0f) / valid_radii.size();
        base.valid = true;
        cout << "[" << img_name << "] 基准校准成功：" << base.radius << " 像素" << endl;
    }
    else
    {
        cout << "[" << img_name << "] 未找到基准弹丸，使用掩膜模式 WWW " << endl;
    }

    return base;
}

// ===================== 弹丸筛选+去重+计数 ==========================================
int countValidBalls(const vector<vector<Point>>& segmented_contours, const BaseParam& base, vector<vector<Point>>& valid_balls)
{
    valid_balls.clear();

    if (base.valid)
{
        // 基准模式：按半径+圆度+面积筛选
        float base_r = base.radius;
        float min_r = base_r * (1 - RADIUS_TOLERANCE);
        float max_r = base_r * (1 + RADIUS_TOLERANCE);
        float base_area = CV_PI * base_r * base_r;

        for (const auto& cnt : segmented_contours)
        {
            double area = contourArea(cnt);
            Point2f center;
            float radius;
            minEnclosingCircle(cnt, center, radius);
            double perimeter = arcLength(cnt, true);
            double circularity = perimeter > 0 ? (4 * CV_PI * area) / (perimeter * perimeter) : 0;

            // 筛选条件：半径在容差内+圆度达标+面积不小于基准
            bool radius_ok = (radius >= min_r && radius <= max_r);
            bool circularity_ok = (circularity >= MIN_CIRCULARITY);
            bool area_ok = (area >= base_area * MIN_AREA_RATIO);

            if (radius_ok && circularity_ok && area_ok)
            {
                valid_balls.push_back(cnt);
            }
        }
    }
    else
    {
        // 掩膜模式：按面积+圆度筛选（无基准时使用）
        for (const auto& cnt : segmented_contours)
        {
            double area = contourArea(cnt);
            double perimeter = arcLength(cnt, true);
            double circularity = perimeter > 0 ? (4 * CV_PI * area) / (perimeter * perimeter) : 0;

            bool area_ok = (area >= MASK_MIN_AREA && area <= MASK_MAX_AREA);
            bool circularity_ok = (circularity >= MIN_CIRCULARITY);

            if (area_ok && circularity_ok)
            {
                valid_balls.push_back(cnt);
            }
        }
    }

    // 去重：避免分割后残留重复轮廓
    vector<vector<Point>> final_balls;
    vector<Point2f> centers;
    for (const auto& cnt : valid_balls)
    {
        Point2f center = getContourCenter(cnt);
        float radius;
        minEnclosingCircle(cnt, center, radius);
        bool is_duplicate = false;

        for (const auto& existing_center : centers)
        {
            if (norm(center - existing_center) < (radius * DUPLICATE_DISTANCE))
            {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate)
        {
            final_balls.push_back(cnt);
            centers.push_back(center);
        }
    }

    valid_balls = final_balls;
    return valid_balls.size();
}

// ===================== 保存中间结果 ========================================
void saveIntermediateResults(const Mat& mask, const Mat& dist_norm, const string& mask_path, const string& dist_path)
{
    // 保存颜色分割后的掩膜
    imwrite(mask_path, mask);
    cout << "颜色分割掩膜已保存：" << mask_path << endl;

    // 保存距离变换图
    Mat dist_save;
    normalize(dist_norm, dist_save, 0, 255, NORM_MINMAX, CV_8UC1);
    imwrite(dist_path, dist_save);
    cout << "距离变换图已保存：" << dist_path << endl;
}

// ===================== 保存最终结果 =========================================
void saveFinalResult(const Mat& bgr, const vector<vector<Point>>& valid_balls, int count, const string& save_path, const BaseParam& base)
{
    Mat final_result = bgr.clone();

    // 绘制掩膜边界
    Mat mask_8u = Mat::zeros(bgr.size(), CV_8UC1);
    for (const auto& ball : valid_balls)
    {
        drawContours(mask_8u, vector<vector<Point>>{ball}, -1, Scalar(255), -1);
    }
    vector<vector<Point>> mask_contours;
    vector<Vec4i> hierarchy;
    findContours(mask_8u, mask_contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(final_result, mask_contours, -1, Scalar(255, 0, 0), 1);

    // 标注每个有效弹丸
    for (const auto& ball : valid_balls)
    {
        Point2f center;
        float radius;
        minEnclosingCircle(ball, center, radius);
        circle(final_result, center, (int)radius, Scalar(0, 0, 255), 2);  // 红色外接圆
        circle(final_result, center, 3, Scalar(255, 0, 0), -1);           // 蓝色圆心
        drawContours(final_result, vector<vector<Point>>{ball}, -1, Scalar(0, 255, 255), 1);  // 黄色轮廓
    }

    // 标注计数结果
    string count_text = "Ball Count: " + to_string(count);
    putText(final_result, count_text, Point(30, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 4);

    // 保存最终结果
    imwrite(save_path, final_result);
    cout << "最终标注结果已保存：" << save_path << endl;

    // 显示结果
    imshow("Final Result: " + save_path, final_result);
    waitKey(0);
}

// ===================== 主函数 ======================================================================================================
int main()
{
    for (int i = 0; i < TASK_FILES.size(); i++)
    {
        // 拼接文件路径
        string img_name = TASK_FILES[i];
        string img_path = IMG_INPUT_DIR + img_name;
        string mask_path = INTERMEDIATE_DIR + MASK_FILES[i];
        string dist_path = INTERMEDIATE_DIR + DIST_FILES[i];
        string watershed_mask_path = INTERMEDIATE_DIR + WATERSHED_MASK_FILES[i];
        string result_path = RESULT_FILES_DIR + RESULT_FILES[i];

        cout << "\n========================================================================================" << endl;
        cout << "正在处理：" << img_name << "（路径：" << img_path << "）" << endl;

        // 1.读取输入图片
        Mat bgr = imread(img_path);
        if (bgr.empty())
        {
            cerr << "无法读取图片 " << img_name << "！  检查图片路径!" << endl;
            continue;
        }

        // 2.颜色分割+形态学优化+层级过滤
        Mat mask = colorSegmentation(bgr);

        // 3.距离变换+极小值区域检测
        Mat dist_norm;
        Mat dist_thresh = generateDistanceTransform(mask, dist_norm);

        // 4.保存中间结果
        saveIntermediateResults(mask, dist_norm, mask_path, dist_path);

        // 5.校准基准弹丸
        BaseParam base = calibrateBaseRadiusPerImage(bgr, mask, img_name);

        // 6.分水岭分割
        Mat watershed_mask;
        vector<vector<Point>> segmented_contours = watershedSegmentation(
            bgr, mask, dist_thresh, watershed_mask,
            base.valid ? base.radius : 0
        );
        cout << "[" << img_name << "] 分水岭分割得到 " << segmented_contours.size() << " 个轮廓" << endl;

        // 保存分水岭后新掩膜
        imwrite(watershed_mask_path, watershed_mask);
        cout << "分水岭分割后掩膜已保存：" << watershed_mask_path << endl;

        // 7.筛选有效弹丸+去重+计数
        vector<vector<Point>> valid_balls;
        int ball_count = countValidBalls(segmented_contours, base, valid_balls);
        cout << "[" << img_name << "] 有效弹丸数量：" << ball_count << endl;

        // 8.保存最终标注结果
        saveFinalResult(bgr, valid_balls, ball_count, result_path, base);
    }

    // 9.关闭所有窗口
    destroyAllWindows();

    cout << "\n====================================================" << endl;
    cout << "所有任务处理完成!!!!" << endl;
    cout << "中间结果在 " << INTERMEDIATE_DIR << "（含3类文件：颜色分割掩膜、距离变换图、分水岭分割掩膜）" << endl;
    cout << "最终结果存储在 " << RESULT_FILES_DIR << "（含标注后的图片，蓝色细框为掩膜边界）" << endl;

    //10. 完成  ฅ՞•ﻌ•՞ฅ

    return 0;
}

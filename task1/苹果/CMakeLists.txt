cmake_minimum_required(VERSION 3.10)  # 最低CMake版本要求
project(AppleDetection)  # 项目名称
# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 查找OpenCV库
find_package(OpenCV REQUIRED)
# 指定源文件
add_executable(apple_detection apple_detection.cpp)
# 链接OpenCV库到可执行文件
target_link_libraries(apple_detection ${OpenCV_LIBS})

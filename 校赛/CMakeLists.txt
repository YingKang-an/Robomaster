# 最低CMake版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(ball_detection)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 打印OpenCV版本
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# 生成可执行文件 "main"
add_executable(main main.cpp)
# 链接OpenCV库到main
target_link_libraries(main ${OpenCV_LIBS})

# 生成可执行文件 "main"
add_executable(1 1.cpp)
# 链接OpenCV库到main
target_link_libraries(1 ${OpenCV_LIBS})


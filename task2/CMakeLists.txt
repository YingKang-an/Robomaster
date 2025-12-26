# 设置cmake最低版本
cmake_minimum_required(VERSION 3.15)
# project命令用于指定cmake工程的名称
project(text)
# 查找依赖包
find_package(OpenCV 4.5.0)
# 将.cpp/.c/.cc文件生成可执行文件
add_executable(text text.cpp)
# # 为target连接库的头文件路径（被link库生成CMakeLists.txt中定义的头文件路径）和对应定义的函数库路径
target_link_libraries(text ${OpenCV_LIBRARIES})

# 将.cpp/.c/.cc文件生成可执行文件
add_executable(main main.cpp)
# # 为target连接库的头文件路径（被link库生成CMakeLists.txt中定义的头文件路径）和对应定义的函数库路径
target_link_libraries(main ${OpenCV_LIBRARIES})
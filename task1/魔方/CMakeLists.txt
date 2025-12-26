#指定编译该项目所需的最低 CMake 版本
cmake_minimum_required(VERSION 3.10)
#定义项目名称
project(1)   #项目名称会影响后续生成的编译文件（比如 Makefile 中的变量）
#指定编译项目时使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD 11)
#作用：强制要求编译器必须支持上面指定的 C++11 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#查找系统中安装的 OpenCV 库
find_package(OpenCV REQUIRED)
#告诉编译器 去哪里找 OpenCV 的头文件
include_directories(${OpenCV_INCLUDE_DIRS})
#定义 要生成的可执行文件 和 对应的源文件
add_executable(main main.cpp)
#定义 要生成的可执行文件 和 对应的源文件
target_link_libraries(main ${OpenCV_LIBS})

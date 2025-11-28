cmake_minimum_required(VERSION 3.8...3.10)
project(teamX_challenge)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(referee_pkg REQUIRED)
find_package(sensor_msgs REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-deprecated-declarations)
endif()

# 手动设置 ONNX Runtime
find_library(ONNXRUNTIME_LIB 
    NAMES onnxruntime
    PATHS /usr/local/lib /usr/lib/x86_64-linux-gnu
    REQUIRED
)

# 查找 ONNX Runtime 头文件
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS /usr/local/include /usr/include
    REQUIRED
)

message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")
message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")

add_executable(vision_node src/vision_node.cpp)

# 添加包含目录
target_include_directories(vision_node PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})

ament_target_dependencies(vision_node
  rclcpp
  std_msgs
  OpenCV 
  geometry_msgs
  cv_bridge
  referee_pkg
  sensor_msgs
)

# 链接库
target_link_libraries(vision_node 
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB}
)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 安装目标
install(TARGETS vision_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装启动文件
install(DIRECTORY 
  launch
  DESTINATION share/${PROJECT_NAME}
)

# 安装配置文件
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
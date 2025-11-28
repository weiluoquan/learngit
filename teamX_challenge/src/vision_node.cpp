#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/object.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <numeric>
#include <map>

using namespace std;
using namespace rclcpp;
using namespace cv;

// YOLOv5 检测结果结构体
struct YOLODetection {
    cv::Rect bbox;
    float confidence;
    int class_id;
};

// YOLOv5 检测器类
class YOLOv5Detector {
public:
    YOLOv5Detector(const std::string& model_path, float conf_threshold = 0.25f, float iou_threshold = 0.45f);
    ~YOLOv5Detector();
    
    std::vector<YOLODetection> detect(const cv::Mat& image);
    void drawDetections(cv::Mat& image, const std::vector<YOLODetection>& detections);
    std::string getClassName(int class_id);

private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    
    std::vector<int64_t> input_shape;
    size_t input_tensor_size;
    
    float conf_threshold;
    float iou_threshold;
    
    cv::Size2f scale;
    cv::Point2f pad;
    
    // 类别名称映射
    std::map<int, std::string> class_names;
    
    cv::Mat preprocess(const cv::Mat& image);
    std::vector<YOLODetection> postprocess(const std::vector<float>& output, const cv::Size& original_shape);
    std::vector<int> nonMaxSuppression(const std::vector<cv::Rect>& boxes, 
                                     const std::vector<float>& scores, 
                                     float iou_threshold);
};

YOLOv5Detector::YOLOv5Detector(const std::string& model_path, float conf_threshold, float iou_threshold)
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5"),
      session(env, model_path.c_str(), Ort::SessionOptions{nullptr}),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
      conf_threshold(conf_threshold), iou_threshold(iou_threshold) {
    
    // 初始化类别名称映射 - 装甲板数字1-5
    class_names = {
        {0, "1"}, {1, "2"}, {2, "3"}, {3, "4"}, {4, "5"}
    };
    
    // 手动设置输入输出名称
    input_names.push_back("images");
    output_names.push_back("output0");
    
    // 获取输入形状
    size_t num_input_nodes = session.GetInputCount();
    for(size_t i = 0; i < num_input_nodes; i++) {
        auto input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        this->input_shape = input_shape;
    }
    
    input_tensor_size = input_shape[1] * input_shape[2] * input_shape[3];
    
    std::cout << "YOLOv5 Model loaded successfully!" << std::endl;
    std::cout << "Input shape: [" << input_shape[0] << ", " << input_shape[1] << ", " 
              << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;
}

YOLOv5Detector::~YOLOv5Detector() {
    // Ort会自动清理资源
}

std::string YOLOv5Detector::getClassName(int class_id) {
    auto it = class_names.find(class_id);
    if (it != class_names.end()) {
        return it->second;
    }
    return "Unknown";
}

cv::Mat YOLOv5Detector::preprocess(const cv::Mat& image) {
    int input_width = input_shape[3];
    int input_height = input_shape[2];
    
    // 获取原始图像尺寸
    int original_width = image.cols;
    int original_height = image.rows;
    
    // Letterbox处理
    float r = std::min(static_cast<float>(input_width) / original_width, 
                      static_cast<float>(input_height) / original_height);
    
    int new_width = static_cast<int>(std::round(original_width * r));
    int new_height = static_cast<int>(std::round(original_height * r));
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    
    // 计算填充
    int dw = input_width - new_width;
    int dh = input_height - new_height;
    dw /= 2;
    dh /= 2;
    
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, dh, input_height - new_height - dh,
                      dw, input_width - new_width - dw, 
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // 保存缩放和填充信息用于后处理
    scale = cv::Size2f(r, r);
    pad = cv::Point2f(dw, dh);
    
    // 转换颜色空间 BGR -> RGB
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    
    // 归一化到 [0, 1]
    cv::Mat float_img;
    padded.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    return float_img;
}

std::vector<YOLODetection> YOLOv5Detector::detect(const cv::Mat& image) {
    // 预处理
    cv::Mat processed = preprocess(image);
    
    // 创建输入张量 - 注意需要将HWC转换为CHW格式
    std::vector<float> input_tensor_values(input_tensor_size);
    
    // 将图像数据从HWC转换为CHW格式
    int channel_length = input_shape[2] * input_shape[3];
    float* input_data = input_tensor_values.data();
    
    // 更高效的CHW转换
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);
    
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data + c * channel_length, channels[c].data, channel_length * sizeof(float));
    }
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    ));
    
    // 推理
    try {
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                        input_names.data(), input_tensors.data(), input_tensors.size(),
                                        output_names.data(), output_names.size());
        
        // 获取输出
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
        std::vector<float> output(output_data, output_data + output_size);
        
        // 后处理
        return postprocess(output, image.size());
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<YOLODetection> YOLOv5Detector::postprocess(const std::vector<float>& output, const cv::Size& original_shape) {
    std::vector<YOLODetection> detections;
    
    // 根据模型输出形状 [1, 25200, 10]，你的模型有5个类别
    const int num_classes = 5;
    const int elements_per_detection = 5 + num_classes;
    
    for (int i = 0; i < 25200; ++i) {
        int base_index = i * elements_per_detection;
        float confidence = output[base_index + 4];
        
        if (confidence < conf_threshold) continue;
        
        // 找到最大概率的类别
        int class_id = -1;
        float max_class_prob = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float class_prob = output[base_index + 5 + j];
            if (class_prob > max_class_prob) {
                max_class_prob = class_prob;
                class_id = j;
            }
        }
        
        float final_confidence = confidence * max_class_prob;
        if (final_confidence < conf_threshold) continue;
        
        // 解析边界框
        float x_center = output[base_index];
        float y_center = output[base_index + 1];
        float width = output[base_index + 2];
        float height = output[base_index + 3];
        
        // 转换为角点坐标
        float x1 = x_center - width / 2;
        float y1 = y_center - height / 2;
        float x2 = x_center + width / 2;
        float y2 = y_center + height / 2;
        
        // 反letterbox变换
        x1 = (x1 - pad.x) / scale.width;
        y1 = (y1 - pad.y) / scale.height;
        x2 = (x2 - pad.x) / scale.width;
        y2 = (y2 - pad.y) / scale.height;
        
        // 确保在图像范围内
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        x2 = std::min(x2, static_cast<float>(original_shape.width));
        y2 = std::min(y2, static_cast<float>(original_shape.height));
        
        float box_width = x2 - x1;
        float box_height = y2 - y1;
        
        if (box_width <= 0 || box_height <= 0) continue;
        
        cv::Rect bbox(static_cast<int>(x1), static_cast<int>(y1), 
                     static_cast<int>(box_width), static_cast<int>(box_height));
        
        detections.push_back({bbox, final_confidence, class_id});
    }
    
    // NMS
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (const auto& det : detections) {
        boxes.push_back(det.bbox);
        scores.push_back(det.confidence);
    }
    
    std::vector<int> indices = nonMaxSuppression(boxes, scores, iou_threshold);
    
    std::vector<YOLODetection> final_detections;
    for (int idx : indices) {
        final_detections.push_back(detections[idx]);
    }
    
    return final_detections;
}

std::vector<int> YOLOv5Detector::nonMaxSuppression(const std::vector<cv::Rect>& boxes, 
                                                 const std::vector<float>& scores, 
                                                 float iou_threshold) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // 按置信度排序
    std::sort(indices.begin(), indices.end(), 
              [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });
    
    std::vector<int> picked;
    
    while (!indices.empty()) {
        int current = indices[0];
        picked.push_back(current);
        
        std::vector<int> remaining;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            
            // 计算IoU
            cv::Rect box1 = boxes[current];
            cv::Rect box2 = boxes[idx];
            
            int x1 = std::max(box1.x, box2.x);
            int y1 = std::max(box1.y, box2.y);
            int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
            int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
            
            int intersection_area = std::max(0, x2 - x1) * std::max(0, y2 - y1);
            int union_area = box1.area() + box2.area() - intersection_area;
            
            float iou = (union_area > 0) ? static_cast<float>(intersection_area) / union_area : 0.0f;
            
            if (iou <= iou_threshold) {
                remaining.push_back(idx);
            }
        }
        
        indices = remaining;
    }
    
    return picked;
}

class VisionNode : public rclcpp::Node {
public:
    VisionNode(string name) : Node(name) {
        RCLCPP_INFO(this->get_logger(), "Initializing VisionNode");

        // 一次性声明所有参数
        declare_parameters();
        
        // 获取参数值
        std::string camera_topic = this->get_parameter("camera_topic").as_string();
        std::string output_topic = this->get_parameter("output_topic").as_string();
        show_debug_window_ = this->get_parameter("show_debug_window").as_bool();
        debug_window_name_ = this->get_parameter("debug_window_name").as_string();

        // 初始化YOLOv5检测器
        initializeYOLODetector();

        Image_sub = this->create_subscription<sensor_msgs::msg::Image>(
            camera_topic, 10,
            bind(&VisionNode::callback_camera, this, std::placeholders::_1));

        Target_pub = this->create_publisher<referee_pkg::msg::MultiObject>(
            output_topic, 10);

        if (show_debug_window_) {
            cv::namedWindow(debug_window_name_, cv::WINDOW_AUTOSIZE);
        }

        RCLCPP_INFO(this->get_logger(), "VisionNode initialized successfully");
        RCLCPP_INFO(this->get_logger(), "Camera topic: %s", camera_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Output topic: %s", output_topic.c_str());
    }

    ~VisionNode() { 
        if (show_debug_window_) {
            cv::destroyWindow(debug_window_name_); 
        }
    }

private:
    // 参数声明函数 - 一次性声明所有参数
    void declare_parameters() {
        // 话题参数
        this->declare_parameter<string>("camera_topic", "/camera/image_raw");
        this->declare_parameter<string>("output_topic", "/vision/target");
        this->declare_parameter<bool>("show_debug_window", true);
        this->declare_parameter<string>("debug_window_name", "Detection Result");
        
        // YOLOv5参数
        this->declare_parameter<string>("yolo_model_path", "/home/weiluoquan/Desktop/learngit/Vision_Arena_2025/src/teamX_challenge/models/best.onnx");
        this->declare_parameter<double>("yolo_confidence_threshold", 0.25);
        this->declare_parameter<double>("yolo_iou_threshold", 0.45);
        this->declare_parameter<bool>("enable_yolo_detection", true);
        
        // 灯条检测参数（在装甲板区域内）
        this->declare_parameter<double>("lightbar_min_area", 10.0);
        this->declare_parameter<double>("lightbar_max_area", 1000.0);
        this->declare_parameter<double>("lightbar_min_aspect_ratio", 2.0);
        this->declare_parameter<double>("lightbar_max_aspect_ratio", 8.0);
        
        // 球体检测参数
        this->declare_parameter<int>("sphere_min_area", 500);
        this->declare_parameter<int>("sphere_min_radius", 15);
        this->declare_parameter<int>("sphere_max_radius", 200);
        this->declare_parameter<double>("sphere_min_circularity", 0.7);
        
        // 矩形检测参数
        this->declare_parameter<int>("rect_min_area", 20);
        this->declare_parameter<int>("rect_max_area", 50000);
        this->declare_parameter<double>("rect_movement_threshold", 1.0);
        
        // 颜色阈值参数
        this->declare_parameter<std::vector<int64_t>>("red_lower1", std::vector<int64_t>{0, 50, 50});
        this->declare_parameter<std::vector<int64_t>>("red_upper1", std::vector<int64_t>{10, 255, 255});
        this->declare_parameter<std::vector<int64_t>>("red_lower2", std::vector<int64_t>{160, 100, 100});
        this->declare_parameter<std::vector<int64_t>>("red_upper2", std::vector<int64_t>{180, 255, 255});
        this->declare_parameter<std::vector<int64_t>>("cyan_lower", std::vector<int64_t>{0, 50, 50});
        this->declare_parameter<std::vector<int64_t>>("cyan_upper", std::vector<int64_t>{180, 255, 255});
        
        // 形态学操作参数
        this->declare_parameter<int>("morph_kernel_size", 3);
        this->declare_parameter<int>("morph_kernel_size_large", 5);
    }

    // 初始化YOLOv5检测器
    void initializeYOLODetector() {
        std::string model_path = this->get_parameter("yolo_model_path").as_string();
        double confidence_threshold = this->get_parameter("yolo_confidence_threshold").as_double();
        double iou_threshold = this->get_parameter("yolo_iou_threshold").as_double();
        
        try {
            yolo_detector_ = std::make_unique<YOLOv5Detector>(model_path, confidence_threshold, iou_threshold);
            RCLCPP_INFO(this->get_logger(), "YOLOv5 detector initialized successfully!");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize YOLOv5 detector: %s", e.what());
        }
    }
    
    // 灯条结构体
    struct LightBar {
        cv::RotatedRect rect;
        std::vector<cv::Point> contour;
        cv::Point topVertex;    // 上顶点
        cv::Point bottomVertex; // 下顶点
        
        LightBar(const cv::RotatedRect& r, const std::vector<cv::Point>& cnt) 
            : rect(r), contour(cnt) {
            // 通过遍历轮廓点找到上下顶点
            findVerticesFromContour();
        }
        
    private:
        // 通过遍历轮廓点找到上下顶点
        void findVerticesFromContour() {
            if (contour.empty()) return;
            
            // 初始化上下顶点
            topVertex = contour[0];
            bottomVertex = contour[0];
            
            // 遍历所有轮廓点，找到y坐标最小和最大的点
            for (const auto& point : contour) {
                // 上顶点是y坐标最小的点
                if (point.y < topVertex.y) {
                    topVertex = point;
                }
                // 下顶点是y坐标最大的点
                if (point.y > bottomVertex.y) {
                    bottomVertex = point;
                }
            }
        }
    };

    // 检测结果结构体
    struct DetectionResult {
        string target_type;
        vector<Point2f> points;
        int armor_number; // 装甲板数字编号
    };

    // YOLOv5装甲板检测函数
    vector<DetectionResult> detectYOLOArmors(Mat& image);
    
    // 在装甲板区域内检测灯条
    vector<LightBar> detectLightBarsInArmor(const cv::Mat& armor_roi, const cv::Rect& armor_bbox);
    
    // 绘制灯条顶点（按左下角开始逆时针编号）
    void drawLightBarVertices(Mat& image, const vector<LightBar>& lightBars, const cv::Rect& armor_bbox);
    
    // 对灯条顶点进行排序（左下->左上->右上->右下）
    vector<Point> sortLightBarVertices(const vector<LightBar>& lightBars, const cv::Rect& armor_bbox);
    
    // sphere球体检测相关函数 
    vector<Point2f> calculateStableSpherePoints(const Point2f &center, float radius);
    vector<DetectionResult> detectSpheres(Mat& image);
    
    // rect矩形检测相关函数 
    vector<Point2f> sortRectanglePoints(vector<Point2f>& points);
    Point2f calculateRectCenter(const vector<Point2f>& points);
    bool isRectangleMoving(const vector<Point2f>& current_points, int rect_index);
    vector<DetectionResult> detectRectangles(Mat& image);

    // 回调函数
    void callback_camera(sensor_msgs::msg::Image::SharedPtr msg);

    // 成员变量声明
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr Image_sub;
    rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr Target_pub;
    std::unique_ptr<YOLOv5Detector> yolo_detector_;
    
    // 保存上一帧的矩形中心点用于移动检测
    vector<Point2f> prev_rect_centers;
    
    // 调试窗口相关参数
    bool show_debug_window_;
    string debug_window_name_;
};

// YOLOv5装甲板检测
vector<VisionNode::DetectionResult> VisionNode::detectYOLOArmors(Mat& image) {
    vector<DetectionResult> results;
    
    if (!yolo_detector_) {
        RCLCPP_WARN(this->get_logger(), "YOLOv5 detector not initialized");
        return results;
    }
    
    bool enable_yolo = this->get_parameter("enable_yolo_detection").as_bool();
    if (!enable_yolo) {
        return results;
    }
    
    try {
        // 使用YOLOv5检测装甲板
        auto yolo_detections = yolo_detector_->detect(image);
        
        // 为每个检测到的装甲板绘制框和数字，并在其内部检测灯条
        for (const auto& det : yolo_detections) {
            // 绘制装甲板边界框
            cv::Scalar color(0, 255, 0); // 绿色
            cv::rectangle(image, det.bbox, color, 2);
            
            // 获取类别名称
            std::string class_name = yolo_detector_->getClassName(det.class_id);
            
            // 在左上角显示装甲板数字
            std::string label = "Armor " + class_name;
            int base_line;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base_line);
            
            cv::Point label_origin(det.bbox.x, det.bbox.y - 10);
            if (label_origin.y < label_size.height) {
                label_origin.y = det.bbox.y + label_size.height;
            }
            
            // 绘制标签背景
            cv::rectangle(image, 
                         cv::Point(label_origin.x, label_origin.y - label_size.height - base_line),
                         cv::Point(label_origin.x + label_size.width, label_origin.y + base_line),
                         color, cv::FILLED);
            
            // 绘制标签文本
            cv::putText(image, label,
                       cv::Point(label_origin.x, label_origin.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            
            // 在装甲板区域内检测灯条
            vector<LightBar> lightBars = detectLightBarsInArmor(image, det.bbox);
            
            // 绘制灯条顶点（按左下角开始逆时针编号）
            drawLightBarVertices(image, lightBars, det.bbox);
            
            // 准备检测结果
            DetectionResult result;
            result.target_type = "armor_red_" + class_name;
            result.armor_number = det.class_id + 1;
            
            // 装甲板的四个角点
            cv::Rect bbox = det.bbox;
            result.points.push_back(Point2f(bbox.x, bbox.y)); // 左上
            result.points.push_back(Point2f(bbox.x + bbox.width, bbox.y)); // 右上
            result.points.push_back(Point2f(bbox.x + bbox.width, bbox.y + bbox.height)); // 右下
            result.points.push_back(Point2f(bbox.x, bbox.y + bbox.height)); // 左下
            
            results.push_back(result);
            
            RCLCPP_INFO(this->get_logger(), "YOLO detected armor %s with confidence %.2f", 
                       class_name.c_str(), det.confidence);
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "YOLOv5 detection error: %s", e.what());
    }
    
    return results;
}

// 在装甲板区域内检测灯条
vector<VisionNode::LightBar> VisionNode::detectLightBarsInArmor(const cv::Mat& image, const cv::Rect& armor_bbox) {
    vector<LightBar> lightBars;
    
    // 提取装甲板区域
    cv::Mat armor_roi = image(armor_bbox).clone();
    
    // 获取颜色阈值参数
    auto red_lower1 = this->get_parameter("red_lower1").as_integer_array();
    auto red_upper1 = this->get_parameter("red_upper1").as_integer_array();
    auto red_lower2 = this->get_parameter("red_lower2").as_integer_array();
    auto red_upper2 = this->get_parameter("red_upper2").as_integer_array();
    int morph_kernel_size = this->get_parameter("morph_kernel_size").as_int();
    
    // 转换为HSV颜色空间进行颜色过滤
    Mat hsv;
    cvtColor(armor_roi, hsv, COLOR_BGR2HSV);
    
    // 红色装甲板（考虑红色的两个区间）
    Mat red_mask1, red_mask2;
    inRange(hsv, 
            Scalar(red_lower1[0], red_lower1[1], red_lower1[2]), 
            Scalar(red_upper1[0], red_upper1[1], red_upper1[2]), 
            red_mask1);
    inRange(hsv, 
            Scalar(red_lower2[0], red_lower2[1], red_lower2[2]), 
            Scalar(red_upper2[0], red_upper2[1], red_upper2[2]), 
            red_mask2);
    Mat color_mask = red_mask1 | red_mask2;
    
    // 形态学操作去噪
    Mat kernel = getStructuringElement(MORPH_RECT, Size(morph_kernel_size, morph_kernel_size));
    morphologyEx(color_mask, color_mask, MORPH_OPEN, kernel);
    morphologyEx(color_mask, color_mask, MORPH_CLOSE, kernel);
    
    // 找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(color_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 获取灯条检测参数
    double min_area = this->get_parameter("lightbar_min_area").as_double();
    double max_area = this->get_parameter("lightbar_max_area").as_double();
    double min_aspect_ratio = this->get_parameter("lightbar_min_aspect_ratio").as_double();
    double max_aspect_ratio = this->get_parameter("lightbar_max_aspect_ratio").as_double();
    
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;
        
        double area = contourArea(contour);
        if (area < min_area || area > max_area) continue;
        
        RotatedRect rect = minAreaRect(contour);
        double length = max(rect.size.width, rect.size.height);
        double width = min(rect.size.width, rect.size.height);
        double aspect_ratio = length / width;
        
        if (aspect_ratio < min_aspect_ratio || aspect_ratio > max_aspect_ratio) continue;
        
        LightBar lightBar(rect, contour);
        lightBars.push_back(lightBar);
    }
    
    return lightBars;
}

// 对灯条顶点进行排序（左下->左上->右上->右下）
vector<Point> VisionNode::sortLightBarVertices(const vector<LightBar>& lightBars, const cv::Rect& armor_bbox) {
    vector<Point> allVertices;
    
    // 收集所有灯条的顶点（转换到全局坐标系）
    for (const auto& lightBar : lightBars) {
        Point topVertex_global(lightBar.topVertex.x + armor_bbox.x, lightBar.topVertex.y + armor_bbox.y);
        Point bottomVertex_global(lightBar.bottomVertex.x + armor_bbox.x, lightBar.bottomVertex.y + armor_bbox.y);
        allVertices.push_back(topVertex_global);
        allVertices.push_back(bottomVertex_global);
    }
    
    // 如果顶点数量不是4个，返回空向量
    if (allVertices.size() != 4) {
        return vector<Point>();
    }
    
    // 计算中心点
    Point center(0, 0);
    for (const auto& vertex : allVertices) {
        center.x += vertex.x;
        center.y += vertex.y;
    }
    center.x /= 4;
    center.y /= 4;
    
    // 将顶点分为四个象限并排序
    vector<Point> sortedVertices;
    
    // 左下象限 (x < center.x, y > center.y)
    vector<Point> bottomLeft;
    // 右下象限 (x >= center.x, y > center.y)
    vector<Point> bottomRight;
    // 右上象限 (x >= center.x, y <= center.y)
    vector<Point> topRight;
    // 左上象限 (x < center.x, y <= center.y)
    vector<Point> topLeft;
    
    for (const auto& vertex : allVertices) {
        if (vertex.x < center.x && vertex.y > center.y) {
            bottomLeft.push_back(vertex);
        } else if (vertex.x >= center.x && vertex.y > center.y) {
            bottomRight.push_back(vertex);
        } else if (vertex.x >= center.x && vertex.y <= center.y) {
            topRight.push_back(vertex);
        } else { // vertex.x < center.x && vertex.y <= center.y
            topLeft.push_back(vertex);
        }
    }
    
    // 对每个象限内的点进行排序
    // 左下象限：按y从大到小，然后x从小到大
    sort(bottomLeft.begin(), bottomLeft.end(),
         [](const Point& a, const Point& b) {
             return (a.y > b.y) || (a.y == b.y && a.x < b.x);
         });
    
    // 右下象限：按x从小到大，然后y从大到小
    sort(bottomRight.begin(), bottomRight.end(),
         [](const Point& a, const Point& b) {
             return (a.x < b.x) || (a.x == b.x && a.y > b.y);
         });
    
    // 右上象限：按y从小到大，然后x从大到小
    sort(topRight.begin(), topRight.end(),
         [](const Point& a, const Point& b) {
             return (a.y < b.y) || (a.y == b.y && a.x > b.x);
         });
    
    // 左上象限：按x从大到小，然后y从小到大
    sort(topLeft.begin(), topLeft.end(),
         [](const Point& a, const Point& b) {
             return (a.x > b.x) || (a.x == b.x && a.y < b.y);
         });
    
    // 合并四个象限，按照左下→左上→右上→右下的顺序
    if (!bottomLeft.empty()) sortedVertices.push_back(bottomLeft[0]);
    if (!topLeft.empty()) sortedVertices.push_back(topLeft[0]);
    if (!topRight.empty()) sortedVertices.push_back(topRight[0]);
    if (!bottomRight.empty()) sortedVertices.push_back(bottomRight[0]);
    
    return sortedVertices;
}

// 绘制灯条顶点（按左下角开始逆时针编号）
void VisionNode::drawLightBarVertices(Mat& image, const vector<LightBar>& lightBars, const cv::Rect& armor_bbox) {
    // 对灯条顶点进行排序
    vector<Point> sortedVertices = sortLightBarVertices(lightBars, armor_bbox);
    
    // 如果排序后的顶点数量不是4个，不进行绘制
    if (sortedVertices.size() != 4) {
        return;
    }
    
    Scalar vertexColor(0, 165, 255); // 橙色
    
    // 绘制顶点并标注序号（从1开始）
    for (int i = 0; i < sortedVertices.size(); i++) {
        // 用橙色圆点标记顶点
        circle(image, sortedVertices[i], 6, vertexColor, -1);
        
        // 标记序号（从1开始）
        string text = to_string(i + 1);
        
        // 在顶点位置绘制序号，稍微偏移避免覆盖点
        Point textPos = sortedVertices[i] + Point(8, -8);
        putText(image, text, textPos, FONT_HERSHEY_SIMPLEX, 0.6, vertexColor, 2);
    }
}

// 球体检测和矩形检测函数保持不变
vector<Point2f> VisionNode::calculateStableSpherePoints(const Point2f &center, float radius) {
    vector<Point2f> points;

    // 简单稳定的几何计算，避免漂移
    // 左、下、右、上
    points.push_back(Point2f(center.x - radius, center.y));  // 左点 (1)
    points.push_back(Point2f(center.x, center.y + radius));  // 下点 (2)
    points.push_back(Point2f(center.x + radius, center.y));  // 右点 (3)
    points.push_back(Point2f(center.x, center.y - radius));  // 上点 (4)

    return points;
}

vector<VisionNode::DetectionResult> VisionNode::detectSpheres(Mat& image) {
    vector<DetectionResult> results;
    
    // 获取球体检测参数
    int sphere_min_area = this->get_parameter("sphere_min_area").as_int();
    int sphere_min_radius = this->get_parameter("sphere_min_radius").as_int();
    int sphere_max_radius = this->get_parameter("sphere_max_radius").as_int();
    double sphere_min_circularity = this->get_parameter("sphere_min_circularity").as_double();
    int morph_kernel_size_large = this->get_parameter("morph_kernel_size_large").as_int();
    
    // 转换到 HSV 空间
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // 红色检测 - 使用稳定的范围
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255), mask2);
    mask = mask1 | mask2;

    // 适度的形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_kernel_size_large, morph_kernel_size_large));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int valid_spheres = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < sphere_min_area) continue;

        // 计算最小外接圆
        Point2f center;
        float radius = 0;
        minEnclosingCircle(contours[i], center, radius);

        // 计算圆形度
        double perimeter = arcLength(contours[i], true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);

        if (circularity > sphere_min_circularity && radius > sphere_min_radius && radius < sphere_max_radius) {
            vector<Point2f> sphere_points = calculateStableSpherePoints(center, radius);

            // 绘制检测到的球体
            cv::circle(image, center, static_cast<int>(radius), cv::Scalar(0, 255, 0), 2);  // 绿色圆圈
            cv::circle(image, center, 3, cv::Scalar(0, 0, 255), -1);  // 红色圆心

            // 绘制球体上的四个点
            vector<string> point_names = {"左", "下", "右", "上"};
            vector<cv::Scalar> point_colors = {
                cv::Scalar(255, 0, 0),    // 蓝色 - 左
                cv::Scalar(0, 255, 0),    // 绿色 - 下
                cv::Scalar(0, 255, 255),  // 黄色 - 右
                cv::Scalar(255, 0, 255)   // 紫色 - 上
            };

            DetectionResult result;
            result.target_type = "sphere";
            result.armor_number = -1;
            
            for (int j = 0; j < 4; j++) {
                cv::circle(image, sphere_points[j], 6, point_colors[j], -1);
                cv::circle(image, sphere_points[j], 6, cv::Scalar(0, 0, 0), 2);

                // 标注序号
                string point_text = to_string(j + 1);
                cv::putText(image, point_text,
                          cv::Point(sphere_points[j].x + 10, sphere_points[j].y - 10),
                          cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
                cv::putText(image, point_text,
                          cv::Point(sphere_points[j].x + 10, sphere_points[j].y - 10),
                          cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);

                // 添加到检测结果
                result.points.push_back(sphere_points[j]);

                RCLCPP_INFO(this->get_logger(),
                          "Sphere %d, Point %d (%s): (%.1f, %.1f)",
                          valid_spheres + 1, j + 1, point_names[j].c_str(),
                          sphere_points[j].x, sphere_points[j].y);
            }

            // 显示半径信息
            string info_text = "R:" + to_string((int)radius);
            cv::putText(image, info_text, cv::Point(center.x - 15, center.y + 5),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

            results.push_back(result);
            valid_spheres++;
            
            RCLCPP_INFO(this->get_logger(),
                      "Found sphere: (%.1f, %.1f) R=%.1f C=%.3f", center.x,
                      center.y, radius, circularity);
        }
    }
    
    return results;
}

struct PointAngle {
    Point2f pt;
    double angle;
};  // 包含点坐标和与中心连线的夹角

vector<Point2f> VisionNode::sortRectanglePoints(vector<Point2f>& points) {
    if (points.size() != 4) return points;

    // 计算中心点
    Point2f center(0, 0);
    for (const auto& p : points) {
        center += p;
    }
    center.x /= 4;
    center.y /= 4;

    // 计算每个点相对于中心点的角度
    vector<PointAngle> angle_points;
    for (const auto& p : points) {
        double angle = atan2(p.y - center.y, p.x - center.x);
        angle_points.push_back({p, angle});
    }

    // 按角度排序
    sort(angle_points.begin(), angle_points.end(),
         [](const PointAngle& a, const PointAngle& b) {
             return a.angle > b.angle;
         });

    // 提取排序后的点
    vector<Point2f> sorted_points;
    for (const auto& pa : angle_points) {
        sorted_points.push_back(pa.pt);
    }

    // 找左下角点（y最大且x最小）
    int left_bottom_idx = 0;
    for (int i = 1; i < 4; i++) {
        if (sorted_points[i].y > sorted_points[left_bottom_idx].y ||
           (sorted_points[i].y == sorted_points[left_bottom_idx].y && sorted_points[i].x < sorted_points[left_bottom_idx].x)) {
            left_bottom_idx = i;
        }
    }

    // 旋转数组，使左下角在第0位
    vector<Point2f> result(4);
    for (int i = 0; i < 4; i++) {
        result[i] = sorted_points[(left_bottom_idx + i) % 4];
    }

    return result;
}

Point2f VisionNode::calculateRectCenter(const vector<Point2f>& points) {
    Point2f center(0, 0);
    for (const auto& p : points) {
        center += p;
    }
    center.x /= points.size();
    center.y /= points.size();
    return center;
}

bool VisionNode::isRectangleMoving(const vector<Point2f>& current_points, int rect_index) {
    double movement_threshold = this->get_parameter("rect_movement_threshold").as_double();
    
    if (prev_rect_centers.size() <= rect_index) {
        // 如果没有上一帧的数据，默认不移动
        return false;
    }
    
    Point2f current_center = calculateRectCenter(current_points);
    Point2f prev_center = prev_rect_centers[rect_index];
    
    // 计算中心点移动距离
    double distance = norm(current_center - prev_center);
    
    return distance > movement_threshold;
}

vector<VisionNode::DetectionResult> VisionNode::detectRectangles(Mat& image) {
    vector<DetectionResult> results;
    
    // 获取矩形检测参数
    int rect_min_area = this->get_parameter("rect_min_area").as_int();
    int rect_max_area = this->get_parameter("rect_max_area").as_int();
    int morph_kernel_size_large = this->get_parameter("morph_kernel_size_large").as_int();
    
    // 转换到 HSV 空间
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // 青色检测 - HSV范围
    cv::Mat cyan_mask;
    auto cyan_lower = this->get_parameter("cyan_lower").as_integer_array();
    auto cyan_upper = this->get_parameter("cyan_upper").as_integer_array();
    
    cv::inRange(hsv, 
                cv::Scalar(cyan_lower[0], cyan_lower[1], cyan_lower[2]), 
                cv::Scalar(cyan_upper[0], cyan_upper[1], cyan_upper[2]), 
                cyan_mask);
    
    // 形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_kernel_size_large, morph_kernel_size_large));
    cv::morphologyEx(cyan_mask, cyan_mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(cyan_mask, cyan_mask, cv::MORPH_OPEN, kernel);

    // 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cyan_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int valid_rectangles = 0;
    
    // 存储当前帧的矩形中心点
    vector<Point2f> current_centers;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < rect_min_area || area > rect_max_area) continue;

        // 多边形逼近
        std::vector<cv::Point> approx;
        double epsilon = 0.01 * cv::arcLength(contours[i], true);
        cv::approxPolyDP(contours[i], approx, epsilon, true);

        // 检查是否是四边形
        if (approx.size() == 4) {
            // 转换为Point2f
            std::vector<cv::Point2f> rect_points;
            for (const auto& p : approx) {
                rect_points.push_back(cv::Point2f(p.x, p.y));
            }

            // 排序顶点：从左上角开始逆时针
            std::vector<cv::Point2f> sorted_points = sortRectanglePoints(rect_points);

            // 计算当前矩形中心并保存
            Point2f current_center = calculateRectCenter(sorted_points);
            current_centers.push_back(current_center);

            // 检测矩形是否移动
            bool is_moving = isRectangleMoving(sorted_points, valid_rectangles);
            string rect_type = is_moving ? "rect_move" : "rect";
            
            // 根据移动状态选择颜色
            cv::Scalar rect_color = is_moving ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0); // 移动为红色，静止为绿色

            // 绘制矩形边框
            for (int j = 0; j < 4; j++) {
                cv::line(image, sorted_points[j], sorted_points[(j + 1) % 4],
                       rect_color, 3);
            }

            // 绘制四个顶点并标注数字
            vector<string> point_name = {"左下", "右下", "右上", "左上"};
            vector<cv::Scalar> point_colors = {
                cv::Scalar(255, 0, 0),    // 蓝色
                cv::Scalar(0, 255, 0),    // 绿色
                cv::Scalar(0, 255, 255),  // 黄色
                cv::Scalar(255, 0, 255)   // 紫色
            };

            DetectionResult result;
            result.target_type = rect_type;
            result.armor_number = -1;
            
            for (int j = 0; j < 4; j++) {
                cv::circle(image, sorted_points[j], 6, point_colors[j], -1);
                cv::circle(image, sorted_points[j], 6, cv::Scalar(0, 0, 0), 2);

                // 标注序号
                string point_text = to_string(j + 1);
                cv::putText(image, point_text,
                          cv::Point(sorted_points[j].x + 10, sorted_points[j].y - 12),
                          cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
                cv::putText(image, point_text,
                          cv::Point(sorted_points[j].x + 10, sorted_points[j].y - 12),
                          cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);

                // 添加到检测结果
                result.points.push_back(sorted_points[j]);
            }

            // 在矩形中心标注类型
            string center_text = rect_type + " " + to_string(valid_rectangles + 1);
            cv::putText(image, center_text,
                      cv::Point(current_center.x - 40, current_center.y),
                      cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
            cv::putText(image, center_text,
                      cv::Point(current_center.x - 40, current_center.y),
                      cv::FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2);

            results.push_back(result);
            valid_rectangles++;
            
            RCLCPP_INFO(this->get_logger(),
                      "Rectangle %d (%s), area: %.1f, center: (%.1f, %.1f)",
                      valid_rectangles + 1, rect_type.c_str(), area,
                      current_center.x, current_center.y);
        }
    }

    // 更新上一帧的中心点数据
    prev_rect_centers = current_centers;
    
    return results;
}

void VisionNode::callback_camera(sensor_msgs::msg::Image::SharedPtr msg) {
    try {
        // 图像转换
        cv_bridge::CvImagePtr cv_ptr;

        // 检测图像编码格式并转换为BGR8
        if (msg->encoding == "rgb8" || msg->encoding == "R8G8B8") {
            cv::Mat image(msg->height, msg->width, CV_8UC3,
                        const_cast<unsigned char *>(msg->data.data()));
            cv::Mat bgr_image;
            cv::cvtColor(image, bgr_image, cv::COLOR_RGB2BGR);
            cv_ptr = std::make_shared<cv_bridge::CvImage>();
            cv_ptr->header = msg->header;
            cv_ptr->encoding = "bgr8";
            cv_ptr->image = bgr_image;
        } else {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }

        cv::Mat image = cv_ptr->image;

        if (image.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty image");
            return;
        }

        // 创建结果图像
        Mat result_image = image.clone();

        // 按照新的顺序运行检测模式：先矩形检测，再球体检测，最后装甲板检测
        vector<DetectionResult> all_results;
        
        // 1. 矩形检测（最先执行）
        vector<DetectionResult> rectangle_results = detectRectangles(result_image);
        all_results.insert(all_results.end(), rectangle_results.begin(), rectangle_results.end());
        
        // 2. 球体检测
        vector<DetectionResult> sphere_results = detectSpheres(result_image);
        all_results.insert(all_results.end(), sphere_results.begin(), sphere_results.end());
        
        // 3. YOLOv5装甲板检测（最后执行，覆盖在矩形检测之上）
        vector<DetectionResult> yolo_armor_results = detectYOLOArmors(result_image);
        all_results.insert(all_results.end(), yolo_armor_results.begin(), yolo_armor_results.end());

        // 显示结果图像
        if (show_debug_window_) {
            cv::imshow(debug_window_name_, result_image);
            cv::waitKey(1);
        }

        // 创建并发布消息
        referee_pkg::msg::MultiObject msg_object;
        msg_object.header = msg->header;
        msg_object.num_objects = all_results.size();

        for (const auto& result : all_results) {
            referee_pkg::msg::Object obj;
            obj.target_type = result.target_type;

            // 每个目标发送4个角点
            for (const auto& point : result.points) {
                geometry_msgs::msg::Point corner;
                corner.x = point.x;
                corner.y = point.y;
                corner.z = 0.0;
                obj.corners.push_back(corner);
            }

            msg_object.objects.push_back(obj);
        }

        Target_pub->publish(msg_object);
        RCLCPP_INFO(this->get_logger(), "Published %u targets (rectangle: %zu, sphere: %zu, YOLO armor: %zu)",
                    msg_object.num_objects, rectangle_results.size(), 
                    sphere_results.size(), yolo_armor_results.size());

    } catch (const cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionNode>("vision_node");
    RCLCPP_INFO(node->get_logger(), "Starting VisionNode");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
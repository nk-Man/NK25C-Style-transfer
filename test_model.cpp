// test_model.cpp
#include "model.h"
#include <iostream>
#include <filesystem>

int main() {

    std::string model_path   = "./style_transfer.pt";
    std::string content_path = "man.jpg";
    std::string style_path   = "sky.jpg";
    std::string output_path  = "output.jpg";

    std::cout << "Working dir: "
              << std::filesystem::current_path()
              << std::endl;

    std::cout << "Exists(" << model_path << ")? "
              << std::filesystem::exists(model_path)
              << std::endl;

    StyleTransferModel model;
    std::cout << ">>> trying to load from: \"" << model_path << "\"\n"
              << ">>> exists? " << std::filesystem::exists(model_path) << std::endl;
    if (!model.loadModel(model_path)) {
        std::cerr << "Failed to load model from: "
                  << model_path << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully.\n";

    // 2. 读取图片（OpenCV BGR 格式）
    cv::Mat content_img = cv::imread(content_path, cv::IMREAD_COLOR);
    cv::Mat style_img   = cv::imread(style_path,   cv::IMREAD_COLOR);
    if (content_img.empty() || style_img.empty()) {
        std::cerr << "Failed to read images:\n"
                  << "  content: " << content_path << "\n"
                  << "  style:   " << style_path   << std::endl;
        return 1;
    }
    std::cout << "Images loaded (content: "
              << content_img.cols << "x" << content_img.rows
              << ", style: "
              << style_img.cols << "x" << style_img.rows
              << ").\n";

    // 3. 推理
    cv::Mat output_img = model.infer(content_img, style_img);
    if (output_img.empty()) {
        std::cerr << "Inference failed, output is empty.\n";
        return 1;
    }
    std::cout << "Inference done. Output size: "
              << output_img.cols << "x" << output_img.rows << "\n";

    // 4. 保存结果
    if (!cv::imwrite(output_path, output_img)) {
        std::cerr << "Failed to save output to: " << output_path << std::endl;
        return 1;
    }
    std::cout << "Output saved to " << output_path << "\n";

    return 0;
}

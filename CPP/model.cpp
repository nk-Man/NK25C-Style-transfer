
#include <iostream>
// 取消 UNICODE，保证 torch 里用的是真正的 fopen
#ifdef UNICODE
#undef UNICODE
#endif
#ifdef _UNICODE
#undef _UNICODE
#endif
#include<torch/torch.h>
#include<string>

#include "model.h"


//对模型类成员函数进行类外定义
//构造函数
StyleTransferModel::StyleTransferModel()
    : is_loaded_(false) {}

StyleTransferModel::~StyleTransferModel() {}//析构函数

//模型加载函数
bool StyleTransferModel::loadModel(const std::string& model_path) {

    try {
        // 也可以直接传 std::filesystem::path
        module_ = torch::jit::load(model_path);
        torch::NoGradGuard no_grad;
        module_.to(torch::kCPU);
        module_.eval();
        is_loaded_ = true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return false;
    }
    return true;
}
//图片预处理函数
torch::Tensor StyleTransferModel::preprocess(const cv::Mat& img) {
    cv::Mat resized, cropped, img_rgb, img_float;
    // 1. Resize to 256x256
    cv::resize(img, resized, cv::Size(256, 256));
    // 2. Center crop to 224x224
    int offset = (256 - 224) / 2;
    cv::Rect roi(offset, offset, 224, 224);
    cropped = resized(roi).clone();
    // 3. Convert BGR to RGB
    cv::cvtColor(cropped, img_rgb, cv::COLOR_BGR2RGB);
    // 4. Convert to float [0,1]
    img_rgb.convertTo(img_float, CV_32F, 1.0 / 255);
    // 5. HWC -> CHW and add batch dim
    auto tensor = torch::from_blob(
        img_float.data, {1, 224, 224, 3}, torch::kFloat32);
    tensor = tensor.permute({0, 3, 1, 2});
    // 6. Normalize
    for (int c = 0; c < 3; ++c) {
        tensor[0][c] = tensor[0][c].sub(mean_[c]).div(std_[c]);
    }
    return tensor.clone();  // ensure memory ownership
}

//图片后处理函数
cv::Mat StyleTransferModel::postprocess(const torch::Tensor& tensor) {
    auto t = tensor.squeeze().detach();  // 3x224x224
    // Denormalize
    for (int c = 0; c < 3; ++c) {
        t[c] = t[c].mul(std_[c]).add(mean_[c]);
    }
    // Clamp to [0,1]
    t = t.clamp(0, 1);
    // to CPU uint8
    t = t.mul(255).to(torch::kU8);
    // CHW -> HWC
    t = t.permute({1, 2, 0});

    int height = t.size(0);
    int width  = t.size(1);
    // Create OpenCV Mat from tensor data
    cv::Mat output(height, width, CV_8UC3);
    std::memcpy(output.data, t.data_ptr(), width * height * 3);
    // Convert RGB back to BGR for display
    cv::Mat output_bgr;
    cv::cvtColor(output, output_bgr, cv::COLOR_RGB2BGR);
    return output_bgr;
}

//模型推理函数
cv::Mat StyleTransferModel::infer(const cv::Mat& content_img, const cv::Mat& style_img) {
    if (!is_loaded_) {
        std::cerr << "Model not loaded! Call loadModel() first." << std::endl;
        return {};
    }
    auto content_t = preprocess(content_img);
    auto style_t   = preprocess(style_img);

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(2);
    inputs.push_back(content_t);
    inputs.push_back(style_t);

    torch::Tensor output_t = module_.forward({inputs}).toTensor();
    return postprocess(output_t);
}








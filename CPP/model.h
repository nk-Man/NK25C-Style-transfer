#ifndef MODEL_H
#define MODEL_H

// 取消 UNICODE，保证 torch 里用的是真正的 fopen
#ifdef UNICODE
#undef UNICODE
#endif
#ifdef _UNICODE
#undef _UNICODE
#endif

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
//这些第三方库导入一定要放在qt相关库的导入的上面！！
#include <QMainWindow>
#include<QProcess>
QT_BEGIN_NAMESPACE
namespace Ui {
class model;
}
QT_END_NAMESPACE

class model : public QMainWindow
{
    Q_OBJECT

public:
    model(QWidget *parent = nullptr);
    ~model();

private:
    Ui::model *ui;
};


//定义模型类，声明里面的成员函数，没有进行定义！
class StyleTransferModel {
public:
    StyleTransferModel();
    ~StyleTransferModel();

    /**
     * Load a TorchScript model from file and prepare for inference
     * @param model_path Path to the serialized model.pt
     * @return true if loaded successfully
     */
    bool loadModel(const std::string& model_path);

    /**
     * Run style transfer inference on a content and style image
     * @param content_img OpenCV BGR image
     * @param style_img   OpenCV BGR image
     * @return stylized output BGR image
     */
    cv::Mat infer(const cv::Mat& content_img, const cv::Mat& style_img);

private:
    torch::Tensor preprocess(const cv::Mat& img);
    cv::Mat postprocess(const torch::Tensor& tensor);

private:
    torch::jit::script::Module module_;
    bool is_loaded_;
    const std::vector<double> mean_ = {0.485, 0.456, 0.406};
    const std::vector<double> std_  = {0.229, 0.224, 0.225};
};
#endif // MODEL_H

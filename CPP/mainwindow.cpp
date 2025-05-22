#include "mainwindow.h"
#include <opencv2/opencv.hpp>





#include <QImageReader>
#include <QDir>
#include <QCoreApplication>
#include <QFileDialog>
#include <QPixmap>
#include <QImage>



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setupUI();
    // 1. 加载 TorchScript 模型
    QString exeDir    = QCoreApplication::applicationDirPath();
    QString modelPath = QDir(exeDir).filePath("style_transfer.jit");
    if (!styleModel_.loadModel(modelPath.toStdString())) {
        // 可选：弹框提示加载失败
    }

    // 2. 连接 Generate 按钮信号到推理槽
    connect(generateButton, &QPushButton::clicked,
            this, &MainWindow::onGenerateClicked);
}

MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    // 设置窗口最小尺寸
    this->setMinimumSize(800, 600);  // 设置最小宽度和高度

    // 设置背景色
    this->setStyleSheet("QMainWindow { background-color: #f4f4f9; }");

    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    QHBoxLayout *imageLayout = new QHBoxLayout();
    QHBoxLayout *buttonLayout = new QHBoxLayout();

    // Content Image UI
    contentImageLabel = new QLabel("No Content Image", this);
    contentImageLabel->setFixedSize(320, 320);  // 增大图片框
    contentImageLabel->setStyleSheet("QLabel { "
                                     "background-color: #eee; "
                                     "border-radius: 15px; "
                                     "border: 1px solid #ddd; "
                                     "box-shadow: 2px 2px 8px rgba(0,0,0,0.1); "
                                     "}");
    contentImageLabel->setAlignment(Qt::AlignCenter);

    contentPromptLabel = new QLabel("请选择内容图", this);
    contentPromptLabel->setAlignment(Qt::AlignCenter);
    contentPromptLabel->setStyleSheet("font-size: 12px; color: #777;");  // 调小文字大小

    QVBoxLayout *contentLayout = new QVBoxLayout();
    contentLayout->addWidget(contentImageLabel);
    contentLayout->addWidget(contentPromptLabel);

    // Style Image UI
    styleImageLabel = new QLabel("No Style Image", this);
    styleImageLabel->setFixedSize(320, 320);  // 增大图片框
    styleImageLabel->setStyleSheet("QLabel { "
                                   "background-color: #eee; "
                                   "border-radius: 15px; "
                                   "border: 1px solid #ddd; "
                                   "box-shadow: 2px 2px 8px rgba(0,0,0,0.1); "
                                   "}");
    styleImageLabel->setAlignment(Qt::AlignCenter);

    stylePromptLabel = new QLabel("请选择风格图", this);
    stylePromptLabel->setAlignment(Qt::AlignCenter);
    stylePromptLabel->setStyleSheet("font-size: 12px; color: #777;");  // 调小文字大小

    QVBoxLayout *styleLayout = new QVBoxLayout();
    styleLayout->addWidget(styleImageLabel);
    styleLayout->addWidget(stylePromptLabel);

    imageLayout->addLayout(contentLayout);
    imageLayout->addLayout(styleLayout);

    // Buttons
    contentButton = new QPushButton("选择内容图", this);
    contentButton->setStyleSheet("QPushButton {"
                                 "background-color: #4CAF50; "
                                 "color: white; "
                                 "border-radius: 12px; "
                                 "font-size: 16px; "
                                 "padding: 10px 20px; "
                                 "box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); "
                                 "}"
                                 "QPushButton:hover {"
                                 "background-color: #45a049; "
                                 "}");

    styleButton = new QPushButton("选择风格图", this);
    styleButton->setStyleSheet("QPushButton {"
                               "background-color: #008CBA; "
                               "color: white; "
                               "border-radius: 12px; "
                               "font-size: 16px; "
                               "padding: 10px 20px; "
                               "box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); "
                               "}"
                               "QPushButton:hover {"
                               "background-color: #007bb5; "
                               "}");

    generateButton = new QPushButton("Generate", this);
    generateButton->setStyleSheet("QPushButton {"
                                  "background-color: #f44336; "
                                  "color: white; "
                                  "border-radius: 12px; "
                                  "font-size: 16px; "
                                  "padding: 10px 20px; "
                                  "box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); "
                                  "}"
                                  "QPushButton:hover {"
                                  "background-color: #e53935; "
                                  "}");

    buttonLayout->addWidget(contentButton);
    buttonLayout->addWidget(styleButton);
    buttonLayout->addWidget(generateButton);

    mainLayout->addLayout(imageLayout);
    mainLayout->addLayout(buttonLayout);

    // Connect signals and slots
    connect(contentButton, &QPushButton::clicked, this, &MainWindow::loadContentImage);
    connect(styleButton, &QPushButton::clicked, this, &MainWindow::loadStyleImage);
}


void MainWindow::loadContentImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, "选择内容图", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!fileName.isEmpty()) {
        loadImageToLabel(contentImageLabel, fileName, contentPromptLabel);
    }
}

void MainWindow::loadStyleImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, "选择风格图", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!fileName.isEmpty()) {
        loadImageToLabel(styleImageLabel, fileName, stylePromptLabel);
    }
}

void MainWindow::loadImageToLabel(QLabel *label, const QString &filePath, QLabel *promptLabel)
{
    QPixmap pixmap(filePath);
    if (!pixmap.isNull()) {
        label->setPixmap(pixmap.scaled(label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        if (promptLabel)
            promptLabel->hide();
    }
}
void MainWindow::onGenerateClicked()
{
    QPixmap cp = contentImageLabel->pixmap();
     QPixmap sp = styleImageLabel->pixmap();
    //  QPixmap → QImage （RGB888）
    QImage cqimg = cp.toImage().convertToFormat(QImage::Format_RGB888);
    QImage sqimg = sp.toImage().convertToFormat(QImage::Format_RGB888);
    // 将 QPixmap 转为 cv::Mat（BGR）
    auto qimgToMat = [](const QPixmap px)->cv::Mat {
        QImage img = px.toImage().convertToFormat(QImage::Format_RGB888);
        cv::Mat mat(img.height(), img.width(), CV_8UC3,
                    const_cast<uchar*>(img.bits()),
                    img.bytesPerLine());
        cv::Mat matBGR;
        cv::cvtColor(mat, matBGR, cv::COLOR_RGB2BGR);
        return matBGR.clone();
    };

    cv::Mat contentMat = qimgToMat(contentImageLabel->pixmap());
    cv::Mat styleMat   = qimgToMat(styleImageLabel->pixmap());

    // 调用模型推理
    cv::Mat resultMat = styleModel_.infer(contentMat, styleMat);

    // 将结果 cv::Mat 转为 QImage 并显示（假设 resultMat 为 BGR，8UC3）
    cv::Mat resultRGB;
    cv::cvtColor(resultMat, resultRGB, cv::COLOR_BGR2RGB);
    QImage resultImg(resultRGB.data,
                     resultRGB.cols, resultRGB.rows,
                     resultRGB.step,
                     QImage::Format_RGB888);
    contentImageLabel->setPixmap(
        QPixmap::fromImage(resultImg)
            .scaled(contentImageLabel->size(),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation));
}

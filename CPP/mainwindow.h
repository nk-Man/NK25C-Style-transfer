#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPixmap>
#include <QFileDialog>
#include <QScrollArea>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void loadContentImage();
    void loadStyleImage();

private:
    QWidget *centralWidget;
    QLabel *contentImageLabel;
    QLabel *styleImageLabel;
    QLabel *contentPromptLabel;
    QLabel *stylePromptLabel;
    QPushButton *contentButton;
    QPushButton *styleButton;
    QPushButton *generateButton;

    void setupUI();
    void loadImageToLabel(QLabel *label, const QString &filePath, QLabel *promptLabel);
};

#endif // MAINWINDOW_H

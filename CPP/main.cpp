#include "mainwindow.h"
#include<torch/script.h>


#include <QApplication>
#include <QLocale>
#include <QTranslator>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString &locale : uiLanguages) {
        const QString baseName = "StyleTransferNet_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName)) {
            a.installTranslator(&translator);
            break;
        }
    }
    // 自动寻找并加载 model
    QString modelFile = QDir(QCoreApplication::applicationDirPath())
                            .filePath("style_transfer.jit");
    torch::jit::Module module = torch::jit::load(modelFile.toStdString());

    MainWindow w(std::move(module));
    w.show();
    return a.exec();
}

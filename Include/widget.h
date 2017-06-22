#include <QtWidgets/QWidget>
#include "ui_widget.h"

class Widget : public QWidget {
Q_OBJECT
public:
    Widget(QWidget* parent = 0);
    ~Widget();

private slots:
    void slotPushButton();

private:
    Ui::Widget ui;
    int m_iCount;
};
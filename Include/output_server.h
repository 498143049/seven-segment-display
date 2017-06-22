//
// Created by dubing on 17-4-29.
//

#ifndef DIGITAL_LINUX_OUTPUT_SERVER_H
#define DIGITAL_LINUX_OUTPUT_SERVER_H
#include <QtCore/QObject>
#include <qwebsocketserver.h>
#include "QtWebSockets/QWebSocket"
class output_server: public QObject {
    Q_OBJECT
public:
    explicit output_server(uint16_t port, QObject *parent = Q_NULLPTR);
    virtual ~output_server();

private:
    void onNewConnection();
    void processTextMessage(QString message);
    void processBinaryMessage(QByteArray message);
    void socketDisconnected();
    QWebSocketServer *m_pWebSocketServer;
    QList<QWebSocket *> m_clients;
};


#endif //DIGITAL_LINUX_OUTPUT_SERVER_H

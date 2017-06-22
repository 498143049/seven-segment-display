//
// Created by dubing on 17-4-29.
//

#include "output_server.h"
#include "web_interface.h"
output_server::output_server(quint16 port, QObject *parent) : QObject(parent), m_pWebSocketServer(Q_NULLPTR) {
    m_pWebSocketServer = new QWebSocketServer(QStringLiteral("sent url Server"), QWebSocketServer::NonSecureMode, this);
    if (m_pWebSocketServer->listen(QHostAddress::LocalHost, port)) {
        qDebug() << "SSL Echo Server listening on port" << port;
        connect(m_pWebSocketServer, &QWebSocketServer::newConnection, this, &output_server::onNewConnection);
    }
}
output_server::~output_server()  {
    m_pWebSocketServer->close();
    qDeleteAll(m_clients.begin(), m_clients.end());
}
void output_server::onNewConnection()
{
    QWebSocket *pSocket = m_pWebSocketServer->nextPendingConnection();

    qDebug() << "Client connected:" << pSocket->peerName() << pSocket->origin();

    connect(pSocket, &QWebSocket::textMessageReceived, this, &output_server::processTextMessage);
    connect(pSocket, &QWebSocket::binaryMessageReceived, this, &output_server::processBinaryMessage);
    connect(pSocket, &QWebSocket::disconnected, this, &output_server::socketDisconnected);
    //这里发送json 数据
    pSocket->sendTextMessage(QString::fromStdString(web_interface::get_content_pic()));
    pSocket->sendTextMessage(QString::fromStdString(web_interface::get_contain_pic("ndata_0")));

    m_clients << pSocket;
}
//! [onNewConnection]

//! [processTextMessage]
void output_server::processTextMessage(QString message)
{
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    if (pClient)
    {
        pClient->sendTextMessage(QString::fromStdString(web_interface::get_contain_pic("ndata_"+message.toStdString())));
    }
}
//! [processTextMessage]

//! [processBinaryMessage]
void output_server::processBinaryMessage(QByteArray message)
{
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    if (pClient)
    {
        pClient->sendBinaryMessage(message);
    }
}
//! [processBinaryMessage]

//! [socketDisconnected]
void output_server::socketDisconnected()
{
    qDebug() << "Client disconnected";
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    if (pClient)
    {
        m_clients.removeAll(pClient);
        pClient->deleteLater();
    }
}
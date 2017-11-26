#include <iostream>
#include "json.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include "Include/DigtalLocate.h"
using namespace cv;
using namespace std;
using namespace std;
using namespace cv::ml;
//#include "Include/widget.h"
ofstream outfile("../python-svm/outn.txt");
int main(int argc, char** argv)
{

//    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
//    QApplication app(argc, argv);
//
//    QDir dir = QDir::current();
//
//    output_server server(2009);
//    Q_UNUSED(server);
//
//    QWebEngineView view;
//    view.load(QUrl::fromLocalFile(dir.filePath("../resources/UI/starter.html")));
//    view.resize(1920, 1080);
//    view.show();
//
//    return app.exec();

//    for(int i=0; i<=261;i++) {
//        int64_t time = cvGetTickCount();
//        DigtalLocate temp =  DigtalLocate("../datasource/ppdata/ppdata_"+to_string(i)+".jpg",1.1);
//        Mat t;
// //       if(mean(temp._gray)[0]>120) {
//            t = tool::char_threshold(temp.const_Mat, temp._gray, temp.name,temp._max,1);
//        } else{
//            t = tool::char_threshold(temp.const_Mat, temp._max, temp.name,2);
//        }
//        tool::DebugOut("test2",temp.name,t);
//        tool::DebugOut("test_gray",temp.name,temp._gray);
//        cout<<i<<" is ok! \t"<<"cost time :"<<(cvGetTickCount()-time)/(cvGetTickFrequency()*1000)<< "ms"<<endl;
//    }


//    for(int i=0; i<=203;i++) {
//        int64_t time = cvGetTickCount();
//        DigtalLocate temp =  DigtalLocate("../datasource/subdata/subdata_"+to_string(i)+".jpg",1.1);
//        temp.get_char();
//        cout<<i<<" is ok! \t"<<"cost time :"<<(cvGetTickCount()-time)/(cvGetTickFrequency()*1000)<< "ms"<<endl;
//    }



    for(int i=37; i<=37;i++) {
        int64_t time = cvGetTickCount();
        DigtalLocate temp =  DigtalLocate("../datasource/ndata/ndata_"+to_string(i)+".jpg",1.1);
        temp.probably_locate();
     //   temp.output_joson();
        cout<<i<<" is ok! \t"<<"cost time :"<<(cvGetTickCount()-time)/(cvGetTickFrequency()*1000)<< "ms"<<endl;
    }

//    for(int i=0; i<=91;i++) {
//        int64_t time = cvGetTickCount();
//        DigtalLocate temp =  DigtalLocate("../datasource/didata/didata_"+to_string(i)+".jpg",1.1);
//        string s = temp.get_char();
//
//        cout<<s<<"||"<<i<<" is ok! \t"<<"cost time :"<<(cvGetTickCount()-time)/(cvGetTickFrequency()*1000)<< "ms"<<endl;
//    }
//        for(int i=21; i<=21;i++) {
//        int64_t time = cvGetTickCount();
//        DigtalLocate temp =  DigtalLocate("../datasource/sdata/sdata_"+to_string(i)+".jpg",1.1);
//        temp.recongize_test();
      //  cout<<i<<" is ok! \t"<<"cost time :"<<(cvGetTickCount()-time)/(cvGetTickFrequency()*1000)<< "ms"<<endl;
//    }
     outfile.close();
//    cout<<"all_down"<<endl;
}

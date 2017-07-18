#include <iostream>
#include "Include/DigtalLocate.h"

//test correct_angle
//测试角度函数
using namespace std;
using namespace cv;
int main() {
    for(int i=0; i<=19;i++) {
        int64_t time = cvGetTickCount();
        DigtalLocate temp =  DigtalLocate("../datasource/test_correct/test_correct_"+to_string(i)+".jpg",1.1);
        double angle;
        Mat best_test;
        threshold(temp._medium,best_test,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
        //曲线输出
        medianBlur(temp._gray,temp._gray,5);
        angle =  tool::correct_error(temp._gray,angle);

        //
        cout<<angle<<endl;
      //  cout<<angle<<" is ok! \t"<<"cost time :"<<(cvGetTickCount()-time)/(cvGetTickFrequency()*1000)<< "ms"<<endl;
    }
    cout<<"I am ok!"<<endl;
}
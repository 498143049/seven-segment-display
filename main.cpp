#include <iostream>
#include "Include/json.h"
#include <opencv2/opencv.hpp>
#include "Include/DigtalLocate.h"
#include "Include/LBP.h"
using namespace std;
using namespace cv;
void dealPicGroup(int num, string outputurl,string name)
{
   //先读取txt　

}
#define  PNumAll 593
#define  NNumAll 371
#define  Pnum 400 // 400
#define  Nnum 240  // 240
using namespace ml;
static Ptr<ml::SVM> svm = ml::SVM::create();
void train( Mat_<float> trainData,Mat_<float> labels,vector<float>weight) {

    svm->setType(ml::SVM::C_SVC);
 //   svm->setGamma(3.5);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setC(15);
    Mat w(weight);
    svm->setClassWeights(w);
//    svm->train(trainData,ml::ROW_SAMPLE,labels);
    auto data = cv::ml::TrainData::create(trainData,ml::ROW_SAMPLE,labels,noArray(),noArray());
    auto c = ParamGrid(0.001,0.1,1.04);
    auto gamma = ParamGrid(1,1,0);
//    auto gamma = ParamGrid(0.01,1,1.04);
    auto p = ParamGrid(1,1,0);
    auto nu = ParamGrid(1,1,0);
    auto coeff = ParamGrid(1,1,0);
    auto degree = ParamGrid(1,1,0);

    svm->trainAuto(data,5,c,gamma,p, nu,coeff,degree, false );
    svm->save("../svm-model/digtal_binary.xml");
}
void trainwap(){
    Mat_<float> trainData;Mat_<float> labels;vector<float> weight;
    for(int i=0; i<=Nnum;i++) {
        DigtalLocate a = DigtalLocate("../datasource/nnndata/nnndata_"+to_string(i)+".jpg",1.1);
        trainData.push_back(a.Get_LBP());
        labels.push_back(0);


    }
    weight.push_back(Pnum+1);
    for(int i=0; i<=Pnum;i++) {
        DigtalLocate a = DigtalLocate("../datasource/pppdata/pppdata_"+to_string(i)+".jpg",1.1);


        trainData.push_back(a.Get_LBP());
        labels.push_back(1);


    }
    weight.push_back(Nnum+1);

    train(trainData,labels,weight);
    cout<<svm->getC()<<"||"<<svm->getGamma()<<endl;
    Mat an;
    svm->predict(trainData,an);
    auto s = (vector<float>)(an.reshape(1, 1));
    auto s2 = (vector<float>)(labels.reshape(1, 1));
    int count=0;
    for(int i=0;i<s2.size();i++){
        if(s[i]==s2[i])
            count++;
        else
            cout<<i<<endl;
    }
    cout<<count<<"||"<<(double)count/s2.size()<<endl;
}
void test(){
    DigtalLocate a = DigtalLocate("../datasource/ppdata/ppdata_"+to_string(1)+".jpg",1.1);
    Mat b;
//    ximgproc::guidedFilter(a.const_Mat,b,)
//    d.filter(a.const_Mat,b);
//    imshow("xx",b);
}
void test_all(){
    Mat_<float> trainData;Mat_<float> labels;vector<float> weight;
    double  count=0;
    double  num=0;
    Ptr<SVM> svm=ml::StatModel::load<SVM>("../svm-model/digtal_binary.xml");
    for(int i=Nnum+1; i<=NNumAll;i++) {
        DigtalLocate a = DigtalLocate("../datasource/nnndata/nnndata_"+to_string(i)+".jpg",1.1);

        Mat ans;
        svm->predict(a.Get_LBP(),ans);
        auto s = (vector<float>)(ans.reshape(1, 1));
        if(s[0]==1){
            tool::DebugOut(STR(TS),to_string(i)+'N',a.imgMat);
            count++;
        }
     //   cout<<i<<endl;
        num++;
    }
    for(int i=Pnum+1; i<=PNumAll;i++) {
        DigtalLocate a = DigtalLocate("../datasource/pppdata/pppdata_"+to_string(i)+".jpg",1.1);
        Mat ans;
        svm->predict(a.Get_LBP(),ans);
        auto s = (vector<float>)(ans.reshape(1, 1));
        if(s[0]==0){
            tool::DebugOut(STR(TS),to_string(i)+'P',a.imgMat);
            count++;
        }

        num++;
    }


    cout<<count<<"||"<<1-(double)count/num<<endl;
}
void rotateImage(Mat &img, Mat &img_rotate,int degree)
{
    //旋转中心为图像中心
    CvPoint2D32f center;
    center.x=float (img.cols/2.0+0.5);
    center.y=float (img.rows/2.0+0.5);
    //计算二维旋转的仿射变换矩阵
    float m[6];
    Mat M = Mat( 2, 3, CV_32F, m );
    M = getRotationMatrix2D( center, degree,1);
    //变换图像，并用黑色填充其余值
    cv::warpAffine(img,img_rotate, M,img.size());
}
void deal(){
    for(int i=0; i<=62;i++) {
        DigtalLocate a = DigtalLocate("../datasource/nndata/nndata_"+to_string(i)+".jpg",1.1);
        auto b = Rect(0,0,a.imgMat.cols/2,a.imgMat.rows/2);
        auto c = cv::Rect(a.imgMat.cols/2,0,a.imgMat.cols/2,a.imgMat.rows/2);
        auto d = cv::Rect(0,a.imgMat.rows/2,a.imgMat.cols/2,a.imgMat.rows/2);
        auto e = cv::Rect(a.imgMat.cols/2,a.imgMat.rows/2,a.imgMat.cols/2,a.imgMat.rows/2);
        Mat f;
        rotateImage(a.imgMat,f,180);
        tool::DebugOut(STR(T),a.name+'A',a.imgMat(b));
        tool::DebugOut(STR(T),a.name+'B',a.imgMat(c));
        tool::DebugOut(STR(T),a.name+'C',a.imgMat(d));
        tool::DebugOut(STR(T),a.name+'D',a.imgMat(e));
        tool::DebugOut(STR(T),a.name+'F',f);

    }
}
//读取图片　//测试导向滤波
void filter(){
    //cv::ximgproc::GuidedFilter::filter

}
int main() {
    //deal();
    trainwap();
    test_all();
   // test();
    return  1;
}
#include <iostream>
#include "Include/DigtalLocate.h"
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>

//test correct_angle
//测试角度函数
//定位函数 返回的是每一个图片的位置大小关系
using namespace std;
using namespace cv;

void test(Mat srcImage){
    Mat midImage,dstImage;//临时变量和目标图的定义

    //【2】显示原始图

    //【3】转为灰度图并进行图像平滑
    cvtColor(srcImage,midImage, COLOR_BGR2GRAY);//转化边缘检测后的图为灰度图
    GaussianBlur( midImage, midImage, Size(3, 3), 2, 2 );
    Mat gray_t = tool::stretch(midImage);  //灰度拉伸
    Mat threholdpic;
    int t = (int)threshold(gray_t,threholdpic,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
    //分水岭算法输出的图像
    Mat threshold_diff =  tool::get_watershed_segmenter_mark(t+20, t-60,gray_t); //用于字符分割
    threshold_diff = tool::get_binnary_by_watershed(threshold_diff,srcImage);

    imshow("pre_o", threshold_diff);
    //【4】进行霍夫圆变换
    vector<Vec3f> circles;
    HoughCircles( threshold_diff, circles, HOUGH_GRADIENT,1.5, 10, 500, 100, 0, 0 );

    //【5】依次在图中绘制出圆
    for( size_t i = 0; i < circles.size(); i++ )
    {
        //参数定义
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        //绘制圆心
        circle( srcImage, center, 3, Scalar(0,255,0), -1, 8, 0 );
        //绘制圆轮廓
        circle( srcImage, center, radius, Scalar(155,50,255), 3, 8, 0 );
    }

    //【6】显示效果图
    imshow("pre_c", srcImage);

    waitKey(0);

}
int main() {
    Mat s1 = imread("../datasource/ndata/judge1.jpg");
    test(s1);
}
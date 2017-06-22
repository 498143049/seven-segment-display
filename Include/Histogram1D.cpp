//
// Created by dubing on 17-5-8.
//

#include "Histogram1D.h"

Histogram1D::Histogram1D()
{
    histSize[0] = 256;//箱子个数设为256
    channels[0] = 0;//使用一个通道，默认为0
    hranges[0] = 0.0;
    hranges[1] = 256.0;
    ranges[0] = hranges;//值范围
}

Histogram1D::~Histogram1D()
{
}

cv::Mat Histogram1D::getHistogram(const cv::Mat& image)
{
    cv::Mat hist;
    cv::calcHist(&image,
                 1,//单幅图像的直方图
                 channels,
                 cv::Mat(),
                 hist,
                 1,//一维直方图
                 histSize,
                 ranges);
    return hist;
}
cv::Mat getImageOfHistogram(const cv::Mat& hist, int zoom);
cv::Mat Histogram1D::getHistogramImage(const cv::Mat& image, int zoom = 1)
{
    cv::Mat hist = getHistogram(image);
    //这里调用静态方法
    return getImageOfHistogram(hist,zoom);
}

cv::Mat getImageOfHistogram(const cv::Mat& hist,
                            int zoom//设置缩放系数
)
{
    double minVal = 0;
    double maxVal = 0;
    cv::minMaxLoc(hist,&minVal,&maxVal,0,0);

    int histSize = hist.rows;

    //用于显示直方图的方形图像
    cv::Mat histImg(histSize*zoom,histSize*zoom,CV_8U,cv::Scalar(255));

    //设置最高点为90%的箱子个数
    int hmax = static_cast<int>(histSize*0.9);

    //画线
    for (int h = 0; h < histSize;h++)
    {
        float binVal= hist.at<float>(h);
        if (binVal > 0)
        {
            int intensity = binVal*hmax / maxVal;
            cv::line(histImg, cv::Point(h*zoom, histSize*zoom), cv::Point(h*zoom, (histSize-intensity)*zoom), cv::Scalar(0), zoom);
        }
    }
    return histImg;
}
//应用查找表相关的函数
cv::Mat Histogram1D::applyLookUp(const cv::Mat& image, const cv::Mat& lookup)
{
    cv::Mat result;
    cv::LUT(image,lookup,result);
    return result;
}
//get方法，获取成员变量histSize
int*  Histogram1D::getHistSize()
{
    return histSize;
}
//set方法，设置成员变量histSize
void  Histogram1D::setHistSize(int n)
{
    histSize[0] = n;
}

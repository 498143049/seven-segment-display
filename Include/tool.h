//
// Created by dubing on 17-4-13.
//

#ifndef DIGITAL_LINUX_TOOL_H
#define DIGITAL_LINUX_TOOL_H

#include <iostream>
#include<opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include <vector>
#include <string>
#include <sstream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include "WatershedSegmenter.hpp"
#include "DigtalLocate.h"
#include <cmath>
using namespace cv;
using namespace std;
const int MAX_GRAY_VALUE = 256;
const int MIN_GRAY_VALUE = 0;
const int PT_WHITE = 255;
const int PT_GRAY = 128;
const int PT_BLACK = 0;

const int SCREEN_WIDTH = 40;//显示区域宽度阈值
const int SCREEN_HEIGHT = 50;//显示区域高度阈值
const int SCREEN_HIGHT_WIDTH_RATE = 1;//显示区域的高宽比阈值
const int SCREEN_DISTANCE = 40;//显示区域之间的距离阈值
const float ROW_EDGE_OFFSET_RATE = 0.1f;//上下边界偏移百分比，往缩小方向

const float LIGHT_RATE = 0.4f;//点亮占比阈值
const int SINGLE_WIDTH = 10;//单个数字宽度阈值
const float ONE_HIGHT_WIDTH_RATE = 2.6f;//数字1图像高宽比阈值
const int SCALING_RATE = 4;//原图到显示缩放率
const int DEFLECTION_ANGLE = 5;//原图相对水平偏转角度
const int LINE_WIDTH = 5;//数码管线宽阈值
//三线扫描交点个数阈值
const int INTERSECTION_COUNT = 4;//三线扫描交点个数阈值
class DigtalLocate;
struct areaRange {
    int x1;
    int y1;//左上角
    int x2;
    int y2;//右下角
    areaRange(){}
    areaRange(Point tl, Point br){x1=tl.x;y1=tl.y;x2=br.x;y2=br.y;}
};
class tool {
public:
    static  vector<int> get_col_number( const Mat& picture,double rate=1);
    static  vector<int> get_row_number( const Mat& picture, double rate=1);
    static  Mat get_hist_show(const vector<int> vec);
    static  vector< pair<int,int> > get_Area( vector<int> &vec,  int);
    static  string DebugOut  (string dir ,string name, Mat resultResized, string id="",int tid=0);
    static  bool is_Inside(Rect& rect1, Rect rect2, double scale);
    static  Rect& thin_rect(Rect &rect, CvSize size);
    static  Rect& scale_rect(Rect &rect, int  scale_x,int  scale_y);
    static  cv::Mat stretch(const cv::Mat& image,int minvalue=0);
    static  std::string getNOByLine( Mat single, vector<areaRange> range, bool atBottom,double t);
    static  bool isIntersected( Mat single, areaRange range );
    static  vector<areaRange> getLineRange( int rows, int cols );
    static  string resultout  (string dir ,string name, Mat resultResized, string subdirName, int tid);
    static void incrementRadon(vector<double> &vt, double pixel, double r);
    static int imageRotate2(InputArray src, OutputArray dst, double angle);
    static double correct_error(Mat &src, double &best_angle,Mat &canny_dst);
    static Mat get_R_mat(Mat src, double angle);
    static string get_name(string urlPath);
    static Mat  get_best_side(DigtalLocate *src,int front_value,int back_value,Mat & mat);
    static void marge(vector<Rect> &candidates);
    static void split_mat(Mat &src,Mat &max, Mat &med,Mat &min);
    static Mat Resize(Mat src,Size si);
    static Mat roatation(Mat src, double angle);
    static Mat char_threshold(Mat srcMat, Mat gray, string name, Mat max,int type);
    static Mat char_threshold2(Mat srcMat, Mat gray, string name, Mat max,int type);
    static cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1);
    static void thin(const Mat &src, Mat &dst, const int iterations);
    static void AdaptiveFindThreshold(const Mat image, double *low, double *high, int aperture_size=3);
    static string location(Mat &src);
    template <typename  T>
    static vector<vector<double >> radon(Mat src,vector<T> angle_array) {
        uint16_t k, m, n;              /* loop counters */
        double angle;             /* radian angle value */
        double cosine, sine;      /* cosine and sine of current angle */

        /* tables for x*cos(angle) and y*sin(angle) */
        double x,y;
        double r, delta;
        int r1;
        double  deg2rad = 3.14159265358979 / 180.0;
        for_each(angle_array.begin(),angle_array.end(),[=](double &a1){a1*=deg2rad;});

        int width = src.cols, height = src.rows;
        vector<double > xCosTable(2*src.cols ,0);
        vector<double > ySinTable(2*src.rows ,0);
        int xOrigin = max(0,(width-1)/2);
        int yOrigin = max(0,(height-1)/2);
        int rFirst = 1;
        int rSize = 2;

        //获取最大的长度 //rize
        int temp1 = height - 1 - yOrigin;
        int temp2 = width - 1 - xOrigin;
        int rLast = (int) ceil(sqrt((double) (temp1*temp1+temp2*temp2))) + 1;
        rFirst = -rLast;
        rSize = rLast - rFirst + 1;

        //对于输入进行度数转弧度
        vector<vector<double >> answer(angle_array.size(),vector<double>(rSize,0));

        for( k = 0; k < angle_array.size(); k++){

            double angle = angle_array[k];

            cosine = cos(angle);
            sine = sin(angle);

            for( n = 0;n<width;n++){
                x = n - xOrigin;
               xCosTable[2*n]   = (x - 0.25)*cosine;   //由极坐标的知识知道，相对于变换的原点，这个就是得到了该点的横坐标
                xCosTable[2*n+1] = (x + 0.25)*cosine;
            }
            for (m = 0; m < height; m++)
            {
                y = yOrigin - m;
                ySinTable[2*m] = (y - 0.25)*sine;   //同理，相对于变换的原点，得到了纵坐标
                ySinTable[2*m+1] = (y + 0.25)*sine;
            }
            for(n=0;n<width;n++) //遍历行
            {
                for(m=0;m<height;m++) //遍历列
                {
                    double t =  src.at<uint8_t >(m,n);
                    if(t!=0) {
                        t *= 0.25;

                        //一个像素点分解成四个临近的像素点进行计算，提高精确度  //r指的是位置,固定r 求出这一段区间内的值的积分
                        //距离为y*sin()+x*cos() 当为４５度是最大即rFirst 加上一个初值也能保证r最小值为０
                        r = xCosTable[2*n] + ySinTable[2*m] - rFirst;
                        incrementRadon(answer[k], t, r);

                        r = xCosTable[2*n+1] + ySinTable[2*m] - rFirst;
                        incrementRadon(answer[k], t, r);

                        r = xCosTable[2*n] + ySinTable[2*m+1] - rFirst;
                        incrementRadon(answer[k], t, r);

                        r = xCosTable[2*n+1] + ySinTable[2*m+1] - rFirst;
                        incrementRadon(answer[k], t, r);
                    }
                }
            }
        }
        return std::move(answer);
    }

};


#endif //DIGITAL_LINUX_TOOL_H

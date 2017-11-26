//
// Created by dubing on 2017/2/27.
//

#ifndef DIGTAL_TEST_DIGTALLOCATE_H
#define DIGTAL_TEST_DIGTALLOCATE_H

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
#include "tool.h"
#include <tuple>

#include "LBP.h"
#include <opencv2/ml.hpp>
using namespace cv;
using namespace std;

#define STR(s) #s


class DigtalLocate {
public:
    friend  class web_interface;
    friend  class tool;
    DigtalLocate(const string urlpath,int type=0);
    DigtalLocate(DigtalLocate *Dig,Rect rect,int id);
    DigtalLocate(string urlPath, double type=0);
    Mat Get_LBP();
    void recongize_test();
   // DigtalLocate(const string urlpath,int m_GaussianBlurSiz=5);
    void atuo_Modify_Parametes();
    vector<cv::Rect> mserExtractor ();
    vector<cv::Rect> extrator_edge ();
    vector<cv::Rect> extrator_edge_differ ();
    Mat  extract_map_by_edge(Mat &src);
    Mat  extera_by_edge_diff(Mat &src,std::vector<cv::Rect> &scandidates);
    Mat  extera_by_edge_1( Mat &src);
    vector<cv::Rect> extra_marge_ara_1(Mat &src);
    vector<cv::Rect> extra_marge_ara( Mat &src);
    vector<cv::Rect> extra_marge_sub_ara( Mat &src);
    void  getsub_char(vector<Rect> &candidates);
    vector<cv::Rect> splite_char(cv::Mat &src, cv::Rect &result);
    Mat splite_char_sub(cv::Mat &src);
    void jujue_char();
    void extract_char();
    vector<cv::Rect>  extrator_threshold();
    Mat  extera_by_edge_threshold(Mat &src,std::vector<cv::Rect> &scandidates);
    void get_pic_rect();
    void output_joson();
    void find();
    void Hist();
   // void output_result();
    void test_char();
    void output_log();
    vector<Rect> probably_locate();
    bool reconize_char();
    vector<cv::Rect> get_digital_area(Mat &src);
    string get_char();
    //inline void output();
    //void Display();

    //int plateLocate(Mat, vector<Mat>& );
    //inline void DebugOut  (string , Mat );
    //static void Resize_t(Mat &resultResized);
    Mat _LBPImage;

public:
    //原图像
    typedef std::function<void (std::string,std::function<std::string (std::string ,std::string, cv::Mat,std::string ,int)>)> debug_fun;
    typedef std::function<void (std::string ,std::string)> log_fun;
    cv::Mat imgMat;
    cv::Mat srcMat;
    cv::Mat yellow;
    cv::Mat green;
    cv::Mat blue;
    cv::Mat black;
    cv::Mat const_Mat;
    vector<pair<string,cv::Rect>> result_ROI;  //存储最后的结果最后结果
    debug_fun output;
    log_fun log;
    //图片的url
    const std::string urlName;
    //最后的选区以及选区为元素可能的概率　
    int m_GaussianBlurSiz=5; //设置高斯模糊值

    std::string name;
    std::string dirname;

    //set for debug
    vector<pair<string,string>> output_mat_url;
    vector<string> output_sub_mat_url;
    string out_put_result_mat;
    stringstream my_stream;

    //参数的给定存储着自己的参数　
    Mat _gray;
    Mat _gray_dif;
    Mat _max;
    Mat _min;
    Mat _medium;
    Mat *best_gray; //绑定最佳的灰度通道,可能指向_gray 也可能指向＿max

//    std::vector<cv::Mat> _channels;


};


#endif //WINDOWS_DIGTAL_TEST_DIGTALLOCATE_H

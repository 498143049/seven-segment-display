//
// Created by dubing on 17-7-18.
//
#include <iostream>
#include <json/json.h>
#include <opencv2/opencv.hpp>
#include "Include/DigtalLocate.h"
#include "Include/LBP.h"
#include <opencv2/ximgproc/edge_filter.hpp>
using namespace std;
using namespace cv;
#define  PNumAll 593
#define  NNumAll 371
#define  Pnum 400 // 400
#define  Nnum 240  // 240
int main(){
    for(int i=0; i<=PNumAll;i++) {
        Mat best_test;double  angel;
        DigtalLocate sub = DigtalLocate("../datasource/pppdata/pppdata_"+to_string(i)+".jpg",1.1);
        threshold(sub._medium,best_test,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
//        tool::DebugOut(STR(finnal6), sub.name, best_test);
////             best_test = tool::char_threshold(sub.const_Mat,t,sub.name,sub._max,0);
////             tool::DebugOut(STR(finnal), sub.name, best_test);
        Mat canny;
        angel = tool::correct_error(best_test, angel,canny);  //整体的角度　best_angle 矫正钱
        tool::DebugOut(STR(canny),sub.name,canny);
        double  dangle = 90-angel;
        if(dangle<=0){
            dangle = 0;
            tool::DebugOut(STR(error), sub.name, sub._gray);
        }
        Mat ss = tool::get_R_mat(best_test,dangle);
        Mat sssrc = tool::get_R_mat(sub.const_Mat,dangle);
        ss = sub.splite_char_sub(ss);
        tool::DebugOut(STR(ss_correct_after), sub.name, ss); //ss矫正后
        tool::DebugOut(STR(ss_correct_after_src), sub.name, sssrc); //ss矫正后
        string result = tool::location(ss);
        cout<<result<<endl;

    }
}
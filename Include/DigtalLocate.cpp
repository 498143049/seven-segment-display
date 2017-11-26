//
// Created by dubing on 2017/2/27.
//
#include "DigtalLocate.h"
#include <opencv2/ml.hpp>
#include "LBP.h"
#include <algorithm>
#include "json.h"
using namespace std;
using namespace cv::ml;




DigtalLocate::DigtalLocate(string urlPath,int type):urlName(urlPath) {
    this->srcMat = imread(urlPath);
    this->imgMat =  this->srcMat.clone();  //srcMat 为不变的
    BOOST_ASSERT((this->imgMat.data)&&"read image error");

    auto s =  [=](string name, string value){this->my_stream<<name<<":"<<value<<endl;};
    log = s;

    log(STR(name),urlPath);

    //处理用户名
    auto n = urlPath.rfind("/");
    if(n == std::string::npos)
        this->name=urlPath;
    else
        this->name=urlPath.substr(n+1);

    n = this->name.rfind(".");
    this->dirname =this->name.substr(0,n);

    if(type==0) {
        get_pic_rect();
        output_log();
        this->output_joson(); //将json 输出
        // out_put_result_mat = ; //输出结果
//        cout<<result_ROI.size()<<endl;
    }
    else if (type==1) {
        extract_char();
        this->imgMat=this->srcMat;
    }
    else if(type==2) {

        extrator_edge_differ();

    }
    else if(type==3) {
        jujue_char();
        this->imgMat=this->srcMat;
    }
    else if(type==4) {
        test_char();
    }
    else if(type==5) {
        find();
    }
    else if(type==6) {
        Hist();
    }

}
extern ofstream outfile;
DigtalLocate::DigtalLocate(string urlPath,double x){
    this->const_Mat = imread(urlPath);
    //构造函数可以对类中的对象进行缩放

    this->imgMat =  this->const_Mat.clone();  //const_Mat 为不变的　img 是为调试做准备
    this->blue=this->const_Mat.clone();
    this->green=this->const_Mat.clone();
    this->yellow=this->const_Mat.clone();
    this->black =  Mat(const_Mat.rows,const_Mat.cols, CV_8UC1, Scalar(0));
    BOOST_ASSERT((this->imgMat.data)&&"read image error");
    this->name=tool::get_name(urlPath);  //
    //this->dirname = name;
    //得到灰度图　最大值图　最小值图　(都是经过滤波后的)


//    Mat HSV;
//    cv::cvtColor(const_Mat, HSV, CV_BGR2HSV);
//    cv::split(HSV, _channels);
    cv::cvtColor(const_Mat, _gray, CV_BGR2GRAY);
  //  medianBlur(_gray,_gray,3);  //

    _max.create(this->const_Mat.size(),CV_8UC1);
    _medium.create(this->const_Mat.size(),CV_8UC1);
    _min.create(this->const_Mat.size(),CV_8UC1);

    tool::split_mat(this->const_Mat, _max, _medium, _min);  //获得3个通道

  //  medianBlur(_channels[2],_channels[2],3);

    //_gray_dif = _max-_gray;   //目前是max(RGB)-mean(RGB) 可以改为max(RGB)-MIN(RGB) 之后的滤波方式也可改为最大值滤波和最小值滤波

    _gray_dif = _max-_min;

//    tool::DebugOut(STR(vt1), name, max);
//    tool::DebugOut(STR(vt2), name, med_m);
//    tool::DebugOut(STR(vt3), name, min);
//    tool::DebugOut(STR(gray), name, _gray);
//    tool::DebugOut(STR(sub), name, max-min);
 //   tool::DebugOut(STR(gray_s), name, gray);
    //vector<Rect> rect = probably_locate();

//    Mat current = tool::Resize(_gray,Size(32,48));
//    tool::DebugOut(STR(current), name, current);
//    LBP LB;
//    Size ResImgSiz(16, 24);
//    Mat LBPImage;
//    LB.ComputeLBPFeatureVector_Rotation_Uniform(current, ResImgSiz, LBPImage);
//    _LBPImage=LBPImage;


    // outfile << LBPImage << endl;
  //  deal_char();

}
DigtalLocate::DigtalLocate(DigtalLocate *Dig,cv::Rect rect,int id){
    //自己构造出自自己//最大值和最小值滤波
    this->name = Dig->name+"_"+to_string(id);
    this->const_Mat=Dig->const_Mat(rect);
    this->imgMat=Dig->imgMat(rect);
    this->yellow = Dig->yellow(rect);
    this->green = Dig->green(rect);
    this->blue = Dig->blue(rect);
    this->_min = Dig->_min(rect);
    this->_max = Dig->_max(rect);
    this->_medium = Dig->_medium(rect);
    this->_gray_dif = Dig->_gray_dif(rect);
    this->_gray = Dig->_gray(rect);
    this->black = Dig->black(rect);
    if(Dig->best_gray==&Dig->_gray)
        this->best_gray = &this->_gray;
    else
        this->best_gray = &this->_max;
}

void DigtalLocate::recongize_test(){
    tool::DebugOut(STR(min), name, _min);
    //中值滤波　
    Mat out;
    medianBlur(_gray,out,3);
    tool::DebugOut(STR(med), name, out);
    Mat th1;
    int t = (int)threshold(out,th1,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
    tool::DebugOut(STR(th1), name, th1);
    cout<<tool::location(th1)<<endl;

   // Mat best_test = tool::char_threshold2(const_Mat,_gray,name,_max,0);
   // tool::DebugOut(STR(best_test), name, best_test);
    //这是使用图割的方式...

}
Mat DigtalLocate::Get_LBP(){
    Mat gray_s = tool::stretch(_gray);
    medianBlur(gray_s,gray_s,3);
    Mat current = tool::Resize(gray_s,Size(32,48));
    tool::DebugOut(STR(gray_t1), name,current);
    LBP LB;
    Size ResImgSiz(8, 16);
    Mat LBPImage;
   // LB.ComputeLBPFeatureVector_256(current, ResImgSiz, LBPImage);
    LB.ComputeLBPFeatureVector_Uniform(current, ResImgSiz, LBPImage);
   // LB.ComputeLBPFeatureVector_Rotation_Uniform(current, ResImgSiz, LBPImage);
    return std::move(LBPImage);
}
bool DigtalLocate::reconize_char(){

    Mat threholdpic;
    int t = (int)threshold(_gray,threholdpic,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
    int avg = static_cast<int >(mean(threholdpic)[0]);
    tool::DebugOut(STR(ALL), name, const_Mat);
    if(avg<205&&avg>20) {
        Ptr<SVM> svm = ml::StatModel::load<SVM>("../svm-model/digtal_binary.xml");
        Mat ans;
        svm->predict(Get_LBP(), ans);
        //
        if(ans.at<float>(0, 0)){
            tool::DebugOut(STR(combine_p), name, const_Mat);
        }else{

          //  cout<<Get_LBP()<<endl;
            tool::DebugOut(STR(combine_S), name, _gray);
            tool::DebugOut(STR(combine_N), name, const_Mat);
        }

        return (bool) ans.at<float>(0, 0);
    }
    else{
        cout<<avg<<endl;

    }
    tool::DebugOut(STR(combine_T), name, const_Mat);
   return false;
}

string  DigtalLocate::get_char(){
    vector<Rect> canditaes;
    Mat threholdpic;
    vector<string> resultArray;
    Rect rt;
    double best_angel=0;
    double angel = -361;
    if(mean(_max)[0]<190) {
      //  best_gray = &_gray;
        best_gray = &_max;  //如果不选择max图像容易操作过分割
    } else{
        best_gray = &_gray;
    }

    Mat gray_t = tool::stretch(*best_gray);  //灰度拉伸
    tool::DebugOut(STR(gray_source), name, *best_gray);  //灰度拉升钱
    tool::DebugOut(STR(gray_stretch), name, gray_t);    //灰度拉升后
    int t = (int)threshold(gray_t,threholdpic,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
    //分水岭算法输出的图像
    Mat threshold_diff =  tool::get_best_side(this, t+80, t-50,gray_t); //用于字符分割
    tool::DebugOut(STR(threshold_diff_sub), name, threshold_diff);
    tool::DebugOut(STR(threshold_diff_ostu), name, threholdpic);

    //将结果扔入阈值分割里面
    //本来可有进行水平字符矫正
    //tool::DebugOut(STR(ostu), name, threshold_diff);

   // double angel = tool::correct_error(threshold_diff,best_angel);  函数有问题．这个会改变传入的值．
    //putText(imgMat,to_string(angel),Point(0,10),1,1,Scalar(255,255,255));

    canditaes = splite_char(threshold_diff,rt);

    vector<Rect> newcanditaes;
    string result;
    for(uint8_t c=0;c<canditaes.size();c++) {
        DigtalLocate sub = DigtalLocate(this,canditaes[c],c);
        bool ist = sub.reconize_char();
      //  cv::rectangle(imgMat, canditaes[c], Scalar(0, 255, 0), 3, 1, 0);
        //cv::rectangle(green, canditaes[c], Scalar(0, 0, 0), 40, 1, 0);
        cv::rectangle(green, canditaes[c], Scalar(0, 255, 0), 20, 1, 0);
        cv::Mat roiImg_raw  = const_Mat(canditaes[c]);
        tool::DebugOut(STR(sub_char), sub.name, sub.const_Mat);
        Mat best_test;
        if(ist){
            //确定是数字
          //  if(mean(sub._gray)[0]>70) {
             //  threshold(sub._gray, best_test, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
            Mat t = sub._gray.clone();

              //  best_test = tool::char_threshold(sub.const_Mat,t,sub.name,0);
//            }
//            else{
//                threshold(sub._max, best_test, 0, 255, CV_THRESH_TRIANGLE | CV_THRESH_BINARY);
//            }


            //用图割的效果更好所以这段分水岭算法的代码算是作废
//            Mat gray_t = tool::stretch(sub._gray);  //灰度拉伸
//            Mat temp;
//            int t3 = (int)threshold(gray_t,temp,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
//            //分水岭算法输出的图像
//            //Mat threshold_diff =  tool::get_best_side(this, t+80, t-50,gray_t); //用于字符分割
//             best_test =  tool::get_best_side(&sub, t3+80, t3-50,gray_t); //对差值图使用 给定一个阈值
//             tool::DebugOut(STR(finnal2), sub.name, best_test);


            /**  这部分是同选择合适的阈值法
//                Mat th1;
//                threshold(sub._gray,th1,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
//                tool::DebugOut(STR(finnal4), sub.name, sub._gray);
//                tool::DebugOut(STR(finnal5), sub.name,th1);
//                if(mean(th1)[0]<50){
//                    threshold(sub._max,th1,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
//                }
//                tool::DebugOut(STR(finnal3), sub.name, th1);
//            best_test = th1;
             **/
            threshold(sub._medium,best_test,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);

            tool::DebugOut(STR(finnal6), sub.name, best_test);
//             best_test = tool::char_threshold(sub.const_Mat,t,sub.name,sub._max,0);
//             tool::DebugOut(STR(finnal), sub.name, best_test);

            sub.black = sub.black|best_test;
            tool::DebugOut(STR(finna7), sub.name, sub.black);
            Mat canny;

            angel = tool::correct_error(best_test, angel,canny);  //整体的角度　best_angle 矫正钱
            tool::DebugOut(STR(canny),sub.name,canny);
            angel = 0;

//            cout<<angel<<endl;
//            if(angel==-361) {
//
//                Mat threshold_diff_c ;
//                cvtColor(threshold_diff,threshold_diff_c, CV_GRAY2BGR);
//                putText(threshold_diff_c,to_string(angel),Point(0,10),1,1,Scalar(255,0,0));
//                tool::DebugOut(STR(threshold_diff_c), name, threshold_diff_c);
//            }
            double  dangle = 90-angel;
            if(dangle<=0){
                dangle = 0;
                tool::DebugOut(STR(error), sub.name, sub._gray);
            }
            Mat ss = tool::get_R_mat(best_test,dangle);
            Mat sssrc = tool::get_R_mat(sub.const_Mat,dangle);
            ss = splite_char_sub(ss);
            tool::DebugOut(STR(ss_correct_after), sub.name, ss); //ss矫正后
            tool::DebugOut(STR(ss_correct_after_src), sub.name, sssrc); //ss矫正后源文件方便展示
            result +=tool::location(ss);
            tool::DebugOut(STR(size_after), sub.name, ss); //ss矫正后
            //result += tool::getNOByLine(ss, tool::getLineRange(best_test.rows, best_test.cols), false,angel);
            resultArray.push_back(result);
          //  tool::DebugOut(STR(sub_candidates_P), name, roiImg_raw, name, c);
            newcanditaes.push_back(canditaes[c]);
        }
    }


    if(result.size()!=0){
     //   cout<<result<<endl;
    }
    tool::DebugOut(STR(imgMat), name, imgMat);
    for(uint8_t c=0;c<newcanditaes.size();c++) {
        cv::rectangle(imgMat, newcanditaes[c], Scalar(255, 0, 0), 3, 1, 0);
       // cv::rectangle(blue, newcanditaes[c], Scalar(0, 0, 0), 40, 1, 0);
     //   cv::rectangle(black, newcanditaes[c], Scalar(255,255,255), -1, 0, 0);
        cv::rectangle(blue, newcanditaes[c], Scalar(255, 0, 0), 20, 1, 0);
     //   cv::Mat roiImg_raw  = const_Mat(newcanditaes[c]);
        //roiImg_raw.copyTo(black);
     //  putText(imgMat,result,Point(30,30),1,1,Scalar(255,255,255),1);  //调试不需要

//        tool::DebugOut(STR(sub_candidates_s), name, roiImg_raw, name, c);
//        tool::DebugOut(STR(sub_candidate_s), name, imgMat);
    }

   return result;
}


vector<Rect> DigtalLocate::probably_locate(){
    const int must_front_high = 150;
    const int must_front_low = 100;
    Mat threshold_diff, threshold_white, combine,Ex_combine;
    vector<Rect> canditaes;    //数码管特征图
    //使用分水岭算法提取有颜色的区域
    // 利用了颜色特征

    threshold_diff =  tool::get_best_side(this, must_front_high, must_front_low,_gray_dif); //对差值图使用给定一个阈值
    cv::threshold(_max,threshold_white,254, 255, CV_THRESH_BINARY);  //曝光过度值图
    combine = threshold_diff|threshold_white;
    cv::morphologyEx(combine, Ex_combine, cv::MORPH_CLOSE, cv::Mat::ones(5, const_Mat.cols/50, CV_8UC1));  //膨胀与腐蚀

    /******deubg******/
    tool::DebugOut(STR(gray_dif), name, _gray_dif);        //差值图
    tool::DebugOut(STR(set_result), name, threshold_diff); //分水岭结果图
    tool::DebugOut(STR(threshold_white), name, threshold_white); //曝光过度图
    tool::DebugOut(STR(before_combine), name, combine); //曝光过度图
    tool::DebugOut(STR(combine), name, Ex_combine);
    /************end********/


    canditaes = get_digital_area(Ex_combine);
    for(unsigned int  c=0;c<canditaes.size();c++) {
       // cv::rectangle(imgMat, canditaes[c], Scalar(0, 255, 255), 5, 1, 0);
        //cv::rectangle(yellow, canditaes[c], Scalar(0, 0, 0), 80, 1, 0);
        cv::rectangle(yellow, canditaes[c], Scalar(0, 255, 255), 80, 1, 0);
        cv::Mat roiImg_raw  = const_Mat(canditaes[c]);
        //tool::DebugOut(STR(first_ROI), name, roiImg_raw);
        //立马输出子图
        tool::DebugOut(STR(sub_candidates), name, roiImg_raw, name, c);
        DigtalLocate sub = DigtalLocate(this,canditaes[c],c);
        string s = sub.get_char();
        //结果
        //调试出去画图结果
        putText(imgMat,s,Point(canditaes[c].x-50,canditaes[c].y-50),1,10,Scalar(255,255,255),10);
        if(s.size()!=0)
            result_ROI.push_back({s,Rect()});
    }





    out_put_result_mat =  tool::DebugOut(STR(out_resutl), name, imgMat);
    tool::DebugOut(STR(yellow), name, yellow);
    tool::DebugOut(STR(blue), name, blue);
    tool::DebugOut(STR(green), name, green);
    tool::DebugOut(STR(black), name, black);
    return std::move(canditaes);
    //构成了candidate　然后画出来
}

template <typename Activation>
double rescale(double x) {
    Activation a;
    return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}
void DigtalLocate::find(){
    Mat gray;
    cv::cvtColor(srcMat, gray, CV_BGR2GRAY );
    Mat dst_gray(100, this->srcMat.cols*100/this->srcMat.rows, CV_8UC1, Scalar(255));


    resize(gray, dst_gray, dst_gray.size());
    cv::morphologyEx(dst_gray, dst_gray, cv::MORPH_CLOSE, cv::Mat::ones(5, 5, CV_8UC1));
    tool::DebugOut(STR(size), name, dst_gray);

    vector<Vec4i>lines;
    Mat canny;
    Canny(dst_gray,canny,100,300,3);

    Mat out = dst_gray.clone();
    tool::DebugOut(STR(canny), name, canny);
    vector<double> test;
    for(int i=-90;i<90;i++) {
//        double  x= i*0.5;
//        test.push_back(x);
        double x= i;
        test.push_back(x/2.0);
        test.push_back(x);
    }
    auto s = tool::radon(canny,test);
    vector<double> max_array;
    for_each(s.begin(),s.end(),[&](vector<double> single){max_array.push_back(*max_element(single.begin(),single.end()));});
    //角度限制在   60~120 度之间
    auto time = max_element(max_array.begin()+60*2,max_array.begin()+120*2+1);
    long  angle = std::distance(time,max_array.begin());
    //
    auto time2 =  max_element(max_array.begin(),max_array.begin()+5*2+1);
    auto time3 = max_element(max_array.end()-5*2,max_array.end());
    double wangle=0;
    if(*time3>*time2){
        wangle =  std::distance(time3,max_array.end());
    }
    else{
        wangle =  std::distance(time2,max_array.begin());
    }
    Mat answer;
    tool::imageRotate2(srcMat,answer,-wangle/2);

    tool::DebugOut(STR(result_s_r), name, answer);


    double angle_t = 90+angle/2;
    cout<<angle_t<<"||"<<-wangle/2<<endl;

    cvtColor(out,out,CV_GRAY2BGR);


}
void DigtalLocate::Hist(){

    const int channels[1]={0};

    const int histSize[1]={256};

    float hranges[2]={0,255};

    const float* ranges[1]={hranges};

    MatND hist;

    Mat gray;
    cv::cvtColor(srcMat, gray, CV_BGR2GRAY );


    cv::mean(gray);
    cv::Mat mean1;
    cv::Mat stdDev;
    cv::meanStdDev(gray, mean1, stdDev);
    double avg =  mean(gray)[0];
    double stddev  = (double)srcMat.cols/srcMat.rows;
   // double stddev = stdDev.ptr<double>(0)[0];
    cout<<avg<<","<<stddev<<endl;
}

void DigtalLocate::test_char() {
    Mat desImg(32, 32, CV_8UC1, Scalar(0));//创建一个全黑的图片

    Mat dst(26, this->srcMat.cols*26/this->srcMat.rows, CV_8UC1, Scalar(255));


      Mat temp,dst2 ;
      cvtColor(srcMat,dst2,CV_RGB2GRAY);
      resize(dst2, dst, dst.size());
      Mat imageROI;
      imageROI = desImg(Rect((32-dst.cols)/2, 3, dst.cols, dst.rows));
      dst.copyTo(imageROI);
      desImg=~desImg;
      tool::DebugOut(STR(desImg), name,desImg);
//    cv::Mat_<uint8_t> resized;
//    cv::resize(desImg, resized, cv::Size(32, 32));
//    vec_t data;
//    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
//                   [=](uint8_t c) { return (255 - c) * (1 - -1.0) / 255.0 + -1.0; });
//    network<sequential> nn;
//    nn.load("LeNet-model");
//
//    // recognize
//    auto res = nn.predict(data);
//    vector<pair<double, int> > scores;
//
//    // sort & print top-3
//    for (int i = 0; i < 10; i++)
//        scores.emplace_back(rescale<tanh_layer>(res[i]), i);
//
//    sort(scores.begin(), scores.end(), greater<pair<double, int>>());
//
//    for (int i = 0; i < 1; i++)
//        cout << scores[i].second << endl;
//
//    tool::DebugOut(STR(out_t), name, desImg, to_string(scores[0].second),0);

//    Ptr<SVM> svm1 = StatModel::load<SVM>("../datasource/mnist_svm.xml");
//    cv::Mat testMat = desImg.clone().reshape(1,1);
//    testMat.convertTo(testMat, CV_32F);
//
 //   int predicted  = svm1->predict(testMat);
    //std::cout << std::endl  << "Recognizing following number -> " << predicted << std::endl << std::endl;

}
//传入的是二值化后的图像
//vector<cv::Rect> DigtalLocate::splite_char_all(cv::Mat &src,cv::Rect &group){
//    //竖直分割,　在进行水平分割
//    const double height = src.rows;
//    //vector<int> vec = tool::get_col_number(src);
//    vector<int> vec_row = tool::get_row_number(src);
//    vector<pair<int, int> > vecty = tool::get_Area(vec_row, src.cols);
//    auto iter = vecty.begin();
//    while (iter != vecty.end()) {  //保证有一定的高度,数码管具有一个最小的高度
//        if ((*iter).second - (*iter).first < 20) iter = vecty.erase(iter);
//        else ++iter;
//    }
//    for(auto s:vecty){
//        int set_height = s.second - s.first;
//        int set_y =s.first;
//        //构造ROI区域
//        Mat col= src(Rect(0,set_y,src.cols,set_height));
//        //
//        vector<int> vec_col = tool::get_row_number(col);
//
//
//    }
//
//    //排除最小广宽度区域　
//}


Mat DigtalLocate::splite_char_sub(cv::Mat &src){
    vector<int> vec = tool::get_col_number(src,0.7);
    //减去小数点的像素
    //for_each(vec.begin(),vec.end(),[&](int s){s=max(0,s-src.rows/10);});
    //vector<pair<int, int> > vect = tool::get_Area(vec, src.rows);
    auto one = std::find_if(vec.begin(),vec.end(),[](int i){ return  i>2;});
    auto two = std::find_if(vec.rbegin(),vec.rend(),[](int i){ return i>2;});
    int set_x=0;
    if(one!=vec.end()){
        set_x = one - vec.begin();
    }
    int set_width = src.cols - set_x;
    if(two!=vec.rend()){
        set_width = src.cols-(two-vec.rbegin())-set_x;
    }
    return src(Rect(set_x,0,set_width,src.rows));

}
vector<cv::Rect> DigtalLocate::splite_char(cv::Mat &src,cv::Rect &group) {

        const double height = src.rows;

        vector<int> vec_row = tool::get_row_number(src);

        vector<pair<int, int> > vect2 = tool::get_Area(vec_row, src.cols);
        auto iter = vect2.begin();
        int set_height=0,set_y=0;
        //if(vect2.size()==0) {

       // }
        while (iter != vect2.end()) {
            if ((*iter).second - (*iter).first < 20) iter = vect2.erase(iter);
            else ++iter;
        }
        if(vect2.size()!=0) {
             set_height = vect2[0].second - vect2[0].first;
            set_y = vect2[0].first;
        } else {
            set_height = 0;
            set_y = 0;
        }



        vector<Rect> candidates;
        if(set_height>0) {
            Mat srcrow = src(Rect(0, set_y, src.cols, set_height));
            vector<int> vec = tool::get_col_number(srcrow);
            vector<pair<int, int> > vect = tool::get_Area(vec, srcrow.rows);

            //画出直方图
            Mat temp_b = tool::get_hist_show(vec);
            tool::DebugOut(STR(hist), name,(temp_b));

            for (auto s: vect) {
                int width = s.second - s.first;
                if (width < set_height / 16.0 || width > set_height * 40.0 / 30) continue;  //hist 分析法
                candidates.push_back(CvRect(s.first, set_y, s.second - s.first, set_height));
            }
        }


        //返回对于的grop
        if(!candidates.empty()) {
            group.y= set_y;
            group.height = set_height;
            group.width = candidates.back().x-candidates.front().x+candidates.back().width;
            group.x = candidates.front().x;
        }
        return std::move(candidates);
}
void DigtalLocate::jujue_char() {
    cv::Mat gray, HSV, fiddle,gray_bar,threshold_value,threshold2,compete,img_morphology_ex;
    std::vector<cv::Mat> channels;
    cv::cvtColor(srcMat, HSV, CV_BGR2HSV);
    cv::cvtColor(srcMat, gray, CV_BGR2GRAY);
    cv::split(HSV, channels);

    medianBlur(gray,gray,5);
    medianBlur(channels[2],channels[2],5);
   // bilateralFilter(gray,fiddle,5, 25*2, 25/2 );

    double MaxValue, MinValue;
    cv::minMaxLoc(channels[2], &MinValue,&MaxValue);
    if(MaxValue-MinValue<100) {
        tool::DebugOut(STR(gray_test), name,(channels[2]));
        return ;
    }

    if(mean(channels[2])[0]>190) {
        threshold(gray,threshold_value,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
    }
    else {
        double t = threshold(channels[2],threshold_value,0, 255, CV_THRESH_TRIANGLE|CV_THRESH_BINARY);

        if((mean(threshold_value)[0]<30)) {
         //   cout<<t<<endl;
            threshold(channels[2],threshold_value,t-40, 255, CV_THRESH_BINARY);
        }
        if((mean(threshold_value)[0]>90)) {
            threshold(channels[2],threshold_value,t+(mean(threshold_value)[0]-90), 255, CV_THRESH_BINARY);
        }
    }

    tool::DebugOut(STR(gray_s), name,gray);
    tool::DebugOut(STR(channels), name,channels[2]);

    //圆检测来判断子通道
    vector<Vec3f> circles;
    HoughCircles( gray, circles, CV_HOUGH_GRADIENT,1.5, 10, 100, 90, 40, 0 );

  //  putText(srcMat,to_string(circles.size()),Point( srcMat.rows/2,srcMat.cols/4),CV_FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0));
//    if(mean(threshold_value)[0])>100;
    tool::DebugOut(STR(srcMat), name,srcMat);
    tool::DebugOut(STR(threshold2), name,threshold_value);

    cv::morphologyEx(threshold_value, img_morphology_ex, cv::MORPH_CLOSE, cv::Mat::ones(6, 1, CV_8UC1));
    tool::DebugOut(STR(img_morphology_ex_s), name,img_morphology_ex);
    //adapt clow split
    vector<int> vec = tool::get_col_number(img_morphology_ex);
    vector< pair<int,int> > vect = tool::get_Area(vec,srcMat.rows);
    vector<Rect> candidates;


    for(auto s: vect) {
        int width = s.second-s.first;
        if(width<20||width>250) continue;  //hist 分析法
         candidates.push_back(CvRect(s.first,0,s.second-s.first,300));
    }


    Mat hist = tool::get_hist_show(vec);
    tool::DebugOut(STR(hist), name, hist);
    Mat scopy = srcMat.clone();
    for(uint8_t i=0;i<candidates.size();i++) {
        //cout<<candidates.size()<<endl;
        cv::rectangle(scopy, candidates[i], Scalar(0, 255, 0), 2, 1, 0);
        Mat temp = srcMat(candidates[i]);
        Mat the = threshold_value(candidates[i]);
        putText(temp,to_string(mean(the)[0]),Point( 0,30),CV_FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0));
        tool::DebugOut(STR(roiImg_raw_s_s), name, temp, dirname, i);
    }
   tool::DebugOut(STR(roi_single_output), name, scopy);
    /*cv::findContours(img_morphology_ex, plate_contours,RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
    Mat scopy = imgMat.clone();
    for (size_t i = 0; i != plate_contours.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
        cv::rectangle(scopy, rect, Scalar(0, 255, 0), 2, 1, 0);
    }
    tool::DebugOut(STR(S_ROI_F), name, scopy);*/


   /* if(mean(threshold_value)[0]>95){
       // threshold(gray,threshold_value,t+55, 255, CV_THRESH_BINARY);
        threshold(gray,threshold_value,0, 255, CV_THRESH_TRIANGLE|CV_THRESH_BINARY);
    }
    tool::DebugOut(STR(threshold_value), name,threshold_value);
    cout<<mean(gray)[0]<<","<<mean(threshold_value)[0]<<"  ,value"<< t <<endl;
    if(mean(threshold_value)[0]>95){
        tool::DebugOut(STR(gray_f), name,gray);
    }
    else{
        tool::DebugOut(STR(gray_t), name,gray);
    }*/
    //canny =  DetectText::textDetection(gray, 0,name);

    //tool::DebugOut(STR(canny), name, canny);

}
void DigtalLocate::getsub_char(vector<Rect> &candidates) {
    vector<Rect> answer;
    //这里可以使用多线程优化
    for(uint8_t i =0;i<candidates.size();i++) {

        Mat src = srcMat(candidates[i]);  //获取子图
        // tool::DebugOut(STR(src_s), name,src,dirname,i);
        cv::Mat gray, HSV, gray_bar,threshold_value,threshold2,compete;
        std::vector<cv::Mat> channels;
        cv::cvtColor(src, HSV, CV_BGR2HSV);
        cv::cvtColor(src, gray, CV_BGR2GRAY);
        cv::split(HSV, channels);

        medianBlur(gray,gray,3);
        medianBlur(channels[2],channels[2],3);
        gray_bar = channels[2]-gray;   //目前是max(RGB)-mean(RGB) 可以改为max(RGB)-MIN(RGB) 之后的滤波方式也可改为最大值滤波和最小值滤波


        tool::DebugOut(STR(gray_bar_row), name,gray_bar,dirname,i);
        tool::DebugOut(STR(higt_channels), name,channels[2],dirname,i);
        cv::threshold(gray_bar,threshold_value,75, 255, CV_THRESH_BINARY);  //参数自己调整
        tool::DebugOut(STR(threshold_value_s), name,threshold_value,dirname,i);
        cv::threshold(channels[2],threshold2,254, 255, CV_THRESH_BINARY);

        compete = threshold_value|threshold2;

        tool::DebugOut(STR(threshold2_s), name,threshold2,dirname,i);
        cv::morphologyEx(compete, compete, cv::MORPH_CLOSE, cv::Mat::ones(5, candidates[i].width/50, CV_8UC1));
        tool::DebugOut(STR(compete_S), name,compete,dirname,i);
        vector<Rect> sub_canditaes;
        //对子图进行合并
        sub_canditaes = extra_marge_sub_ara(compete);
        Mat sub_candid = imgMat.clone();
        vector<Mat> mat_row;
        Mat x = srcMat.clone();
        for(uint8_t c=0;c<sub_canditaes.size();c++) {

            Rect a(sub_canditaes[c].x+candidates[i].x,sub_canditaes[c].y+candidates[i].y,sub_canditaes[c].width,sub_canditaes[c].height);
            cv::rectangle(sub_candid, a, Scalar(0, 255, 0), 3, 1, 0);
            cv::Mat roiImg_raw  = x(a);
            mat_row.push_back(roiImg_raw);
            tool::DebugOut(STR(sub_candidate), name, sub_candid);
            tool::DebugOut(STR(sub_raw_t), name, roiImg_raw, dirname, c+100*i);
        }



        for(uint8_t j=0;j<sub_canditaes.size();j++)  {
          //  int current = sub_canditaes[j].width * sub_canditaes[j].height;
          //  if(current<rect_area_min||sub_canditaes[j].width>0.65*srcMat.cols) continue;
            //内容分析
            cv::Mat mean1;
            cv::Mat stdDev;
            Mat sub_channels(channels[2](sub_canditaes[j]));
            Mat sub_gray(gray(sub_canditaes[j]));
            cv::meanStdDev(sub_channels, mean1, stdDev);

            double avg = mean1.ptr<double>(0)[0];
            double stddev = stdDev.ptr<double>(0)[0];
           // if(avg>190) continue;
            //gray的平局灰度值分析

            putText(mat_row[j],to_string((int)stddev)+"||"+to_string((int)avg),Point(0,10),1,1,Scalar(255,255,255));
            tool::DebugOut(STR(sub_raw_text), name, mat_row[j], dirname, j+100*i);

            if(stddev<30&&(avg<210)) {

                continue;
            };
            if(stddev<47&&(avg<175)) {
                continue;
            };


            cv::meanStdDev(sub_gray, mean1, stdDev);
            avg = mean1.ptr<double>(0)[0];
            stddev = stdDev.ptr<double>(0)[0];
            putText(mat_row[j],to_string((int)avg)+"||"+to_string((int)stddev),Point(0,20),1,1,Scalar(0,0,0));
            tool::DebugOut(STR(sub_raw_text_g), name, mat_row[j], dirname, j+100*i);
            if(avg>185) continue;

            vector<Vec3f> circles;
            HoughCircles( sub_gray, circles, CV_HOUGH_GRADIENT,1.5, 10, 100, 100, 40, 0 );
           // cout<<circles.size()<<endl;
            if(circles.size()>1) {

                continue;
            };

            //find the best thresh
            Mat best_threshold,best_gray_pic;
            if(mean(sub_channels)[0]>180) {
                best_gray_pic = sub_gray;
                threshold(sub_gray, best_threshold,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
            }
            else {
                best_gray_pic = sub_channels;
                double temp_threshold = threshold(sub_channels,best_threshold,0, 255, CV_THRESH_TRIANGLE|CV_THRESH_BINARY);
                double mean_value = mean(best_threshold)[0];
                //std::cout<<temp_threshold<<"||"<<mean_value<<"||"<<100*i+j<<endl;
                if((mean_value<35)) {
                    threshold(sub_channels, best_threshold, temp_threshold - 40, 255, CV_THRESH_BINARY);
                }
                else if(mean_value>90) {
                    temp_threshold = threshold(sub_channels,best_threshold,temp_threshold+(mean_value-90), 255, CV_THRESH_BINARY);
                    double mean_value = mean(best_threshold)[0];
                    if(mean_value>90)
                        threshold(sub_channels,best_threshold,temp_threshold+(mean_value-90), 255, CV_THRESH_BINARY);
                     //  std::cout<<temp_threshold<<"||"<<mean_value<<"||"<<100*i+j<<endl;
                }
            }

            //if(mean(best_threshold)[0]>200) continue;
            if(mean(best_threshold)[0]>200||mean(best_threshold)[0]<20) continue;

            output_sub_mat_url.push_back(tool::DebugOut(STR(src_channe), name,sub_channels,dirname,100*i+j));
            output_sub_mat_url.push_back(tool::DebugOut(STR(src_threshold), name,best_threshold,dirname,100*i+j));
            output_sub_mat_url.push_back(tool::DebugOut(STR(src_gray), name,best_gray_pic,dirname,100*i+j));

            Mat best_test = best_threshold.clone();
            double best_angel;
            Mat canny;
            double angel = tool::correct_error(best_test,best_angel,canny);
          //  cvtColor(best_test, best_test, CV_GRAY2BGR);
          //  cout<<best_angel<<endl;
            best_angel=best_angel;
            putText(best_test,to_string(angel),Point(0,10),1,1,Scalar(255,255,255));
            output_sub_mat_url.push_back(tool::DebugOut(STR(best_test), name,best_test,dirname,100*i+j));
            if(best_angel>20&&best_angel<120) continue;
            if(best_angel>170&&best_angel<340) continue;
           // best_threshold = best_test;


            Rect group,group2;
            vector<Rect> char_candidate = splite_char(best_threshold,group);  //函数内部已经重新修正了rect 的值
            vector<Rect> true_char_candidate = splite_char(best_test,group2);
            int c=0; string result;
          //  cout<<char_candidate.size()<<endl;
            for(auto t:char_candidate) {
                Mat sresult = best_threshold(t);
                double avg =  mean(sresult)[0];
                double stddev  = (double)sresult.cols/sresult.rows;


                // / if(avg>175) continue;

                if(avg<50&&stddev<0.3) continue;
                Mat ss = tool::get_R_mat(sresult,angel);
                result += tool::getNOByLine(ss, tool::getLineRange(sresult.rows, sresult.cols), false,angel);

                output_sub_mat_url.push_back(tool::DebugOut(STR(ss_ss), name,ss,dirname,250*i+50*j+(++c)));
                output_sub_mat_url.push_back(tool::DebugOut(STR(src_threshold_s), name,sresult,dirname,250*i+50*j+(++c)));
                output_sub_mat_url.push_back(tool::DebugOut(STR(src_gray_s), name,best_gray_pic(t),dirname,250*i+50*j+(++c)));
                t.x += sub_canditaes[j].x + candidates[i].x;
                t.y += sub_canditaes[j].y + candidates[i].y;
                answer.push_back(t);
            }
            group.x+=candidates[i].x+sub_canditaes[j].x;
            group.y+=candidates[i].y+sub_canditaes[j].y;
            if(!result.empty()){
                result_ROI.push_back({result,group});
            }
            Mat single = srcMat.clone();
            for(auto s:result_ROI) {
                cv::rectangle(single, s.second, Scalar(0, 255, 0), 10, 1, 0);
               // putText(single,to_string(s.first),Point(s.second.x,s.second.y+10),1,10,Scalar(255,255,255),10);
            }
            out_put_result_mat = tool::DebugOut(STR(out_result), name,single);
        }


    /*    for(int j=0;j<subcanditaes.size();j++) {
            subcanditaes[j].x+=candidates[i].x;
            subcanditaes[j].y+=candidates[i].y;
            //tool::scale_rect(subcanditaes[i],1,1);

        }*/
    }
    candidates.swap(answer);
}
void DigtalLocate::extract_char(){

    cv::Mat gray,med_blur,threshold1,img_morphology_ex,gray_neg,gray_bar,gray_neg_blur,img_threshold,canny,compete,edage;
    cv::Mat HSV;
    std::vector<cv::Mat> channels;
    cv::cvtColor(srcMat, HSV, CV_BGR2HSV);
    cv::cvtColor(srcMat, gray, CV_BGR2GRAY);
    // 通道分离
    cv::split(HSV, channels);
//    tool::DebugOut(STR(channel1_s), name,channels[0]);
//    tool::DebugOut(STR(channel2_S), name,channels[1]);
//    tool::DebugOut(STR(channel3_S), name,channels[2]);

    medianBlur(gray,gray,3);
   // medianBlur(channels[2],channels[2],3);
    gray_bar = channels[2]-gray;
    tool::DebugOut(STR(gray_s), name,gray);
    tool::DebugOut(STR(channel3_s), name, channels[2]);
    tool::DebugOut(STR(gray_bar_s), name, gray_bar);
    threshold(gray_bar,threshold1,75, 255, CV_THRESH_BINARY);
    threshold(channels[2],img_threshold,254, 255, CV_THRESH_BINARY);
    tool::DebugOut(STR(img_threshold_s), name, img_threshold);
    tool::DebugOut(STR(threshold1_s), name, threshold1);
    compete = threshold1|img_threshold;
    tool::DebugOut(STR(compete), name, compete);
    cv::morphologyEx(compete, img_morphology_ex, cv::MORPH_CLOSE, cv::Mat::ones(20, 20, CV_8UC1));
    //cv::morphologyEx(img_morphology_ex, img_morphology_ex, cv::MORPH_CLOSE, cv::Mat::ones(5, 2, CV_8UC1));
    tool::DebugOut(STR(img_morphology_ex_S), name, img_morphology_ex);

    vector<Rect> candidates;
    candidates = extra_marge_sub_ara(img_morphology_ex);

    Mat scopy = srcMat.clone();
    bool t = false;
    for(uint8_t i=0;i<candidates.size();i++) {

        cv::rectangle(scopy, candidates[i], Scalar(0, 255, 255), 2, 1, 0);
        Mat roiImg_raw(gray(candidates[i]));

        if(mean(roiImg_raw)[0]<60){   //输出H图
            roiImg_raw = channels[2](candidates[i]);
        }

        threshold(roiImg_raw,roiImg_raw,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
        tool::DebugOut(STR(roiImg_raw), name, roiImg_raw, dirname, i);
        t=true;
    }
    if(t)
      tool::DebugOut(STR(ROI_EDGE), name, scopy);



    return;

//    threshold(gray,gray,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
//    canny =  DetectText::textDetection(channels[2], 0,name);
//
//    tool::DebugOut(STR(canny), name, canny);
    //图像滤波

//
//    std::vector<std::vector<cv::Point> > plate_contours;
//    threshold(med_blur,threshold1,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
//    tool::DebugOut(STR(threshold_S), name,threshold1);
//    cv::morphologyEx(threshold1, img_morphology_ex, cv::MORPH_CLOSE, cv::Mat::ones(6, 1, CV_8UC1));
//    tool::DebugOut(STR(img_morphology_ex_s), name,img_morphology_ex);
//    //adapt clow split
//    vector<int> vec = tool::get_col_number(img_morphology_ex);
//    vector< pair<int,int> > vect = tool::get_Area(vec);
//    vector<Rect> candidates(vect.size());


//    for(auto s: vect) {
//        int width = s.second-s.first;
//        if(width<10||width>80||width==imgMat.cols-1) continue;
//        candidates.push_back(CvRect(s.first,0,s.second-s.first,100));
//    }


//    Mat hist = tool::get_hist_show(vec);
//    tool::DebugOut(STR(hist), name, hist);
//    Mat scopy = imgMat.clone();
//    for(auto single:candidates) {
//        cv::rectangle(scopy, single, Scalar(0, 255, 0), 2, 1, 0);
//    }
//    tool::DebugOut(STR(roi_single_output), name, scopy);
//    /*cv::findContours(img_morphology_ex, plate_contours,RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
//    Mat scopy = imgMat.clone();
//    for (size_t i = 0; i != plate_contours.size(); ++i)
//    {
//        // 求解最小外界矩形
//        cv::Rect rect = cv::boundingRect(plate_contours[i]);
//        cv::rectangle(scopy, rect, Scalar(0, 255, 0), 2, 1, 0);
//    }
//    tool::DebugOut(STR(S_ROI_F), name, scopy);*/

}
/*void DigtalLocate::extrator_threshold(){
    cv::Mat HSV,med_blur,gray,img_threshold;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad;
    std::vector<cv::Mat> channels;
    cv::cvtColor(imgMat, HSV, CV_BGR2HSV);

    cv::split(HSV, channels);
    tool::DebugOut(STR(channel1), name,channels[0]);
    tool::DebugOut(STR(channel2), name,channels[1]);//像素值高,　说明发刚
    tool::DebugOut(STR(channel3), name,channels[2]);
    medianBlur(channels[2],med_blur,11);
    //局部二值话
    int blockSize = 25;
    int constValue = 10;
    cv::Mat local, local1;

   //单阈值法
    //　240
    threshold(med_blur, img_threshold,245 , 255,THRESH_TOZERO); //二值化

    cv::adaptiveThreshold(img_threshold, local, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
    tool::DebugOut(STR(img_threshold), name,img_threshold);

    //cv::adaptiveThreshold(med_blur, local1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
    //threshold(img_threshold,local,0, 255, THRESH_TRIANGLE|CV_THRESH_BINARY);


    tool::DebugOut(STR(local), name,local);
    tool::DebugOut(STR(local1), name,local1);

//    Sobel( med_blur, grad_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT );
//    Sobel( med_blur, grad_y, CV_8UC1, 0, 1, 3, 1, 0, BORDER_DEFAULT );
//    convertScaleAbs( grad_x, abs_grad_x );
//    convertScaleAbs( grad_y, abs_grad_y );
//    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
//    threshold(grad,grad,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);




   // threshold(med_blur,grad_x,0, 255, CV_THRESH_TRIANGLE|CV_THRESH_BINARY);

    cv::Mat mserClosedMat;
    cv::morphologyEx(local, mserClosedMat,
                     cv::MORPH_CLOSE, cv::Mat::ones(10, 1, CV_8UC1));

    tool::DebugOut(STR(result), name,mserClosedMat);
    std::vector<std::vector<cv::Point> > plate_contours;
    cv::findContours(mserClosedMat, plate_contours,RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    // 候选车牌区域判断输出

    std::vector<cv::Rect> candidates;
    const int imagemin = 0.0001*imgMat.rows * imgMat.cols;
    for (size_t i = 0; i != plate_contours.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
        int app = rect.width*rect.height;
        if(app<imagemin||rect.width>0.7*imgMat.rows||rect.height>0.3*imgMat.cols) continue;
        candidates.push_back(rect);
        //逻辑限制
    }
    //对candidtes 进行合并
    std::vector<cv::Rect> new_candidates;
    sort(candidates.begin(), candidates.end(), [](const cv::Rect &a,const cv::Rect &b){return (a.x<b.x);});
   // stable_sort(candidates.begin(), candidates.end(), [](const cv::Rect &a,const cv::Rect &b){return (a.y+a.height/2)<(b.y+b.height/2);});
    vector<int> status(candidates.size(), 0);
    if(candidates.size()!=0) {

        for(int i=0;i<candidates.size();++i) {
            if(status[i] == 1) continue;
            status[i] = 0;
            Rect target = candidates[i];
            for(int j=i+1;j<candidates.size();++j) {

                if(status[j] == 0) {
                    int current_value = candidates[j].y + candidates[j].height / 2;
                    if (abs(target.y + target.height / 2 - current_value) < 30  && abs(target.x-candidates[j].x)<(max(target.width, candidates[j].width)+candidates[j].height+candidates[j].width)) {

                        status[j] = 1;
                        if (target.height > candidates[j].height) {
                            target.width = abs(target.x - (candidates[j].x + candidates[j].width));
                        } else {
                            target.height = candidates[j].height;
                            target.y = candidates[j].y;
                            target.width = abs(target.x - (candidates[j].x + candidates[j].width));
                        }
                    }
                }
            }
            new_candidates.push_back(target);
        }

    }
//    cout<<"------"<<new_candidates.size()<<endl;
    for(int i=0;i<new_candidates.size();i++) {
        cv::rectangle(imgMat, new_candidates[i], Scalar(0, 255, 0), 2, 1, 0);
        cv::Mat roiImg, image_threshold, rsize, roiImg1;
        roiImg1 = imgMat(new_candidates[i]);
        roiImg = med_blur(new_candidates[i]);

        threshold(roiImg, image_threshold, 0, 255, THRESH_TRIANGLE | CV_THRESH_BINARY);
        Size size((int) (roiImg.cols * (100.0 / roiImg.rows)), 100);

        resize(roiImg, rsize, size);
        //cv::morphologyEx(rsize, rsize,cv::MORPH_CLOSE, cv::Mat::ones(5, 5, CV_8UC1));
        tool::DebugOut(STR(roiImg_the_resize), name, rsize, dirname, i + new_candidates.size());
        tool::DebugOut(STR(roiImg_T), name, roiImg1, dirname, i);
    }
    tool::DebugOut(STR(ROI_T), name, imgMat);

//    std::vector<std::vector<cv::Point>> contours ;
//    //获取轮廓不包括轮廓内的轮廓
//    cv::findContours(local , contours ,
//                     CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE) ;
//    cv::Mat result(local.size() , CV_8U , cv::Scalar(255)) ;
//    cv::drawContours(result , contours ,-1 , cv::Scalar(0) , -1) ;
//    tool::DebugOut(STR(result), name,result);
//    std::vector<std::vector<cv::Point>> allContours ;
//    cv::Mat allContoursResult(grad.size() , CV_8U , cv::Scalar(255)) ;
//    cv::findContours(grad , allContours ,
//                     CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE) ;
//    cv::drawContours(allContoursResult , allContours ,-1 ,
//                     cv::Scalar(0) , 2) ;



}*/
void DigtalLocate::atuo_Modify_Parametes() {

}

Mat  DigtalLocate::extera_by_edge_1(Mat &src) {
    //configure_parametes after　input it in class
    const int  erode_size=3, morphology_ex_size_x=5,morphology_ex_size_y=17;   //增大可以减少小的点

    const int rect_area_min = 0.0001*src.rows * src.cols;  //0.0001
    const float  max_width = 0.7*src.rows, min_width = 0,max_height = 0.3*src.cols,min_height = 0, min_rate = 0.3, max_rate = 5;
    const float thin_rate = 0;


    cv::Mat img_adapt_threshold, img_erode, img_morphology_ex;
    std::vector<cv::Rect> candidates;
    std::vector<std::vector<cv::Point> > plate_contours;
    cv::Mat allContoursResult(src.size(), CV_8U, cv::Scalar(0));

//    cv::adaptiveThreshold(src, img_adapt_threshold, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, get<0>(adapt_threshold_parameters), get<1>(adapt_threshold_parameters));

    //clear ioslate point
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(erode_size,erode_size));
    erode(img_adapt_threshold,img_erode,element);
    tool::DebugOut(STR(img_erode_t), name,img_erode);
    cv::morphologyEx(img_erode, img_morphology_ex, cv::MORPH_CLOSE, cv::Mat::ones(morphology_ex_size_x, morphology_ex_size_y, CV_8UC1));

    tool::DebugOut(STR(img_morphology_ex), name,img_morphology_ex);

    cv::findContours(img_morphology_ex, plate_contours,RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);

    tool::DebugOut(STR(img_erode_s), name,img_erode);
    //limit

    for (size_t i = 0; i != plate_contours.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
        int app = rect.width*rect.height;
        double rate = (double)rect.width/rect.height;

        cout<<app<<"|||"<<rect_area_min<<rate<<endl;

       if(app<rect_area_min||rect.width>max_width||rect.height>max_height||rate<min_rate||rate>max_rate) continue;
        candidates.push_back(rect);
    }
    for(auto single:candidates) {
        tool::thin_rect(single, CvSize(single.width*thin_rate,single.height*thin_rate));
        rectangle(allContoursResult,single,255,-1);
    }


    //输出记录的值

    return allContoursResult;


}

//search_by_edge
Mat  DigtalLocate::extract_map_by_edge(Mat &src) {
    //configure_parametes after　input it in class
    //设置一些不可能为包围盒区域内的点
    const int   morphology_ex_size=9;   //增大可以减少小的点
    const double rect_area_min = 0.003*src.rows * src.cols;  //限定数码管的最小面积
    const double  min_rate = 0.2, max_rate = 5;              //限定数码管的宽高比
    int adapt_threshold_parameters_b = 21;                  //　滤波区域大小　可随着缩小的比例减小而增大．
    int adapt_threshold_parameters_v = 10;                //限定数码管的阈值,　如果边缘弱可以通过减少这个值．


    cv::Mat img_adapt_threshold, img_erode, img_morphology_ex;
    std::vector<cv::Rect> candidates;
    std::vector<std::vector<cv::Point> > plate_contours;

    cv::Mat allContoursResult(src.size(), CV_8UC1, cv::Scalar(0));  //创建全黑色的图片

    cv::adaptiveThreshold(src, img_adapt_threshold, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, adapt_threshold_parameters_b ,adapt_threshold_parameters_v);

//    output_mat_url.push_back({STR(img_adapt_threshold_t),tool::DebugOut(STR(img_adapt_threshold_t), name,img_adapt_threshold)});

    cv::morphologyEx(img_adapt_threshold, img_morphology_ex, cv::MORPH_CLOSE, cv::Mat::ones(1, morphology_ex_size, CV_8UC1));

    output_mat_url.push_back({STR(img_morphology_ex_t),tool::DebugOut(STR(img_morphology_ex_t), name,img_morphology_ex)});

    cv::findContours(img_morphology_ex, plate_contours,RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i != plate_contours.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
        int app = rect.width*rect.height;
        double rate = (double)rect.width/rect.height;

       if(app<rect_area_min||rate<min_rate||rate>max_rate) continue;
        candidates.push_back(rect);
        rectangle(allContoursResult,rect,255,-1);
    }

    return allContoursResult;

}
//src 是灰度图
Mat  DigtalLocate::extera_by_edge_threshold(Mat &src,std::vector<cv::Rect> &scandidates){
    const int blockSize = 101, constValue = 20, erode_size=19, morphology_ex_size=71;   //增大可以减少小的点
    cv::Mat img_adapt_threshold, img_erode, img_morphology_ex;
    std::vector<cv::Rect> candidates;
    std::vector<std::vector<cv::Point> > plate_contours;
    cv::Mat allContoursResult(src.size(), CV_8U, cv::Scalar(0));
    threshold(src,img_adapt_threshold,254, 254, CV_THRESH_BINARY);
    output_mat_url.push_back({STR(img_adapt_threshold_t),tool::DebugOut(STR(img_adapt_threshold_t), name,img_adapt_threshold)});
    output_mat_url.push_back({STR(img_adapt_threshold_t),tool::DebugOut(STR(img_adapt_threshold_t), name,img_adapt_threshold)});


    cv::morphologyEx(img_adapt_threshold, img_morphology_ex, cv::MORPH_CLOSE, cv::Mat::ones(1, morphology_ex_size, CV_8UC1));
//
//    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(erode_size,erode_size));
//    erode(img_morphology_ex,img_erode,element);
//    img_morphology_ex = img_erode;

    output_mat_url.push_back({STR(img_erode),tool::DebugOut(STR(img_erode), name,img_erode)});
    output_mat_url.push_back({STR(img_morphology_ex_t),tool::DebugOut(STR(img_morphology_ex_t), name,img_morphology_ex)});
    return  img_adapt_threshold;
}

Mat  DigtalLocate::extera_by_edge_diff(Mat &src,std::vector<cv::Rect> &scandidates) {
    //configure_parametes after　input it in class
    const int blockSize = 101, constValue = 20, erode_size=19, morphology_ex_size=21;   //增大可以减少小的点

//    const int rect_area_min = 0.0005*src.rows * src.cols;
//    const float  max_width = 0.7*src.rows, min_width = 0,max_height = 0.25*src.cols,min_height = 0, min_rate = 0.5, max_rate = 5;
//    const float thin_rate = 0;

    cv::Mat img_adapt_threshold, img_erode, img_morphology_ex;
    std::vector<cv::Rect> candidates;
    std::vector<std::vector<cv::Point> > plate_contours;
    cv::Mat allContoursResult(src.size(), CV_8U, cv::Scalar(0));

   // cv::adaptiveThreshold(src, img_adapt_threshold, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, get<0>(adapt_threshold_parameters), get<1>(adapt_threshold_parameters));

   // threshold(src,img_erode,10, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
    threshold(src,img_adapt_threshold,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);

    output_mat_url.push_back({STR(img_adapt_threshold_t),tool::DebugOut(STR(img_adapt_threshold_t), name,img_adapt_threshold)});


    cv::morphologyEx(img_adapt_threshold, img_morphology_ex, cv::MORPH_CLOSE, cv::Mat::ones(1, morphology_ex_size, CV_8UC1));
//
//    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(erode_size,erode_size));
//    erode(img_morphology_ex,img_erode,element);
//    img_morphology_ex = img_erode;

    output_mat_url.push_back({STR(img_erode),tool::DebugOut(STR(img_erode), name,img_erode)});
    output_mat_url.push_back({STR(img_morphology_ex_t),tool::DebugOut(STR(img_morphology_ex_t), name,img_morphology_ex)});
//
//
//    cv::findContours(img_morphology_ex, plate_contours,RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
//
//    //limit  //这里进行程序优化,程序优化的准则就是如果父轮廓满足要求则就采用父轮廓,否则才放弃付轮廓采用子轮廓
//
//    for (size_t i = 0; i != plate_contours.size(); ++i)
//    {
//        // 求解最小外界矩形
//        cv::Rect rect = cv::boundingRect(plate_contours[i]);
//        int app = rect.width*rect.height;
//        double rate = (double)rect.width/rect.height;
//
//        if(app<rect_area_min||rect.width>max_width||rect.height>max_height||rate<min_rate||rate>max_rate) continue;
//        candidates.push_back(rect);
//    }
//    for(auto single:candidates) {
//        tool::thin_rect(single, CvSize(single.width*thin_rate,single.height*thin_rate));
//        rectangle(allContoursResult,single,255,-1);
//    }
//    scandidates.swap(candidates);
    return allContoursResult;

}
//这个函数必须保证只能拥有1个数码管不能拥有多个
vector<cv::Rect> DigtalLocate::get_digital_area(Mat &src){
    const double rect_area_min = 0.0003*const_Mat.rows * const_Mat.cols;  //先定数码管区域大小
    const double rect_area_max = 0.4*const_Mat.rows * const_Mat.cols;     //先定数码管区域大小
    const double max_height = 0.4*const_Mat.rows;
    const double max_width = 0.75*const_Mat.cols;
    std::vector<std::vector<cv::Point> > plate_contours;
    std::vector<cv::Rect> candidates;

    cv::findContours(src, plate_contours,RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i != plate_contours.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
        int app = rect.width*rect.height;
        double rate = (double)rect.width/rect.height;
        if(app<rect_area_min||app>rect_area_max||rect.width>max_width||rect.height>max_height||rate>8) continue;
        candidates.push_back(rect);
    }
    tool::marge(candidates);
    return std::move(candidates);
}
vector<cv::Rect> DigtalLocate::extra_marge_sub_ara(Mat &src) {

    //先定数码管区域的最大最小值,以及比例. 最大和最小值和原图有关而不是输入图像
    double rect_area_min = 0.0002*srcMat.rows * srcMat.cols;  //先定数码管区域大小
    double rect_area_max = 0.4*srcMat.rows * srcMat.cols;     //先定数码管区域大小
    double max_height = 0.4*srcMat.rows;
    double max_width = 0.75*srcMat.cols;

    std::vector<std::vector<cv::Point> > plate_contours;
    std::vector<cv::Rect> candidates;

    cv::findContours(src, plate_contours,RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);


    for (size_t i = 0; i != plate_contours.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
//        int app = rect.width*rect.height;
//        double rate = (double)rect.width/rect.height;
//        //limit ||rate<min_rate||rate>max_rate
//        if(app<rect_area_min||app>rect_area_max||rect.width>max_width||rect.height>max_height) continue;
         candidates.push_back(rect);
    }
    //区域merger
      std::vector<cv::Rect> new_candidates;
      sort(candidates.begin(), candidates.end(), [](const cv::Rect &a,const cv::Rect &b){return (a.x<b.x);});
      vector<int> status(candidates.size(), 0);
      if(candidates.size()!=0) {

          for(uint8_t i=0;i<candidates.size();++i) {
              if(status[i] == 1) continue;
              status[i] = 0;
              Rect target = candidates[i];
              for(uint8_t j=i+1;j<candidates.size();++j) {

                  if(status[j] == 0) {
                      int current_value = candidates[j].y + candidates[j].height / 2;
                      std::array<int, 4> ar={target.width, target.height, candidates[j].height,candidates[j].width};
                      sort(ar.begin(),ar.end());
                      if (abs(target.y + target.height / 2 - current_value) < 40  && abs(target.x-candidates[j].x)<(target.width+min((ar[1]+ar[2])/3,160))) {

                          status[j] = 1;
                          if (target.height > candidates[j].height) {
                              target.width = abs(target.x - (candidates[j].x + candidates[j].width));
                          } else {
                              target.height = candidates[j].height;
                              target.y = candidates[j].y;
                              target.width = abs(target.x - (candidates[j].x + candidates[j].width));
                          }
                      }
                  }
              }
              new_candidates.push_back(target);
          }

      }
    candidates.clear();
    //限制尺寸
    for (size_t i = 0; i != new_candidates.size(); ++i) {
        int app = new_candidates[i].width*new_candidates[i].height;
        if(app<rect_area_min||app>rect_area_max||new_candidates[i].width>max_width||new_candidates[i].height>max_height) continue;
        candidates.push_back(new_candidates[i]);
    }


//      */
    return candidates;
    //merger　结束
}

//在边缘像素中
vector<cv::Rect> DigtalLocate::extra_marge_ara(Mat &src) {
    std::vector<std::vector<cv::Point> > plate_contours;
    std::vector<cv::Rect> candidates;
    cv::findContours(src, plate_contours,RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < plate_contours.size(); ++i) {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
        candidates.push_back(rect);
    }
    return std::move(candidates);
    //merger　结束
}

vector<cv::Rect> DigtalLocate::extra_marge_ara_1(Mat &src) {
    const  int morphology_ex_size_x=40, morphology_ex_size_y=5;
    const int rect_area_min = 0.0001*src.rows * src.cols;
    const float  max_width = 0.9*src.rows, min_width = 0,max_height = 0.7*src.cols,min_height = 0, min_rate = 0.3, max_rate = 5;
    std::vector<std::vector<cv::Point> > plate_contours;
    std::vector<cv::Rect> candidates;
    Mat img_morphology_ex;
   // cv::morphologyEx(src, img_morphology_ex, cv::MORPH_CLOSE, cv::Mat::ones(morphology_ex_size_x, morphology_ex_size_y, CV_8UC1));

   // tool::DebugOut(STR(img_morphology_ex), name,img_morphology_ex);
    cv::findContours(src, plate_contours,RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i != plate_contours.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
        int app = rect.width*rect.height;
        double rate = (double)rect.width/rect.height;

        //limit
        if(app<rect_area_min||rect.width>max_width||rect.height>max_height||rate<min_rate||rate>max_rate) continue;
        candidates.push_back(rect);
    }
    //区域merger
    std::vector<cv::Rect> new_candidates;
    sort(candidates.begin(), candidates.end(), [](const cv::Rect &a,const cv::Rect &b){return (a.x<b.x);});
    vector<int> status(candidates.size(), 0);
    if(candidates.size()!=0) {

        for(uint8_t i=0;i<candidates.size();++i) {
            if(status[i] == 1) continue;
            status[i] = 0;
            Rect target = candidates[i];
            for(int j=i+1;j<candidates.size();++j) {

                if(status[j] == 0) {
                    int current_value = candidates[j].y + candidates[j].height / 2;
                    if (abs(target.y + target.height / 2 - current_value) < 30  && abs(target.x-candidates[j].x)<(target.width+min(max(target.width, candidates[j].width),150))) {

                        status[j] = 1;
                        if (target.height > candidates[j].height) {
                            target.width = abs(target.x - (candidates[j].x + candidates[j].width));
                        } else {
                            target.height = candidates[j].height;
                            target.y = candidates[j].y;
                            target.width = abs(target.x - (candidates[j].x + candidates[j].width));
                        }
                    }
                }
            }
            new_candidates.push_back(target);
        }

    }
    candidates.clear();
    const int  image_min_end = 0.00075*src.rows * src.cols;
    for(auto single : new_candidates){
        int app = single.width*single.height;
        if(app<image_min_end) continue;
        candidates.push_back(single);
    }
    return candidates;
    //merger　结束
}
vector<cv::Rect> DigtalLocate::extrator_edge() {

    cv::Mat dst,gray,edage;
    std::vector<cv::Rect> candidates;

    //缩小加快操作
    dst = Mat::zeros(this->srcMat.rows / 8, this->srcMat.cols / 8, CV_8UC3);
    resize(this->srcMat, dst, dst.size());

    cv::cvtColor(dst, gray, CV_BGR2GRAY);
    medianBlur(gray,gray,3);     //save

// output_mat_url.push_back({STR(gray), tool::DebugOut(STR(gray), name, gray)});

    edage = extract_map_by_edge(gray);

    output_mat_url.push_back({STR(edage_map), tool::DebugOut(STR(edage), name, edage)});

    candidates = extra_marge_ara(edage);

    for(uint8_t i=0;i<candidates.size();++i) {
        tool::scale_rect(candidates[i],8,8);
        cv::rectangle(imgMat, candidates[i], Scalar(0, 255, 255), 10, 1, 0);
    }
    output_mat_url.push_back({STR(edage), tool::DebugOut(STR(ROI_EDGE), name, imgMat)});
    return   std::move(candidates);

}
vector<cv::Rect> DigtalLocate::extrator_edge_differ() {

    cv::Mat HSV,gray,med_blur, edage;
    std::vector<cv::Mat> channels;
    std::vector<cv::Rect> candidates,candidates1;
    cv::cvtColor(imgMat, HSV, CV_BGR2HSV);
    cv::cvtColor(imgMat, gray, CV_BGR2GRAY);
    // 通道分离
    cv::split(HSV, channels);

    output_mat_url.push_back({STR(channel1), tool::DebugOut(STR(channel1), name, channels[0])});
    output_mat_url.push_back({STR(channel2), tool::DebugOut(STR(channel2), name, channels[1])});//像素值高,　说明发刚
    output_mat_url.push_back({STR(channel3), tool::DebugOut(STR(channel3), name, channels[2])});
    output_mat_url.push_back({STR(gray), tool::DebugOut(STR(gray), name, gray)});
    cv::Mat ab = channels[2]-gray;
    threshold(ab,ab,40, 255, CV_THRESH_BINARY);
    output_mat_url.push_back({STR(test), tool::DebugOut(STR(test), name, ab)});
    //threshold(ab,ab,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
    //output_mat_url.push_back({STR(test1), tool::DebugOut(STR(test1), name, ab)});
    //这里使用
    //gray = ab;
    //gray = channels[1];extera_by_edge
    medianBlur(ab,ab,9);     //save
    std::vector<cv::Rect> a;

    //edage = extera_by_edge_diff(ab,candidates);
    edage = extera_by_edge_threshold(channels[2],candidates);
   // output_mat_url.push_back({STR(edage),tool::DebugOut(STR(edage), name, edage)});

//    candidates = extra_marge_ara(edage);
//
//    Mat scopy = imgMat.clone();
//    for(auto s : candidates) {
//        cv::rectangle(scopy, s, Scalar(0, 255, 255), 5, 1, 0);
//    }
//    tool::DebugOut(STR(ROI_EDGE), name, scopy);
    return   candidates;

}

vector<cv::Rect>  DigtalLocate::extrator_threshold(){
    cv::Mat HSV,med_blur,gray,img_threshold,edage;
    std::vector<cv::Mat> channels;
    std::vector<cv::Rect> candidates;

    cv::cvtColor(imgMat, HSV, CV_BGR2HSV);
    cv::split(HSV, channels);
    tool::DebugOut(STR(channel1), name,channels[0]);
    tool::DebugOut(STR(channel2), name,channels[1]);//像素值高,　说明发刚
    tool::DebugOut(STR(channel3), name,channels[2]);
    medianBlur(channels[2],med_blur,7);
    //局部二值话
    threshold(med_blur, img_threshold,245 , 255,THRESH_TOZERO); //二值化
    edage = extera_by_edge_1(img_threshold);
    tool::DebugOut(STR(edage_threshold), name, edage);
    candidates = extra_marge_ara_1(edage);
    Mat scopy = imgMat.clone();
    for(auto s : candidates) {
        cv::rectangle(scopy, s, Scalar(0, 255, 255), 5, 1, 0);
    }
    tool::DebugOut(STR(ROI_Threshold), name, scopy);
    return  candidates;
}

void DigtalLocate::get_pic_rect() {
    std::vector<cv::Rect> final_candidate;

    final_candidate = extrator_edge();


    getsub_char(final_candidate);
    for(uint8_t i=0;i<final_candidate.size();i++) {
        cv::rectangle(imgMat, final_candidate[i], Scalar(0, 255, 0), 3, 1, 0);
        cv::Mat roiImg_raw  = srcMat(final_candidate[i]);
        tool::DebugOut(STR(sub_raw), name, roiImg_raw, dirname, i);
    }
    output_mat_url.push_back({STR(ROI_CHAR), tool::DebugOut(STR(ROI_CHAR), name, imgMat)});

}
/*vector<cv::Rect> DigtalLocate::mserExtractor(int a) {
    cv::Mat HSV,gray,med_blur,img_erode,img_threshold;
    std::vector<cv::Rect> candidates;

    cv::cvtColor(imgMat, HSV, CV_BGR2HSV);
    cv::cvtColor(imgMat, gray, CV_BGR2GRAY);
    // 通道分离
    std::vector<cv::Mat> channels;
    cv::split(HSV, channels);
    tool::DebugOut(STR(channel1), name,channels[0]);
    tool::DebugOut(STR(channel2), name,channels[1]);//像素值高,　说明发刚
    tool::DebugOut(STR(channel3), name,channels[2]);
    //gray = channels[1];

    tool::DebugOut(STR(gray), name,gray);
    medianBlur(gray,med_blur,7);

   // threshold(med_blur, med_blur,220 , 255,THRESH_TOZERO); //增加一个单阈值











    //边界搜索
    Mat img_threshold1,img_threshold2;
    threshold(channels[2], img_threshold1,160 , 255,THRESH_BINARY); //二值化

  //  img_threshold2 = img_threshold1&allContoursResult;
    img_threshold2 = allContoursResult;
    tool::DebugOut(STR(img_threshold1), name, img_threshold1);

    tool::DebugOut(STR(allContoursResult), name, allContoursResult);
    std::vector<std::vector<cv::Point> > plate_contours_1;
    cv::findContours(img_threshold2, plate_contours_1,RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    cv::morphologyEx(img_threshold2, img_threshold2,
                     cv::MORPH_CLOSE, cv::Mat::ones(31, 71, CV_8UC1));
    tool::DebugOut(STR(img_threshold2), name, img_threshold2);
    candidates.clear();
    imagemin = 0.001*imgMat.rows * imgMat.cols;
    for (size_t i = 0; i != plate_contours_1.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours_1[i]);
        int app = rect.width*rect.height;

        double rate = (double)rect.width/rect.height;
        if(app<imagemin||rect.width>0.7*imgMat.rows||rect.height>0.3*imgMat.cols||rate<0.5||rate>5) continue;
        candidates.push_back(rect);
        //逻辑限制
    }
    //output child pic

    //区域和并
    std::vector<cv::Rect> new_candidates;
    sort(candidates.begin(), candidates.end(), [](const cv::Rect &a,const cv::Rect &b){return (a.x<b.x);});
    vector<int> status(candidates.size(), 0);
    if(candidates.size()!=0) {

        for(int i=0;i<candidates.size();++i) {
            if(status[i] == 1) continue;
            status[i] = 0;
            Rect target = candidates[i];
            for(int j=i+1;j<candidates.size();++j) {

                if(status[j] == 0) {
                    int current_value = candidates[j].y + candidates[j].height / 2;
                    if (abs(target.y + target.height / 2 - current_value) < 30  && abs(target.x-candidates[j].x)<(target.width+min(min(target.width, candidates[j].width)*1/3,150))) {

                        status[j] = 1;
                        if (target.height > candidates[j].height) {
                            target.width = abs(target.x - (candidates[j].x + candidates[j].width));
                        } else {
                            target.height = candidates[j].height;
                            target.y = candidates[j].y;
                            target.width = abs(target.x - (candidates[j].x + candidates[j].width));
                        }
                    }
                }
            }
            new_candidates.push_back(target);
        }

    }
    new_candidates.swap(candidates);

    //end
    imagemin = 0.0075*imgMat.rows * imgMat.cols;
    for(int i=0;i<candidates.size();i++){
        //limit
        int app = candidates[i].width*candidates[i].height;
        if(app<imagemin) continue;
        cout<<app<<"|||"<<imagemin<<endl;
        //end

        cv::Mat roiImg,image_threshold,rsize,roiImg1;
        roiImg = med_blur(candidates[i]);

        threshold(roiImg,image_threshold,0, 255, THRESH_OTSU|CV_THRESH_BINARY);
        threshold(roiImg,roiImg1,0, 255, THRESH_TRIANGLE|CV_THRESH_BINARY);
        // Size size((int)(roiImg.cols*(100.0 / roiImg.rows)),100);
        //  resize(roiImg,rsize,size);
        //cv::morphologyEx(rsize, rsize,cv::MORPH_CLOSE, cv::Mat::ones(5, 5, CV_8UC1));
        tool::DebugOut(STR(roiImg), name, image_threshold,dirname,i+candidates.size());
        tool::DebugOut(STR(roiImg1), name, imgMat(candidates[i]),dirname,i+candidates.size());
        cv::rectangle(imgMat, candidates[i], Scalar(0, 255, 0), 2, 1, 0);
        // tool::DebugOut(STR(roiImg), name, roiImg1,dirname,i);
    }
    cout<<candidates.size()<<endl;
    tool::DebugOut(STR(ROI), name, imgMat);
//    std::vector<std::vector<cv::Point> > regContours;
//    std::vector<std::vector<cv::Point> > charContours;
//    //parameter
//
//    const int imageArea = 0.3*imgMat.rows * imgMat.cols;   //可以调整以防输入区域过大
//    const int imagemin = 0.0001*imgMat.rows * imgMat.cols;
//
//    cv::Ptr<MSER> mesr1 = cv::MSER::create(3, imagemin,imageArea, 0.1); //创建mers //第一个增加可以减少重影  //第四个越大则范围越广
//    std::vector<cv::Rect> bboxes1;
//    mesr1->detectRegions(med_blur, regContours, bboxes1);
//
//    cv::Mat mserMapMat =cv::Mat::zeros(imgMat.size(), CV_8UC1);
//
//    for(int i=0;i<bboxes1.size();i++)
//    {
//        auto cvsingle = bboxes1[i];
//        double wh_ratio = cvsingle.width / double(cvsingle.height);
//        if(wh_ratio>1) {
//            candidates.push_back(cvsingle);
//            for(auto point:regContours[i])
//            {
//                mserMapMat.at<uchar>(point) = 255;
//            }
//        }
//    }
//    for(auto cvsingle : candidates) {
//        cv::rectangle(imgMat, cvsingle, Scalar(0, 255, 0), 2, 1, 0);
//    }
//    cout<<candidates.size()<<endl;
//    tool::DebugOut(STR(mserMapMat), name,mserMapMat);
//    tool::DebugOut(STR(ROI), name, imgMat);
    return candidates;
}*/

//通过mser提取图像
vector<cv::Rect> DigtalLocate::mserExtractor(){
    // HSV空间转换
    cv::Mat gray, gray_neg,gray_bar,gray_neg_blur,img_threshold;
    cv::Mat HSV;
    std::vector<cv::Rect> candidates;
    cv::cvtColor(imgMat, HSV, CV_BGR2HSV);
    // 通道分离
    std::vector<cv::Mat> channels;
    cv::split(HSV, channels);

    gray=channels[2];
    tool::DebugOut(STR(channel1), name,channels[0]);
    tool::DebugOut(STR(channel2), name,channels[1]);//像素值高,　说明发刚
    tool::DebugOut(STR(channel3), name,channels[2]);
    tool::DebugOut(STR(neg_channel2), name, 255-channels[1]);
    gray_neg_blur=255-channels[1];
    //输出据均衡化的直方图
    Mat temp;
    equalizeHist(channels[2], temp);
    tool::DebugOut(STR(temp_blance), name,temp);
    threshold(gray_neg_blur,gray_neg_blur,0, 200, CV_THRESH_OTSU|CV_THRESH_BINARY);
    tool::DebugOut(STR(neg_channel2_2), name,gray_neg_blur);
    GaussianBlur( gray, gray_bar , Size(9,9) , 4, 4, BORDER_DEFAULT );   //如果高斯模糊变大对细线无法处理

    gray_neg =  255-gray_bar;
    tool::DebugOut(STR(gray_neg), name,gray_neg);

    threshold(gray_neg, img_threshold,85 , 255,THRESH_BINARY);
    // if this is back other
    tool::DebugOut(STR(img_threshold), name, img_threshold);
    gray_neg = img_threshold;

    std::vector<std::vector<cv::Point> > regContours;
    std::vector<std::vector<cv::Point> > charContours;
    //parameter
    const double imageArea = 0.01*gray_neg.rows * gray_neg.cols;   //可以调整以防输入区域过大
    const double imagemin = 0.00015*gray_neg.rows * gray_neg.cols;

//    cv::Ptr<MSER> mesr1 = cv::MSER::create(0.01, imagemin,imageArea, 0.5); //创建mers //第一个增加可以减少重影  //第四个越大则范围越广
////    cv::Ptr<MSER> mesr2 = cv::MSER::create(2, 2, 400, 0.1, 0.3);
//    std::vector<cv::Rect> bboxes1;
////    std::vector<cv::Rect> bboxes2;
//    mesr1->detectRegions(gray_neg, regContours, bboxes1);
////    cv::morphologyEx(local, mserClosedMat,
//  //                   cv::MORPH_CLOSE, cv::Mat::ones(10, 1, CV_8UC1));
//    cv::Mat mserMapMat =cv::Mat::zeros(imgMat.size(), CV_8UC1);
//
//    for(auto singlepointarry:regContours)
//    {
//        for(auto point:singlepointarry)
//        {
//            mserMapMat.at<uchar>(point) = 255;
//        }
//    }

    // 闭操作连接缝隙
  /*  cv::Mat mserClosedMat;
    cv::morphologyEx(mserMapMat, mserClosedMat,
                     cv::MORPH_CLOSE, cv::Mat::ones(20, 1, CV_8UC1));
    std::vector<std::vector<cv::Point> > plate_contours;
    cv::findContours(mserClosedMat, plate_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    tool::DebugOut(STR(close), name, mserClosedMat);
    tool::DebugOut(STR(mserMapMat), name, mserMapMat);
    //去除不满足条件的
    std::vector<cv::Rect> candidates1;
    for (size_t i = 0; i != plate_contours.size(); ++i)
    {
        // 求解最小外界矩形
        cv::Rect rect = cv::boundingRect(plate_contours[i]);
        // 宽高比例
        double wh_ratio = rect.width / double(rect.height);
        // 不符合尺寸条件判断
        if((wh_ratio<4)&&(wh_ratio>0.2))
            candidates1.push_back(rect);
    }*/

//    for(auto cvsingle : bboxes1)
//    {
//        double wh_ratio = cvsingle.width / double(cvsingle.height);
//        if((wh_ratio<3)&&(wh_ratio>0.3))
//        {
//            candidates.push_back(cvsingle);
//        }
//    }
//  //  groupRectangles(candidates, 1 );
//   for(auto cvsingle : candidates) {
//       cv::rectangle(imgMat, cvsingle, Scalar(0, 255, 0), 2, 1, 0);
//   }
//    tool::DebugOut(STR(ROI), name, imgMat);
    return candidates;

}
void DigtalLocate::output_joson() {
    Json::Value arrayObj;   // 构建对象
    arrayObj["type"]="value";
    arrayObj["result"] = out_put_result_mat;
    string value;
    for(auto s:result_ROI) {
        value+=s.first+",";
    }
    arrayObj["result_value"] = value.substr(0,value.size()-1);
    for(auto s:output_mat_url) {
        Json::Value item;
        item["name"] = s.first;
        item["url"] = s.second;
        arrayObj["debug"].append(std::move(item));
    }

    for(auto s:output_sub_mat_url) {
        arrayObj["suburl"].append(s);
    }

    Json::FastWriter writer;
    std::string out2 = writer.write(arrayObj);
    //讲字符串写入文件
    boost::format frm_dir("../datasource/logfile/%d.json");
    frm_dir %name;
    ofstream outfile;
    outfile.open(frm_dir.str(), ios::out  | ios::trunc );
    if (outfile.is_open())
    {
        outfile << out2;
        outfile.close();
    }
}
//vector<Mat> Get_Sub_Mat(Mat srcImage,double scale,string name){
//    CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 3);  //检查是否是三通道图像
//    Size ResImgSiz = Size(srcImage.cols*scale, srcImage.rows*scale);
//    Mat ResImg = Mat(ResImgSiz, srcImage.type());
//
//    resize(srcImage, ResImg, ResImgSiz, CV_INTER_CUBIC); //创建比较小的图像
//    const int must_front_high = 150;
//    const int must_front_low = 100;
//    Mat threshold_diff, threshold_white, combine,Ex_combine;
//    vector<Rect> canditaes;    //数码管特征图
//    //使用分水岭算法提取有颜色的区域
//    // 利用了颜色特征
//    Mat mask = tool::get_watershed_segmenter_mark(must_front_high,must_front_low,_gray_dif);
//    Mat result = tool::get_binnary_by_watershed(mask,srcImage)
//
//    cv::threshold(_max,threshold_white,254, 255, CV_THRESH_BINARY);  //曝光过度值图
//    combine = threshold_diff|threshold_white;
//    cv::morphologyEx(combine, Ex_combine, cv::MORPH_CLOSE, cv::Mat::ones(5, const_Mat.cols/50, CV_8UC1));  //膨胀与腐蚀
//
//    /******deubg******/
//    tool::DebugOut(STR(gray_dif), name, _gray_dif);        //差值图
//    tool::DebugOut(STR(set_result), name, threshold_diff); //分水岭结果图
//    tool::DebugOut(STR(threshold_white), name, threshold_white); //曝光过度图
//    tool::DebugOut(STR(before_combine), name, combine); //曝光过度图
//    tool::DebugOut(STR(combine), name, Ex_combine);
//    /************end********/
//
//
//    canditaes = get_digital_area(Ex_combine);
//
//
//
//}
void DigtalLocate::output_log(){

    boost::format frm_dir("../datasource/log/%d.txt");
    frm_dir %dirname;
    ofstream outfile;
    outfile.open(frm_dir.str(), ios::out  | ios::trunc );
    if (outfile.is_open())
    {
        outfile <<my_stream.str();
        outfile.close();
    }
}
/*void  DigtalLocate::Resize_t(Mat &resultResized)
{
    //归一化尺寸  可以使用static
     Size ResImgSiz(32,48);
     Mat ResImg;
     resize(resultResized, ResImg, ResImgSiz, CV_INTER_CUBIC);
     //调整完毕后
    //cv::Mat mask(ResImg);
    //ResImg.copyTo(imageROI,mask);
    resultResized=ResImg;
    //压缩数据
    //然后把剩下的部分在在黑色的区域左中区域。
}*/





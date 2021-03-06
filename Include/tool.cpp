//
// Created by dubing on 17-4-13.
//

#include "tool.h"
#include "Histogram1D.h"

//#define NDEBUG

//返回分水岭后的图像

Mat  tool::get_best_side(DigtalLocate *src,int front_value,int back_value,Mat & mat){
    Mat front,back;
    cv::threshold(mat,front,front_value, 255, CV_THRESH_BINARY);  //参数自己调整
    cv::threshold(mat,back,back_value, 255, CV_THRESH_BINARY);  //参数自己调整
    back=~back;
    Mat scr(src->const_Mat.size(),CV_8UC1,Scalar(128));
    cv::Mat markers(src->const_Mat.size(),CV_8UC1,cv::Scalar(0));
    markers+=front;
    scr.copyTo(markers,back);
    tool::DebugOut(STR(Segment_markers), src->name, markers);

    WatershedSegmenter segmenter1;  //实例化一个分水岭分割方法的对象
    segmenter1.setMarkers(markers);//设置算法的标记图像，使得水淹过程从这组预先定义好的标记像素开始
    segmenter1.process(src->const_Mat);     //传入待分割原图

    Mat answer =  segmenter1.getSegmentation();
   // tool::DebugOut(STR(answer1), src->name, answer);
    threshold(answer,answer,129,255,CV_THRESH_BINARY); //把128的也设置为0　
   // tool::DebugOut(STR(answer), src->name, answer);
    return std::move(answer);
}
void tool::split_mat(Mat &src,Mat &max, Mat &med,Mat &min){

    uchar cmax,cmed,cmin;//currentPoint;
    for (int i=0;i<src.rows;++i)
    {
        for (int j=0;j<src.cols;++j)
        {
            cmax=src.at<Vec3b>(i,j)[0];
            cmed=src.at<Vec3b>(i,j)[1];
            cmin=src.at<Vec3b>(i,j)[2];
            if(cmin>cmed) swap(cmin,cmed);
            if(cmed>cmax) swap(cmax,cmed);
            if(cmin>cmed) swap(cmin,cmed);

            max.at<uchar>(i,j)=cmax;
            med.at<uchar>(i,j)=cmed;
            min.at<uchar>(i,j)=cmin;

        }
    }
  //  return std::move(result);
}


void tool::marge(vector<Rect> &candidates){
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
    candidates.swap(new_candidates);
}
vector<int> tool::get_row_number( const Mat& picture ) {

    vector<int> result(picture.rows,0);
    for (int i = 0; i < picture.rows; i++)
        for (int j = 0; j < picture.cols; j++)
            if (picture.at<uchar>(i, j) > 0)
                result[i] += 1;
    return std::move(result);

}
vector<int> tool::get_col_number( const Mat& picture ) {

    vector<int> result(static_cast<uint64_t>(picture.cols), 0);
    for (int j = 0; j < picture.cols; j++)
        for (int i = 0; i < picture.rows; i++)
            if (picture.at<uchar>(i, j) > 0)
                result[j] += 1;
    return std::move(result);

}
Mat tool::get_hist_show(const vector<int> vec) {

    int value_max = *max_element(vec.begin(),vec.end());
    Mat verticalProjectionMat(value_max, static_cast<int>(vec.size()), CV_8UC1);
    for (auto i = 0; i < value_max; i++)
        for (size_t j = 0; j < vec.size(); j++)
            verticalProjectionMat.at<uchar>(i, j) = 0;

    /*将直方图的曲线设为黑色*/
    for (size_t i = 0; i <  vec.size(); i++)
        for (size_t j = 0; j < vec[i]; j++)
            verticalProjectionMat.at<uchar>(value_max - 1 - j, i) = 255;
    return std::move(verticalProjectionMat);

}
vector< pair<int,int> > tool::get_Area( vector<int> &vec, int hight) {
    int start_index = 0;//记录进入字符区的索引
    int end_index = 0;//记录进入字符区的索引
    bool start = false;//是否遍历到了字符区内
    vector<pair<int,int>> vp;
    int min_value = *min_element(vec.begin(),vec.end());
    for_each (vec.begin(), vec.end(), [min_value,hight](int &i){i=max(i-min(min_value, hight/5)-hight/30-3,0);});
    int filterTime=0;
    for (size_t i = 0; i < vec.size(); ++i) {
        if(vec[i]!=0&&!start) {
            filterTime=0;
            start =true;
            start_index = i;
            continue;
        }
        if(start&&vec[i]==0) {
            filterTime++;
            if(filterTime==2) {  //之前参数为５
                filterTime=0;
                end_index = i;
                vp.push_back(make_pair<int, int>(std::move(start_index), std::move(end_index)));
                start = false;
            }
        }
    }
    if(start) {
        end_index = static_cast<int>(vec.size())-1;
        vp.push_back(make_pair<int,int>(std::move(start_index),std::move(end_index)));
    }
    return vp;

}
vector<areaRange> tool::getLineRange( int rows, int cols )
{
    vector<areaRange> area;
    areaRange range[7];
    /*
    _________
   |    1    |
   | 6     2 |
   |    7    |
   | 5     3 |
   |    4    |
    ��������

    ������ͼ�����߶�
    */
    range[0].x1 = cols / 2;
    range[0].y1 = 0;
    range[0].x2 = cols / 2;
    range[0].y2 = rows / 3;

    range[1].x1 = cols / 2;
    range[1].y1 = rows / 3;
    range[1].x2 = cols;
    range[1].y2 = rows / 3;

    range[2].x1 = cols  / 2;
    range[2].y1 = rows * 2 / 3;
    range[2].x2 = cols;
    range[2].y2 = rows * 2 / 3;

    range[3].x1 = cols / 2;
    range[3].y1 = rows * 2 / 3;
    range[3].x2 = cols  / 2;
    range[3].y2 = rows;

    range[4].x1 = 0;
    range[4].y1 = rows * 2 / 3;
    range[4].x2 = cols / 2;
    range[4].y2 = rows * 2 / 3;

    range[5].x1 = 0;
    range[5].y1 = rows / 3;
    range[5].x2 = cols / 2;
    range[5].y2 = rows / 3;

    range[6].x1 = cols / 2;
    range[6].y1 = rows / 3;
    range[6].x2 = cols / 2;
    range[6].y2 = rows * 2 / 3;

    for (int i = 0; i < 7; i++)
    {
        area.push_back(range[i]);
    }
    return area;
}
bool tool::isIntersected( Mat single, areaRange range )
{
    int count = 0;
    if (range.x1==range.x2)
    {
        for (int i = range.y1; i < range.y2; i++)
        {
            if (single.at<uchar>(i,range.x1) == PT_WHITE)
                count++;
            if (count >= INTERSECTION_COUNT)
                return true;
            if (single.at<uchar>(i,range.x1) == PT_BLACK)
                count = 0;
        }
    }
    if (range.y1 == range.y2)
    {
        for (int j = range.x1; j < range.x2; j++)
        {
            if (single.at<uchar>(range.y1,j) == PT_WHITE)
                count++;
            if (count >= INTERSECTION_COUNT)
                return true;
            if (single.at<uchar>(range.y1,j) == PT_BLACK)
                count = 0;
        }
    }
    return false;
}
string tool::get_name(string urlPath){
    auto n = urlPath.rfind("/");
    string name;
    if(n == std::string::npos)
        name=urlPath;
    else
        name=urlPath.substr(n+1);

    n = name.rfind(".");
    name =name.substr(0,n);
    return std::move(name);
}
std::string tool::getNOByLine( Mat single, vector<areaRange> range, bool atBottom , double t)
{
    /*
    _________
   |    1    |
   | 6     2 |
   |    7    |
   | 5     3 |
   |    4    |
    ��������

    �����ж�1��ͼ���߿��ȴ�����ֵ��Ϊ1
    ������ͼ�����߶Σ��ཻ��0��ʾ

    ����	����������d1-d7��	������
    2		12457				0010010
    3		12347				0000110
    4		2367				1001100
    5		13467				0100100
    6		134567				0100000
    7		123					0001111
    8		1234567				0000000
    9		123467				0000100
    0		123456				0000001
    ���ݶ����Ʊ���ȷ�������ж�˳��Ϊ1 3 4 6 7 2 5
    d1==1 Y 4
    N
    d3==1 Y 2
    N
    d4==1 Y 7
    N
    d6==1 Y 3
    N
    d7==1 Y 0
    N
    d2==1   Y   d5==1 Y 5
    N           N
    d5==1 Y9    6
    N
    8
*/
    if(range.size() == 0)
        return "";
    float heightDivWidth = (float)single.rows/single.cols;
    cout<<heightDivWidth<<endl;
    if (atBottom)
        //if (heightDivWidth > 0.8 && heightDivWidth < 1.2)
        return "";
    if (heightDivWidth > ONE_HIGHT_WIDTH_RATE)//�����ȴ���һ����ֵ����Ϊ1��С����
        //return ((float)getForegroundPt(single)/(single.rows*single.cols))<0.3? ".":"1";//������С��������0.2��ΪС����
        return "1";

    if (!isIntersected(single, range[0]) && !isIntersected(single, range[3]))
        return "4";
    else
    {
        if (!isIntersected(single, range[2]))
            return "2";
        else
        {
            if (!isIntersected(single, range[3]) && !isIntersected(single, range[5]))
                return "7";
            else
            {
                if (!isIntersected(single, range[5])) {
                    if(!isIntersected(single, range[6]))
                        return "7";
                    else
                        return "3";
                }
                else
                {
                    if (!isIntersected(single, range[6]))
                        return "0";
                    else
                    {
                        if (!isIntersected(single, range[1]))
                        {
                            if (!isIntersected(single, range[4]))
                                return "5";
                            else
                                return "6";
                        }
                        else
                        {
                            if (!isIntersected(single, range[4]))
                                return "9";
                            else
                                return "8";
                        }
                    }

                }

            }

        }
    }
}
Rect& tool::thin_rect(Rect &rect, CvSize size) {
    rect.width = rect.width + size.width;
    rect.height = rect.height + size.height;
    Point pt;
    pt.x = cvRound(size.width/2.0);
    pt.y = cvRound(size.height/2.0);
    rect.x = (rect.x-pt.x);
    rect.y = (rect.y-pt.y);
    return rect;
}
Rect& tool::scale_rect(Rect &rect, int  scale_x,int  scale_y) {
    rect.width*=scale_x;
    rect.height*=scale_y;
    rect.x*=scale_x;
    rect.y*=scale_y;
    return  rect;
}
bool tool::is_Inside(Rect& rect1, Rect rect2, double scale)  {
    rect2 = thin_rect(rect2,cvSize(static_cast<int>(rect2.width*scale), static_cast<int>(rect2.height*scale)));
    return (rect1 == (rect1&rect2));
}

cv::Mat tool::stretch(const cv::Mat& image,int minvalue)
{
    Histogram1D h;
    cv::Mat hist = h.getHistogram(image);
    //找到直方图的左边限值
    int imin = 0;
    for (; imin < h.getHistSize()[0];imin++)
    {
        if (hist.at<float>(imin)>minvalue)
        {
            break;
        }
    }
    //找到直方图的右边限值
    int imax = h.getHistSize()[0]-1;
    for (; imax >= 0; imax--)
    {
        if (hist.at<float>(imax)>minvalue)
        {
            break;
        }
    }

    cv::Mat lut(1, 256, CV_8U);
    //构建查找表
    for (int i = 0; i < 256;i++)
    {
        if (i < imin)
            lut.at<uchar>(i) = 0;
        else if (i>imax)
            lut.at<uchar>(i) = 255;
        else
        {
            lut.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin));
        }
    }
    cv::Mat result = Histogram1D::applyLookUp(image,lut);
    return result;
}
string tool::resultout  (string dir ,string name, Mat resultResized, string subdirName, int tid) {
    //可以先判断文件夹在不在,然后可以创建
    boost::format frm_dir("../datasource/result/%s");
    frm_dir %dir;
    boost::filesystem::path p(frm_dir.str());
    if(! boost::filesystem::is_directory(p))
        boost::filesystem::create_directory(p);

    if(subdirName!=""){
        boost::format frm_dir_1("../datasource/result/%s/%s");
        frm_dir_1 %dir %subdirName;
        boost::filesystem::path p_1(frm_dir_1.str());
        if(! boost::filesystem::is_directory(p_1))
            boost::filesystem::create_directory(p_1);
        boost::format frm("../datasource/tmp/%s/%s/debug_resize_%d_%s");
        frm %dir %subdirName %tid %name ;
        imwrite(frm.str(), resultResized);

        return boost::filesystem::absolute(boost::filesystem::path(frm.str())).string();
    }
    else{
        boost::format frm("../datasource/result/%s/debug_resize_%s");
        frm %dir %name ;
        imwrite(frm.str(), resultResized);
        return boost::filesystem::absolute(boost::filesystem::path(frm.str())).string();
    }

}

string tool::DebugOut  (string dir ,string name, Mat resultResized, string subdirName, int tid) {
#ifndef  NDEBUG
    //可以先判断文件夹在不在,然后可以创建
    boost::format frm_dir("../datasource/tmp/%s");
    frm_dir %dir;
    boost::filesystem::path p(frm_dir.str());
    if(! boost::filesystem::is_directory(p))
        boost::filesystem::create_directory(p);

    if(subdirName!=""){
        boost::format frm_dir_1("../datasource/tmp/%s/%s");
        frm_dir_1 %dir %subdirName;
        boost::filesystem::path p_1(frm_dir_1.str());
        if(! boost::filesystem::is_directory(p_1))
            boost::filesystem::create_directory(p_1);
        boost::format frm("../datasource/tmp/%s/%s/debug_%d_%s.jpg");
        frm %dir %subdirName %tid %name ;
        imwrite(frm.str(), resultResized);

        return boost::filesystem::absolute(boost::filesystem::path(frm.str())).string();
    }
    else{
        boost::format frm("../datasource/tmp/%s/debug_%s.jpg");
        frm %dir %name ;
        imwrite(frm.str(), resultResized);
        return boost::filesystem::absolute(boost::filesystem::path(frm.str())).string();
    }
#else
    return "s";

#endif
}
//引用传递vector 进行改变
void tool::incrementRadon(vector<double> &vt, double pixel, double r) {
    int r1;
    double delta;

    r1 = (int) r;   //对于每一个点，r值不同，所以，通过这种方式，可以把这一列中相应行的元素的值给赋上
    delta = r - r1;
    vt[r1] += pixel * (1.0 - delta); //radon变换本来就是通过记录目标平面上某一点的被映射后点的积累厚度来反推原平面的直线的存在性的，故为+=
    vt[r1+1] += pixel * delta;  //两个点互相配合，提高精度
}
////输入是提取边缘后的图像,输入的是 vector 里面是弧度(应该是弧度或者角度)?

//template <typename  T>
//vector<vector<double >> tool::radon(Mat src,vector<T> angle_array) {
//    int k, m, n;              /* loop counters */
//    double angle;             /* radian angle value */
//    double cosine, sine;      /* cosine and sine of current angle */
//    double *pr;               /* points inside output array */
//    double *pixelPtr;         /* points inside input array */
//    double pixel;             /* current pixel value */
//
//    /* tables for x*cos(angle) and y*sin(angle) */
//    double x,y;
//    double r, delta;
//    int r1;
//
//    int width = src.cols, height = src.rows;
//    vector<double > xCosTable(2*src.cols ,0);
//    vector<double > ySinTable(2*src.rows ,0);
//    int xOrigin = max(0,(width-1)/2);
//    int yOrigin = max(0,(height-1)/2);
//    int rFirst = 1;
//    int rSize = 2;
//
//    //获取最大的长度 //rize
//    temp1 = M - 1 - yOrigin;
//    temp2 = N - 1 - xOrigin;
//    rLast = (int) ceil(sqrt((double) (temp1*temp1+temp2*temp2))) + 1;
//    rFirst = -rLast;
//    rSize = rLast - rFirst + 1;
//
//    //对于输入进行度数转弧度
//    vector<vector<double >> answer(angle_array.size(),vector<double>(rSize,0));
//
//    for( k = 0; k < angle_array.size(); k++){
//
//        double angle = angle_array[k];
//
//        cosine = cos(angle);
//        sine = sin(angle);
//
//        for( n = 0;n<width;n++){
//            x = n - xOrigin;
//            xCosTable[2*n]   = (x - 0.25)*cosine;   //由极坐标的知识知道，相对于变换的原点，这个就是得到了该点的横坐标
//            xCosTable[2*n+1] = (x + 0.25)*cosine;
//        }
//        for (m = 0; m < height; m++)
//        {
//            y = yOrigin - m;
//            ySinTable[2*m] = (y - 0.25)*sine;   //同理，相对于变换的原点，得到了纵坐标
//            ySinTable[2*m+1] = (y + 0.25)*sine;
//        }
//        for(int i=0;i<height;i++) //遍历行
//        {
//            for(int j=0;j<width;j++) //遍历列
//            {
//               float t =  (float)src.at<unsigned short>(i,j);
//               if(t!=0) {
//                    pixel *= 0.25;
//
//                    //一个像素点分解成四个临近的像素点进行计算，提高精确度  //r指的是位置,固定r 求出这一段区间内的值的积分
//                    //距离为y*sin()+x*cos() 当为４５度是最大即rFirst 加上一个初值也能保证r最小值为０
//                    r = xCosTable[2*n] + ySinTable[2*m] - rFirst;
//                    incrementRadon(answer[k], pixel, r);
//
//                    r = xCosTable[2*n+1] + ySinTable[2*m] - rFirst;
//                    incrementRadon(answer[k], pixel, r);
//
//                    r = xCosTable[2*n] + ySinTable[2*m+1] - rFirst;
//                    incrementRadon(answer[k], pixel, r);
//
//                    r = xCosTable[2*n+1] + ySinTable[2*m+1] - rFirst;
//                    incrementRadon(answer[k], pixel, r);
//                }
//            }
//        }
//    }
//}
//Mat tool::darkChannel(Mat src) {
//    Mat rgbmin = Mat::zeros(src.rows, src.cols, CV_8UC1);
//    Mat dark = Mat::zeros(src.rows, src.cols, CV_8UC1);
//    Vec3b intensity;
//
//    for (int m = 0; m < src.rows; m++) {
//        for (int n = 0; n < src.cols; n++) {
//            intensity = src.at<Vec3b>(m, n);
//            rgbmin.at<uchar>(m, n) = min(min(intensity.val[0], intensity.val[1]), intensity.val[2]);
//        }
//    }
//}
//vector<cv::Rect> tool::sub_Area(cv::Mat src,cv::Mat gray){
//    //通过HSI 颜色分析除去没有颜色的图片即黑色与白色
//    //得到HSV　图然后与灰度图相减　得到的就是
//    cv::cvtColor(src, HSV, CV_BGR2HSV);
//    cv::split(HSV, channels);
//    medianBlur(channels[2],channels[2],9);     //save
//    output_mat_url.push_back({STR(channel1), tool::DebugOut(STR(channel1), name, channels[0])});
//    output_mat_url.push_back({STR(channel2), tool::DebugOut(STR(channel2), name, channels[1])});//像素值高,　说明发刚
//    output_mat_url.push_back({STR(channel3), tool::DebugOut(STR(channel3), name, channels[2])});
//}

//返回的是倾斜的角度
//输入二值图像,图像原路矫正,而且
double tool::correct_error(Mat &src,double &best_angle){

    Mat canny;
    vector<double> max_array; //存储最大值的素组
    vector<double> array_p;   //生成的角度素组P

    double angle_Horizontal;  //水平角度
    double angle_vertical;    //竖直角度

    //canny 边缘检测(减少数量可以加速速度)
    Canny(src,canny,100,300,3);
    //初始化数组
    for(int i=-90;i<90;i++) {
//        double  x= i*0.5;
//        test.push_back(x);
        double x= i;
        array_p.push_back(x/2.0);
        array_p.push_back(x);
    }

    //获取最后返回的数组
    auto s = tool::radon(canny,array_p);
    //找到每一个角度的最大值
    for_each(s.begin(),s.end(),[&](vector<double> single){max_array.push_back(*max_element(single.begin(),single.end()));});
    //角度限制在   60~120 度之间
    angle_vertical = std::distance(max_element(max_array.begin()+60*2,max_array.begin()+120*2+1),max_array.begin());
    //
    auto time2 =  max_element(max_array.begin(),max_array.begin()+5*2+1);
    auto time3 = max_element(max_array.end()-5*2,max_array.end());

    if(*time3>*time2){
        angle_Horizontal =  std::distance(time3,max_array.end());
    }
    else{
        angle_Horizontal =  std::distance(time2,max_array.begin());
    }
    tool::imageRotate2(src,src,-angle_Horizontal/2);

  //  tool::DebugOut(STR(result_s_r), name, answer);

    best_angle=std::distance(max_array.begin(),max_element(max_array.begin(),max_array.end()));
   // best_angle = angle_vertical;
    angle_vertical = 90+angle_vertical/2;
    return angle_vertical;
}



int tool::imageRotate2(InputArray src, OutputArray dst, double angle)
{
    Mat input = src.getMat();
    if( input.empty() ) {
        return -1;
    }

    //得到图像大小
    int width = input.cols;
    int height = input.rows;

    //计算图像中心点
    Point2f center;
    center.x = width / 2.0;
    center.y = height / 2.0;

    //获得旋转变换矩阵
    double scale = 1.0;
    Mat trans_mat = getRotationMatrix2D( center, -angle, scale );

    //计算新图像大小
    double angle1 = angle  * CV_PI / 180. ;
    double a = sin(angle1) * scale;
    double b = cos(angle1) * scale;
    double out_width = height * fabs(a) + width * fabs(b);
    double out_height = width * fabs(a) + height * fabs(b);

    //仿射变换
    warpAffine( input, dst, trans_mat, Size(out_width, out_height));

    return 0;
}
Mat tool::roatation(Mat src, double angle)
{

    Mat dst;
    cv::Size src_sz = src.size();
    cv::Size dst_sz(src_sz.height, src_sz.width);
    int len = std::max(src.cols, src.rows);

//指定旋转中心
    cv::Point2f center(len / 2., len / 2.);

//获取旋转矩阵（2x3矩阵）
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

//根据旋转矩阵进行仿射变换
    cv::warpAffine(src, dst, rot_mat, dst_sz);
    return dst;

}
Mat tool::get_R_mat(Mat src, double angle) {
    Point2f srcTri[3];
    Point2f dstTri[3];
    int height = src.rows;
    int p_width = (int) (height*tan(CV_PI/180*(angle))/2.0);
    srcTri[0] = Point2f( 0,0 );
    srcTri[1] = Point2f( src.cols - 1, 0 );
    srcTri[2] = Point2f( 0, src.rows - 1 );

    dstTri[0] = Point2f( -p_width+p_width, 0 );
    dstTri[1] = Point2f( src.cols - 1 - p_width+p_width, 0 );
    dstTri[2] = Point2f( p_width+p_width, src.rows - 1 );
    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = getAffineTransform( srcTri, dstTri );
    Mat warp_dst;
    Size targetSize(src.cols+2*p_width,src.rows);
    warpAffine( src, warp_dst, warp_mat,targetSize );
    return warp_dst;

}
Mat tool::Resize(Mat src,Size si){

    Mat desImg(si, CV_8UC1, Scalar(255));//创建一个全黑的图片
    //可能会大于3/2 的该特殊处理这种情况
    Mat dst;
//    if(src.cols/(double)src.rows<0.3)
//        dst = Mat(si.height-1, src.cols*(si.height-1)/src.rows, CV_8UC1, Scalar(255));
//    else
    dst =  Mat(si.height, si.width, CV_8UC1, Scalar(255));
//    if(src.cols*si.height<src.rows*si.width)
//
//    else
//         dst = Mat(src.cols*(si.width-1)/src.cols, si.width-1, CV_8UC1, Scalar(255));

    resize(src, dst, dst.size());
//    Mat imageROI;
//    imageROI = desImg(Rect(0, 0, dst.cols, dst.rows)); //3 是固定值
//    dst.copyTo(imageROI);

    return  dst;
}

Mat tool::char_threshold(Mat srcMat, Mat gray,string name,Mat max,int type){
    Mat threshold_value,threshold_value2,threshold_value3,threshold_value4;
    //
    Mat binary,binary2;
   // gray = tool::stretch(gray);
    DebugOut(STR(gray_t),name,gray);
   // medianBlur(gray,gray,5);
    //medianBlur(max,max,5);
    int t;
    //if(type==1) {
    t = (int) threshold(gray, binary, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
    int t2 = (int) threshold(max, binary2, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);

    //大津阈值法

    //}
//    else{
//        t = (int) threshold(gray, binary, 0, 255, CV_THRESH_TRIANGLE | CV_THRESH_BINARY);
//    }
    vector<int> v;
    v=(vector<int>)gray.reshape(1,1);
    sort(v.begin(),v.end());

    int th1 = (int)v.size()*0.1;
    int th2 = t;

    //int th3 = (int)v.size()*0.4;
    int th3 = t2;
    th2=min(th2,254);  //防止超过255
    threshold(gray,threshold_value2,th2, 255, CV_THRESH_BINARY); //前景
    threshold(gray,threshold_value3,v[th1], 255, CV_THRESH_BINARY); //背景
    threshold(max,threshold_value,th3, 255, CV_THRESH_BINARY); //可能的前景
    threshold_value3=~threshold_value3;
    DebugOut(STR(threshold_value3),name,threshold_value3);
    //int th3 = (int)threshold(gray,threshold_value,0, 255, CV_THRESH_TRIANGLE|CV_THRESH_BINARY);
    // threshold_value4=~threshold_value3;
    cv::Mat mask1(srcMat.size(),CV_8UC1,Scalar(1));
    cv::Mat mask2(srcMat.size(),CV_8UC1,Scalar(2));
    cv::Mat mask3(srcMat.size(),CV_8UC1,Scalar(3));
    cv::Mat mask0(srcMat.size(),CV_8UC1,Scalar(0));
    cv::Mat bgmodle, fgmodel;
    cv::Mat result(srcMat.size(),CV_8UC1,Scalar(2)); //2可能的背景

    //for debug
    cv::Mat dmask1(srcMat.size(),CV_8UC3,Scalar(255,255,255));//白色th2 前景
    cv::Mat dmask2(srcMat.size(),CV_8UC3,Scalar(255,0,0));   //蓝色可能的北京
    cv::Mat dmask3(srcMat.size(),CV_8UC3,Scalar(0,255,255));//黄色可能的前景
    cv::Mat dmask0(srcMat.size(),CV_8UC3,Scalar(0,0,0));  //黑色　th3

    cv::Mat dresult(srcMat.size(),CV_8UC3,Scalar(255,0,0)); //2可能的背景

    dmask3.copyTo(dresult,threshold_value);
    dmask1.copyTo(dresult,threshold_value2);
    dmask0.copyTo(dresult,threshold_value3);

    //  cv::imshow("resultsss", threshold_value);
    mask3.copyTo(result,threshold_value);
    mask1.copyTo(result,threshold_value2);
    mask0.copyTo(result,threshold_value3);


    Rect rect;
    Mat result1;
    //  cv::compare(result, cv::GC_FGD, result1, cv::CMP_EQ);
    cv::grabCut(srcMat, result,rect, bgmodle, fgmodel, 20, cv::GC_INIT_WITH_MASK);
    Mat FGD,PR_FGD,PR_BGD;
    cv::compare(result, cv::GC_FGD, FGD, cv::CMP_EQ);
    cv::compare(result, cv::GC_PR_FGD, PR_FGD, cv::CMP_EQ);
    cv::compare(result, cv::GC_PR_BGD, PR_BGD, cv::CMP_EQ);

    cv::Mat display(srcMat.size(),CV_8UC3,Scalar(0,0,0)); //2可能的背景
    dmask1.copyTo(display,FGD);
    dmask2.copyTo(display,PR_BGD);
    dmask3.copyTo(display,PR_FGD);
    DebugOut(STR(display),name,display);


//    result1=temp|result1;
//    if(mean(temp)[0]<55){
//        temp=result1;
//    }
//    DebugOut(STR(out),name,temp);
    DebugOut(STR(mark_grap_cut),name,dresult);
//    DebugOut(STR(out_ts),name,result1);
//    cout<<name<<"||"<<t<<"||"<<mean(temp)[0]<<endl;
//    cv::imshow("resultss", dresult);
//    cv::imshow("result", result);
//    waitKey(0);
    return std::move(FGD);
    //  cv::Mat foreground(srcMat.size(), CV_8UC3, cv::Scalar(0, 0, 0));

     //  srcMat.copyTo(foreground, result);
     //cv::imshow("result", result);
//    cv::imshow("result1", result1);

    //   cv::imshow("结果", foreground);
    //   result1.copyTo(result,threshold_value2);
//    imshow("sss",threshold_value);
    //   imshow("ssss",threshold_value2);
    //  imshow("sssss",threshold_value3);
//    imshow("ssssss",result);
    cout<<th3<<endl;
}
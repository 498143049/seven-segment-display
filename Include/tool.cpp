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
vector<int> tool::get_row_number( const Mat& picture, double rate) {

    vector<int> result(picture.rows,0);
    for (int i = 0; i < picture.rows*rate; i++)
        for (int j = 0; j < picture.cols; j++)
            if (picture.at<uchar>(i, j) > 0)
                result[i] += 1;
    return std::move(result);

}
vector<int> tool::get_col_number( const Mat& picture,double rate ) {

    vector<int> result(static_cast<uint64_t>(picture.cols), 0);
    for (int j = 0; j < picture.cols; j++)
        for (int i = 0; i < picture.rows*rate; i++)
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
double tool::correct_error(Mat &src,double &best_angle,Mat &canny_dst){

    vector<double> max_array; //存储最大值的素组
    vector<double> array_p;   //生成的角度素组P

    double angle_Horizontal;  //水平角度
    double angle_vertical;    //竖直角度

    double low_thresh = 0.0;
    double high_thresh = 0.0;

    tool::AdaptiveFindThreshold(src,&low_thresh,&high_thresh);
    //canny 边缘检测检测边缘更浅
    Canny(src,canny_dst,low_thresh,high_thresh,3);
    //初始化数组　
    //构造角度素组
    for(int i=-25;i<25;++i)
        for(int j=0;j<10;++j)
            array_p.push_back(i+j*0.1);


    //获取最后返回的数组　
    DebugOut("test","canyytest",canny_dst);
    auto s = tool::radon(canny_dst,array_p);
    //找到每一个角度的最大值　//为什么不计算累加值
    for(auto single:s){
        max_array.push_back(*max_element(single.begin(),single.end()));
    }
    //角度限制在   60~120 度之间
//    for(auto i:max_array){
//        cout<<i<<" ";
//    }
//    cout<<endl;
    angle_vertical = std::distance(max_array.begin(),max_element(max_array.begin(),max_array.end()));


    best_angle=0;
    angle_vertical = 90-25+angle_vertical/10.0;
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


Mat tool::char_threshold2(Mat srcMat, Mat gray,string name,Mat max,int type){
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

    vector<int> v;
    v=(vector<int>)gray.reshape(1,1);
    sort(v.begin(),v.end());

    int th1 = (int)v.size()*0.1;
    int th2 = t;

    //int th3 = (int)v.size()*0.4;
    th2=min(th2,254);  //防止超过255
    threshold(gray,threshold_value2,th2, 255, CV_THRESH_BINARY); //前景

    threshold(gray,threshold_value3,v[th1], 255, CV_THRESH_BINARY); //背景
   // threshold(max,threshold_value,th3, 255, CV_THRESH_BINARY); //可能的前景
    threshold_value3=~threshold_value3;
    DebugOut(STR(threshold_value3),name,threshold_value3);

    threshold_value3.at<Vec3b>(threshold_value3.rows/4,threshold_value3.cols/2) = 255;
    threshold_value3.at<Vec3b>(threshold_value3.rows/4+1,threshold_value3.cols/2) = 255;
    threshold_value3.at<Vec3b>(threshold_value3.rows/4,threshold_value3.cols/2+1) = 255;
    threshold_value3.at<Vec3b>(threshold_value3.rows/4-1,threshold_value3.cols/2) = 255;
    threshold_value3.at<Vec3b>(threshold_value3.rows/4,threshold_value3.cols/2+1) = 255;


    threshold_value3.at<Vec3b>(3*threshold_value3.rows/4,threshold_value3.cols/2) = 255;
    threshold_value3.at<Vec3b>(3*threshold_value3.rows/4+1,threshold_value3.cols/2) = 255;
    threshold_value3.at<Vec3b>(3*threshold_value3.rows/4,threshold_value3.cols/2+1) = 255;
    threshold_value3.at<Vec3b>(3*threshold_value3.rows/4-1,threshold_value3.cols/2) = 255;
    threshold_value3.at<Vec3b>(3*threshold_value3.rows/4,threshold_value3.cols/2+1) = 255;
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

    //dmask3.copyTo(dresult,threshold_value);
    dmask1.copyTo(dresult,threshold_value2);
    dmask0.copyTo(dresult,threshold_value3);

    //  cv::imshow("resultsss", threshold_value);
    //mask3.copyTo(result,threshold_value);
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


    DebugOut(STR(mark_grap_cut),name,dresult);
    return std::move(FGD);
}

cv::Mat tool::thinImage(const cv::Mat & src, const int maxIterations) {
    assert(src.type() == CV_8UC1);
    cv::Mat dst;
    int width  = src.cols;
    int height = src.rows;
    src.copyTo(dst);
    int count = 0;  //记录迭代次数
    while (true)
    {
        count++;
        if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达
            break;
        std::vector<uchar *> mFlag; //用于标记需要删除的点
        //对点标记
        for (int i = 0; i < height ;++i)
        {
            uchar * p = dst.ptr<uchar>(i);
            for (int j = 0; j < width; ++j)
            {
                //如果满足四个条件，进行标记
                //  p9 p2 p3
                //  p8 p1 p4
                //  p7 p6 p5
                uchar p1 = p[j];
                if (p1 != 1) continue;
                uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
                uchar p8 = (j == 0) ? 0 : *(p + j - 1);
                uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
                uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
                uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
                uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
                uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
                uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
                if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
                {
                    int ap = 0;
                    if (p2 == 0 && p3 == 1) ++ap;
                    if (p3 == 0 && p4 == 1) ++ap;
                    if (p4 == 0 && p5 == 1) ++ap;
                    if (p5 == 0 && p6 == 1) ++ap;
                    if (p6 == 0 && p7 == 1) ++ap;
                    if (p7 == 0 && p8 == 1) ++ap;
                    if (p8 == 0 && p9 == 1) ++ap;
                    if (p9 == 0 && p2 == 1) ++ap;

                    if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
                    {
                        //标记
                        mFlag.push_back(p+j);
                    }
                }
            }
        }

        //将标记的点删除
        for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
        {
            **i = 0;
        }

        //直到没有点满足，算法结束
        if (mFlag.empty())
        {
            break;
        }
        else
        {
            mFlag.clear();//将mFlag清空
        }

        //对点标记
        for (int i = 0; i < height; ++i)
        {
            uchar * p = dst.ptr<uchar>(i);
            for (int j = 0; j < width; ++j)
            {
                //如果满足四个条件，进行标记
                //  p9 p2 p3
                //  p8 p1 p4
                //  p7 p6 p5
                uchar p1 = p[j];
                if (p1 != 1) continue;
                uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
                uchar p8 = (j == 0) ? 0 : *(p + j - 1);
                uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
                uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
                uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
                uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
                uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
                uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

                if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
                {
                    int ap = 0;
                    if (p2 == 0 && p3 == 1) ++ap;
                    if (p3 == 0 && p4 == 1) ++ap;
                    if (p4 == 0 && p5 == 1) ++ap;
                    if (p5 == 0 && p6 == 1) ++ap;
                    if (p6 == 0 && p7 == 1) ++ap;
                    if (p7 == 0 && p8 == 1) ++ap;
                    if (p8 == 0 && p9 == 1) ++ap;
                    if (p9 == 0 && p2 == 1) ++ap;

                    if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
                    {
                        //标记
                        mFlag.push_back(p+j);
                    }
                }
            }
        }

        //将标记的点删除
        for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
        {
            **i = 0;
        }

        //直到没有点满足，算法结束
        if (mFlag.empty())
        {
            break;
        }
        else
        {
            mFlag.clear();//将mFlag清空
        }
    }
    return dst;
}
void tool::thin(const Mat &src, Mat &dst, const int iterations)
{
    const int height =src.rows -1;
    const int width  =src.cols -1;

    //拷贝一个数组给另一个数组
    if(src.data != dst.data)
    {
        src.copyTo(dst);
    }


    int n = 0,i = 0,j = 0;
    Mat tmpImg;
    uchar *pU, *pC, *pD;
    bool isFinished = false;

    for(n=0; n<iterations; n++)
    {
        dst.copyTo(tmpImg);
        isFinished =false;   //一次 先行后列扫描 开始
        //扫描过程一 开始
        for(i=1; i<height;  i++)
        {
            pU = tmpImg.ptr<uchar>(i-1);
            pC = tmpImg.ptr<uchar>(i);
            pD = tmpImg.ptr<uchar>(i+1);
            for(int j=1; j<width; j++)
            {
                if(pC[j] > 0)
                {
                    int ap=0;
                    int p2 = (pU[j] >0);
                    int p3 = (pU[j+1] >0);
                    if (p2==0 && p3==1)
                    {
                        ap++;
                    }
                    int p4 = (pC[j+1] >0);
                    if(p3==0 && p4==1)
                    {
                        ap++;
                    }
                    int p5 = (pD[j+1] >0);
                    if(p4==0 && p5==1)
                    {
                        ap++;
                    }
                    int p6 = (pD[j] >0);
                    if(p5==0 && p6==1)
                    {
                        ap++;
                    }
                    int p7 = (pD[j-1] >0);
                    if(p6==0 && p7==1)
                    {
                        ap++;
                    }
                    int p8 = (pC[j-1] >0);
                    if(p7==0 && p8==1)
                    {
                        ap++;
                    }
                    int p9 = (pU[j-1] >0);
                    if(p8==0 && p9==1)
                    {
                        ap++;
                    }
                    if(p9==0 && p2==1)
                    {
                        ap++;
                    }
                    if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7)
                    {
                        if(ap==1)
                        {
                            if((p2*p4*p6==0)&&(p4*p6*p8==0))
                            {
                                dst.ptr<uchar>(i)[j]=0;
                                isFinished = true;
                            }

                            //   if((p2*p4*p8==0)&&(p2*p6*p8==0))
                            //    {
                            //         dst.ptr<uchar>(i)[j]=0;
                            //         isFinished =TRUE;
                            //    }

                        }
                    }
                }

            } //扫描过程一 结束


            dst.copyTo(tmpImg);
            //扫描过程二 开始
            for(i=1; i<height;  i++)  //一次 先行后列扫描 开始
            {
                pU = tmpImg.ptr<uchar>(i-1);
                pC = tmpImg.ptr<uchar>(i);
                pD = tmpImg.ptr<uchar>(i+1);
                for(int j=1; j<width; j++)
                {
                    if(pC[j] > 0)
                    {
                        int ap=0;
                        int p2 = (pU[j] >0);
                        int p3 = (pU[j+1] >0);
                        if (p2==0 && p3==1)
                        {
                            ap++;
                        }
                        int p4 = (pC[j+1] >0);
                        if(p3==0 && p4==1)
                        {
                            ap++;
                        }
                        int p5 = (pD[j+1] >0);
                        if(p4==0 && p5==1)
                        {
                            ap++;
                        }
                        int p6 = (pD[j] >0);
                        if(p5==0 && p6==1)
                        {
                            ap++;
                        }
                        int p7 = (pD[j-1] >0);
                        if(p6==0 && p7==1)
                        {
                            ap++;
                        }
                        int p8 = (pC[j-1] >0);
                        if(p7==0 && p8==1)
                        {
                            ap++;
                        }
                        int p9 = (pU[j-1] >0);
                        if(p8==0 && p9==1)
                        {
                            ap++;
                        }
                        if(p9==0 && p2==1)
                        {
                            ap++;
                        }
                        if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7)
                        {
                            if(ap==1)
                            {
                                //   if((p2*p4*p6==0)&&(p4*p6*p8==0))
                                //   {
                                //         dst.ptr<uchar>(i)[j]=0;
                                //         isFinished =TRUE;
                                //    }

                                if((p2*p4*p8==0)&&(p2*p6*p8==0))
                                {
                                    dst.ptr<uchar>(i)[j]=0;
                                    isFinished = true;
                                }

                            }
                        }
                    }

                }

            } //一次 先行后列扫描完成
            //如果在扫描过程中没有删除点，则提前退出
            if(isFinished == false)
            {
                break;
            }
        }

    }
}
//如果是1的话,直接输出结果 //输入处理过的二值图片
#define  need_pixels 2
/**
 *             HORIZ_UP
 *
 * VERT_LEFT_UP          VERT_RIGHT_UP
 *
 *             HORIZ_MID
 *
 * VERT_LEFT_DOWN         VERT_RIGHT_DOWN
 *
 *             HORIZ_DOWN
 *
 */
#define HORIZ_UP 1
#define VERT_LEFT_UP 2
#define VERT_RIGHT_UP 4
#define HORIZ_MID 8
#define VERT_LEFT_DOWN 16
#define VERT_RIGHT_DOWN 32
#define HORIZ_DOWN 64
#define ALL_SEGS 127
#define DECIMAL 128
#define MINUS 256

/* digits */
#define D_OTHER_SIX (ALL_SEGS & ~VERT_LEFT_UP)
#define D_U (ALL_SEGS & ~(HORIZ_MID |HORIZ_UP) )
#define D_P (ALL_SEGS & ~(HORIZ_DOWN |VERT_RIGHT_DOWN) )
#define D_H (ALL_SEGS & ~(HORIZ_UP |HORIZ_DOWN) )
#define Mid (ALL_SEGS & ~(HORIZ_UP |HORIZ_DOWN|VERT_LEFT_UP|VERT_RIGHT_UP|VERT_LEFT_DOWN|VERT_RIGHT_DOWN) )


#define D_ZERO (ALL_SEGS & ~HORIZ_MID)
#define D_ONE (VERT_RIGHT_UP | VERT_RIGHT_DOWN)
#define D_TWO (ALL_SEGS & ~(VERT_LEFT_UP | VERT_RIGHT_DOWN))
#define D_THREE (ALL_SEGS & ~(VERT_LEFT_UP | VERT_LEFT_DOWN))
#define D_FOUR (ALL_SEGS & ~(HORIZ_UP | VERT_LEFT_DOWN | HORIZ_DOWN))
#define D_FIVE (ALL_SEGS & ~(VERT_RIGHT_UP | VERT_LEFT_DOWN))
#define D_SIX (ALL_SEGS & ~VERT_RIGHT_UP)
#define D_SEVEN (HORIZ_UP | VERT_RIGHT_UP | VERT_RIGHT_DOWN)
#define D_ALTSEVEN (VERT_LEFT_UP | D_SEVEN)
#define D_EIGHT ALL_SEGS
#define D_NINE (ALL_SEGS & ~VERT_LEFT_DOWN)
#define D_ALTNINE (ALL_SEGS & ~(VERT_LEFT_DOWN | HORIZ_DOWN))
#define D_DECIMAL DECIMAL
#define D_MINUS MINUS
#define D_HEX_A (ALL_SEGS & ~HORIZ_DOWN)
#define D_HEX_b (ALL_SEGS & ~(HORIZ_UP | VERT_RIGHT_UP))
#define D_HEX_C (ALL_SEGS & ~(VERT_RIGHT_UP | HORIZ_MID | VERT_RIGHT_DOWN))
#define D_HEX_c (HORIZ_MID | VERT_LEFT_DOWN | HORIZ_DOWN)
#define D_HEX_d (ALL_SEGS & ~(HORIZ_UP | VERT_LEFT_UP))
#define D_HEX_E (ALL_SEGS & ~(VERT_RIGHT_UP | VERT_RIGHT_DOWN))
#define D_HEX_F (ALL_SEGS & ~(VERT_RIGHT_UP | VERT_RIGHT_DOWN | HORIZ_DOWN))
#define D_UNKNOWN 0


string tool::location(Mat &src){
    float heightDivWidth = (float)src.rows/src.cols;
    if (heightDivWidth > ONE_HIGHT_WIDTH_RATE) //还需要验证一下
    {
        double t = cv::mean(src)[0];
        if(t>90) {
            return "1";
        }
        else
        {
            return "";
        }
    }

    //读取图片
    //模板滤波
    if(src.rows<40) return " ";
    /* Mat templates = imread("../datasource/template/template.jpg",IMREAD_GRAYSCALE); //灰度图
    Size targetsize(src.cols, src.rows);
    resize(templates,templates,targetsize);
    src = src&templates; */


    uint8_t type;
    int middle=0, quarter=0, three_quarters=0; /* scanlines */
    //if(width/heiht<2.3) return "1";
    int d_height=src.rows; /* height of digit */
    int third=1; /* in which third we are */
    int half;
    int found_pixels=0; /* how many pixels are already found */
    middle = src.cols/2;
    for(int i=0;i<src.rows;i++){
        uchar * pData1=src.ptr<uchar>(i);
        if(pData1[middle]>1){
            found_pixels++;
          //  pData1[middle]=0;

        }
        //判断3个格子并记录
        /* pixels in first third count towards upper segment */
        if(i >= d_height/3 && third == 1) {
            if(found_pixels >= need_pixels) {
                type |= HORIZ_UP; /* add upper segment */
            }
            found_pixels = 0;
            third++;
        } else if(i>=2*d_height/3 && third == 2) {
            /* pixels in second third count towards middle segment */
            if(found_pixels >= need_pixels) {
                type |= HORIZ_MID; /* add middle segment */
            }
            found_pixels = 0;
            third++;
        }
    }
    /* found_pixels contains pixels of last third */
    if(found_pixels >= need_pixels) {
        type |= HORIZ_DOWN; /* add lower segment */
    }
    found_pixels = 0;
    half=1; /* in which half we are */
    quarter = src.rows/4;

    //第一个竖线
    uchar * pData1=src.ptr<uchar>(quarter);
    for (int j=0; j<src.cols; ++j) {
        if(pData1[j]>1) {
            found_pixels++;
        }
        if(j >= middle && half == 1) {
            if(found_pixels >= need_pixels) {
                type |= VERT_LEFT_UP;
            }
            found_pixels = 0;
            half++;
        }
    }
    if(found_pixels >= need_pixels) {
        type |= VERT_RIGHT_UP;
    }
    found_pixels = 0;
    half = 1;
    /* check lower vertical segments */
    half=1; /* in which half we are */
    three_quarters = 3*src.rows/ 4;
    pData1=src.ptr<uchar>(three_quarters);
    for (int j=0; j<src.cols; ++j) {
        if(pData1[j]>1) {
            found_pixels++;
        }
        if(j >= middle && half == 1) {
            if(found_pixels >= need_pixels) {
                type |= VERT_LEFT_DOWN;
            }
            found_pixels = 0;
            half++;
        }
    }
    if(found_pixels >= need_pixels) {
        type |= VERT_RIGHT_DOWN;
    }
  //  cout<<bitset<sizeof(int)*8>(type)<<endl;
    found_pixels = 0;
    switch(type) {
        case D_ZERO: return "0"; break;
        case D_ONE: return "1"; break;
        case D_TWO: return "2"; break;
        case D_THREE: return "3"; break;
        case D_FOUR: return "4"; break;
        case D_FIVE: return "5"; break;
        case D_SIX: return "6"; break;
        case D_SEVEN: /* fallthrough */
        case D_ALTSEVEN: return "7"; break;
        case D_EIGHT: return "8"; break;
        case D_NINE: /* fallthrough */
        case D_ALTNINE: return "9"; break;
        case D_DECIMAL: return "."; break;
        case D_MINUS: return "-"; break;
        case D_HEX_A: return "a"; break;
        case D_HEX_b: return "b"; break;
        case D_HEX_C: /* fallthrough */
        case D_HEX_c: return "c"; break;
        case D_HEX_d: return "d"; break;
        case D_HEX_E: return "e"; break;
        case D_HEX_F: return "f"; break;
        case D_U: return "U";break;
        case D_OTHER_SIX: return "6"; break;
        case D_P:return "P"; break;
        case D_H:return "H"; break;
        case Mid:return "-"; break;
        default: return "X";
    }
    //根据type 判断
}
// 仿照matlab，自适应求高低两个门限
void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high)
{
    CvSize size;
    IplImage *imge=0;
    int i,j;
    CvHistogram *hist;
    int hist_size = 255;
    float range_0[]={0,256};
    float* ranges[] = { range_0 };
    double PercentOfPixelsNotEdges = 0.5;//调整的参数
    size = cvGetSize(dx);
    imge = cvCreateImage(size, IPL_DEPTH_32F, 1);
    // 计算边缘的强度, 并存于图像中
    float maxv = 0;
    for(i = 0; i < size.height; i++ )
    {
        const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
        float* _image = (float *)(imge->imageData + imge->widthStep*i);
        for(j = 0; j < size.width; j++)
        {
            _image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
            maxv = maxv < _image[j] ? _image[j]: maxv;

        }
    }
    if(maxv == 0){
        *high = 0;
        *low = 0;
        cvReleaseImage( &imge );
        return;
    }

    // 计算直方图
    range_0[1] = maxv;
    hist_size = (int)(hist_size > maxv ? maxv:hist_size);
    hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
    cvCalcHist( &imge, hist, 0, NULL );
    int total = (int)(size.height * size.width * PercentOfPixelsNotEdges);
    float sum=0;
    int icount = hist->mat.dim[0].size;

    float *h = (float*)cvPtr1D( hist->bins, 0 );
    for(i = 0; i < icount; i++)
    {
        sum += h[i];
        if( sum > total )
            break;
    }
    // 计算高低门限
    *high = (i+1) * maxv / hist_size ;
    *low = *high * 0.4;
    cvReleaseImage( &imge );
    cvReleaseHist(&hist);
}
//自适应canny 参数
void tool::AdaptiveFindThreshold(const Mat image, double *low, double *high, int aperture_size)
{
    cv::Mat src = image;
    const int cn = src.channels();
    cv::Mat dx(src.rows, src.cols, CV_16SC(cn));
    cv::Mat dy(src.rows, src.cols, CV_16SC(cn));

    //使用soble 得到边缘
    cv::Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);

    CvMat _dx = dx, _dy = dy;
    _AdaptiveFindThreshold(&_dx, &_dy, low, high);

}



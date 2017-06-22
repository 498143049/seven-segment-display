//
// Created by dubing on 17-4-29.
//

#include "web_interface.h"



#include <string>
using namespace std;
string web_interface::get_content_pic() {
    Json::Value arrayObj;   // 构建对象

    auto path = boost::filesystem::current_path();
    //cout<<p.filename()<<endl;
    path = path.parent_path().append("/datasource/ndata");
    //boost::filesystem::path  path("/media/dubing/liunxFile/back_up_t/digital_linux/datasource/ndata");  //获取图片的文件

    boost::filesystem::recursive_directory_iterator beg_iter(path);
    boost::filesystem::recursive_directory_iterator end_iter;
    vector<string> vect;

    for (; beg_iter != end_iter; ++beg_iter) {
        if (boost::filesystem::is_regular_file(*beg_iter)) {
            vect.push_back(beg_iter->path().string());
        }
    }
    sort(vect.begin(),vect.end(),[](string s1, string s2){
        boost::smatch mat1,mat2;
        boost::regex reg( "-?[0-9]\\d*" );    //查找字符串里的数字
        boost::regex_search(s1,mat1,reg);
        boost::regex_search(s2,mat2,reg);
        string s3 = mat1[0].str();
        string s4 = mat2[0].str();
        return boost::lexical_cast<int>(s4)>boost::lexical_cast<int>(s3);
    });
    arrayObj["type"] = "content";
    for(auto s:vect){
        arrayObj["url"].append(s);
    }

    Json::FastWriter writer;
    std::string out2 = writer.write(arrayObj);
    return std::move(out2);
}

string web_interface::get_contain_pic(string name) {
    //判断对于的有没有json文件
    boost::format frm_dir("../datasource/logfile/%d.json");
    frm_dir %name;
    //如果没有则调用处理图像的文件
    //调用对应的初始化]
    boost::filesystem::path p_1(frm_dir.str());
    if(!boost::filesystem::exists(p_1)) {
        DigtalLocate temp =    DigtalLocate("../datasource/ndata/" + name + ".jpg", 1.1);
        temp.probably_locate();
        temp.output_joson();
    }

    std::ifstream t(frm_dir.str());
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());

    return std::move(contents);

}
//void web_interface::output_joson(DigtalLocate *tmp) {
//    Json::Value arrayObj;   // 构建对象
//
//    arrayObj["result"] = tmp->out_put_result_mat;
//    for(auto s:tmp->output_mat_url) {
//        Json::Value item;
//        item["name"] = s.first;
//        item["url"] = s.second;
//        arrayObj["debug"].append(std::move(item));
//    }
//
//    for(auto s:tmp->output_sub_mat_url) {
//        arrayObj["suburl"].append(s);
//    }
//
//    Json::FastWriter writer;
//    std::string out2 = writer.write(arrayObj);
//    //讲字符串写入文件
//    boost::format frm_dir("../datasource/tmp/logfile/%d.json");
//    frm_dir %tmp->dirname;
//    ofstream outfile;
//    outfile.open(frm_dir.str(), ios::out  | ios::trunc );
//    if (outfile.is_open())
//    {
//        outfile << out2;
//        outfile.close();
//    }
//}
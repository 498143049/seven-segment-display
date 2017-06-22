//////////////////////////////////////////////////////////////////////////
//  SVMTest.h
// 2016-12-12,by QQ
//
// Please contact me if you find any bugs, or have any suggestions.
// Contact:
//      Telephone:15366105857
//      Email:654393155@qq.com
//      Blog: http://blog.csdn.net/qianqing13579
//////////////////////////////////////////////////////////////////////////

#ifndef __SVMTEST__
#define __SVMTEST__

#include "opencv2/ml.hpp"
//#include"../Utility/CommonUtility.h"
//#include"../Utility/LogInterface.h"
#include<fstream>
#include"LBP.h"
using namespace cv::ml;

// if you do not need log,comment it,just like :#define LOG_WARN_SVM_TEST(...)                  //LOG4CPLUS_MACRO_FMT_BODY ("SVMTest", WARN_LOG_LEVEL, __VA_ARGS__)
#define LOG_DEBUG_SVM_TEST(...)                //LOG4CPLUS_MACRO_FMT_BODY ("SVMTest", DEBUG_LOG_LEVEL, __VA_ARGS__)
#define LOG_ERROR_SVM_TEST(...)                 //LOG4CPLUS_MACRO_FMT_BODY ("SVMTest", ERROR_LOG_LEVEL, __VA_ARGS__)
#define LOG_INFO_SVM_TEST(...)                    //LOG4CPLUS_MACRO_FMT_BODY ("SVMTest", INFO_LOG_LEVEL, __VA_ARGS__)
#define LOG_WARN_SVM_TEST(...)                  //LOG4CPLUS_MACRO_FMT_BODY ("SVMTest", WARN_LOG_LEVEL, __VA_ARGS__)
//#define CONFIG_FILE                                              "./Resource/Configuration.xml"
#define CELL_SIZE   16

class SVMTest
{
public:
    SVMTest(const string &_trainDataFileList,
            const string &_testDataFileList,
            const string &_svmModelFilePath,
            const string &_predictResultFilePath,
            SVM::Types svmType, // See SVM::Types. Default value is SVM::C_SVC.
            SVM::KernelTypes kernel,
            double c, // For SVM::C_SVC, SVM::EPS_SVR or SVM::NU_SVR. Default value is 0.
            double coef,  // For SVM::POLY or SVM::SIGMOID. Default value is 0.
            double degree, // For SVM::POLY. Default value is 0.
            double gamma, // For SVM::POLY, SVM::RBF, SVM::SIGMOID or SVM::CHI2. Default value is 1.
            double nu,  // For SVM::NU_SVC, SVM::ONE_CLASS or SVM::NU_SVR. Default value is 0.
            double p // For SVM::EPS_SVR. Default value is 0.
    );
    bool Initialize();
    virtual ~SVMTest();

    void Train();
    void Predict();

private:
    string trainDataFileList;
    string testDataFileList;
    string svmModelFilePath;
    string predictResultFilePath;

    // SVM
    Ptr<SVM> svm;

    // feature extracting(HOG,LBP,Haar,etc)
    LBP lbp;


};


#endif // SVMTEST
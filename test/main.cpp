#include <iostream>
#include <vector>
#include "../Include/DigtalLocate.h"


using namespace std;
using namespace cv;
int main() {
    cout<<"I am ok!"<<endl;
    for(int i=0; i<=17 ;i++) {
        DigtalLocate("../datasource/single/single_"+to_string(i)+".jpg",0);
    }
    cout<<"I am ok!！！"<<endl;
}
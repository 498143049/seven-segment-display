//
// Created by dubing on 17-4-29.
//

#ifndef DIGITAL_LINUX_WEB_INTERFACE_H
#define DIGITAL_LINUX_WEB_INTERFACE_H

#include <json/json.h>
#include <string>
#include<boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include "DigtalLocate.h"
using namespace std;
class web_interface {
public:
    static string get_content_pic();
    static string get_contain_pic(string name);


};


#endif //DIGITAL_LINUX_WEB_INTERFACE_H

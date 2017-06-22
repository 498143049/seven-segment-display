# encoding:UTF-8
# 修改图片名称,名称为文件夹名＋后缀名
import sys,os
import shutil

def modify_pic_name(path):
    # 所有文件夹，第一个字段是次目录的级别
    files = os.listdir(path)

    for dir in files:
        if(os.path.isdir(os.path.join(path,dir))):
            # 排除隐藏文件夹。因为隐藏文件夹过多
            if(dir[0] == '.'):
                pass
            else:
                #寻找文件夹里面的图片
                    dirpath = os.path.join(path, dir)
                    jpgs = os.listdir(dirpath)
                    for i,jpg in enumerate(jpgs):
                        if (os.path.isfile(os.path.join(dirpath, jpg))):
                            if('.jpg' in jpg):
                                    newname=dir + '_' + str(i) + '.jpg'
                                    shutil.copyfile(os.path.join(dirpath, jpg), os.path.join(path ,newname))
if __name__ == '__main__':
    modify_pic_name(sys.path[0])
    print('ok')

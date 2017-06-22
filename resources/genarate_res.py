# encoding:UTF-8
import sys,os

def write_down(prefix,f,path):
    files = os.listdir(path)
    f.write(u'<qresource> \n')
    for dir in files:
     if not os.path.isdir(os.path.join(path,dir)):
        f.write((u'<file>'+prefix+"/"+dir+'</file>\n'))
    f.write(u'</qresource> \n')
    for dir in files:
     if os.path.isdir(os.path.join(path,dir)):
	if not dir[0]=='.':
          write_down(prefix+"/"+dir,f,os.path.join(path,dir))
   

def genarate_res(path):
    # 所有文件夹，第一个字段是次目录的级别
    files = os.path.join(path,"UI")
    f = open('resources.qrc', 'w+')
    f.write(u'<!DOCTYPE RCC>\n<RCC version="1.0">\n')
    write_down("UI",f,files);
    f.write(u'</RCC>')
    f.close()

if __name__ == '__main__':
    genarate_res(sys.path[0])
    print('ok')


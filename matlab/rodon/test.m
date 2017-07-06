function RadonTest()  
  
fileName='1.jpg';  
srcImage=imread(fileName);  
grayImage=rgb2gray(srcImage);  

cannyImage=edge(grayImage,'canny');  
level = graythresh(grayImage);
 BW = im2bw(grayImage,level);
theta=0:180;  
[R,x]=radon(cannyImage,theta); 
[X,Y]=meshgrid(theta,x);
surf(X,Y,R)
end  


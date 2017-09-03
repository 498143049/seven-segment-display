function RadonTest()  
  
fileName='1.jpg';  
srcImage=imread(fileName);  
grayImage=rgb2gray(srcImage);  

cannyImage=edge(grayImage,'canny');  
level = graythresh(grayImage);
 BW = im2bw(grayImage,level);
theta=-25:0.1:25;
[R,x]=radon(cannyImage,theta); 
[X,Y]=meshgrid(theta,x);
surf(X,Y,R)
C = max(R)
plot(C)
end  


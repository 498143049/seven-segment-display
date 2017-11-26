%% Read image from file
inImg = imread('3.jpg');
inImg = im2double(rgb2gray(inImg));
% inImg = imresize(inImg, [764, 1800], 'bilinear');
%% Spectral Residual
myFFT = fft2(inImg);
myLogAmplitude = log(abs(myFFT));
myPhase = angle(myFFT);
mySmooth = imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
mySpectralResidual = myLogAmplitude - mySmooth;
saliencyMap = abs(ifft2(exp(mySpectralResidual + i*myPhase))).^2;
%% After Effect
saliencyMap = imfilter(saliencyMap, fspecial('disk', 3));
saliencyMap = mat2gray(saliencyMap);
imshow(saliencyMap, []);
%%%%%%
level = graythresh(saliencyMap);%%matlab 自带的自动确定阈值的方法，大津法，类间方差
BW = im2bw(saliencyMap,level);%%用得到的阈值直接对图像进行二值化
imshow(BW)
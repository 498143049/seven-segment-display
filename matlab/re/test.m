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
level = graythresh(saliencyMap);%%matlab �Դ����Զ�ȷ����ֵ�ķ�������򷨣���䷽��
BW = im2bw(saliencyMap,level);%%�õõ�����ֱֵ�Ӷ�ͼ����ж�ֵ��
imshow(BW)
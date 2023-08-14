% This file is for you to be familar with MATLAB and an image
% Created by:   Qian Kemao
% Last update:  10/08/2021

%% 1: You can use MATLAB like a calculator

a=1
b=10
c=a+b
d=sin(30)
e=sin(30/180*pi)

%% 2: what are pixels

%read in an image
a=imread('Singapore.jpg');
%see the size (I always do)
[m n k]=size(a)
%display it
imagesc(a); axis on;
%zoom in to see the pixels by selecting "+" tool to magnify the image

%% 3A: Understand the subsmapling

close all
figure;imshow(uint8(a(1:1:end,1:1:end,:)))
figure;imshow(uint8(a(1:2:end,1:2:end,:)))
figure;imshow(uint8(a(1:4:end,1:4:end,:)))
figure;imshow(uint8(a(1:8:end,1:8:end,:)))
figure;imshow(uint8(a(1:16:end,1:16:end,:)))


%% 3B: imshow vs imagesc (I use the latter much more)

close all
figure;imagesc(uint8(a(1:1:end,1:1:end,:)))
figure;imagesc(uint8(a(1:2:end,1:2:end,:)))
figure;imagesc(uint8(a(1:4:end,1:4:end,:)))
figure;imagesc(uint8(a(1:8:end,1:8:end,:)))
figure;imagesc(uint8(a(1:16:end,1:16:end,:)))

%% 4A: See the color and understand how it is represented

a(100,100,:)

%% 4B: Play more on color

temp(:,:,1)=ones(256,256)*200;
temp(:,:,2)=ones(256,256)*0;
temp(:,:,3)=ones(256,256)*200;
imagesc(uint8(temp))

%% 5A: off the light - use a naive and simple example 
% to show the power of image processing

clear all; close all
%read in the image
a=double(imread('Singapore.jpg'));
%check the size
[m n k]=size(a);
%make a copy in b
b=a;

%change the pixel intensity: if R+G+B is larger than 300, I think it is a
%light source and make it less bright (set to 60)
tic
for i=1:m
    for j=1:n
        if (a(i,j,1)+a(i,j,2)+a(i,j,3))>300
           b(i,j,:)=60;
        end
    end
end
toc %check the computing time

%show the result
figure; imagesc(uint8(a));
figure; imagesc(uint8(b));

%% 5B: off the light again

%same as before
clear all; close all
a=double(imread('Singapore.jpg'));
[m n k]=size(a);
b=a;

%matrix-based operation: MATLAB prefers matrix-based operatioin
tic
%get a matrix where pixels are bright
mask=(a(:,:,1)+a(:,:,2)+a(:,:,3))>300;
%change the values for all masked pixels
b(:,:,1)=b(:,:,1).*(1-mask)+60*mask;
b(:,:,2)=b(:,:,2).*(1-mask)+60*mask;
b(:,:,3)=b(:,:,3).*(1-mask)+60*mask;
toc

%same as before
figure; imagesc(uint8(a));
figure; imagesc(uint8(b));

%%
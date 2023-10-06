%% 2.1 Contrast Stretching

% 2.1.a Converting to Grayscale
Pc = imread('mrt-train.jpg');
whos Pc % 320x443x3 indicates 3 channels, meaning RGB image
P = rgb2gray(Pc); % convert to grayscale
whos P
%imwrite(P, 'grayscale-mrt-train.jpg');

% 2.1.b View Image
imshow(P); % View this image % save image

% 2.1.c Minimum and Maximum Intensities
min(P(:)), max(P(:)) % Check the min & max intensities

% 2.1.d Contrast Stretching
[min_inten, max_inten] = deal(double(min(P(:))), double(max(P(:))));
P2 = (255/(max_inten-min_inten)) * imsubtract(P,min_inten);

min(P2(:)), max(P2(:)) % Check image P2 min & max intensities 

% 2.1.e Redisplay Image
imshow(uint8(P2))
%imwrite(P2, 'contrastStretched-mrt-train.jpg'); % save image

%% 2.2 Histogram Equalization

% 2.2.a Image Intensity Histogram
Pc = imread('mrt-train.jpg');
P = rgb2gray(Pc); % convert to grayscale
imhist(P,10); % histogram of P using 10 bins
imhist(P,256); % histogram of P using 256 bins

% 2.2.b Histogram Equalization
P3 = histeq(P,255); % histogram equalization 
%imwrite(P3, 'equalized-mrt-train.jpg'); % save image
imhist(P3,10); % histogram of P3 using 10 bins
imhist(P3,256); % histogram of P3 using 256 bins

% 2.2.c
P3_rerun = histeq(P3,255); % histogram equalization 
imhist(P3_rerun,10); % histogram of P3_rerun using 10 bins
imhist(P3_rerun,256); % histogram of P3_rerun using 256 bins

%% 2.3 Linear Spatial Filtering 

% 2.3.a Generate filters
x = -2:2; % [-2, -1, 0, 1, 2] 5x5 kernel
y = -2:2; % [-2, -1, 0, 1, 2] 5x5 kernel
[X,Y] = meshgrid(x,y);
% filter for 2.3.a.(i)
sigma_square = 1.0^2; % sigma square for i
e_term_one = exp(-1 * ((X.*X) + (Y.*Y)) / (2*(sigma_square)));
filter_one = (e_term_one) / (2*pi*sigma_square);
filter_one = filter_one ./ sum(filter_one(:));
sum(filter_one(:)) % sanity check to add up to 1
mesh(filter_one)

% filter for 2.3.a.(ii)
sigma_square = 2.0^2; % sigma square for ii
e_term_two = exp(-1 * ((X.*X) + (Y.*Y)) / (2*(sigma_square)));
filter_two = (e_term_two) / (2*pi*sigma_square);
filter_two = filter_two ./ sum(filter_two(:));
sum(filter_two(:)) % sanity check to add up to 1
mesh(filter_two)

% 2.3.b View Image
imshow('lib-gn.jpg');

% 2.3.c Filtering Gaussian Noise
original_img_gn = imread('lib-gn.jpg');
convolved_one = uint8(conv2(original_img_gn, filter_one));
imshow(convolved_one);
%imwrite(convolved_one, 'convolved_gn_one.jpg'); % save image

convolved_two = uint8(conv2(original_img_gn, filter_two));
imshow(convolved_two);
%imwrite(convolved_two, 'convolved_gn_two.jpg'); % save image

% 2.3.d View Image
imshow('lib-sp.jpg');

% 2.3.e Filtering Speckle Noise
original_img_sp = imread('lib-sp.jpg');
convolved_one = uint8(conv2(original_img_sp, filter_one));
imshow(convolved_one);
%imwrite(convolved_one, 'convolved_sp_one.jpg'); % save image

convolved_two = uint8(conv2(original_img_sp, filter_two));
imshow(convolved_two);
%imwrite(convolved_two, 'convolved_sp_two.jpg'); % save image

%% 2.4 Median Filtering 
imshow('lib-gn.jpg');

% gaussian noise
% 3x3 kernel
original_img_gn = imread('lib-gn.jpg');
median_filtered_3_gn =  medfilt2(original_img_gn, [3,3]);
imshow(median_filtered_3_gn);
%imwrite(median_filtered_3_gn, 'median_filtered_gn_3.jpg'); % save image

% 5x5 kernel
median_filtered_5_gn =  medfilt2(original_img_gn, [5,5]);
imshow(median_filtered_5_gn);
%imwrite(median_filtered_5_gn, 'median_filtered_gn_5.jpg'); % save image

% speckle noise
% 3x3 kernel
original_img_sp = imread('lib-sp.jpg');
median_filtered_3_sp =  medfilt2(original_img_sp, [3,3]);
imshow(median_filtered_3_sp);
%imwrite(median_filtered_3_sp, 'median_filtered_sp_3.jpg'); % save image

% 5x5 kernel
median_filtered_5_sp =  medfilt2(original_img_sp, [5,5]);
imshow(median_filtered_5_sp);
%imwrite(median_filtered_5_sp, 'median_filtered_sp_5.jpg'); % save image

%% 2.5 Suppressing Noise Interference Patterns
% 2.5.a display image
imshow('pck-int.jpg');

% 2.5.b power specturm with fftshift
original_img_pck = imread('pck-int.jpg');
F = fft2(original_img_pck); % fast fourier transform, gets kernel coefficients
S = abs(F).^2; %calculate power spectrum
imagesc(fftshift(S.^0.1));
colormap('default');

% 2.5.c power specturm without fftshift
imagesc(S.^0.1);
colormap('default');

% 2.5.d zeroing 5x5 neighbouthood of 2 peaks
x_1 = 9;
y_1 = 241;
x_2 = 249;
y_2 = 17;

F(y_1-2:y_1+2, x_1-2:x_1+2) = 0; % +-2 cos 5x5 kernel
F(y_2-2:y_2+2, x_2-2:x_2+2) = 0; % +-2 cos 5x5 kernel
S = abs(F).^2; %calculate power spectrum
imagesc(fftshift(S.^0.1));
colormap('default');

% 2.5.e Compute the inverse Fourier transform 
imshow(uint8(ifft2(F)));
%imwrite(uint8(ifft2(F)), 'inverse_fft_zeroed_pck_int.jpg'); % save image

original_img_pck = imread('pck-int.jpg');
F = fft2(original_img_pck); % fast fourier transform, gets kernel coefficients
x_1 = 9;
y_1 = 241;
x_2 = 249;
y_2 = 17;

% zero out line extending from peaks (Single pixel thick line)
F(:,x_1) = 0;
F(y_1,:) = 0;
F(:,x_2) = 0;
F(y_2,:) = 0;

S = abs(F).^2; %calculate power spectrum
imagesc(fftshift(S.^0.1));
colormap('default');
imshow(uint8(ifft2(F)));
%imwrite(uint8(ifft2(F)), 'inverse_fft_zeroed_lines_pck_int_singleLINE.jpg'); % save image

%reread image: Now, increase the thickness of lines
original_img_pck = imread('pck-int.jpg');
F = fft2(original_img_pck); % fast fourier transform, gets kernel coefficients

% zero out line extending from peaks (3 pixels thick line)
[w, h]=size(F);
F(:,x_1-1:x_1+1) = zeros(w,3);
F(y_1-1:y_1+1,:) = zeros(3,h);
F(:,x_2-1:x_2+1) = zeros(w,3);
F(y_2-1:y_2+1,:) = zeros(3,h);

S = abs(F).^2; %calculate power spectrum
imagesc(fftshift(S.^0.1));
colormap('default');
imshow(uint8(ifft2(F)));
%imwrite(uint8(ifft2(F)), 'inverse_fft_zeroed_lines_pck_int_thicker_line.jpg'); % save image

% 2.5.f Free the Primate: Clear the fence
original_primate = imread('primate-caged.jpg');
original_primate_gray = rgb2gray(original_primate);

F_primate = fft2(original_primate_gray); % fast fourier transform, gets kernel coefficients
S = abs(F_primate).^2; %calculate power spectrum
imagesc(fftshift(S.^0.1));
colormap('default');

imagesc(S.^0.1);
colormap('default');

% zeroed coordinates
x1 = 11; y1 = 252;
x2 = 22; y2 = 248;
x3 = 246; y3 = 8;
x4 = 234; y4 = 12;
x5 = 235; y5 = 10;
x6 = 32; y6 = 244;

a = 4; % controls kernel size
F_primate(y1-a : y1+a, x1-a : x1+a) = 0;
F_primate(y2-a : y2+a, x2-a : x2+a) = 0;
F_primate(y3-a : y3+a, x3-a : x3+a) = 0;
F_primate(y4-a : y4+a, x4-a : x4+a) = 0;
F_primate(y5-a : y5+a, x5-a : x5+a) = 0;
F_primate(y6-a : y6+a, x6-a : x6+a) = 0;


S = abs(F_primate).^2; %calculate power spectrum
imagesc(S.^0.1);
colormap('default');

imshow(real(uint8(ifft2(F_primate))));
%imwrite(real(uint8(ifft2(F_primate))), 'inverse_fft_zeroed_Monkey_9x9.jpg'); % save im

%% 2.6 Undoing Perspective Distortion of Planar Surface 
% 2.6.b read image
P = imread('book.jpg');
imshow(P);

% 2.6.b read coords
[X Y] = ginput(4) 
X_A4 = [0; 210; 210; 0];
Y_A4 = [0; 0; 297; 297];

% 2.6.c Setup the matrix
A = [[X(1), Y(1), 1, 0, 0, 0, -X_A4(1)*X(1), -X_A4(1)*Y(1)];
    [0, 0, 0, X(1), Y(1), 1, -Y_A4(1)*X(1), -Y_A4(1)*Y(1)];
    [X(2), Y(2), 1, 0, 0, 0, -X_A4(2)*X(2), -X_A4(2)*Y(2)];
    [0, 0, 0, X(2), Y(2), 1, -Y_A4(2)*X(2), -Y_A4(2)*Y(2)];
    [X(3), Y(3), 1, 0, 0, 0, -X_A4(3)*X(3), -X_A4(3)*Y(3)];
    [0, 0, 0, X(3), Y(3), 1, -Y_A4(3)*X(3), -Y_A4(3)*Y(3)];
    [X(4), Y(4), 1, 0, 0, 0, -X_A4(4)*X(4), -X_A4(4)*Y(4)];
    [0, 0, 0, X(4), Y(4), 1, -Y_A4(4)*X(4), -Y_A4(4)*Y(4)];];

v = [X_A4(1); Y_A4(1); X_A4(2); Y_A4(2); 
    X_A4(3); Y_A4(3); X_A4(4); Y_A4(4)];

u = A \ v; %computes u = A-1v
U = reshape([u;1], 3, 3)' % convert to the normal matrix form

% Verify correctness by transforming the original coordinates
w = U*[X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));

% 2.6.d Warp the image
T = maketform('projective', U');
P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);

% 2.6.e Display the image
imshow(P2);
%imwrite(P2, 'book_warped.jpg'); % save image

% 2.6.f
% Read the image
originalImage = imread('book.jpg');

% Convert from RGB to HSV color space
hsvRightHalf = rgb2hsv(originalImage);

% Define the HSV range for the orange color
% Adjusted HSV range as needed to captures more orange shades
lowerOrange = [0.01, 0.1, 0.1]; 
upperOrange = [0.06, 0.8, 1.0]; 

% mask out the orange
orangeMask = (hsvRightHalf(:,:,1) >= lowerOrange(1) ...
    & hsvRightHalf(:,:,1) <= upperOrange(1)) ...
    & (hsvRightHalf(:,:,2) >= lowerOrange(2) ...
    & hsvRightHalf(:,:,2) <= upperOrange(2)) ...
    & (hsvRightHalf(:,:,3) >= lowerOrange(3) ...
    & hsvRightHalf(:,:,3) <= upperOrange(3));

% Apply the filtered component mask to the binary image
%filteredImage = originalImage;
%filteredImage(repmat(~orangeMask, [1, 1, 3])) = 0; % perform masking
%imshow(filteredImage);
%imwrite(filteredImage, 'indentified_computer_screen_part1.jpg'); % save image

% Filter out small noise regions in the mask using bwareaopen
minOrangeRegionArea = 21;  % Adjust as needed
filteredMask = bwareaopen(orangeMask, minOrangeRegionArea);

% Calculate the area of orange regions in the filtered mask
orangeRegionStats = regionprops(filteredMask, 'Area');

% Initialize the output image with the original image
outputImage = originalImage;

% Create a binary mask to keep only the dense orange regions
denseOrangeMask = false(size(filteredMask));
for i = 1:numel(orangeRegionStats)
    if orangeRegionStats(i).Area >= minOrangeRegionArea
        % Mark the region as dense in the dense orange mask
        denseOrangeMask = denseOrangeMask | (filteredMask == i);
    end
end

%outputImage(repmat(~denseOrangeMask, [1, 1, 3])) = 0; % apply threshold
%imshow(outputImage);
%imwrite(outputImage, 'indentified_computer_screen_part2.jpg'); % save image

% Use dilation to expand the mask across neighboring pixels
se = strel('rectangle', [5, 8]);  % Adjust the disk size as needed
dilatedMask = imdilate(filteredMask, se);

combinedMask = denseOrangeMask | dilatedMask;
% Apply the dense orange mask to the output image
outputImage(repmat(~combinedMask, [1, 1, 3])) = 0;  % Set non-orange pixels to black

%imshow(outputImage);
%imwrite(outputImage, 'indentified_computer_screen_part3.jpg'); % save image

% Label connected components in the binary image
cc = bwconncomp(outputImage);

% Define the minimum required pixel count threshold
minPixelCount = 1000;  % Adjust as needed

% Initialize a mask to keep only connected components meeting the threshold
filteredComponentMask = false(size(outputImage));

% Iterate through connected components
for i = 1:cc.NumObjects
    % Calculate the size (number of pixels) of the connected component
    componentSize = numel(cc.PixelIdxList{i});
    
    % Check if the component size meets the threshold
    if componentSize >= minPixelCount
        % If it meets the threshold, mark it in the mask
        filteredComponentMask(cc.PixelIdxList{i}) = true;
    end
end

% Apply the filtered component mask to the binary image
filteredBinaryImage = outputImage;
filteredBinaryImage(~filteredComponentMask) = 0;

imshow(filteredBinaryImage);
%imwrite(filteredBinaryImage, 'indentified_computer_sscreen.jpg'); % save image
%% CZ4003-Lab 2 Codes: LEE YEW CHUAN MICHAEL

% 3.1 Edge Detection
% 3.1.a Convert to Grayscale & View image 
Pc = imread('macritchie.jpg');
P = rgb2gray(Pc); % convert to grayscale
imshow(P); % View this image
%imwrite(P, 'grayscale-macritchie.jpg');

% 3.1.b Create 3x3 vertical and horizontal Sobel filters & filter img
sobel_ver = [-1 0 1;
           -2 0 2;
           -1 0 1];

sobel_hor = [-1 -2 -1;
            0  0  0;
            1  2  1];

% vertical filtered img
ver_filtered = conv2(P, sobel_ver); % apply vertical sobel
imshow(uint8(ver_filtered));
%imwrite(uint8(ver_filtered), 'macritchie_vertical_sobel.jpg');

% horizontal filtered img
hor_filtered = conv2(P, sobel_hor); % apply horizontal sobel
imshow(uint8(hor_filtered));
%imwrite(uint8(hor_filtered), 'macritchie_horizontal_sobel.jpg');

% 3.1.c Generate a combined edge img
E = ver_filtered.^2 + hor_filtered.^2;
imshow(uint8(E));
%imwrite(uint8(E), 'macritchie_combinedSquared_sobel.jpg');

% 3.1.d Theshold the edge img
thresh_list = {100, 500, 1000, 5000, 10000, 50000, 100000, 500000};

for t = 1:length(thresh_list)
    Et = E>thresh_list{t};
    figure;
    imshow(Et);
    title('thresh'+string(thresh_list{t}));
end

% 3.1.e Canny edge
tl = 0.04;
th = 0.1;
sigma = 1.0;

E = edge(P , 'canny', [tl, th], sigma);
imshow(E)

sigma_list = {1.0, 2.0, 3.0, 4.0, 5.0};
for s = 1:length(sigma_list)
    E = edge(P , 'canny', [tl, th], s);
    figure;
    imshow(E);
    title('Sigma'+string(sigma_list{s}));
end

% varying tl
tl_list = {0.01, 0.02, 0.06, 0.08};
for t = 1:length(tl_list)
    E = edge(P , 'canny', [tl_list{t}, th], sigma);
    figure;
    imshow(E);
    title('tl'+string(tl_list{t}));
end

%% 3.2 Line finding using Hough Transform
% 3.2.a Ruse the edge image computed with canny algorithm
tl = 0.04;
th = 0.1;
sigma = 1.0;
Pc = imread('macritchie.jpg');
P = rgb2gray(Pc); % convert to grayscale
E = edge(P , 'canny', [tl, th], sigma);
imshow(E);

% 3.2.b use Radon transform
[H, xp] = radon(E);
figure;
imagesc(uint8(H));
colormap(gca,hot);

% 3.2.c Find the peak
[radius, theta] = find(H == max(H(:)));

% 3.2.d convert theta radius line presentation to normal line form
radius = xp(radius);
[A, B] = pol2cart(theta*pi/180, radius);
B = -B; 

[numRows, numCols] = size(P);
x_center_c = numCols/2;
y_center_c = numRows/2;

C = A*(A+x_center_c) + B*(B+y_center_c);

% 3.2.e compute yl and yr 
xl = 0;
xr = numCols - 1;

yl = (C - A*xl)/B;
yr = (C - A*xr)/B;

% 3.2.f Display and superimpose your line
imshow(P);
line([xl,xr], [yl,yr]);

%% 3.3 3D Stereo
% 3.3.a Write the disparity map algorithm
% Written in other file called disparity_map.m

% 3.3.b Converting both images to grayscale
Pl = imread("corridorl.jpg");
Pl = rgb2gray(Pl);
figure
imshow(Pl);

Pr = imread("corridorr.jpg");
Pr = rgb2gray(Pr);
figure
imshow(Pr);

% 3.3.c Run your algorithm on the two images
D = disparity_map(double(Pl), double(Pr), 11, 11);
figure
imshow(-D,[-15 15]);

% 3.3.d Re-run on triclops
Tl = imread("triclopsi2l.jpg");
Tl = rgb2gray(Tl);
figure
imshow(Tl);

Tr = imread("triclopsi2r.jpg");
Tr = rgb2gray(Tr);
figure
imshow(Tr);

D_triclops = disparity_map(double(Tl), double(Tr), 11, 11);
figure
imshow(-D_triclops,[-15 15]);
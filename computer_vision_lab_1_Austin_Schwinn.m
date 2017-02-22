close all; clear all; clc;

%% PART 1 - Construct necessary form of images

% a. Read the image
img = imread('chessboard00.png');
img = im2double(img);
% b. Compute the image derivative Ix and Iy.
[Ix,Iy] = imgradientxy(img, 'sobel');
% c. Generate a Gaussian filter of size 9*9 and standard deviation 2.
gauss_filt = fspecial('gaussian',[9 9],2);
% d. Apply the Gaussian filter to smooth the images Ix*Ix, Iy*Iy and Ix*Iy.
Ixx = imfilter(Ix.*Ix,gauss_filt,'same');
Ixy = imfilter(Ix.*Iy,gauss_filt,'same');
Iyy = imfilter(Iy.*Iy,gauss_filt,'same');
% e. Display the results
figure; imshow(mat2gray(Ixx));
figure; imshow(mat2gray(Ixy));
figure; imshow(mat2gray(Iyy));

%% PART 2 - Compute Matrix E which contains for every point the value  
% of the smaller eigenvalue of auto correlation matrix M
% a. Compute E
	% Initialize E. Then, for each pixel, 
    [h,w] = size(Ixx);
    E = zeros(h,w);
	% (1) build matrix M considering a window of size 3*3, 
    window = eye(3);
    Ixx_windowed = imfilter(Ixx, window, 'same');
    Ixy_windowed = imfilter(Ixy, window, 'same');
    Iyy_windowed = imfilter(Iyy, window,'same');
    for a = 1:h
        for b = 1:w
            M = [Ixx_windowed(a,b),Ixy_windowed(a,b);Ixy_windowed(a,b),Iyy_windowed(a,b)];
                % (2) Compute eigenvalues of the matrix,
                e = eig(M);
                % (3) save the smaller eigenvalue in E
                E(a,b) = min(e);
        end
    end
% b. Display results
figure, imshow(mat2gray(E));


%% PART 3 - Compute Matrix R which contains for every point the cornerness score

% a. Compute R
	% Initialize R. Then, for each pixel,
    R = zeros(h,w);
	% (1) build matrix M, 
    for c = 1:h
        for d = 1:w
            M = [Ixx_windowed(c,d),Ixy_windowed(c,d);Ixy_windowed(c,d),Iyy_windowed(c,d)];
            % (2) Compute the trace and the determinant of M,
            Trace = trace(M);
            Det = det(M);
            k = 0.04;
            % (3) save the result of equation 3 in R
            R(c,d) = (Det - (k*Trace^2));
        end
    end
% b. Display results
figure, imshow(mat2gray(R));


%% PART 4 - Select for E and R the 81 most salient points.

% a. Write a function to obtain the 81 most salient points of E and R,
points = 81;
[~,sortR] = sort(R(:),'descend');
[~,sortE] = sort(E(:),'descend');
[Rx,Ry] = ind2sub([h w],sortR);
[Ex,Ey] = ind2sub([h w],sortE);
% and their coordinates features(p_x, p_y)
figure;
imshow(img);
hold on;
xlabel('Max 81 Points');
for i = 1:points
    plot(Rx(i),Ry(i),'r+');
end
hold off;
figure;
imshow(img);
hold on;
xlabel('Max 81 Points');
for i = 1:points
    plot(Ex(i),Ey(i),'r+');
end
hold off;

%% Part 5 - Build a function to carry out non-maximal suppression for E and R

% Select the 81 most salient points using a non-maximal suppression of
% 11x11 pixals 

% a. Apply non maximal syppression with a window of 11*11
NMS_R = ordfilt2(R,11*11,ones(11,11));
NMS_R2 = (R == NMS_R) & (R > (max(R(:) * 0.1)));
NMS_E = ordfilt2(E,11*11,ones(11,11));
NMS_E2 = (E == NMS_E) & (E > (max(E(:) * 0.1)));
% b. Get the 81 most salient points and their coordinates, in the same way
% as part 4
[~,sortR] = sort(NMS_R2(:),'descend');
[~,sortE] = sort(NMS_E2(:),'descend');
[Rx,Ry] = ind2sub([h w],sortR);
[Ex,Ey] = ind2sub([h w],sortE);
% c. Display the selected points on the top of the original image
figure;
imshow(img);
hold on;
xlabel('NMS Max 81 Points');
for i = 1:points
    plot(Rx(i),Ry(i),'r+');
end
hold off;
figure;
imshow(img);
hold on;
xlabel('NMS Max 81 Points');
for i = 1:points
    plot(Ex(i),Ey(i),'r+');
end
hold off;
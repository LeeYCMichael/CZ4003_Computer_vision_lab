function map = test_map(imgl, imgr, dim1, dim2)
    
imgl = double(imgl); % Convert imgs to double
imgr = double(imgr);

[height, width] = size(imgl); 
max_range = 15;
row = floor(dim1/2);
col = floor(dim2/2);

map = ones(size(imgl));

for i = 1+row:height-row
    for j = 1+col:width-col
        I = imgl(i-row:i+row, j-col:j+col);

        min_ssd = inf;
        min_diff = 0;
        lower_bound = max(1+row, j - max_range); 
        upper_bound = min(j+max_range, width - col);
        for k = lower_bound:upper_bound

            T = imgr(i-row:i+row, k-col:k+col);
            
            ssd = (I - T).^2;
            ssd = sum(ssd(:));
            if ssd < min_ssd
                min_ssd = ssd;
                min_diff = k;
            end
        end
        map(i, j) = j - min_diff;
    end
end
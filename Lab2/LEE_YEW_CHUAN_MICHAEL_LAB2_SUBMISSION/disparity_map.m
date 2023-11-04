function map = disparity_map(imgl, imgr, dim1, dim2)
    
[height, width] = size(imgl); 
max_range = 14;
row = floor(dim1/2);
col = floor(dim2/2);

map = ones(size(imgl));

for h = 1+row:height
    min_row = max(1, h - row);
    max_row = min(height, h + row);

    for c = 1+col:width
        min_col = max(1, c - col);
        max_col = min(width, c + col);
        
        I = imgl(min_row:max_row, min_col:max_col);

        min_ssd = inf;
        min_diff = 0;
        lower_bound = max(-max_range, 1 - min_col); 
        upper_bound = min(max_range, width - max_col);
        for k = lower_bound:upper_bound
            T = imgr(min_row:max_row, min_col + k:max_col + k);
            
            ssd = (I - T).^2;
            ssd = sum(ssd(:));
            if ssd < min_ssd
                min_ssd = ssd;
                min_diff = k + c;
            end
        end
        map(h, c) = min_diff - c;
    end
end
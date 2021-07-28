function output=Call_BaSiC(input_img,output_img,is_IF)

addpath(genpath('dctool'));
img = imread(input_img);
img_dim = size(size(img));
if img_dim(2)>=3
    img = rgb2gray(img);
end

if is_IF
    %     [flatfield, darkfield] = BaSiC(img,'darkfield','true');
    %     basefluor = BaSiC_basefluor(img,flatfield,darkfield);
    %     img_corr = (double(img)-darkfield)./flatfield - basefluor;
    
    flatfield = BaSiC(img);
    basefluor =  BaSiC_basefluor(img,flatfield);
    img_corr = double(img)./flatfield - basefluor;
    
    %         [flatfield, darkfield] = BaSiC(img,'darkfield','true');
    %         basefluor = BaSiC_basefluor(img,flatfield,darkfield);
    %         img_corr = (double(img)-darkfield);
else
    flatfield = BaSiC(img);
    img_corr = double(img)./flatfield;
    %         [flatfield, darkfield] = BaSiC(img,'darkfield','true');
    %         basefluor = BaSiC_basefluor(img,flatfield,darkfield);
    %         img_corr = (double(img)-darkfield);
end

if isa(img,'uint8')
    imwrite(uint8(img_corr),output_img);
elseif isa(img,'uint16')
    imwrite(uint16(img_corr),output_img);
end

output=true;
end
function output=Task_Draw(All_well_fractal,main_path,my_title,draw_path)
% Only after image acquisition processing and Fractal!
% Only darw image fractal

%---- input Experiment_Plan.csv file and Fractal.csv file
%main_path = 'G:\CD46\PROCESSING\';
key_Fractal_file = 'AllFractal.mat';
Experiment_Plan_name = 'Experiment_Plan.csv';
Fractal_name = 'Fractal.csv';
Experiment_Plan = importfile([main_path,'\',Experiment_Plan_name]);
well_numbers = size(Experiment_Plan,1); % well number
CHIR_tabulate_summary = tabulate(Experiment_Plan.chir);
CHIR=CHIR_tabulate_summary(find(CHIR_tabulate_summary(:,2)~=0),1)'; % chir conditions

%---- Calculate the mean CHIR_Fractal and do curve fit
CHIR_Fractal =[]; % chir original line
for i=1:length(CHIR)
    chir_index = find(Experiment_Plan.chir==CHIR(i));
    CHIR_Fractal = [CHIR_Fractal;mean(All_well_fractal(chir_index,:))];
end
CHIR_Fractal_nor=CHIR_Fractal./CHIR_Fractal(:,1); % chir original line normalized
CF_CHIR_Fractal = my_curve_fit(CHIR_Fractal);% chir CF line normalized
CF_CHIR_Fractal_nor = my_curve_fit(CHIR_Fractal_nor);% chir CF line normalized

close all;
cd(draw_path);
save(key_Fractal_file,'-v7.3');
Results=my_curve_fit_plot4(my_title,CHIR,CHIR_Fractal,CF_CHIR_Fractal,CHIR_Fractal_nor,CF_CHIR_Fractal_nor,true);
output=1;

end


function ExperimentPlan = importfile(filename, dataLines)
%  IMPORTFILE 从文本文件中导入数据
%  EXPERIMENTPLAN = IMPORTFILE(FILENAME)读取文本文件 FILENAME 中默认选定范围的数据。
%  以表形式返回数据。
%
%  EXPERIMENTPLAN = IMPORTFILE(FILE, DATALINES)按指定行间隔读取文本文件 FILENAME
%  中的数据。对于不连续的行间隔，请将 DATALINES 指定为正整数标量或 N×2 正整数标量数组。
%
%  示例:
%  ExperimentPlan = importfile("C:\Users\Kitty\Documents\Desktop\Fractal_\test\Experiment_Plan.csv", [2, Inf]);
%
% 如果不指定 dataLines，请定义默认范围
if nargin < 2
    dataLines = [2, Inf];
end

opts = delimitedTextImportOptions("NumVariables", 10);

% 指定范围和分隔符
opts.DataLines = dataLines;
opts.Delimiter = ",";
% 指定列名称和类型
opts.VariableNames = ["VarName1", "medium", "IPS_type", "density", "truly", "chir", "chir_hour", "rest_hour", "IF_intensity", "IF_human"];
opts.VariableTypes = ["int32", "int32", "int32", "double", "double", "int32", "double", "double", "double", "double"];
% 指定文件级属性
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
% 指定变量属性
opts = setvaropts(opts, ["VarName1", "medium", "IPS_type", "chir"], "TrimNonNumeric", true);
opts = setvaropts(opts, ["VarName1", "medium", "IPS_type", "chir"], "ThousandsSeparator", ",");
% 导入数据
ExperimentPlan = readtable(filename, opts);

end


function output=my_curve_fit_plot4(my_title,chir,CHIR_SSSS,CFM_cells,CHIR_SSSS_nor,CFM_cells_nor,save_img)
% plot the 4 subplot pictures and save


%---- define the x axle, the color and other parameters
disp(['CHIR is : ', num2str(chir)]);
chir_cell =cell(1,length(chir));
for i=1:length(chir)
    chir_cell(1,i)= {['CHIR-',num2str(chir(i))]};
end
[~,points_num]=size(CHIR_SSSS);
x0=[0:1:points_num-1];
x=[0:0.1:points_num-1];
if ismember(1,chir) || ismember(3,chir) || ismember(5,chir) || ismember(7,chir) || ismember(9,chir) || ismember(11,chir) || ismember(13,chir) || max(chir)>14
    color_key=[[1:size(lines,1)]',lines];
else
    color_key=[0,0,0,0;
        2,0.341,0.341,0.461;
        4,0.311,0.311,0.95;
        6,0.511,0.85,0.95;
        8,0.211,0.90,0.211;
        10,0.950,0.666,0.355;
        12,0.95,0.261,0.261;
        14,0.88,0.161,0.88];
end
my_LineWidth=1.5;
threshold=1e-7;
output=ones(length(chir),3)*(-1);
img_size = [0,0,1920,1080];

%---- define the figure 1
this_figure_handle=figure();
%---- first picture: Original Fractal
subplot(2,2,1);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x0,CHIR_SSSS(i,:)','LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Original: ' ,num2str(chir),' CHIR Fractal']);
plot_all_points(CHIR_SSSS);

%---- second picture: Fitting Fractal
subplot(2,2,2);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x,CFM_cells{i}(x),'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Curve Fitting : ' ,num2str(chir),' CHIR Fractal']);
for i =1:length(chir)
    %     find_ip(x,CFM_cells{i}(x),threshold);
    find_max_gradient(x,CFM_cells{i}(x));
end

%---- third picture: Normalized Original Fractal
subplot(2,2,3);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x0,CHIR_SSSS_nor(i,:)','LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Original: ' ,num2str(chir),' CHIR Fractal']);
plot_all_points(CHIR_SSSS_nor);

%---- fourth picture: Normalized Fitting Fractal
subplot(2,2,4);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x,CFM_cells_nor{i}(x),'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Curve Fitting : ' ,num2str(chir),' CHIR Fractal']);
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    [value,i__]=find_max_gradient(x,y);
    disp(['The CHIR ',num2str(chir(i)),' max gradient is: ',num2str(value,'%e'),' & The y is: ',num2str(y(i__),'%e')]);
    output(i,:)=[x(i__),y(i__),value];
end


%---- save image
if  save_img
    set(this_figure_handle,'position',img_size);
    saveas(this_figure_handle,[my_title,'_FC_slope_.png']);
end


%---- Draw Original phase space image 2
this_figure_handle=figure();
subplot(2,2,1);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x0,CHIR_SSSS(i,:)','LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Original: ' ,num2str(chir),' CHIR Fractal']);
plot_all_points(CHIR_SSSS);
subplot(2,2,3);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    y=CHIR_SSSS(i,:)';
    y__1=diff(y,1);
    plot(x0(1:end-1),y__1,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Original Slope: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,2);hold on; grid on;
for i =1:length(chir)
    y=CHIR_SSSS(i,:)';
    y__1=diff(y,1);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y(1:end-1),y__1,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Original Phase space image: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,4);hold on; grid on;
for i =1:length(chir)
    y=CHIR_SSSS(i,:)';
    y__1=diff(y,1);
    if y__1(1)<0
        temp_n = -y__1(1);
    elseif y__1(1)==0
        temp_n = 1;
    else
        temp_n = y__1(1);
    end
    y__1_nor=y__1./temp_n; %  normalized
    %  y__1_nor=abs(y__1_nor);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y(1:end-1),y__1_nor,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Original Phase space image: ' ,num2str(chir),' CHIR Fractal']);

if  save_img
    set(this_figure_handle,'position',img_size);
    saveas(this_figure_handle,[my_title,'_FC_Phase_space_.png']);
end


%---- Draw Normalized Original phase space image 3
this_figure_handle=figure();
subplot(2,2,1);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x0,CHIR_SSSS_nor(i,:)','LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Original: ' ,num2str(chir),' CHIR Fractal']);
plot_all_points(CHIR_SSSS_nor);
subplot(2,2,3);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    y=CHIR_SSSS_nor(i,:)';
    y__1=diff(y,1);
    plot(x0(1:end-1),y__1,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Original Slope: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,2);hold on; grid on;
for i =1:length(chir)
    y=CHIR_SSSS_nor(i,:)';
    y__1=diff(y,1);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y(1:end-1),y__1,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Original Phase space image: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,4);hold on; grid on;
for i =1:length(chir)
    y=CHIR_SSSS_nor(i,:)';
    y__1=diff(y,1);
    if y__1(1)<0
        temp_n = -y__1(1);
    elseif y__1(1)==0
        temp_n = 1;
    else
        temp_n = y__1(1);
    end
    y__1_nor=y__1./temp_n; %  normalized
    %  y__1_nor=abs(y__1_nor);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y(1:end-1),y__1_nor,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Original Phase space image: ' ,num2str(chir),' CHIR Fractal']);

if  save_img
    set(this_figure_handle,'position',img_size);
    saveas(this_figure_handle,[my_title,'_FC_Normalized_Phase_space_.png']);
end


%---- Draw Curve Fitting phase space image 4
this_figure_handle=figure();
subplot(2,2,1);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x,CFM_cells{i}(x),'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Curve Fitting : ' ,num2str(chir),' CHIR Fractal']);
for i =1:length(chir)
    find_max_gradient(x,CFM_cells{i}(x));
end
subplot(2,2,3);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells{i}(x));
    y__1=diff(y,1);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x(1:end-1),y__1,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Curve Fitting Slope: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,2);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells{i}(x));
    y__1=diff(y,1);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y(1:end-1),y__1,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Phase space image: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,4);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells{i}(x));
    y__1=diff(y,1);
    if y__1(1)<0
        temp_n = -y__1(1);
    elseif y__1(1)==0
        temp_n = 1;
    else
        temp_n = y__1(1);
    end
    y__1_nor=y__1./temp_n; %  normalized
    % y__1_nor=abs(y__1_nor);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y(1:end-1),y__1_nor,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Phase space image: ' ,num2str(chir),' CHIR Fractal']);

if  save_img
    set(this_figure_handle,'position',img_size);
    saveas(this_figure_handle,[my_title,'_FFC_Phase_space_.png']);
end


%---- Draw Normalized Curve Fitting phase space image 5
this_figure_handle=figure();
subplot(2,2,1);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x,CFM_cells_nor{i}(x),'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Curve Fitting : ' ,num2str(chir),' CHIR Fractal']);
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    [value,i__]=find_max_gradient(x,y);
    %disp(['The CHIR ',num2str(chir(i)),' max gradient is: ',num2str(value,'%e'),' & The y is: ',num2str(y(i__)-1,'%e')]);
end
subplot(2,2,3);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    y__1=diff(y,1);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x(1:end-1),y__1,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Curve Fitting Slope: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,2);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    y__1=diff(y,1);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y(1:end-1),y__1,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Phase space image: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,4);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    y__1=diff(y,1);
    if y__1(1)<0
        temp_n = -y__1(1);
    elseif y__1(1)==0
        temp_n = 1;
    else
        temp_n = y__1(1);
    end
    y__1_nor=y__1./temp_n; %  normalized
    %  y__1_nor=abs(y__1_nor);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y(1:end-1),y__1_nor,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Phase space image: ' ,num2str(chir),' CHIR Fractal']);

if  save_img
    set(this_figure_handle,'position',img_size);
    saveas(this_figure_handle,[my_title,'_FFC_Normalized_Phase_space_.png']);
end


%---- Draw Normalized Curve Fitting  extend phase space image 6
this_figure_handle=figure();
subplot(2,2,1);hold on; grid on;
for i =1:length(chir)
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x,CFM_cells_nor{i}(x),'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Curve Fitting : ' ,num2str(chir),' CHIR Fractal']);
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    [value,i__]=find_max_gradient(x,y);
    %disp(['The CHIR ',num2str(chir(i)),' max gradient is: ',num2str(value,'%e'),' & The y is: ',num2str(y(i__)-1,'%e')]);
end
subplot(2,2,3);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    y__2=diff(y,2);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(x(1:end-2),y__2,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Curve Fitting Slope: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,2);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    y__1=diff(y,1);
    y__2=diff(y,2);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y__1(1:end-1),y__2,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Phase space image: ' ,num2str(chir),' CHIR Fractal']);
subplot(2,2,4);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    y__1=diff(y,1);
    if y__1(1)<0
        temp_n = -y__1(1);
    elseif y__1(1)==0
        temp_n = 1;
    else
        temp_n = y__1(1);
    end
    y__1_nor=y__1./temp_n; %  normalized
    % y__1_nor=abs(y__1_nor);
    y__2=diff(y,2);
    if y__2(1)<0
        temp_n = -y__2(1);
    elseif y__2(1)==0
        temp_n = 1;
    else
        temp_n = y__2(1);
    end
    y__2_nor=y__2./temp_n; %  normalized
    % y__2_nor=abs(y__2_nor);
    color_key_index = find(color_key(:,1)==chir(i));
    plot(y__1_nor(1:end-1),y__2_nor,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Phase space image: ' ,num2str(chir),' CHIR Fractal']);

if  save_img
    set(this_figure_handle,'position',img_size);
    saveas(this_figure_handle,[my_title,'_FFC_12_Phase_space_.png']);
end


%---- Draw Normalized Curve Fitting  extend phase space 3D image 7
this_figure_handle=figure();
subplot(1,2,1);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    y__1=diff(y,1);
    if y__1(1)<0
        temp_n = -y__1(1);
    elseif y__1(1)==0
        temp_n = 1;
    else
        temp_n = y__1(1);
    end
    y__1_nor=y__1./temp_n; %  normalized
    % y__1_nor=abs(y__1_nor);
    y__2=diff(y,2);
    if y__2(1)<0
        temp_n = -y__2(1);
    elseif y__2(1)==0
        temp_n = 1;
    else
        temp_n = y__2(1);
    end
    y__2_nor=y__2./temp_n; %  normalized
    % y__2_nor=abs(y__2_nor);
    color_key_index = find(color_key(:,1)==chir(i));
    plot3(x(1:end-1),y(1:end-1),y__1_nor,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Phase space image: ' ,num2str(chir),' CHIR Fractal']);
subplot(1,2,2);hold on; grid on;
for i =1:length(chir)
    y=(CFM_cells_nor{i}(x));
    y__1=diff(y,1);
    if y__1(1)<0
        temp_n = -y__1(1);
    elseif y__1(1)==0
        temp_n = 1;
    else
        temp_n = y__1(1);
    end
    y__1_nor=y__1./temp_n; %  normalized
    %  y__1_nor=abs(y__1_nor);
    y__2=diff(y,2);
    if y__2(1)<0
        temp_n = -y__2(1);
    elseif y__2(1)==0
        temp_n = 1;
    else
        temp_n = y__2(1);
    end
    y__2_nor=y__2./temp_n; %  normalized
    %  y__2_nor=abs(y__2_nor);
    color_key_index = find(color_key(:,1)==chir(i));
    plot3(y(1:end-2),y__1_nor(1:end-1),y__2_nor,'LineWidth',my_LineWidth,'color',color_key(color_key_index,2:end));
end
legend(chir_cell);
title([my_title,' Normalized Phase space image: ' ,num2str(chir),' CHIR Fractal']);

if  save_img
    set(this_figure_handle,'position',img_size);
    saveas(this_figure_handle,[my_title,'_FFC_3D_Phase_space_.png']);
end


end

function [value,i__]=find_max_gradient(x,y)
% find max gradient of a curve and label Y value (and the max gradient)

y__2=diff(y,1);
[value,i__]=max(y__2);
hold on;
plot(x(i__),y(i__),'.','MarkerSize',9,'color',[0.9,0.15,0.9]);
% text(x(i__),y(i__),[num2str(value,'%e'),',',num2str(y(i__)-1,'%e')],'color','k','FontSize',9)
text(x(i__),y(i__),['y:',num2str(y(i__)-1,'%e')],'color','k','FontSize',9)

end


function plot_all_points(CHIR_SSSS)
%  plot on figure dot ,must 'hold on;'
%  '.','MarkerSize',9,'color',[0,0,1]

[chir_num,points_num]=size(CHIR_SSSS);
x=[0:1:points_num-1];
for i =1:chir_num
    plot(x,CHIR_SSSS(i,:)','.','MarkerSize',9,'color',[0,0,1]);
end

end


function output = my_curve_fit(input)
%  Create a cell{} of fit object

name='my_title';
[chir_num,x_num]=size(input);
x=[0:x_num-1];
output=cell(chir_num,1);
for i=1:chir_num
    [this_CFM,GN]=createFit(x, input(i,:), name,false);
    output(i)={this_CFM};
end

end


function [fitresult, gof] = createFit(x, y,my_title,show)
%  CREATEFIT(X,Y)
%  Create a object of fit.
%
%  Data for my_title fit:
%      X Input : x
%      Y Output: y
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%   Fit: my_title.
[xData, yData] = prepareCurveData( x, y );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.SmoothingParam = 0.996655955330749;

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

if show
    % Plot fit with data.
    figure( 'Name', my_title);
    h = plot( fitresult, xData, yData );
    legend( h, 'y vs. x', my_title, 'Location', 'NorthEast', 'Interpreter', 'none' );
    % Label axes
    xlabel( 'x', 'Interpreter', 'none' );
    ylabel( 'y', 'Interpreter', 'none' );
    title(my_title);
    grid on
end

end

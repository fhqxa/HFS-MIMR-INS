%% Main Function
clear;clc;close all;
str1={'F194'};
%str1={'DD'};
%str1={'ilsvrc65'};
%str1={'Cifar100'};
%str1={'SUN' };
%str1={'page' };
%str1={'VOCResnet50'};
m = length(str1);
alpha=0.1; %类间距
beta=1;  %正交
lambda=10;
maxIte =10;
flag=1;
for i = 1:m
    filename = [str1{i} 'Train'];
    load (filename);
    [X,Y]=creatSubTablezh(data_array, tree); 
    %clear data_array;
    %tic;
    %% others
    %[feature] = MIMR(X, Y, tree, lambda, alpha,  beta, maxIte,flag);%正交+类间距
    %[feature] = HiFSRR02(X, Y, tree, 10, 1, 1, 0);
    %% TKDE
    tic;
    cor=corr(data_array(:,1:end-1)',data_array(:,1:end-1)','type','pearson');
    [X,Y,~,cor]=create_SubTable(data_array, tree,cor);
    %[Yd] = create_hier_distribution(Y,tree,cor,0.05,0.05);
    [Yd] = create_hier_distribution(Y,tree,cor,0.2,0.15);
    [treecor] = get_treecor(tree); %结点之间的相似度
    [sibcor] = get_sibcor(cor,Y,tree); %兄弟之间的相似度
    %Feature selection
    [feature,W] = HFSLDL(X, Yd, tree, 10, treecor,sibcor,0.2,0.1,0);
   toc;               
    %% Test feature
   testFile = [str1{i}, 'Test.mat'];
   load (testFile);
   [accuracyMean, accuracyStd, F_LCAMean, FHMean, TIEmean, TestTime] = HierSVMPredictionBatchFeature(data_array, tree, feature)
end
fprintf('---- completed ----\n');

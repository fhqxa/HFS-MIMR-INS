%% 10-fold Hierarchical SVM
%% Written by Hong Zhao
% 2017-4-11
% Modified by Hong Zhao on May 16th, 2017.
%% Last modified by Hong Zhao with hierarchical SVM on 2018-2-8.
%% Input
% data: the dataset with feature and label;
% numFolds: 10-fold or 5-fold;
% tree: the hierarchical structure of classes;
% flag: the classficiation with different features is used when flag=1;
% feature: the feature subset for each node;
% numberSel: the number of selected feature;
% indices: it depends on numFolds.
%% Output
function [accuracyMean,accuracyStd,F_LCAMean,FHMean,TIEmean] = FS_Kflod_TopDownSVMClassifier(data, numFolds, tree, feature, numberSel,indices)
% if(length(varargin) == 4)
%     data = varargin{1};
%     numFolds = varargin{2};
%     tree = varargin{3};
%     flag = varargin{4};
%  %   indices = varargin{5};
% else
%     if(length(varargin)==5)
%         data = [varargin{1},varargin{2}];
%         numFolds = varargin{3};
%         tree = varargin{4};
%         flag = varargin{5};
%         %   indices = varargin{5};
%     end
% end
[M,N]=size(data);
accuracy_k = zeros(1,numFolds);
% indices = crossvalind('Kfold',data(1:M,N),numFolds);%//进行随机分包 for k=1:10//交叉验证k=10，10个包轮流作为测试集
%     save indices10001 indices;
%        load indices1000;

for k = 1:numFolds  
    testID = (indices == k);%//获得test集元素在数据集中对应的单元编号
    trainID = ~testID;%//train集元素的编号为非test元素的编号
    test_data = data(testID,1:N-1);
    test_label = data(testID,N);
    train_data = data(trainID,:);
    %没有train_target吗？
    
    %% Creat sub table
    [trainDataMod, trainLabelMod] = creatSubTablezh(train_data, tree);
    %将train_data拆分成data  和 label（每个类下面的训练样本）
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Train classifiers of all internal nodes
    numNodes = length(tree(:,1));%ZH: The total of all nodes.
    for i = 1:numNodes
        if (~ismember(i, tree_LeafNode(tree)))%  遍历结点，当此时的结点不是叶子结点的时候
            %提取出相应的分类器的训练数据和label
            trainLabel = trainLabelMod{i};
            trainData = trainDataMod{i};
%           trainLabel = sparse(trainLabelMod{i});
%           trainData = sparse(trainDataMod{i});
            selFeature = feature{i}(1:numberSel);
            [modelSVM{i}]  = svmtrain(trainLabel, trainData(:,selFeature), '-c 1 -t 0 -q'); 
            %是训练得到的模型，是一个结构体（如果参数中用到-v，得到的就不是结构体，对于分类问题，得到的是交叉检验下的平均分类准确率；对于回归问题，得到的是均方误差）。
            %https://blog.csdn.net/qq_22475211/article/details/100537586
         %[model{i}]  = train(double(trainLabel), sparse(trainData(:,selFeature(1:numberSel))), '-c 2 -s 0 -B 1 -q');
        end
    end
    
    %%           Prediction       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [predict_label] = FS_topDownSVMPrediction(test_data, modelSVM, tree,feature,numberSel) ;

    %%          Envaluation       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [PH(k), RH(k), FH(k)] = EvaHier_HierarchicalPrecisionAndRecall(test_label,predict_label',tree);
   [P_LCA(k),R_LCA(k),F_LCA(k)] = EvaHier_HierarchicalLCAPrecisionAndRecall(test_label,predict_label',tree);
   TIE(k) = EvaHier_TreeInducedErrorNormalize(test_label,predict_label',tree);
   accuracy_k(k) = EvaHier_HierarchicalAccuracy(test_label,predict_label', tree);%王煜
    
end
accuracyMean = mean(accuracy_k);
accuracyStd = std(accuracy_k);
F_LCAMean=mean(F_LCA);
FHMean=mean(FH);
TIEmean=mean(TIE);
end
%% Top-down prediction
% Written by Yu Wang 
% Modified by Hong Zhao
% 2017-4-11
%% Inputs:
% input_data: training data without labels
% model: 
% tree: the tree hierarchy
%% Output
%                          FS_topDownSVMPrediction(test_data, modelSVM, tree,feature,numberSel) ;
function [predict_label] = FS_topDownSVMPrediction(input_data, model, tree, feature,numberSel)
    [m,~]=size(input_data);
    root = find(tree(:,1)==0);     
	for j=1:m %The number of samples
%% 先从树根开始
        selFeature = feature{root}(1:numberSel);
         [currentNode] = svmpredict(input_data(j,end),input_data(j,selFeature), model{root},'-q');%预测得到的标签
                       % svmpredict(test_label,       test_matrix,              model,      ['libsvm_options'])
%         [~,~,d_v] = predict(1,sparse(input_data(j,selFeature)), model{root}, '-b 1 -q');%第一个参数没有作用。
%         currentNode=model{root}.Label(currentNodeID) ;
    %% 递归调用中间层直到叶子结点
       while(~ismember(currentNode,tree_LeafNode(tree)))%预测得到的标签不是叶子结点
          selFeature = feature{currentNode}(1:numberSel);
          [currentNode] = svmpredict(input_data(j,end),input_data(j,selFeature), model{currentNode},'-q');
%           [~,~,d_v] = predict(1,  sparse(input_data(j,selFeature)),  model{currentNode}, '-b 1 -q');
%           [~,currentNodeID] = max(d_v);
%           currentNode=model{currentNode}.Label(currentNodeID);
        end
        predict_label(j)=currentNode;        
    end %%endfor    
end

%% creatSubTable
% Written by Yu Wang
% Modified by Hong Zhao
% 2017-4-11
%% Creat subtable创建子表
function [DataMod,LabelMod]=creatSubTablezh(dataset, tree)

Data = dataset(:,1:end-1);  %提取了dataset的数据部分
Label =  dataset(:,end);    %提取了dataset的标签部分
[numTrain,~] = size(dataset);  %数据样本数目（行）
internalNodes = tree_InternalNodes(tree); %内部结点
indexRoot = tree_Root(tree);   % The root of the tree
noLeafNode =[internalNodes;indexRoot];  %非叶子：内部+根

for i = 1:length(noLeafNode)   %遍历每一个非叶子结点
    
    %%%%获得训练集中是当前非叶子结点的后代的数据和标签-------同一大类
    cur_descendants = tree_Descendant(tree, noLeafNode(i)); %获得当前非叶子结点的后代结点 
    ind_d = 1; % index for id subscript increment
    id = [];   % data whose labels belong to the descendants of the current nodes 此时后代标签的在Label索引
    for n = 1:numTrain   %遍历每一个训练数据集
        if (ismember(Label(n), cur_descendants) ~= 0)  %如果此时训练样本的标签 属于 这个非叶子结点后代的标签
            %如果此时训练样本 是 这个非叶子结点的后代
            id(ind_d) =  n;      %那么把这个样本的索引放在id[]中
            ind_d = ind_d +1;    %下标后移一个
        end
    end
    
    %把训练集中根据不同非叶子结点的后代进行分类 %Label_Uni_Sel：在训练集中属于这个结点后代的标签
    Label_Uni_Sel = Label(id,:); 
    DataSel = Data(id,:);     %select relative training data for the current classifier
    numTrainSel = size(Label_Uni_Sel,1);  
    
    %%%把训练集中属于当前非叶子结点的后代分配给类标签
    %训练集中的非叶子结点后代中的叶子结点标签分配上一类标签，返回上一层分类
    LabelUniSelMod = label_modify_MLNP(Label_Uni_Sel, noLeafNode(i), tree);
    
    % Get the sub-training set containing only relative nodes
    ind_tdm = 1;
    index = [];     % data whose labels belong to the children of the current nodes
    children_set = get_children_set(tree, noLeafNode(i));
    for ns = 1:numTrainSel  %遍历
        if (ismember(LabelUniSelMod(ns), children_set) ~= 0)
            %如果每个训练样本所属于的类 是 当前非叶子结点的分类
        
            index(ind_tdm) =  ns;  
            %将这个样本的序号存入到index中
            ind_tdm = ind_tdm +1;
        end
    end
    % Find the sub-training set of relative to-be-classified nodes
    DataMod{noLeafNode(i)} = DataSel(index, :);  %每个类样本数*特征数
    LabelMod{noLeafNode(i)} = LabelUniSelMod(index, :);
end
end
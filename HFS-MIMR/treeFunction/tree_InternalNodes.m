%% Return the index of the middle node.
%% Author: Hong Zhao
%% Date: 2016-5-14
%% Example: 
% load tree;
% return internal nodes and root node
% middleNode = tree_InternalNode( tree )

function [ middleNode ] = tree_InternalNodes( tree )
treeParent=tree(:,1)';  %tree有两列，返回第一列（父节点）
index=find(treeParent==0); %根结点的索引
lengthTree = length(treeParent); %length（）数组长度（即行数或列数中的较大值）
middleNode = [];

%根据同一个父节点的标签一样，删除treeParent中相同的元素，把相同元素存入到middleNode找到中间结点
while length(treeParent)~=0   
    middleNode=[middleNode; treeParent(1)]; 
    label=find(treeParent==treeParent(1));  
    treeParent(label)=[];   %每次都会删除相同元素
end
middleNode = middleNode(1:end-1); %去除最后一个结点0

middleNode(find(middleNode==index))=[];%去除根结点
end


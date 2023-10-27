function [ output_args ] = conversionY01_extend( Y, numCol )
    gnd = Y;  %Y{noLeafNode(5)}
    ClassLabel = unique(Y);   %Count the number of labels
    nClass = length(ClassLabel); %类别个数
    [nSmp,~] = size(Y); %
    Y = eye(nClass,nClass);%单位矩阵
    Z = zeros(nSmp,numCol);
    for i=1:nClass  %遍历每个类别
        idx = find(gnd==ClassLabel(i)); %在每个类下的样本中 分别 找到对应类别，索引存到idx
        Z(idx,1:nClass) = repmat(Y(i,1:nClass),length(idx),1);  %Z中每个样本所对应1的位置就是所属于的类
        %将Y(i,1:nClass)在每一列上重复length(idx)次
    end    
    output_args= Z;%Z中每个样本所对应1的位置就是所属于的类
end


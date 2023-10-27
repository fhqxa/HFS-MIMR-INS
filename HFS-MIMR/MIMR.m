%% 最大化类间距离和最小化类内冗余
%% 在生物数据集上10 0.1 0.1
%% 在图像数据集上10 1 1
%% 发表在2023年的INS上
%% (1) T = 20 on VOC and F194;
%% (2) T = 30 on DD; and (3) T = 10 on the remaining datasets.
function [feature_slct] = MIMR(X, Y, tree, lambda, alpha,  beta, maxIte,flag) 
%rand('seed', 1);
internalNodes = tree_InternalNodes(tree);
    indexRoot = tree_Root(tree);% The root of the tree
    noLeafNode =[internalNodes;indexRoot];    
    numSelected = size(X{indexRoot},2);
    for i = 1:length(noLeafNode)
        ClassLabel = unique(Y{noLeafNode(i)});
        m(noLeafNode(i)) = length(ClassLabel);
    end
    maxm = max(m);
    %% initialize
    for j = 1:length(noLeafNode)
        [~,d] = size(X{noLeafNode(j)}); % get the number of features
        Y{noLeafNode(j)} = conversionY01_extend(Y{noLeafNode(j)},maxm);% extend 2 to [1 0]
        W{noLeafNode(j)} = ones(d, maxm);
    end
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:maxIte
        %% initialization on each non-LeafNode
        for j = 1:length(noLeafNode)
            D{noLeafNode(j)} = diag(0.5./max(sqrt(sum(W{noLeafNode(j)}.*W{noLeafNode(j)},2)),eps));
            XX{noLeafNode(j)} = X{noLeafNode(j)}' * X{noLeafNode(j)};
            XY{noLeafNode(j)} = X{noLeafNode(j)}' * Y{noLeafNode(j)};
        end
        %% Update the root node
        W{indexRoot} = (XX{indexRoot} + lambda * D{indexRoot} + beta *ones(d) -beta*eye(d))\XY{indexRoot};
        %% Update the internal nodes
        for j = 1:length(internalNodes)
            U1 = zeros(d,d);
            U2 = zeros(d,d);
            leafNode = tree_LeafNode(tree);
            siblingNodes = [];
            siblingNodes = tree_Sibling(tree,internalNodes(j));  % 兄弟结点
            siblingNodes = setdiff(siblingNodes,leafNode); %delete the leaf node.
            for jj = 1:length(siblingNodes)
                U1= W{siblingNodes(jj)}*W{siblingNodes(jj)}'; 
                U2= W{siblingNodes(jj)}*eye(maxm);
            end
             W{internalNodes(j)} =(XX{internalNodes(j)} + lambda * D{internalNodes(j)}+ beta *(ones(d) -eye(d))+ alpha * U1)\ (XY{internalNodes(j)} + alpha * U2);        
        end   
       if (flag ==1)
        obj(i)=1/2*(norm(X{indexRoot}*W{indexRoot}-Y{indexRoot}))^2+lambda*L21(W{indexRoot})+beta*trace(ones(d)*W{indexRoot}*W{indexRoot}'-W{indexRoot}*W{indexRoot}');
        for j = 1:length(internalNodes)
            currentSibling = tree_Sibling(tree,internalNodes(j));
            currentSibling=setdiff(currentSibling,leafNode);
            for  jj = 1:length(currentSibling)
                S = sum(norm(W{internalNodes(j)}'*W{currentSibling(jj)}-eye(maxm))^2); 
            end
         obj(i)=obj(i)+1/2* (norm(X{internalNodes(j)}*W{internalNodes(j)}-Y{internalNodes(j)}))^2 +lambda/2*L21(W{internalNodes(j)})+beta*trace(ones(d)*W{internalNodes(j)}*W{internalNodes(j)}'-W{internalNodes(j)}*W{internalNodes(j)}')+alpha*S;
        end
      end 
        

   end        
 
    %% Objective vlue
    for i = 1:length(noLeafNode)
        W1=W{noLeafNode(i)};
        W{noLeafNode(i)} = W1(:,1:m(noLeafNode(i)));
    end

    clear W1;
    for j = 1: length(noLeafNode)
        tempVector = sum(W{noLeafNode(j)}.^2, 2);
        [~, value] = sort(tempVector, 'descend'); % sort tempVecror (W) in a descend order
        clear tempVector;
        feature_slct{noLeafNode(j)} = value(1:numSelected);
    end
    if (flag == 1)
    fontsize = 20;
    figure1 = figure('Color',[1 1 1]);
    axes1 = axes('Parent',figure1,'FontSize',fontsize,'FontName','Times New Roman');

    plot(obj,'LineWidth',4,'Color',[0 0 1]);
    xlim(axes1,[0.8 10]);
%     ylim(axes1,[16000,36000]);%Cifar
% set(gca,'yscale','log') 
    set(gca,'FontName','Times New Roman','FontSize',fontsize);
    xlabel('Iteration number');
    ylabel('Objective function value');
    end
end
%% Get the positive label set of the current node, data with these labels
%% can be considered as positive examples for classifier training
%% 

%分三种情况（叶子、根、内部）来找到孩子结点的叶子结点
function [pos_label_set] = get_pos_label_MLNP(tree, cur_node)
cur_ancestor = tree_Ancestor(tree, cur_node); 
cur_descendants = tree_Descendant(tree, cur_node);
leaf_nodes_set = tree_LeafNode(tree);

% If the node is root node--  根第一个孩子的后代中的叶子结点
if (isempty(cur_ancestor))  %代表着cur_ancestor为空，即cur_node是根节点   
    ind = find(tree(:,1) == cur_node);  % ind：cur_node(根)的孩子结点的索引
    root_children_node = ind(1);  % root_children_node：根的第一个孩子
    all_pos_nodes = tree_Descendant(tree, root_children_node); %all_pos_nodes：根节点第一个孩子的后代
    index_root_ret = 1;  % 这个是下标索引
    for i = 1:length(all_pos_nodes) % 遍历根第一个孩子的所有后代
        %isLeaf返回第一个孩子后代中属于叶子结点的索引
        isLeaf = find(all_pos_nodes(i) == leaf_nodes_set); 
        if (~isempty(isLeaf))   % 是叶子结点
            %把这个后代放到pos_label_set中
            pos_label_set(index_root_ret) = all_pos_nodes(i);  
            index_root_ret = index_root_ret + 1; 
        end
    end
    
% the current node is a leaf node   
elseif (isempty(cur_descendants))  
    pos_label_set = cur_node;
    
% the current node is a internal node    ---返回这个内部结点的叶子结点后代
else                               
    all_pos_nodes = tree_Descendant(tree, cur_node); 
    index_internal_ret = 1;  %下标索引
    for i = 1:length(all_pos_nodes)  %遍历所有的后代
        %返回这个内部结点后代结点是叶子结点的索引
        isLeaf = find(all_pos_nodes(i) == leaf_nodes_set);  
        if (~isempty(isLeaf))  
            pos_label_set(index_internal_ret) = all_pos_nodes(i);
            index_internal_ret = index_internal_ret + 1;
        end
    end
end
end
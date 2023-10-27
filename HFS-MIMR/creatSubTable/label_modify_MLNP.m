%% Find the leaf nodes of the current node (if has), and assign the labels of these leaf nodes 
%% to the label of the current class   将这些叶节点的标签分配给当前类的标签
%     孩子的叶子结点 变成 孩子结点--- 叶子结点的标签分配给当前类

function[label_mod] = label_modify_MLNP(labelSet, cur_node, tree)
label_mod = labelSet;  
% Find the corresponding leaf nodes of every child node
children_set = get_children_set(tree, cur_node);   % Direct children nodes of the current node
%遍历每一个孩子结点
for c =1:length(children_set)  
    %返回是后代中的叶子结点
    pos_label_set = get_pos_label_MLNP(tree, children_set(c));  % Relative leaf nodes of the children nodes
    for tl = 1:length(label_mod) %遍历后代的标签
        %如果label_mod(tl)是后代中的叶子结点，
        if (ismember(label_mod(tl), pos_label_set) ~= 0)  
            label_mod(tl) = children_set(c);  
        end
    end
end
end

%原来是labelSet，label_mod = labelSet，在tree中找到cur_node的孩子结点children_set。
%遍历所有的孩子结点通过get_pos_label_MLNP(）得到每个孩子结点的的叶子结点pos_label_set
%再继续遍历label_mod，如果满足label_mod（t1）属于pos_label_set的条件
%将此时的children_set赋值给label_mod，也就是说用孩子结点代替孩子结点的的叶子结点
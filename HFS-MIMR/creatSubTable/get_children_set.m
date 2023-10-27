%% Get the direct children nodes of the current node

function children_set = get_children_set(tree, cur_node)
cur_descendants = tree_Descendant(tree, cur_node); 

if (isempty(cur_descendants))
    children_set = [];  %叶子结点
else      %非叶子结点
    children_set = find(tree(:,1) == cur_node);  
end
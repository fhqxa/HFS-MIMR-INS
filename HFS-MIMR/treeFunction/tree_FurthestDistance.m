
% lenDistance is the number of edges between two furthest distant nodes in the tree.
% root is node 1, and root points to 0
function [ lenDistance ] = tree_FurthestDistance(tree)
a = max(tree(:,2));
temp = find(tree(:,2)==a);
tree(temp(1),:) = [];
b = max(tree(:,2));
lenDistance = a + b;   
end


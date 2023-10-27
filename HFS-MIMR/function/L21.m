%% Compute the L_{2,1}-norm of matrix X
%
% 
function a = L21(X)
%
% Input:
% X: matrix (d x n) (x1', ..., xn')'
%
% Output:
% a = ||x1||_{2} + ... + ||xn||_{2}

a = sum(sqrt(sum(X.*X,2)));
%a=sum(A)  %列求和
%b=sum(A,2) %行求和
%c=sum(A(:)) %矩阵求和

end
    
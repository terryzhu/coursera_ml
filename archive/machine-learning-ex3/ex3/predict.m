function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% 这道题目我是40%靠打印维度+50%参考网上答案做的, 具体而言, 需要注意
% a = theta * X
% 其中X是 当前level激活单元数量为行数,样本数量为列数的矩阵
% 而theta则是下一个level激活单元数量为行数,当前激活数量为列数的矩阵

A1 = [ones(1, m); X' ];

X2 = sigmoid(Theta1 * A1);

A2 = [ones(1, m); X2 ];

A3 = sigmoid(Theta2 * A2);

[x, p] = max(A3' , [], 2);     % max函数用于获取每一行的最大值。

% =========================================================================


end

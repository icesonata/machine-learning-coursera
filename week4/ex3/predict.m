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
a1 = [ones(size(X)(1), 1) X];	% X is m by n
a2 = sigmoid(Theta1*a1')';	% Theta1 is 25 by (n+1) while a1 is m by (n+1)
a2 = [ones(size(a2)(1), 1)  a2];	% a2 updated to m by 26 while Theta2 is 10 by (25+1)
a3 = sigmoid(Theta2*a2')';	% a3 is m by 10
[M, p] = max(a3,[],2);		% get max result and indices based on 2-d space (maximum of each row);
% p now is m by 1 with values from 1 to 10






% =========================================================================


end

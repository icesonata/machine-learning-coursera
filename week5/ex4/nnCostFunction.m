function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a1 = [ones(m,1) X];
a2 = sigmoid(Theta1*a1');
a2 = [ones(1,size(a2,2)); a2];
a3 = sigmoid(Theta2*a2);
% transform labels vector into 0's and 1's vectors.
% y_prime = zeros(size(a3,1),size(y,1));
% for i=1:size(y_prime,2)
%     y_prime(y(i),i) = 1;
% endfor
y_prime = (1:num_labels) == y;
y_prime = y_prime';
%
J = (1/m) * sum(sum(-y_prime.*log(a3)-(1-y_prime).*log(1-a3)));

% implement regularization
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% size(Theta1_grad)
% size(Theta2_grad)
% Backpropagation
for t=1:m
    % step 1
    a1 = X(t, :)';
    a1 = [1; a1];
    z2 = Theta1*a1;     % a1 == z1
    a2 = sigmoid(z2);
    a2 = [ones(1,size(a2,2)); a2];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    % step 2
    % compute delta_3
    delta3 = a3 - y_prime(:, t);
    % step 3
    % ignore delta2_0
    delta2 = Theta2'*delta3.*[1; sigmoidGradient(z2)];
    % step 4
    % removing 0th element of delta 2
    delta2 = delta2(2:end);
    % assigning to corresponding Theta_grad
    Theta1_grad = Theta1_grad + delta2 * a1';
    Theta2_grad = Theta2_grad + delta3 * a2';
end
% step 5
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% regularized grad
Theta1_grad = Theta1_grad + (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

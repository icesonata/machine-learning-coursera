function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

steps = [0.01 0.03 0.1 0.3 1 3 10 30];
predict_error = zeros(length(steps));
for i=1:length(steps)
    C = steps(i);   % choosing C
    for j=1:length(steps)
        sigma = steps(j);   % choosing sigma
        % create model using applied C and sigma
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        % predict on the cross validation set 
        predictions = svmPredict(model, Xval);
        % compute the prediction error
        predict_error(i, j) = mean(double(predictions ~= yval));
    end
end
% get min values' indices for each row (columns index) 
[min_err_row, idx_js] = min(predict_error, [], 2);
% get minimum value's index (row index)
[min_err_col, idx_i] = min(min_err_row);
% assigning efficient C and sigma values
C = steps(idx_i);
sigma = steps(idx_js(idx_i));

% =========================================================================

end

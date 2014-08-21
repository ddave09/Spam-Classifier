function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
common_pool = [0.01 0.03 0.1 0.3 1 3 10 30];
scp = size(common_pool,2);
x1 = [0 0.2 -0.4 0.4]; x2 = [0.2 0.4 -0.6 0.6];
error = 0;
count = 0;
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

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model,Xval);
error = mean(double(predictions ~= yval));

for i= 1:scp,
    for j=1:scp,
        count = count+1;
        model= svmTrain(X, y, common_pool(i), @(x1, x2) gaussianKernel(x1, x2, common_pool(j)));
        predictions = svmPredict(model,Xval);
        error_cur = mean(double(predictions ~= yval));
        if(error > error_cur)
            error = error_cur;
            C = common_pool(i);
            sigma = common_pool(j);
        end
    end
end





% =========================================================================

end

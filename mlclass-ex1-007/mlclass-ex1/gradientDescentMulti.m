function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
theta_size = length(theta); % number of parameters in vector theta
J_history = zeros(num_iters, 1);

const_val = alpha * 1/m;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %tmp = 0;
    %delta = zeros(theta_size,1);
    %for i = 1:m
    %  tmp = X(i, :) * theta - y(i);
    %  for j = 1:theta_size
    %    delta(j) = delta(j) + tmp * X(i,j);
    %endfor
    %for i = 1:theta_size
    %  theta(i) = theta(i) - const_val * delta(i);
    %endfor
    theta = theta - const_val * X'* (X*theta-y);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

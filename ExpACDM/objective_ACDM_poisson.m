function [f] = objective_ACDM_poisson(phi, X, j, qj, K)
% 
% object of minimization per j, as a function of beta
% 
% @param phi: N x 2^K probability matrix computed in the E-step
% @param X: observed responses
% @param j : integer between 1 and J
% @param qj: j-th row of Q
% @param K: number of attributes

f = @obj;
    function [value] = obj(x)
        beta = x;
        index = [1, 1+find(qj)];
        beta_long = zeros(K+1, 1);
        beta_long(index) = beta;
        
        tmp = 0;
        for a = 0:(2^K-1)
            eta = log([1, binary(a, K)] * beta_long);
            tmp = tmp + eta' * ftn_T(X(:,j)') * phi(:, a+1) - ftn_A(eta')*sum(phi(:,a+1));
        end
        value = - tmp;      % minus sign to convert to minimization
    end
end
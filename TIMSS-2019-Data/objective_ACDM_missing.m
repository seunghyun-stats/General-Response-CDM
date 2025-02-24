function [f] = objective_ACDM_missing(phi, X, j, qj, K, index_i)
% object of minimization per j
% f: R^{S_j} x R -> R
% qj: j-th row of Q

f = @obj;
    function [value] = obj(x) % obj(beta, gamma) % object of minimization
        beta = x(1:end-1); gamma = x(end);
        index = [1, 1+find(qj)];
        beta_long = zeros(K+1, 1);
        beta_long(index) = beta;
        
        tmp = 0;
        for a = 0:(2^K-1)
            eta = ftn_h([[1, binary(a, K)] * beta_long; gamma]);
            tmp = tmp + (eta)' * ftn_T(X(index_i,j)') * phi(index_i, a+1) - ftn_A((eta)')*sum(phi(index_i,a+1));
        end
        value = - tmp; % minus sign to convert to minimization
    end
end
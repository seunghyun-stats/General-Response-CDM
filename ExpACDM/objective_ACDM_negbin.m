function [f] = objective_ACDM_negbin(phi, X, j, qj, K)

f = @obj;
    function [value] = obj(x)
        beta = x(1:end-1); gamma = x(end);
        index = [1, 1+find(qj)];
        beta_long = zeros(K+1, 1);
        beta_long(index) = beta;
        
        tmp = 0;
        for a = 1:2^K
            r_j = [1, binary(a-1, K)] * beta_long * gamma / (1 - gamma);
            tmp = tmp + sum( phi(:,a) .* (log(nchoosek_prac(X(:,j) + r_j - 1, X(:,j))) ...
                + X(:,j) * log(1-gamma) + r_j * log(gamma)) );
        end
        value = - tmp; % minus sign to convert to minimization
    end
end
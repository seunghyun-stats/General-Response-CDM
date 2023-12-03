function [f] = objective_DINA_negbin_min(phi, Xj, ideal_resp_j, K)
% object of minimization per j
% solve via fzero
% Xj = X(:,j)

f = @obj;
    function [value] = obj(x) % object of minimization, r > 0
        r = x(1);
        p = x(2);
        % N = size(Xj, 1);
        % n_in = 2^K;
        
        value = - (phi * ideal_resp_j')' * (Xj*log(1 - p) + log(p)*r + log(gamma(Xj + r)/gamma(r)));
        % value = - sum(phi .* repmat(ideal_resp_j, N, 1) .* ( repmat(Xj, 1, n_in) * log(1 - p) + repmat(log(p)*r, N, n_in) + repmat(log(gamma(Xj+r)) - log(gamma(r)), 1, n_in)), 'all');
    end
end

% Xj = X(:,j); ideal_resp_j = ideal_resp(j,:);
% fff = objective_DINA_negbin_1(phi, Xj, ideal_resp_j, K)
% fff([4, 0.5])

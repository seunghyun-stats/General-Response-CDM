function [nu, mu_0, mu_1, sigma_0, sigma_1, loglik] = get_EM_DINA_lognormal_missing(X, Q, A_in, nu_in, mu_0_in, mu_1_in, sigma_0_in, sigma_1_in)

% mu_0_in : length J vector
% EM algorithm given Q

[J, K] = size(Q);
N = size(X, 1);
index = 1 - isnan(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%
n_in = size(A_in, 1); % 2^K
ideal_resp = prod(bsxfun(@power, reshape(A_in, [1 n_in K]), ...
    reshape(Q, [J 1 K])), 3); % ideal response matrix Gamma
% prod_X = prod(X, 2);
%%%%%%%%%%%%%%%%%%%%%%%%%%

nu = nu_in;
mu_0 = mu_0_in;
mu_1 = mu_1_in;
sigma_0 = sigma_0_in;
sigma_1 = sigma_1_in;

err = 1;
itera = 0;
loglik = 0;

% abs(err)
iter_indicator = (err > 5*1e-2 && itera < 1000);

while iter_indicator
    old_loglik = loglik;
    
    %% E-step
    phi = zeros(N, n_in);
    psi = zeros(n_in, 1);

    for i = 1:N
        for a = 1:2^K
            index_j = find(index(i,:));
            mu = log(X(i, index_j))' - mu_1(index_j) .* ideal_resp(index_j, a) - mu_0(index_j) .* (1 - ideal_resp(index_j, a));
            sigma = sigma_1(index_j) .* ideal_resp(index_j, a) + sigma_0(index_j) .* (1 - ideal_resp(index_j, a));
            phi(i, a) = exp( - sum(mu.^2 ./ (2*sigma.^2))) * nu(a);
        end
        tmp = sum(phi(i, :));
        phi(i,:) = phi(i,:)/tmp;
    end
    
    %% M-step
    % updating nu
    for a = 1:2^K
        psi(a) = sum(phi(:,a));
    end
    nu = psi / sum(psi);
    
%   supposedly faster code that results in NA's
%   alpha_rc =  repmat(nu', N, 1) .* exp(1 -((log(X) - repmat(mu_1', N, 1)).^2 ./ repmat(sigma_1', N, 1).^2 * ideal_resp + ...
%   (log(X) - repmat(mu_0', N, 1)).^2 ./ repmat(sigma_0', N, 1).^2 * (1 - ideal_resp) )/2);
%    alpha_rc = bsxfun(@rdivide, alpha_rc, sum(alpha_rc, 3));
%    nu = sum(bsxfun(@times, alpha_rc, prop_resp), 1);
    
    % updating item parameters
    for j = 1:J
        index_i = find(index(:,j));
        mu_1_denom = sum(phi(index_i,:) .* repmat(ideal_resp(j,:),size(index_i,1),1), 'all');
        mu_0_denom = sum(phi(index_i,:) .* repmat(1 - ideal_resp(j,:),size(index_i,1),1), 'all');
        mu_1_num = sum(phi(index_i,:) .* repmat(ideal_resp(j,:),size(index_i,1),1) .* repmat(log(X(index_i,j)),1,n_in), 'all');
        mu_0_num = sum(phi(index_i,:) .* repmat(1 - ideal_resp(j,:),size(index_i,1),1) .* repmat(log(X(index_i,j)),1,n_in), 'all');

        mu_1(j) = mu_1_num / mu_1_denom;
        mu_0(j) = mu_0_num / mu_0_denom;

        sigma_1_num = sum(phi(index_i,:) .* repmat(ideal_resp(j,:),size(index_i,1),1) .* repmat( (log(X(index_i,j)) ...
            - mu_1(j)).^2 ,1,n_in), 'all');
        sigma_0_num = sum(phi(index_i,:) .* repmat(1 - ideal_resp(j,:),size(index_i,1),1) .* repmat( (log(X(index_i,j)) ...
            - mu_0(j)).^2 ,1,n_in), 'all');

        sigma_1(j) = sqrt(sigma_1_num / mu_1_denom);
        sigma_0(j) = sqrt(sigma_0_num / mu_0_denom);
    end
    
    % update log-lik
    tmp = 0;
    exponent = zeros(n_in, 1);
    for i = 1:N
        index_j = find(index(i,:));
        for a = 1:n_in
            exponent(a) = sum(ideal_resp(index_j,a) .* ( (log(X(i,index_j))' - mu_1(index_j)) .^2 ./ (sigma_1(index_j).^2)/2 + log(sigma_1(index_j)) ) ...
                + (1 - ideal_resp(index_j,a)) .* ( (log(X(i,index_j))' - mu_0(index_j)) .^2 ./ (sigma_0(index_j).^2)/2 + log(sigma_0(index_j)) ) );
        end
        tmp = tmp + log(nu' * exp(-exponent));
    end
    loglik = tmp;
    err = (loglik - old_loglik);
    iter_indicator = (abs(err) > 5* 1e-2 && itera<1000); % abs

    itera = itera + 1;
    fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);
end

end
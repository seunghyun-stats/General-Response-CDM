function [nu, mu_0, mu_1, sigma_0, sigma_1, phi, loglik] = get_EM_DINA_lognormal(X, Q, A_in, nu_in, mu_0_in, mu_1_in, sigma_0_in, sigma_1_in)
%
% This EM algorithm estimates the lognormal-DINA parameters
% 
% @param X          : observed response matrix (N * J)
% @param Q          : Q-matrix (size J * K)
% @param A_in       : all possible latent configurations (size 2^K * K)
% @param nu_in      : proportion parameter initial value (len 2^K)
% @param mu_0_in    : normal mean item parameter initial values (len J)
% @param sigma_0_in : normal variance item parameter initial values (len J)


[J, K] = size(Q);

N = size(X, 1);
n_in = size(A_in, 1); % 2^K
ideal_resp = prod(bsxfun(@power, reshape(A_in, [1 n_in K]), ...
    reshape(Q, [J 1 K])), 3); % ideal response matrix Gamma

nu = nu_in;
mu_0 = mu_0_in;
mu_1 = mu_1_in;
sigma_0 = sigma_0_in;
sigma_1 = sigma_1_in;

err = 1;
itera = 0;
loglik = 0;

iter_indicator = (abs(err) > 5*1e-2 && itera < 1000);

while iter_indicator
    old_loglik = loglik;

    %% E-step
    phi = zeros(N, n_in);
    psi = zeros(n_in, 1);

    for i = 1:N
        for a = 1:2^K
            mu = log(X(i,:))' - mu_1 .* ideal_resp(:, a) - mu_0 .* (1 - ideal_resp(:, a));
            sigma = sigma_1 .* ideal_resp(:, a) + sigma_0 .* (1 - ideal_resp(:, a));
            phi(i, a) = exp( - sum(mu.^2 ./ (sigma.^2) / 2)) * nu(a);
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
    
    % updating item parameters
    for j = 1:J
        mu_1_denom = ideal_resp(j,:)*psi;
        mu_0_denom = (1-ideal_resp(j,:))*psi;
        mu_1_num = sum(phi .* repmat(ideal_resp(j,:),N,1) .* repmat(log(X(:,j)),1,n_in), 'all');
        mu_0_num = sum(phi .* repmat(1 - ideal_resp(j,:),N,1) .* repmat(log(X(:,j)),1,n_in), 'all');

        mu_1(j) = mu_1_num / mu_1_denom;
        mu_0(j) = mu_0_num / mu_0_denom;

        sigma_1_num = sum(phi .* repmat(ideal_resp(j,:),N,1) .* repmat( (log(X(:,j)) ...
            - mu_1(j)).^2 ,1,n_in), 'all');
        sigma_0_num = sum(phi .* repmat(1 - ideal_resp(j,:),N,1) .* repmat( (log(X(:,j)) ...
            - mu_0(j)).^2 ,1,n_in), 'all');

        sigma_1(j) = sigma_1_num / mu_1_denom;
        sigma_0(j) = sigma_0_num / mu_0_denom;
    end
    
    % updating log-lik
    tmp = 0;
    exponent = zeros(n_in, 1);
    for i = 1:N
        for a = 1:n_in
            exponent(a) = exp(- sum(ideal_resp(:,a) .* ( (log(X(i,:))' - mu_1) .^2 ./ (sigma_1.^2)/2 + log(sigma_1) ) ...
                + (1 - ideal_resp(:,a)) .* ( (log(X(i,:))' - mu_0) .^2 ./ (sigma_0.^2)/2 + log(sigma_0) ) ) );
        end
        tmp = tmp + log(nu' * exponent);
    end
    loglik = tmp;
    err = (loglik - old_loglik);
    iter_indicator = (abs(err) > 5* 1e-2 && itera<1000);

    itera = itera + 1;
    % fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);
end

end
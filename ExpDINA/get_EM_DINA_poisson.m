function [nu, lam_0, lam_1, phi, loglik] = get_EM_DINA_poisson(X, Q, A_in, nu_in, lam_0_in, lam_1_in)
%
% This EM algorithm estimates the Poisson-DINA parameters
% 
% @param X          : observed response matrix (N * J)
% @param Q          : Q-matrix (size J * K)
% @param A_in       : all possible latent configurations (size 2^K * K)
% @param nu_in      : proportion parameter initial value (len 2^K)
% @param lam_0_in   : item parameter initial value (len J)


[J, K] = size(Q);
N = size(X, 1);

n_in = size(A_in, 1); % 2^K
ideal_resp = prod(bsxfun(@power, reshape(A_in, [1 n_in K]), ...
    reshape(Q, [J 1 K])), 3); % ideal response matrix Gamma

nu = nu_in;
lam_0 = lam_0_in;
lam_1 = lam_1_in;

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
            lam = lam_1 .* ideal_resp(:, a) + lam_0 .* (1 - ideal_resp(:, a));
            phi(i, a) = exp( - sum(lam)) * prod(lam.^(X(i,:)')) * nu(a);
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
        num_1 = 0; num_0 = 0;
        for a = 1:n_in
            num_1 = num_1 + sum(phi(:,a) .* X(:,j)) * ideal_resp(j,a);
            num_0 = num_0 + sum(phi(:,a) .* X(:,j)) * (1-ideal_resp(j,a));
        end
        denom_1 = ideal_resp(j,:)*psi;
        denom_0 = (1-ideal_resp(j,:))*psi;
        lam_1(j) = num_1 / denom_1;
        lam_0(j) = num_0 / denom_0;
    end
    
    % updating log-lik
    tmp = 0;
    exponent = zeros(n_in, 1);
    for i = 1:N
        for a = 1:n_in
            exponent(a) = prod(ideal_resp(:,a) .* ( exp(-lam_1) .* lam_1 .^ (X(i,:)') ) ...
                + (1 - ideal_resp(:,a)) .* ( exp(-lam_0) .* lam_0 .^(X(i,:)')) );
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
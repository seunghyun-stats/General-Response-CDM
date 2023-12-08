function [nu, beta, gamma, loglik, phi] = get_EM_ACDM_negbin(X, Q, A_in, nu_in, beta_in, gamma_in)
%
% This EM algorithm estimates the nebgin-ACDM model parameters

[J, K] = size(Q);
N = size(X, 1);
n_in = size(A_in, 1);

nu = nu_in;
beta = beta_in;
gamma = gamma_in;
S = sum(Q, 2);

index = cell(J,1);
for j = 1:J
    index{j} = [1, 1+find(Q(j,:))];
end

err = 1;
itera = 0;
loglik = 0;
r = zeros(J,1);
iter_indicator = (abs(err) > 5*1e-2 && itera < 1000);

options = optimset('Display', 'off'); 

while iter_indicator
    %% E-step
    old_loglik = loglik;

    phi = zeros(N, n_in);
    psi = zeros(n_in, 1);
    for i = 1:N
        for a = 1:n_in
            r = (beta * [1, binary(a-1, K)]') .* gamma ./ (1 - gamma);
            tmp_j = prod(nchoosek_prac(X(i,:)'+r-1, X(i,:)') .* (1-gamma).^ (X(i,:)') .* gamma .^r );
            phi(i, a) = tmp_j * nu(a);
        end
        phi(i,:) = phi(i,:)/sum(phi(i, :));
    end
    
    %% M-step
    % updating nu
    for a = 1:n_in
        psi(a) = sum(phi(:,a));
    end
    nu = psi / sum(psi);
    
    % updating item parameters
    for j = 1:J
        f = objective_ACDM_negbin(phi, X, j, Q(j,:), K);
        opt = fmincon(f, [nonzeros(beta(j,:)); gamma(j)], [], [], [], [], ...
            0.4*ones(1, S(j) + 2), [10*ones(1, S(j) + 1), 1], [], options);
        beta_tmp = opt(1:end-1); gamma(j) = opt(end);
        beta(j,index{j}) = beta_tmp;
    end
    
    % update log-lik
    tmp = 0;
    exponent = zeros(n_in, 1);
    for i = 1:N
        for a = 1:n_in
            r = (beta * [1, binary(a-1, K)]') .* gamma ./ (1 - gamma);
            exponent(a) = prod(nchoosek_prac(X(i,:)'+r-1, X(i,:)') .* (1-gamma).^ (X(i,:)') .* gamma.^r );
        end
        tmp = tmp + log(nu' * exponent);
    end
    loglik = tmp;
    err = (loglik - old_loglik);
    iter_indicator = (abs(err) > 5* 1e-2 && itera<60);
    itera = itera + 1;
    fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);
end

end
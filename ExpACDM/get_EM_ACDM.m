function [nu, beta, gamma, loglik, itera] = get_EM_ACDM(X, Q, A_in, nu_in, beta_in, gamma_in)
%
% This EM algorithm estimates the ExpACDM model parameters
%
% For implementation, please use functions ftn_h, ftn_A, ftn_T
% corresponding to your parametric family
%
% Currently, the code optimization settings are set to the lognormal-ACDM
% (or normal-ACDM) model. To fit the Poisson-ACDM model, please use the
% optimization settings that are currently commented out.
% 
% @param X          : observed response matrix (N * J)
% @param Q          : Q-matrix (size J * K)
% @param A_in       : all possible latent configurations (size 2^K * K)
% @param nu_in      : proportion parameter initial value (len 2^K)
% @param beta_in    : beta initial value (size J * K+1)
% @param gamma_in   : dispersion parameter initial value (len J)

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

iter_indicator = (abs(err) > 5*1e-2 && itera < 1000);
options = optimset('Display', 'off'); 

% optimization constraints (lower, upper bound) for [beta(j,:), gamma(j)]
% this may vary per each exponential family
lb = cell(J,1); ub = cell(J,1); 

% constraints for lognormal
for j = 1:J
    lb{j} = [-2, zeros(1, S(j) + 1)];
    ub{j} = [4*ones(1, S(j) + 1), 2];
end

% constraints for Poisson
% for j = 1:J
%     lb{j} = zeros(1, S(j) + 1);
%     ub{j} = 3*ones(1, S(j) + 1);
% end

while iter_indicator
    %% E-step
    old_loglik = loglik;
    
    phi = zeros(N, n_in);
    psi = zeros(n_in, 1);
    exponent = zeros(n_in, 1);
    for i = 1:N
        for a = 1:n_in
            eta = ftn_h([[1, binary(a-1, K)]*beta'; gamma'])';
            exponent(a) = (diag(eta * ftn_T(X(i,:))) - ftn_A(eta))' * ones(J, 1) ; 
        end
        phi(i,:) = exp(exponent) .* nu;
        tmp = sum(phi(i, :));
        phi(i,:) = phi(i,:)/tmp;
    end
    
    %% M-step
    % updating nu
    for a = 1:n_in
        psi(a) = sum(phi(:,a));
    end
    nu = psi / sum(psi);
    

    % updating item parameters
    % the code depends on whether the dispersion parameter is used or not

    for j = 1:J
        %% updates for the lognormal-ACDM
        f = objective_ACDM_lognormal(phi, X, j, Q(j,:), K);
        opt = fmincon(f, [nonzeros(beta(j,:)); gamma(j)]', [], [], [], [], ...
            lb{j}, ub{j}, [], options);
        beta_tmp = opt(1:end-1); gamma(j) = opt(end);
        beta(j,index{j}) = beta_tmp;

        %% updates for the Poisson-ACDM
        % f = objective_ACDM_poisson(phi, X, j, Q(j,:), K);
        % beta(j,index{j}) = fmincon(f, nonzeros(beta(j,:))', [], [], [], [], ...
        %     lb{j}, ub{j}, [], options);
    end
    
    % updating log-lik
    tmp = 0;
    exponent = zeros(n_in, 1);
    for i = 1:N
        for a = 1:n_in
            eta = ftn_h([[1, binary(a-1, K)]*beta'; gamma'])';
            exponent(a) = sum( diag( eta * ftn_T(X(i,:))) - ftn_A(eta) );
        end
        tmp = tmp + log(nu' * exp(exponent));
    end
    loglik = tmp;
    err = (loglik - old_loglik);
    iter_indicator = (abs(err) > 5* 1e-2 && itera<1000);
    itera = itera + 1;
    fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);
end

end
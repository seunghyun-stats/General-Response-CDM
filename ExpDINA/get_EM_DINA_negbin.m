function [nu, r_0, r_1, p_0, p_1, phi, loglik] = get_EM_DINA_negbin(X, Q, A_in, nu_in, r_0_in, r_1_in, p_0_in, p_1_in)
%
% This EM algorithm estimates the negbin-DINA parameters
% 
% @param X          : observed response matrix (N * J)
% @param Q          : Q-matrix (size J * K)
% @param A_in       : all possible latent configurations (size 2^K * K)
% @param nu_in      : proportion parameter initial value (len 2^K)
% @param r_0_in     : negbin item parameter initial values (len J)
% @param p_0_in     : negbin item parameter initial values (len J)


[J, K] = size(Q);

N = size(X, 1);
n_in = size(A_in, 1); % 2^K
ideal_resp = prod(bsxfun(@power, reshape(A_in, [1 n_in K]), ...
    reshape(Q, [J 1 K])), 3); % ideal response matrix Gamma

nu = nu_in;
r_0 = r_0_in;
r_1 = r_1_in;
p_0 = p_0_in;
p_1 = p_1_in;

err = 1;
itera = 0;
loglik = 0;

iter_indicator = (abs(err) > 5*1e-2 && itera < 1000);
options = optimset('Display', 'off'); 

while iter_indicator
    old_loglik = loglik;
    
    %% E-step
    phi = zeros(N, n_in);
    psi = zeros(n_in, 1);

    for i = 1:N
        for a = 1:2^K
            r = r_1 .* ideal_resp(:, a) + r_0 .* (1 - ideal_resp(:, a));
            p = p_1 .* ideal_resp(:, a) + p_0 .* (1 - ideal_resp(:, a));
            phi(i, a) = prod(nchoosek_prac(X(i,:)' + r - 1, X(i,:)')) * ...
                prod(p .^ r) * prod((1-p) .^ (X(i,:)')) * nu(a);
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
        f = objective_DINA_negbin_min(phi, X(:,j), ideal_resp(j,:), K);
        opt = fmincon(f, [3, 0.5], [], [], [], [], [1, 0.05], [6, 0.95], [], options);
        r_1(j) = opt(1); p_1(j) = opt(2);

        g = objective_DINA_negbin_min(phi, X(:,j), 1 - ideal_resp(j,:), K);
        opt = fmincon(g, [1, 0.5], [], [], [], [], [0.1, 0.05], [3, 0.95], [], options);
        r_0(j) = opt(1); p_0(j) = opt(2);
    end

    % updating log-lik
    tmp = 0;
    exponent = zeros(n_in, 1);
    for i = 1:N
        for a = 1:n_in
            r = r_1 .* ideal_resp(:, a) + r_0 .* (1 - ideal_resp(:, a));
            p = p_1 .* ideal_resp(:, a) + p_0 .* (1 - ideal_resp(:, a));
            exponent(a) = prod(nchoosek_prac(X(i,:)' + r - 1, X(i,:)')) * ...
                prod(p .^ r) * prod((1-p) .^ (X(i,:)'));
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
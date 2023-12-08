function [X, X_A] = generate_X_ACDM_poisson(N, nu_true, delta_true, Q, A)
%
% This function generates data (X, A) from the lognormal-ACDM model
% 
% @param N          : sample size
% @param nu_true    : proportion parameter vector (len 2^K)
% @param delta_true : item parameters beta (J * K+1 coefficient matrix)
% @param Q          : Q-matrix (size J * K)
% @param A          : Q-matrix (size J * K)

[~, K] = size(Q);

% generate multinomial counts
counts = mnrnd(N, nu_true);
X_A = zeros(N, 1);
n = 1;

% A stores empirical CDF for categories: 1, ..., n_in
for a = 1:length(nu_true)
    X_A(n:(n+counts(a)-1), 1:K) = repmat(A(a, 1:K), counts(a), 1);
    n = n+counts(a);
end
X_A = X_A(randperm(N), 1:K);

% generate ACDM parameters
mu_correct = [ones(N, 1), X_A] * delta_true';

% generate responses
X = poissrnd(mu_correct);

end
function [X, X_A] = generate_X_ACDM_negbin(N, nu_true, delta_true, Q, A, p)
%
% This function generates data (X, A) from the negbin-ACDM model
% 
% @param N          : sample size
% @param nu_true    : proportion parameter vector (len 2^K)
% @param delta_true : item parameters beta (J * K+1 coefficient matrix)
% @param Q          : Q-matrix (size J * K)

[J, K] = size(Q);

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
X = zeros(N,J);
for i = 1:N
    for j = 1:J
        X(i,j) = nbinrnd(mu_correct(i,j)*p(j)/(1-p(j)), p(j));
    end
end
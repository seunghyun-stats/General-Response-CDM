function [X, X_A] = generate_X_ACDM_normal(N, nu_true, delta_true, Q, A, sigma)

[J, K] = size(Q);

counts = mnrnd(N, nu_true);
X_A = zeros(N, 1);
n = 1;
for a = 1:length(nu_true)
    X_A(n:(n+counts(a)-1), 1:K) = repmat(A(a, 1:K), counts(a), 1);
    n = n+counts(a);
end
X_A = X_A(randperm(N), 1:K);

mu_correct = [ones(N, 1), X_A] * delta_true';

s = repmat(sigma', N, 1);
X = mu_correct + normrnd(0, s, N, J);

end
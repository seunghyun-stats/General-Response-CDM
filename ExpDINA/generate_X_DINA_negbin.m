function [X, A] = generate_X_DINA_negbin(N, nu_true, r_0, r_1, p_0, p_1, Q)
%
% This function generates data (X, A) from the negbin-DINA model
% 
% @param N          : sample size
% @param nu_true    : proportion parameter vector (len 2^K)
% @param r_0, 1     : negbin integer item parameter vector (len J)
% @param p_0, 1     : negbin proportion item parameter vector (len J)
% @param Q          : Q-matrix (size J * K)

[J, K] = size(Q);
counts = mnrnd(N, nu_true);
A = zeros(N, 1);
n = 1;

for a = 1:length(nu_true)
    A(n:(n+counts(a)-1)) = a-1;
    n = n+counts(a);
end

A = A(randperm(N));

% convert each number in A (ranging from 0 to 2^K-1) to binary form, size N by K
% each row of A stores the attribute profile for each individual
A = binary(A, K);

% ideal response matrix, size N by J
xi = prod(bsxfun(@power, reshape(A, [N 1 K]), reshape(Q, [1 J K])), 3);

X_0 = zeros(N,J);
X_1 = zeros(N,J);
for j = 1:J
    X_0(:,j) = nbinrnd(r_0(j), p_0(j), N, 1);
    X_1(:,j) = nbinrnd(r_1(j), p_1(j), N, 1);
end
X = X_0 .* (1 - xi) + X_1 .* xi;

end